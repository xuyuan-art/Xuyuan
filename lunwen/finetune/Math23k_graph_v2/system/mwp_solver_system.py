#!/usr/bin/env python3
# coding: utf-8

import argparse
import copy
import json
import os
import pickle
import re
import sys
import threading
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.expressions_transfer import compute_prefix_expression, out_expression_list
from src.models import GenerateNode, GraphFusionEncoder as V2GraphFusionEncoder, Merge, Prediction
from src.pre_data import (
    _safe_change_num,
    DEFAULT_ZH_BERT_PATH,
    get_single_example_graph,
    load_raw_data,
    prepare_data_23k_graph,
    transfer_num,
)
import src.train_and_evaluate as train_eval


class LegacyRelationGraphModule(nn.Module):
    def __init__(self, hidden_size: int, relation_num: int = 5):
        super().__init__()
        self.relation_num = relation_num
        self.relation_linears = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(relation_num)])
        self.out_linear = nn.Linear(hidden_size * relation_num, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, node_embeddings: torch.Tensor, relation_graph: torch.Tensor) -> torch.Tensor:
        relation_graph = relation_graph.float()
        outputs = []
        for ridx, linear in enumerate(self.relation_linears):
            adj = relation_graph[:, ridx]
            degree = adj.sum(-1, keepdim=True).clamp(min=1.0)
            agg = torch.bmm(adj, node_embeddings) / degree
            outputs.append(torch.relu(linear(agg)))
        merged = torch.cat(outputs, dim=-1)
        merged = self.out_linear(merged)
        merged = self.dropout(merged)
        return self.norm(node_embeddings + merged)


def collate_bert_encodings(bert_encoding):
    if isinstance(bert_encoding, Mapping):
        return bert_encoding
    length_max = max(item["input_ids"].squeeze().size(0) for item in bert_encoding)
    input_ids = []
    attention_mask = []
    for item in bert_encoding:
        input_id = item["input_ids"].squeeze()
        mask = item["attention_mask"].squeeze()
        zeros = torch.zeros(length_max - input_id.size(0))
        input_ids.append(torch.cat([input_id.long(), zeros.long()]))
        attention_mask.append(torch.cat([mask.long(), zeros.long()]))
    return {"input_ids": torch.stack(input_ids, dim=0).long(), "attention_mask": torch.stack(attention_mask, dim=0).long()}


def _prepare_bert_inputs(bert_encoding, device):
    batched = collate_bert_encodings(bert_encoding)
    input_ids = batched["input_ids"].long().to(device)
    attention_mask = batched["attention_mask"].long().to(device)
    return input_ids, attention_mask


def get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size, hidden_size):
    device = encoder_outputs.device
    indices = []
    sen_len = encoder_outputs.size(0)
    masked_index = []
    temp_1 = [1 for _ in range(hidden_size)]
    temp_0 = [0 for _ in range(hidden_size)]
    for b in range(batch_size):
        for i in num_pos[b]:
            indices.append(i + b * sen_len)
            masked_index.append(temp_0)
        indices += [0 for _ in range(len(num_pos[b]), num_size)]
        masked_index += [temp_1 for _ in range(len(num_pos[b]), num_size)]
    indices = torch.LongTensor(indices).to(device)
    masked_index = torch.ByteTensor(masked_index).view(batch_size, num_size, hidden_size).to(device)
    all_outputs = encoder_outputs.transpose(0, 1).contiguous()
    all_embedding = all_outputs.view(-1, encoder_outputs.size(2))
    all_num = all_embedding.index_select(0, indices)
    all_num = all_num.view(batch_size, num_size, hidden_size)
    return all_num.masked_fill_(masked_index.bool(), 0.0)


class LegacyGraphFusionEncoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=2, dropout=0.5, relation_num=5):
        super().__init__()
        self.hidden_size = hidden_size
        self.bert_rnn = BertModel.from_pretrained(DEFAULT_ZH_BERT_PATH)
        self.graph_module = LegacyRelationGraphModule(hidden_size, relation_num=relation_num)
        self.fusion_gate = nn.Linear(hidden_size * 2, hidden_size)

    def _fuse_numbers_back(self, bert_output, enhanced_numbers, num_pos):
        seq_len = bert_output.size(0)
        batch_size = bert_output.size(1)
        hidden_size = bert_output.size(2)
        batch_outputs = []
        for bidx in range(batch_size):
            token_output = bert_output[:, bidx, :]
            valid_positions = [p for p in num_pos[bidx] if 0 <= p < seq_len]
            node_count = min(len(valid_positions), enhanced_numbers.size(1))
            if node_count == 0:
                batch_outputs.append(token_output)
                continue
            pos_tensor = torch.tensor(valid_positions[:node_count], dtype=torch.long, device=bert_output.device)
            bert_slice = token_output.index_select(0, pos_tensor)
            graph_slice = enhanced_numbers[bidx, :node_count, :]
            gate = torch.sigmoid(self.fusion_gate(torch.cat([bert_slice, graph_slice], dim=-1)))
            fused_slice = gate * graph_slice + (1 - gate) * bert_slice
            token_output = token_output.scatter(0, pos_tensor.unsqueeze(1).expand(-1, hidden_size), fused_slice)
            batch_outputs.append(token_output)
        return torch.stack(batch_outputs, dim=1)

    def forward(self, input_seqs, input_lengths, bert_encoding, batch_graph=None, num_pos=None, hidden=None):
        device = next(self.bert_rnn.parameters()).device
        input_ids, attention_mask = _prepare_bert_inputs(bert_encoding, device)
        bert_output = self.bert_rnn(input_ids, attention_mask=attention_mask)[0].transpose(0, 1)
        if batch_graph is None or num_pos is None:
            problem_output = bert_output.mean(0)
            return bert_output, problem_output

        if not torch.is_tensor(batch_graph):
            batch_graph = torch.tensor(batch_graph, dtype=torch.float32, device=device)
        else:
            batch_graph = batch_graph.to(device=device, dtype=torch.float32)

        batch_size = len(num_pos)
        num_size = batch_graph.size(-1)
        num_encoder_outputs = get_all_number_encoder_outputs(bert_output, num_pos, batch_size, num_size, self.hidden_size)
        enhanced_numbers = self.graph_module(num_encoder_outputs, batch_graph)
        fused_output = self._fuse_numbers_back(bert_output, enhanced_numbers, num_pos)
        problem_output = fused_output.mean(0)
        return fused_output, problem_output

    def evaluate(self, input_seqs, input_lengths, bert_encoding, batch_graph=None, num_pos=None):
        return self.forward(input_seqs, input_lengths, bert_encoding, batch_graph=batch_graph, num_pos=num_pos, hidden=None)


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir_path(path: str):
    if path.endswith(os.sep):
        return path
    return path + os.sep


def get_train_test_fold(ori_path, prefix, data, pairs, group):
    train_path = os.path.join(ori_path, "train" + prefix)
    valid_path = os.path.join(ori_path, "valid" + prefix)
    test_path = os.path.join(ori_path, "test" + prefix)
    train = read_json(train_path)
    valid = read_json(valid_path)
    test = read_json(test_path)
    train_id = [item["id"] for item in train]
    valid_id = [item["id"] for item in valid]
    test_id = [item["id"] for item in test]
    train_fold = []
    valid_fold = []
    test_fold = []
    for item, pair, g in zip(data, pairs, group):
        pair = list(pair)
        pair.append(g["group_num"])
        pair = tuple(pair)
        if item["id"] in train_id:
            train_fold.append(pair)
        elif item["id"] in test_id:
            test_fold.append(pair)
        else:
            valid_fold.append(pair)
    return train_fold, test_fold, valid_fold


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    result = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            result[key[7:]] = value
        else:
            result[key] = value
    return result


def _find_model_bundle(model_dir: str, epoch: Optional[int] = None) -> str:
    required = ["encoder_graph", "predict_graph", "generate_graph", "merge_graph"]
    if all(os.path.isfile(os.path.join(model_dir, name)) for name in required):
        return model_dir
    ckpt_root = os.path.join(model_dir, "epoch_checkpoints")
    if not os.path.isdir(ckpt_root):
        raise FileNotFoundError("No model bundle found in %s" % model_dir)
    if epoch is not None:
        candidate = os.path.join(ckpt_root, "epoch_%03d" % epoch)
        if all(os.path.isfile(os.path.join(candidate, name)) for name in required):
            return candidate
        raise FileNotFoundError("Checkpoint epoch_%03d not found under %s" % (epoch, ckpt_root))
    epoch_dirs = sorted(
        [d for d in os.listdir(ckpt_root) if re.fullmatch(r"epoch_\d{3}", d)],
        key=lambda x: int(x.split("_")[1]),
    )
    if not epoch_dirs:
        raise FileNotFoundError("No epoch checkpoints in %s" % ckpt_root)
    candidate = os.path.join(ckpt_root, epoch_dirs[-1])
    if all(os.path.isfile(os.path.join(candidate, name)) for name in required):
        return candidate
    raise FileNotFoundError("Latest checkpoint bundle is incomplete: %s" % candidate)


def _detect_encoder_layout(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Union[int, str]]:
    v2_pattern = re.compile(r"^graph_module\.layers\.(\d+)\.relation_linears\.(\d+)\.weight$")
    legacy_pattern = re.compile(r"^graph_module\.relation_linears\.(\d+)\.weight$")

    v2_layer_ids = set()
    v2_relation_ids = set()
    legacy_relation_ids = set()
    for key in state_dict.keys():
        match_v2 = v2_pattern.match(key)
        if match_v2:
            v2_layer_ids.add(int(match_v2.group(1)))
            v2_relation_ids.add(int(match_v2.group(2)))
            continue
        match_legacy = legacy_pattern.match(key)
        if match_legacy:
            legacy_relation_ids.add(int(match_legacy.group(1)))

    if v2_layer_ids:
        return {
            "arch": "v2",
            "graph_layers": max(v2_layer_ids) + 1,
            "relation_num": max(v2_relation_ids) + 1 if v2_relation_ids else 5,
        }
    if legacy_relation_ids:
        return {"arch": "legacy", "graph_layers": 1, "relation_num": max(legacy_relation_ids) + 1}
    return {"arch": "legacy", "graph_layers": 1, "relation_num": 5}


class Math23kSolverSystem:
    NUM_PATTERN = re.compile(r"\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    INLINE_FRACTION_PATTERN = re.compile(r"(?<![\d(])(\d{1,3}\s*/\s*\d{1,3})(?![\d/)])")
    CN_FRACTION_PATTERN = re.compile(r"([零〇一二两三四五六七八九十百千万\d]+)\s*分之\s*([零〇一二两三四五六七八九十百千万\d]+)")
    CN_DIGIT_MAP = {
        "零": 0,
        "〇": 0,
        "一": 1,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    CN_UNIT_MAP = {"十": 10, "百": 100, "千": 1000, "万": 10000}
    OPERATOR_DISPLAY = {"+": "+", "-": "-", "*": "×", "/": "÷", "^": "^"}
    RELATION_DISPLAY = {
        1: ("greater_than", "大于"),
        2: ("less_than", "小于"),
        4: ("adjacent", "相邻"),
    }

    def __init__(
        self,
        data_dir: str,
        model_dir: str,
        epoch: Optional[int] = None,
        device: str = "cpu",
        beam_size: int = 5,
        hidden_size: int = 768,
        embedding_size: int = 128,
        relation_num: int = 5,
        meta_cache_path: Optional[str] = None,
    ):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.epoch = epoch
        self.beam_size = beam_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.relation_num = relation_num
        self.device = torch.device(device)
        self.encoder_arch = "unknown"
        self.encoder_graph_layers = 0
        self.lock = threading.Lock()
        train_eval.USE_CUDA = self.device.type == "cuda"

        self.tokenizer = BertTokenizer.from_pretrained(DEFAULT_ZH_BERT_PATH)
        self._build_runtime_meta(meta_cache_path=meta_cache_path)
        self._load_models()

    @classmethod
    def _normalize_user_text(cls, text: str) -> str:
        normalized = text.replace("（", "(").replace("）", ")").replace("／", "/")

        def _replace_cn_fraction(match: re.Match) -> str:
            den_raw = match.group(1)
            num_raw = match.group(2)
            den = cls._parse_cn_number(den_raw)
            num = cls._parse_cn_number(num_raw)
            if den is None or num is None or den == 0:
                return match.group(0)
            return f"({num}/{den})"

        def _wrap_fraction(match: re.Match) -> str:
            frac = match.group(1).replace(" ", "")
            return f"({frac})"

        normalized = cls.CN_FRACTION_PATTERN.sub(_replace_cn_fraction, normalized)
        normalized = cls.INLINE_FRACTION_PATTERN.sub(_wrap_fraction, normalized)
        return normalized

    @classmethod
    def _parse_cn_number(cls, raw: str) -> Optional[int]:
        token = raw.strip()
        if not token:
            return None
        if token.isdigit():
            return int(token)
        token = token.replace("两", "二")
        if token in cls.CN_DIGIT_MAP:
            return cls.CN_DIGIT_MAP[token]
        if all(ch in cls.CN_DIGIT_MAP for ch in token):
            return int("".join(str(cls.CN_DIGIT_MAP[ch]) for ch in token))

        total = 0
        section = 0
        number = 0
        used = False
        for ch in token:
            if ch in cls.CN_DIGIT_MAP:
                number = cls.CN_DIGIT_MAP[ch]
                used = True
                continue
            unit = cls.CN_UNIT_MAP.get(ch)
            if unit is None:
                return None
            used = True
            if unit == 10000:
                section += number
                if section == 0:
                    section = 1
                total += section * unit
                section = 0
                number = 0
            else:
                if number == 0:
                    number = 1
                section += number * unit
                number = 0
        if not used:
            return None
        return total + section + number

    @classmethod
    def _normalize_expression_token(cls, token: str) -> str:
        t = str(token).strip().replace("（", "(").replace("）", ")").replace("／", "/")
        t = re.sub(r"\s+", "", t)
        mixed = re.fullmatch(r"(\d+)\((\d+)/(\d+)\)", t)
        if mixed:
            return f"({mixed.group(1)} + {mixed.group(2)}/{mixed.group(3)})"
        mixed = re.fullmatch(r"(\d+)\(\((\d+)\)/\((\d+)\)\)", t)
        if mixed:
            return f"({mixed.group(1)} + {mixed.group(2)}/{mixed.group(3)})"
        return t

    @classmethod
    def _strip_outer_brackets(cls, expr: str) -> str:
        text = expr.strip()
        while text.startswith("(") and text.endswith(")"):
            depth = 0
            wrapped = True
            for idx, ch in enumerate(text):
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth < 0:
                        wrapped = False
                        break
                if depth == 0 and idx < len(text) - 1:
                    wrapped = False
                    break
            if not wrapped or depth != 0:
                break
            text = text[1:-1].strip()
        return text

    @classmethod
    def prefix_tokens_to_infix_text(cls, prefix_tokens: Optional[List[str]]) -> Optional[str]:
        if not prefix_tokens:
            return None
        tokens = [str(x).strip() for x in prefix_tokens if str(x).strip()]
        if not tokens:
            return None

        def _parse(index: int):
            if index >= len(tokens):
                return None, index
            tk = tokens[index]
            if tk in cls.OPERATOR_DISPLAY:
                left_expr, next_index = _parse(index + 1)
                if left_expr is None:
                    return None, next_index
                right_expr, next_index = _parse(next_index)
                if right_expr is None:
                    return None, next_index
                op = cls.OPERATOR_DISPLAY[tk]
                return f"({left_expr} {op} {right_expr})", next_index
            return cls._normalize_expression_token(tk), index + 1

        expression, consumed = _parse(0)
        if expression is None or consumed != len(tokens):
            return None
        return cls._strip_outer_brackets(expression)

    @classmethod
    def _parse_numeric_token(cls, token: str) -> Optional[float]:
        raw = str(token).strip()
        if not raw:
            return None
        norm = raw.replace("（", "(").replace("）", ")").replace("／", "/")
        norm = re.sub(r"\s+", "", norm)
        try:
            if norm.endswith("%"):
                base = cls._parse_numeric_token(norm[:-1])
                if base is None:
                    return None
                return float(base) / 100.0

            mixed = re.fullmatch(r"([+-]?\d+)\((\d+)/(\d+)\)", norm)
            if mixed:
                whole = float(mixed.group(1))
                num = float(mixed.group(2))
                den = float(mixed.group(3))
                if den == 0:
                    return None
                sign = -1.0 if whole < 0 else 1.0
                return whole + sign * (num / den)

            frac = re.fullmatch(r"\(?([+-]?\d+(?:\.\d+)?)\s*/\s*([+-]?\d+(?:\.\d+)?)\)?", norm)
            if frac:
                num = float(frac.group(1))
                den = float(frac.group(2))
                if den == 0:
                    return None
                return num / den

            return float(norm)
        except Exception:
            return None

    @classmethod
    def _format_numeric_value(cls, value: Optional[float]) -> Optional[str]:
        if value is None:
            return None
        try:
            fval = float(value)
        except Exception:
            return None
        if not np.isfinite(fval):
            return None
        if abs(fval - round(fval)) < 1e-9:
            return str(int(round(fval)))
        text = f"{fval:.10f}".rstrip("0").rstrip(".")
        return text if text else "0"

    @classmethod
    def _apply_operator(cls, operator: str, left: Optional[float], right: Optional[float]) -> Optional[float]:
        if left is None or right is None:
            return None
        try:
            lval = float(left)
            rval = float(right)
            if operator == "+":
                return lval + rval
            if operator == "-":
                return lval - rval
            if operator == "*":
                return lval * rval
            if operator == "/":
                if abs(rval) < 1e-12:
                    return None
                return lval / rval
            if operator == "^":
                return lval ** rval
            return None
        except Exception:
            return None

    @classmethod
    def _format_formula_operand(cls, expr: str) -> str:
        text = cls._strip_outer_brackets(str(expr).strip())
        if re.search(r"\s[+\-×÷^]\s", text):
            return f"({text})"
        return text

    @classmethod
    def _build_relation_edges(
        cls,
        numbers: List[str],
        graph: Optional[np.ndarray],
    ) -> List[Dict[str, object]]:
        if graph is None:
            return []
        arr = np.asarray(graph)
        if arr.ndim == 4:
            if arr.shape[0] == 0:
                return []
            arr = arr[0]
        if arr.ndim != 3 or arr.shape[0] < 5:
            return []

        num_count = min(len(numbers), int(arr.shape[1]), int(arr.shape[2]))
        if num_count <= 0:
            return []

        edges: List[Dict[str, object]] = []
        for ridx, (rtype, rlabel) in cls.RELATION_DISPLAY.items():
            if ridx >= arr.shape[0]:
                continue
            for i in range(num_count):
                for j in range(num_count):
                    if i == j:
                        continue
                    if arr[ridx, i, j] <= 0.5:
                        continue
                    if rtype == "adjacent" and j < i:
                        continue
                    left = str(numbers[i])
                    right = str(numbers[j])
                    if rtype == "greater_than":
                        desc = f"{left} > {right}"
                    elif rtype == "less_than":
                        desc = f"{left} < {right}"
                    else:
                        desc = f"{left} 与 {right} 在题干中相邻出现"
                    edges.append(
                        {
                            "type": rtype,
                            "label": rlabel,
                            "from_index": int(i),
                            "from_token": left,
                            "to_index": int(j),
                            "to_token": right,
                            "description": desc,
                        }
                    )
        return edges

    def _decode_output_id(self, out_id: int, numbers: Optional[List[str]] = None) -> Dict[str, object]:
        idx = int(out_id)
        if 0 <= idx < int(self.output_lang.n_words):
            vocab_token = str(self.output_lang.index2word[idx])
        else:
            vocab_token = f"ID_{idx}"

        resolved_token = vocab_token
        if vocab_token.startswith("N") and vocab_token[1:].isdigit():
            num_idx = int(vocab_token[1:])
            if numbers and 0 <= num_idx < len(numbers):
                resolved_token = str(numbers[num_idx])
        return {"id": idx, "vocab_token": vocab_token, "resolved_token": resolved_token}

    def _decode_output_ids(self, ids: List[int], numbers: Optional[List[str]] = None) -> List[Dict[str, object]]:
        return [self._decode_output_id(i, numbers=numbers) for i in ids]

    @torch.no_grad()
    def _evaluate_tree_with_trace(
        self,
        input_batch: List[int],
        input_length: int,
        num_pos: List[int],
        bert_input: Dict[str, torch.Tensor],
        batch_graph: Optional[np.ndarray],
        numbers: Optional[List[str]] = None,
        beam_size: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> Tuple[List[int], Dict[str, object]]:
        beam_size = int(beam_size if beam_size is not None else self.beam_size)
        max_length = int(max_length if max_length is not None else train_eval.MAX_OUTPUT_LENGTH)
        device = self.device

        seq_mask = torch.ByteTensor(1, input_length).fill_(0).to(device)
        input_var = torch.LongTensor(input_batch).unsqueeze(1).to(device)
        num_mask = torch.ByteTensor(1, len(num_pos) + len(self.generate_num_ids)).fill_(0).to(device)
        padding_hidden = torch.FloatTensor([0.0 for _ in range(self.predict.hidden_size)]).unsqueeze(0).to(device)

        self.encoder.eval()
        self.predict.eval()
        self.generate.eval()
        self.merge.eval()

        encoder_outputs, problem_output = self.encoder.evaluate(
            input_var, [input_length], bert_input, batch_graph=batch_graph, num_pos=[num_pos]
        )

        node_stacks = [[train_eval.TreeNode(_)] for _ in problem_output.split(1, dim=0)]
        batch_size = 1
        num_size = len(num_pos)
        all_nums_encoder_outputs = get_all_number_encoder_outputs(
            encoder_outputs, [num_pos], batch_size, num_size, self.encoder.hidden_size
        )
        num_start = self.output_lang.num_start
        embeddings_stacks = [[] for _ in range(batch_size)]
        left_childs = [None for _ in range(batch_size)]

        beams = [
            {
                "score": 0.0,
                "node_stack": train_eval.copy_list(node_stacks),
                "embedding_stack": train_eval.copy_list(embeddings_stacks),
                "left_childs": train_eval.copy_list(left_childs),
                "out": [],
            }
        ]

        trace_steps: List[Dict[str, object]] = []
        stopped_reason = "max_length_reached"

        for t in range(max_length):
            current_beams: List[Dict[str, object]] = []
            step_trace: Dict[str, object] = {
                "step_index": int(t + 1),
                "active_beam_count": 0,
            }
            topk_logged = False
            while len(beams) > 0:
                # `beams` is kept in descending score order, so pop(0) expands
                # the current best beam first and makes logged top-k consistent.
                b = beams.pop(0)
                if len(b["node_stack"][0]) == 0:
                    current_beams.append(b)
                    continue

                step_trace["active_beam_count"] = int(step_trace["active_beam_count"]) + 1
                num_score, op, current_embeddings, current_context, current_nums_embeddings = self.predict(
                    b["node_stack"],
                    b["left_childs"],
                    encoder_outputs,
                    all_nums_encoder_outputs,
                    padding_hidden,
                    seq_mask,
                    num_mask,
                )
                out_score = nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)
                topv, topi = out_score.topk(beam_size)

                if not topk_logged:
                    candidates = []
                    local_scores = topv.squeeze(0).tolist()
                    local_ids = topi.squeeze(0).tolist()
                    for score, out_id in zip(local_scores, local_ids):
                        token_meta = self._decode_output_id(int(out_id), numbers=numbers)
                        candidates.append(
                            {
                                "token": token_meta,
                                "local_logp": float(score),
                                "cum_logp": float(b["score"] + float(score)),
                            }
                        )
                    step_trace["best_beam_input_score"] = float(b["score"])
                    step_trace["topk_candidates_from_best_beam"] = candidates
                    topk_logged = True

                for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                    current_node_stack = train_eval.copy_list(b["node_stack"])
                    current_left_childs = []
                    current_embeddings_stacks = train_eval.copy_list(b["embedding_stack"])
                    current_out = copy.deepcopy(b["out"])

                    out_token = int(ti)
                    current_out.append(out_token)
                    current_node_stack[0].pop()

                    if out_token < num_start:
                        generate_input = torch.LongTensor([out_token]).to(device)
                        left_child, right_child, node_label = self.generate(
                            current_embeddings, generate_input, current_context
                        )
                        current_node_stack[0].append(train_eval.TreeNode(right_child))
                        current_node_stack[0].append(train_eval.TreeNode(left_child, left_flag=True))
                        current_embeddings_stacks[0].append(train_eval.TreeEmbedding(node_label[0].unsqueeze(0), False))
                    else:
                        current_num = current_nums_embeddings[0, out_token - num_start].unsqueeze(0)
                        while (
                            len(current_embeddings_stacks[0]) > 0
                            and current_embeddings_stacks[0][-1].terminal
                        ):
                            sub_stree = current_embeddings_stacks[0].pop()
                            op_node = current_embeddings_stacks[0].pop()
                            current_num = self.merge(op_node.embedding, sub_stree.embedding, current_num)
                        current_embeddings_stacks[0].append(train_eval.TreeEmbedding(current_num, True))

                    if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                        current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
                    else:
                        current_left_childs.append(None)

                    current_beams.append(
                        {
                            "score": float(b["score"] + float(tv)),
                            "node_stack": current_node_stack,
                            "embedding_stack": current_embeddings_stacks,
                            "left_childs": current_left_childs,
                            "out": current_out,
                        }
                    )

            beams = sorted(current_beams, key=lambda x: x["score"], reverse=True)[:beam_size]
            if not beams:
                stopped_reason = "no_beam_left"
                break

            best_out_ids = [int(x) for x in beams[0]["out"]]
            step_trace["best_beam_score"] = float(beams[0]["score"])
            step_trace["best_prefix_ids"] = best_out_ids
            step_trace["best_prefix_tokens"] = self._decode_output_ids(best_out_ids, numbers=numbers)
            if best_out_ids:
                step_trace["chosen_token"] = self._decode_output_id(best_out_ids[-1], numbers=numbers)
            trace_steps.append(step_trace)

            all_finished = True
            for b in beams:
                if len(b["node_stack"][0]) != 0:
                    all_finished = False
                    break
            if all_finished:
                stopped_reason = "all_beams_finished"
                break

        pred_ids = [int(x) for x in (beams[0]["out"] if beams else [])]
        trace = {
            "beam_size": int(beam_size),
            "max_length": int(max_length),
            "stopped_reason": stopped_reason,
            "steps": trace_steps,
            "predicted_ids": pred_ids,
            "predicted_tokens": self._decode_output_ids(pred_ids, numbers=numbers),
        }
        return pred_ids, trace

    @classmethod

    def build_explainability_payload(
        cls,
        prefix_tokens: Optional[List[str]],
        numbers: Optional[List[str]] = None,
        num_pos_bert: Optional[List[int]] = None,
        graph: Optional[np.ndarray] = None,
    ) -> Dict[str, object]:
        tokens = [str(tok).strip() for tok in (prefix_tokens or []) if str(tok).strip()]
        number_list = [str(item) for item in (numbers or [])]

        payload: Dict[str, object] = {
            "infix_expression": cls.prefix_tokens_to_infix_text(tokens),
            "summary": "",
            "number_nodes": [],
            "tree_nodes": [],
            "tree_edges": [],
            "steps": [],
            "relation_edges": [],
            "final_result": None,
            "final_result_text": None,
        }
        if not tokens:
            payload["summary"] = "表达式为空，无法生成推理步骤。"
            return payload

        number_nodes = []
        for idx, token in enumerate(number_list):
            item: Dict[str, object] = {
                "index": int(idx),
                "token": token,
                "value": cls._parse_numeric_token(token),
            }
            if num_pos_bert and idx < len(num_pos_bert):
                item["bert_pos"] = int(num_pos_bert[idx])
            number_nodes.append(item)
        payload["number_nodes"] = number_nodes

        operators = set(cls.OPERATOR_DISPLAY.keys())
        node_seq = 0
        node_map: Dict[str, Dict[str, object]] = {}
        tree_nodes: List[Dict[str, object]] = []
        tree_edges: List[Dict[str, object]] = []

        def _new_id(prefix: str) -> str:
            nonlocal node_seq
            node_seq += 1
            return f"{prefix}_{node_seq}"

        def _parse(index: int):
            if index >= len(tokens):
                return None, index
            tk = tokens[index]
            if tk in operators:
                left_id, next_idx = _parse(index + 1)
                if left_id is None:
                    return None, next_idx
                right_id, next_idx = _parse(next_idx)
                if right_id is None:
                    return None, next_idx
                node_id = _new_id("op")
                symbol = cls.OPERATOR_DISPLAY.get(tk, tk)
                node_map[node_id] = {
                    "id": node_id,
                    "kind": "operator",
                    "token": tk,
                    "symbol": symbol,
                    "left": left_id,
                    "right": right_id,
                }
                tree_nodes.append({"id": node_id, "kind": "operator", "text": symbol})
                tree_edges.append({"from": node_id, "to": left_id, "role": "left"})
                tree_edges.append({"from": node_id, "to": right_id, "role": "right"})
                return node_id, next_idx

            node_id = _new_id("num")
            disp = cls._strip_outer_brackets(cls._normalize_expression_token(tk))
            node_map[node_id] = {
                "id": node_id,
                "kind": "number",
                "token": tk,
                "text": disp,
                "value": cls._parse_numeric_token(tk),
            }
            tree_nodes.append({"id": node_id, "kind": "number", "text": disp, "value": node_map[node_id]["value"]})
            return node_id, index + 1

        root_id, consumed = _parse(0)
        if root_id is None or consumed != len(tokens):
            payload["summary"] = "表达式解析失败，无法生成推理步骤。"
            return payload

        steps: List[Dict[str, object]] = []
        step_idx = 0

        def _eval(node_id: str):
            nonlocal step_idx
            node = node_map[node_id]
            if node["kind"] == "number":
                return str(node["text"]), node["value"]

            left_expr, left_val = _eval(str(node["left"]))
            right_expr, right_val = _eval(str(node["right"]))
            operator = str(node["token"])
            symbol = str(node["symbol"])
            result_val = cls._apply_operator(operator, left_val, right_val)
            result_text = cls._format_numeric_value(result_val) or "无法计算"
            left_disp = cls._format_formula_operand(left_expr)
            right_disp = cls._format_formula_operand(right_expr)
            expr = cls._strip_outer_brackets(f"({left_disp} {symbol} {right_disp})")
            step_idx += 1
            steps.append(
                {
                    "step": int(step_idx),
                    "node_id": node_id,
                    "operator": operator,
                    "operator_symbol": symbol,
                    "left_expr": left_expr,
                    "right_expr": right_expr,
                    "left_value": left_val,
                    "right_value": right_val,
                    "result": result_val,
                    "result_text": result_text,
                    "formula": f"{left_disp} {symbol} {right_disp} = {result_text}",
                }
            )
            return expr, result_val

        _, final_result = _eval(root_id)

        relation_edges = cls._build_relation_edges(number_list, graph)
        payload["tree_nodes"] = tree_nodes
        payload["tree_edges"] = tree_edges
        payload["steps"] = steps
        payload["relation_edges"] = relation_edges
        payload["final_result"] = final_result
        payload["final_result_text"] = cls._format_numeric_value(final_result)

        summary_items = [f"识别到 {len(number_list)} 个数字"]
        summary_items.append(f"生成 {len(steps)} 步计算")
        if relation_edges:
            summary_items.append(f"提取 {len(relation_edges)} 条数字关系")
        payload["summary"] = "，".join(summary_items) + "。"
        return payload

    def _build_runtime_meta(self, meta_cache_path: Optional[str]):
        cache_path = meta_cache_path or os.path.join(self.model_dir, "runtime_meta_23k_graph.pkl")
        if os.path.isfile(cache_path):
            with open(cache_path, "rb") as f:
                meta = pickle.load(f)
            self.input_lang = meta["input_lang"]
            self.output_lang = meta["output_lang"]
            self.generate_nums = meta["generate_nums"]
            self.copy_nums = meta["copy_nums"]
            return

        math23k_path = os.path.join(self.data_dir, "Math_23K.json")
        math23k_processed_path = os.path.join(self.data_dir, "Math_23K_processed.json")
        ori_path = ensure_dir_path(self.data_dir)
        prefix = "23k_processed.json"

        data = load_raw_data(math23k_path)
        group_data = read_json(math23k_processed_path)
        pairs, generate_nums, copy_nums = transfer_num(data)
        pairs = [(p[0], train_eval.from_infix_to_prefix(p[1]), p[2], p[3]) for p in pairs]
        train_fold, test_fold, valid_fold = get_train_test_fold(ori_path, prefix, data, pairs, group_data)
        input_lang, output_lang, train_pairs, test_pairs = prepare_data_23k_graph(
            train_fold, test_fold, 5, generate_nums, copy_nums, tree=True
        )
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.generate_nums = generate_nums
        self.copy_nums = copy_nums

        with open(cache_path, "wb") as f:
            pickle.dump(
                {
                    "input_lang": self.input_lang,
                    "output_lang": self.output_lang,
                    "generate_nums": self.generate_nums,
                    "copy_nums": self.copy_nums,
                },
                f,
            )

    def _load_models(self):
        bundle_dir = _find_model_bundle(self.model_dir, self.epoch)
        self.bundle_dir = bundle_dir

        enc_sd = _strip_module_prefix(torch.load(os.path.join(bundle_dir, "encoder_graph"), map_location="cpu"))
        pred_sd = _strip_module_prefix(torch.load(os.path.join(bundle_dir, "predict_graph"), map_location="cpu"))
        gen_sd = _strip_module_prefix(torch.load(os.path.join(bundle_dir, "generate_graph"), map_location="cpu"))
        merge_sd = _strip_module_prefix(torch.load(os.path.join(bundle_dir, "merge_graph"), map_location="cpu"))

        layout = _detect_encoder_layout(enc_sd)
        self.encoder_arch = str(layout.get("arch", "legacy"))
        self.encoder_graph_layers = int(layout.get("graph_layers", 1))
        self.relation_num = int(layout.get("relation_num", self.relation_num))

        fusion_gate_w = enc_sd.get("fusion_gate.weight")
        if isinstance(fusion_gate_w, torch.Tensor) and fusion_gate_w.ndim == 2:
            self.hidden_size = int(fusion_gate_w.size(0))
        generate_embed_w = gen_sd.get("embeddings.weight")
        if isinstance(generate_embed_w, torch.Tensor) and generate_embed_w.ndim == 2:
            self.embedding_size = int(generate_embed_w.size(1))

        if self.encoder_arch == "v2":
            self.encoder = V2GraphFusionEncoder(
                input_size=self.input_lang.n_words,
                embedding_size=self.embedding_size,
                hidden_size=self.hidden_size,
                relation_num=self.relation_num,
                graph_layers=self.encoder_graph_layers,
                graph_dropout=0.1,
                graph_relation_dropout=0.0,
                active_relation_ids=list(range(self.relation_num)),
                disable_graph=False,
            ).to(self.device)
        else:
            self.encoder = LegacyGraphFusionEncoder(
                input_size=self.input_lang.n_words,
                embedding_size=self.embedding_size,
                hidden_size=self.hidden_size,
                relation_num=self.relation_num,
            ).to(self.device)

        self.predict = Prediction(
            hidden_size=self.hidden_size,
            op_nums=self.output_lang.n_words - self.copy_nums - 1 - len(self.generate_nums),
            input_size=len(self.generate_nums),
        ).to(self.device)
        self.generate = GenerateNode(
            hidden_size=self.hidden_size,
            op_nums=self.output_lang.n_words - self.copy_nums - 1 - len(self.generate_nums),
            embedding_size=self.embedding_size,
        ).to(self.device)
        self.merge = Merge(hidden_size=self.hidden_size, embedding_size=self.embedding_size).to(self.device)

        self.encoder.load_state_dict(enc_sd, strict=True)
        self.predict.load_state_dict(pred_sd, strict=True)
        self.generate.load_state_dict(gen_sd, strict=True)
        self.merge.load_state_dict(merge_sd, strict=True)

        self.encoder.eval()
        self.predict.eval()
        self.generate.eval()
        self.merge.eval()
        self.generate_num_ids = [self.output_lang.word2index[num] for num in self.generate_nums]

    def _text_to_graph_inputs(
        self,
        text: str,
    ) -> Tuple[List[str], List[str], List[int], Dict[str, torch.Tensor], np.ndarray, Dict[str, object]]:
        raw_text = text.strip()
        normalized_text = self._normalize_user_text(raw_text)
        if not normalized_text:
            raise ValueError("text is empty")

        nums = []
        tokens = []
        matches = []
        last = 0
        for match in self.NUM_PATTERN.finditer(normalized_text):
            start, end = match.span()
            chunk = normalized_text[last:start]
            for ch in chunk:
                if ch.isspace():
                    continue
                tokens.append(ch)
            nums.append(match.group())
            matches.append({"text": match.group(), "start": int(start), "end": int(end)})
            tokens.append("NUM")
            last = end
        tail = normalized_text[last:]
        for ch in tail:
            if ch.isspace():
                continue
            tokens.append(ch)
        if not nums:
            raise ValueError("no numbers detected in text")

        bert_tokens = ["n" if t == "NUM" else t for t in tokens]
        bert_input = self.tokenizer(bert_tokens, is_split_into_words=True, return_tensors="pt", add_special_tokens=False)
        num_pos = []
        for idx, token_id in enumerate(bert_input["input_ids"].squeeze()):
            if self.tokenizer.convert_ids_to_tokens(int(token_id)) == "n":
                num_pos.append(idx)

        if len(num_pos) != len(nums):
            aligned = min(len(num_pos), len(nums))
            num_pos = num_pos[:aligned]
            nums = nums[:aligned]
            matches = matches[:aligned]
            if not nums:
                raise ValueError("number alignment failed after tokenization")

        num_values = _safe_change_num(nums)
        graph = get_single_example_graph(num_values)
        preprocess_trace = {
            "original_text": raw_text,
            "normalized_text": normalized_text,
            "matched_numbers": matches,
            "tokens_with_num_marker": tokens,
            "bert_tokens": bert_tokens,
            "num_pos_bert": [int(x) for x in num_pos],
            "num_values": [float(x) for x in num_values],
        }
        return bert_tokens, nums, num_pos, bert_input, graph, preprocess_trace

    @torch.no_grad()
    def solve(self, text: str) -> Dict[str, object]:
        with self.lock:
            bert_tokens, nums, num_pos, bert_input, graph, preprocess_trace = self._text_to_graph_inputs(text)
            input_length = int(bert_input["input_ids"].squeeze().size(0))
            input_batch = [0 for _ in range(input_length)]
            pred_ids, decoder_trace = self._evaluate_tree_with_trace(
                input_batch=input_batch,
                input_length=input_length,
                num_pos=num_pos,
                bert_input=bert_input,
                batch_graph=graph,
                numbers=nums,
                beam_size=self.beam_size,
            )
            prefix_tokens = out_expression_list(pred_ids, self.output_lang, nums)
            value = None
            if prefix_tokens:
                try:
                    value = compute_prefix_expression(prefix_tokens)
                except Exception:
                    value = None

            predicted_value = None
            if value is not None:
                predicted_value = float(value)
                if predicted_value == predicted_value and abs(predicted_value - round(predicted_value)) < 1e-9:
                    predicted_value = float(int(round(predicted_value)))

            expression_postprocess: Optional[Dict[str, object]] = None
            if os.environ.get("MWP_INCLUDE_POSTPROCESS_EXPLAIN", "1").strip() == "1":
                expression_postprocess = self.build_explainability_payload(
                    prefix_tokens=prefix_tokens,
                    numbers=nums,
                    num_pos_bert=num_pos,
                    graph=graph,
                )
            graph_nodes = []
            for idx, token in enumerate(nums):
                graph_nodes.append(
                    {
                        "index": int(idx),
                        "token": str(token),
                        "value": self._parse_numeric_token(str(token)),
                        "bert_pos": int(num_pos[idx]) if idx < len(num_pos) else None,
                    }
                )
            explainability = {
                "type": "model_trace",
                "source_note": "以下预处理、关系图、解码轨迹来自模型真实执行过程。",
                "preprocess": preprocess_trace,
                "graph_view": {
                    "relation_channels": ["all", "greater_than", "less_than", "different_index", "adjacent"],
                    "nodes": graph_nodes,
                    "edges": self._build_relation_edges(nums, graph),
                },
                "decoder_trace": decoder_trace,
                "final": {
                    "predicted_ids": [int(x) for x in pred_ids],
                    "predicted_prefix_tokens": [str(x) for x in (prefix_tokens or [])],
                    "predicted_prefix_text": " ".join(prefix_tokens) if prefix_tokens else None,
                    "predicted_infix_text": self.prefix_tokens_to_infix_text(prefix_tokens),
                    "predicted_value": predicted_value,
                },
            }
            if expression_postprocess is not None:
                explainability["expression_postprocess"] = expression_postprocess

            return {
                "text": text,
                "model_bundle_dir": self.bundle_dir,
                "encoder_arch": self.encoder_arch,
                "encoder_graph_layers": self.encoder_graph_layers,
                "encoder_relation_num": self.relation_num,
                "numbers": nums,
                "num_pos_bert": num_pos,
                "bert_tokens": bert_tokens,
                "predicted_ids": [int(x) for x in pred_ids],
                "predicted_prefix_tokens": prefix_tokens,
                "predicted_prefix_text": " ".join(prefix_tokens) if prefix_tokens else None,
                "predicted_infix_text": self.prefix_tokens_to_infix_text(prefix_tokens),
                "predicted_value": predicted_value,
                "explainability": explainability,
            }


def make_handler(solver: Math23kSolverSystem):
    class SolverHandler(BaseHTTPRequestHandler):
        def _write_json(self, code: int, payload: Dict[str, object]):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/health"):
                self._write_json(
                    200,
                    {
                        "ok": True,
                        "service": "math23k-solver",
                        "model_bundle_dir": solver.bundle_dir,
                        "encoder_arch": solver.encoder_arch,
                        "encoder_graph_layers": solver.encoder_graph_layers,
                        "encoder_relation_num": solver.relation_num,
                    },
                )
                return
            self._write_json(404, {"ok": False, "error": "not found"})

        def do_POST(self):
            if self.path != "/solve":
                self._write_json(404, {"ok": False, "error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                payload = json.loads(raw.decode("utf-8"))
                text = str(payload.get("text", "")).strip()
                if not text:
                    self._write_json(400, {"ok": False, "error": "text is required"})
                    return
                result = solver.solve(text)
                self._write_json(200, {"ok": True, "result": result})
            except Exception as exc:
                self._write_json(500, {"ok": False, "error": str(exc)})

        def log_message(self, format: str, *args):
            return

    return SolverHandler


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Math23k graph model inference system")
    parser.add_argument("--data_dir", type=str, default=str(ROOT / "data"))
    parser.add_argument("--model_dir", type=str, default="/root/autodl-tmp/finetune_models-zh-v2")
    parser.add_argument("--epoch", type=int, default=None, help="Optional epoch number under epoch_checkpoints")
    parser.add_argument("--device", type=str, default="cpu", help="e.g. cpu, cuda, cuda:0")
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--meta_cache_path", type=str, default=None)
    parser.add_argument("--text", type=str, default="", help="Solve one question and exit")
    parser.add_argument("--serve", action="store_true", help="Run HTTP service")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18081)
    return parser


def main():
    args = build_arg_parser().parse_args()
    solver = Math23kSolverSystem(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        epoch=args.epoch,
        device=args.device,
        beam_size=args.beam_size,
        meta_cache_path=args.meta_cache_path,
    )
    print("[solver] loaded bundle:", solver.bundle_dir)
    print("[solver] device:", args.device)
    print("[solver] encoder_arch:", solver.encoder_arch)
    print("[solver] encoder_graph_layers:", solver.encoder_graph_layers)
    print("[solver] encoder_relation_num:", solver.relation_num)
    print("[solver] output_vocab_size:", solver.output_lang.n_words)
    if args.text:
        result = solver.solve(args.text)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.serve:
        handler = make_handler(solver)
        server = ThreadingHTTPServer((args.host, args.port), handler)
        print("[solver] serving on http://%s:%d" % (args.host, args.port))
        server.serve_forever()
        return
    print("No action specified. Use --text for one-shot solve or --serve for HTTP service.")


if __name__ == "__main__":
    main()
