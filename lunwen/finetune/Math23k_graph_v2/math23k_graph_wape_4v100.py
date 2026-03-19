# coding: utf-8
from src.train_and_evaluate import *
from src.models import *
import time
import torch.optim
from src.expressions_transfer import *
import json
import hashlib
import pickle
import torch
import os
import argparse
import numpy as np
import random
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import BertTokenizer


def read_json(path):
    with open(path, 'r') as f:
        file = json.load(f)
    return file


def _is_distributed():
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ and int(os.environ.get("WORLD_SIZE", 1)) > 1


def _is_primary():
    return int(os.environ.get("RANK", "0")) == 0


class TransparentDDP(DDP):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--embedding_size", type=int, default=128)
parser.add_argument("--hidden_size", type=int, default=768)
parser.add_argument("--n_epochs", type=int, default=80)
parser.add_argument("--learning_rate", type=float, default=5e-5)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw"])
parser.add_argument("--head_lr_scale", type=float, default=1.0)
parser.add_argument("--beam_size", type=int, default=5)
parser.add_argument("--n_layers", type=int, default=2)
parser.add_argument("--step_size", type=int, default=30)
parser.add_argument("--lr_gamma", type=float, default=0.5)
parser.add_argument("--expected_world_size", type=int, default=4)
parser.add_argument("--eval_every", type=int, default=5)
parser.add_argument("--save_dir", type=str, default="models_graph_wape_4v100")
parser.add_argument("--save_each_epoch", type=int, default=1)
parser.add_argument("--epoch_ckpt_dir", type=str, default="epoch_checkpoints")
parser.add_argument("--epoch_ckpt_every", type=int, default=10)
parser.add_argument("--data_dir", type=str, default="./data")
parser.add_argument("--ori_path", type=str, default=None)
parser.add_argument("--prefix", type=str, default="23k_processed.json")
parser.add_argument("--math23k_path", type=str, default="Math_23K.json")
parser.add_argument("--math23k_processed_path", type=str, default="Math_23K_processed.json")
parser.add_argument("--ape_path", type=str, default="ape_simple_train.json")
parser.add_argument("--ape_test_path", type=str, default="ape_simple_test.json")
parser.add_argument("--ape_id", type=str, default="ape_simple_id.txt")
parser.add_argument("--ape_test_id", type=str, default="ape_simple_test_id.txt")
parser.add_argument("--overlap_id", type=str, default="overlap.txt")
parser.add_argument("--test_overlap_id", type=str, default="test_overlap.txt")
parser.add_argument("--ape_train_ratio", type=float, default=1.0)
parser.add_argument("--sample_seed", type=int, default=2026)
parser.add_argument("--dump_explain_dir", type=str, default="")
parser.add_argument("--graph_relation_num", type=int, default=5)
parser.add_argument("--graph_layers", type=int, default=2)
parser.add_argument("--graph_dropout", type=float, default=0.5)
parser.add_argument("--graph_relation_dropout", type=float, default=0.0)
parser.add_argument("--graph_active_relations", type=str, default="all")
parser.add_argument("--disable_graph", type=int, default=0)
parser.add_argument("--graph_log_every", type=int, default=100)
parser.add_argument("--ddp_find_unused_encoder", type=int, default=1)
parser.add_argument("--ddp_find_unused_heads", type=int, default=0)
parser.add_argument("--ddp_head_static_graph", type=int, default=1)
parser.add_argument("--dataset_cache_enable", type=int, default=1)
parser.add_argument("--dataset_cache_dir", type=str, default=".cache_preprocessed")
parser.add_argument("--dataset_cache_refresh", type=int, default=0)
parser.add_argument("--dataset_cache_tag", type=str, default="")
args = parser.parse_args()


def maybe_wrap_encoder_dp(model, use_ddp):
    if use_ddp:
        return model
    use_encoder_dp = os.environ.get("MWP_ENCODER_DP", "1") != "0"
    if USE_CUDA and use_encoder_dp and torch.cuda.device_count() > 1:
        return TransparentDataParallel(model)
    return model


def unwrap_parallel(model):
    while isinstance(model, (torch.nn.DataParallel, DDP)):
        model = model.module
    return model


def save_graph_model_bundle(save_root, encoder, predict, generate, merge):
    os.makedirs(save_root, exist_ok=True)
    torch.save(unwrap_parallel(encoder).state_dict(), os.path.join(save_root, "encoder_graph"))
    torch.save(unwrap_parallel(predict).state_dict(), os.path.join(save_root, "predict_graph"))
    torch.save(unwrap_parallel(generate).state_dict(), os.path.join(save_root, "generate_graph"))
    torch.save(unwrap_parallel(merge).state_dict(), os.path.join(save_root, "merge_graph"))


def _build_rank_batch_indices(num_batches, world_size, rank):
    if world_size <= 1:
        return list(range(num_batches))
    total_size = ((num_batches + world_size - 1) // world_size) * world_size
    indices = list(range(num_batches))
    if total_size > num_batches:
        indices.extend(indices[: total_size - num_batches])
    return indices[rank:total_size:world_size]


def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(args.data_dir, path)


def ensure_dir_path(path):
    if path.endswith(os.sep):
        return path
    return path + os.sep


def parse_active_relation_ids(spec, relation_num):
    if relation_num <= 0:
        raise ValueError("graph_relation_num must be positive.")
    if spec is None:
        return list(range(relation_num))
    clean_spec = spec.strip().lower()
    if clean_spec in ("", "all"):
        return list(range(relation_num))

    relation_ids = []
    for token in clean_spec.split(","):
        token = token.strip()
        if token == "":
            continue
        ridx = int(token)
        if ridx < 0 or ridx >= relation_num:
            raise ValueError(
                "graph_active_relations contains out-of-range id %d, expected [0, %d]."
                % (ridx, relation_num - 1)
            )
        relation_ids.append(ridx)

    relation_ids = sorted(set(relation_ids))
    if not relation_ids:
        raise ValueError("graph_active_relations resolved to an empty relation set.")
    return relation_ids


def format_graph_stats(stats):
    if not stats:
        return ""
    parts = [
        "fusion_gate=%.4f" % stats.get("fusion_gate_mean", 0.0),
        "problem_gate=%.4f" % stats.get("problem_gate_mean", 0.0),
        "valid_node_ratio=%.4f" % stats.get("valid_node_ratio", 0.0),
    ]
    relation_weights = stats.get("relation_weights")
    if relation_weights:
        relation_str = ",".join(
            "%d:%.3f" % (idx, float(weight))
            for idx, weight in enumerate(relation_weights)
        )
        parts.append("rel_w=[%s]" % relation_str)
    return " ".join(parts)


def _safe_prefix_value(prefix_tokens):
    if prefix_tokens is None:
        return None
    try:
        value = compute_prefix_expression(prefix_tokens)
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _to_token_list(tokenizer, bert_input):
    token_ids = bert_input["input_ids"].squeeze().tolist()
    if isinstance(token_ids, int):
        token_ids = [token_ids]
    return tokenizer.convert_ids_to_tokens(token_ids)


def _read_id_list(path):
    with open(path, "r", encoding="utf-8") as f:
        return [str(i) for i in f.read().split()]


def _subsample_pairs(pairs, keep_ratio, seed):
    if keep_ratio >= 1.0:
        return list(pairs)
    if keep_ratio <= 0.0 or not pairs:
        return []
    keep_count = max(1, int(len(pairs) * keep_ratio))
    rng = random.Random(seed)
    keep_indices = set(rng.sample(range(len(pairs)), keep_count))
    return [pair for idx, pair in enumerate(pairs) if idx in keep_indices]


def _file_signature(path):
    abs_path = os.path.abspath(path)
    stat = os.stat(abs_path)
    return {
        "path": abs_path,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _build_preprocess_cache_key(dependency_paths, config):
    key_payload = {
        "cache_version": 2,
        "dependencies": [_file_signature(path) for path in dependency_paths],
        "config": config,
    }
    digest = hashlib.md5(
        json.dumps(key_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return digest, key_payload


def _atomic_pickle_save(obj, target_path):
    tmp_path = f"{target_path}.tmp.{os.getpid()}"
    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, target_path)


def _pickle_load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _build_preprocessed_graph_payload(
    ori_path,
    prefix,
    math23k_path,
    ape_path,
    ape_test_path,
    ape_id,
    ape_test_id,
    overlap_id,
    test_overlap_id,
    ape_train_ratio,
    sample_seed,
):
    id_list = _read_id_list(ape_id)
    test_id_list = _read_id_list(ape_test_id)
    overlap_list = _read_id_list(overlap_id)
    test_overlap_list = _read_id_list(test_overlap_id)

    data = (
        load_raw_data(math23k_path)
        + raw_data_new(ape_path, id_list, overlap_list)
        + raw_data_new(ape_test_path, test_id_list, test_overlap_list)
    )
    pairs, generate_nums, copy_nums = transfer_num(data)
    pairs = [(p[0], from_infix_to_prefix(p[1]), p[2], p[3]) for p in pairs]
    (
        train_fold_23k,
        test_fold_23k,
        valid_fold_23k,
        train_fold_ape,
        test_fold_ape,
    ) = get_train_test_fold_wape(
        ori_path,
        prefix,
        data,
        pairs,
        set(id_list),
        set(test_id_list),
    )
    sampled_train_fold_ape = _subsample_pairs(train_fold_ape, ape_train_ratio, sample_seed)
    train_fold = train_fold_23k + sampled_train_fold_ape

    input_lang, output_lang, train_pairs, test_pairs_23k = prepare_data_23k_graph(
        train_fold, test_fold_23k, 5, generate_nums, copy_nums, tree=True
    )
    if valid_fold_23k:
        _, _, _, valid_pairs_23k = prepare_data_23k_graph(
            train_fold, valid_fold_23k, 5, generate_nums, copy_nums, tree=True
        )
    else:
        valid_pairs_23k = []
    if test_fold_ape:
        _, _, _, test_pairs_ape = prepare_data_23k_graph(
            train_fold, test_fold_ape, 5, generate_nums, copy_nums, tree=True
        )
    else:
        test_pairs_ape = []

    payload = {
        "input_lang": input_lang,
        "output_lang": output_lang,
        "train_pairs": train_pairs,
        "valid_pairs_23k": valid_pairs_23k,
        "test_pairs_23k": test_pairs_23k,
        "test_pairs_ape": test_pairs_ape,
        "generate_nums": generate_nums,
        "copy_nums": copy_nums,
        "data_stats": {
            "train_23k": len(train_fold_23k),
            "train_ape": len(train_fold_ape),
            "sampled_train_ape": len(sampled_train_fold_ape),
            "valid_23k": len(valid_fold_23k),
            "test_23k": len(test_fold_23k),
            "test_ape": len(test_fold_ape),
        },
    }
    return payload


def get_train_test_fold_wape(ori_path, prefix, data, pairs, ape_train_ids, ape_test_ids):
    mode_train = 'train'
    mode_valid = 'valid'
    mode_test = 'test'
    train_path = os.path.join(ori_path, mode_train + prefix)
    valid_path = os.path.join(ori_path, mode_valid + prefix)
    test_path = os.path.join(ori_path, mode_test + prefix)
    train = read_json(train_path)
    valid = read_json(valid_path)
    test = read_json(test_path)
    train_id = [item['id'] for item in train]
    valid_id = [item['id'] for item in valid]
    test_id = [item['id'] for item in test]
    train_fold_23k = []
    valid_fold_23k = []
    test_fold_23k = []
    train_fold_ape = []
    test_fold_ape = []
    for item, pair in zip(data, pairs):
        if item.get("type") == "23k":
            if item["id"] in train_id:
                train_fold_23k.append(pair)
            elif item["id"] in test_id:
                test_fold_23k.append(pair)
            else:
                valid_fold_23k.append(pair)
            continue
        if item.get("type") != "ape":
            continue
        item_id = str(item["id"])
        if item_id in ape_train_ids:
            train_fold_ape.append(pair)
        elif item_id in ape_test_ids:
            test_fold_ape.append(pair)
    return train_fold_23k, test_fold_23k, valid_fold_23k, train_fold_ape, test_fold_ape


def main():
    use_ddp = _is_distributed()
    if use_ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        backend = os.environ.get("MWP_DDP_BACKEND", "")
        if backend:
            backend = backend.lower()
        else:
            backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        local_rank = int(os.environ.get("MWP_CUDA_DEVICE", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        rank = 0
        world_size = 1
    if args.expected_world_size > 0 and world_size != args.expected_world_size and _is_primary():
        print(
            "[warn] world_size=%d, expected_world_size=%d (set --expected_world_size 0 to disable this check)"
            % (world_size, args.expected_world_size)
        )

    batch_size = args.batch_size
    embedding_size = args.embedding_size
    hidden_size = args.hidden_size
    n_epochs = args.n_epochs
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    beam_size = args.beam_size
    n_layers = args.n_layers
    ori_path = ensure_dir_path(args.ori_path if args.ori_path else args.data_dir)
    prefix = args.prefix
    math23k_path = resolve_path(args.math23k_path)
    ape_path = resolve_path(args.ape_path)
    ape_test_path = resolve_path(args.ape_test_path)
    ape_id = resolve_path(args.ape_id)
    ape_test_id = resolve_path(args.ape_test_id)
    overlap_id = resolve_path(args.overlap_id)
    test_overlap_id = resolve_path(args.test_overlap_id)
    if args.ape_train_ratio < 0.0 or args.ape_train_ratio > 1.0:
        raise ValueError("ape_train_ratio must be in [0.0, 1.0].")
    active_relation_ids = parse_active_relation_ids(args.graph_active_relations, args.graph_relation_num)
    if _is_primary():
        os.makedirs(args.save_dir, exist_ok=True)
        if args.dump_explain_dir:
            os.makedirs(args.dump_explain_dir, exist_ok=True)
        print(
            "[graph_cfg] disable_graph=%s relation_num=%d graph_layers=%d graph_dropout=%.2f "
            "relation_dropout=%.2f active_relations=%s ckpt_every=%d ape_train_ratio=%.2f "
            "optimizer=%s lr=%.6g head_lr_scale=%.2f step_size=%d gamma=%.2f"
            % (
                str(bool(args.disable_graph)),
                args.graph_relation_num,
                args.graph_layers,
                args.graph_dropout,
                args.graph_relation_dropout,
                active_relation_ids,
                max(1, args.epoch_ckpt_every),
                args.ape_train_ratio,
                args.optimizer,
                learning_rate,
                args.head_lr_scale,
                args.step_size,
                args.lr_gamma,
            )
        )
    epoch_ckpt_root = os.path.join(args.save_dir, args.epoch_ckpt_dir)
    if args.save_each_epoch and _is_primary():
        os.makedirs(epoch_ckpt_root, exist_ok=True)

    train_split_path = os.path.join(ori_path, "train" + prefix)
    valid_split_path = os.path.join(ori_path, "valid" + prefix)
    test_split_path = os.path.join(ori_path, "test" + prefix)
    cache_enabled = bool(args.dataset_cache_enable)
    cache_dir = args.dataset_cache_dir
    if not os.path.isabs(cache_dir):
        cache_dir = os.path.abspath(cache_dir)
    cache_config = {
        "tokenizer_path": DEFAULT_ZH_BERT_PATH,
        "ape_train_ratio": float(args.ape_train_ratio),
        "sample_seed": int(args.sample_seed),
        "trim_min_count": 5,
        "tree": True,
        "prefix": prefix,
        "cache_tag": args.dataset_cache_tag,
    }
    cache_key, _ = _build_preprocess_cache_key(
        [
            math23k_path,
            ape_path,
            ape_test_path,
            ape_id,
            ape_test_id,
            overlap_id,
            test_overlap_id,
            train_split_path,
            valid_split_path,
            test_split_path,
        ],
        cache_config,
    )
    cache_path = os.path.join(cache_dir, f"math23k_graph_wape_4v100_{cache_key}.pkl")

    preprocess_payload = None
    if cache_enabled:
        if _is_primary():
            os.makedirs(cache_dir, exist_ok=True)
            if bool(args.dataset_cache_refresh) and os.path.exists(cache_path):
                os.remove(cache_path)
                print(f"[cache] removed={cache_path}")
            print(f"[cache] enabled key={cache_key} path={cache_path}")

        if os.path.exists(cache_path):
            preprocess_payload = _pickle_load(cache_path)
            if _is_primary():
                print("[cache] hit -> loaded preprocessed dataset")
        else:
            if _is_primary():
                print("[cache] miss -> building preprocessed dataset on all ranks (first run)")
            preprocess_payload = _build_preprocessed_graph_payload(
                ori_path=ori_path,
                prefix=prefix,
                math23k_path=math23k_path,
                ape_path=ape_path,
                ape_test_path=ape_test_path,
                ape_id=ape_id,
                ape_test_id=ape_test_id,
                overlap_id=overlap_id,
                test_overlap_id=test_overlap_id,
                ape_train_ratio=args.ape_train_ratio,
                sample_seed=args.sample_seed,
            )
            if _is_primary():
                _atomic_pickle_save(preprocess_payload, cache_path)
                print("[cache] saved preprocessed dataset")
    else:
        if _is_primary():
            print("[cache] disabled -> building preprocessed dataset every run")
        preprocess_payload = _build_preprocessed_graph_payload(
            ori_path=ori_path,
            prefix=prefix,
            math23k_path=math23k_path,
            ape_path=ape_path,
            ape_test_path=ape_test_path,
            ape_id=ape_id,
            ape_test_id=ape_test_id,
            overlap_id=overlap_id,
            test_overlap_id=test_overlap_id,
            ape_train_ratio=args.ape_train_ratio,
            sample_seed=args.sample_seed,
        )

    input_lang = preprocess_payload["input_lang"]
    output_lang = preprocess_payload["output_lang"]
    train_pairs = preprocess_payload["train_pairs"]
    valid_pairs_23k = preprocess_payload.get("valid_pairs_23k", [])
    test_pairs_23k = preprocess_payload["test_pairs_23k"]
    test_pairs_ape = preprocess_payload["test_pairs_ape"]
    generate_nums = preprocess_payload["generate_nums"]
    copy_nums = preprocess_payload["copy_nums"]

    if _is_primary():
        data_stats = preprocess_payload["data_stats"]
        print(
            "[data] train_23k=%d train_ape=%d sampled_train_ape=%d valid_23k=%d test_23k=%d test_ape=%d"
            % (
                data_stats["train_23k"],
                data_stats["train_ape"],
                data_stats["sampled_train_ape"],
                data_stats["valid_23k"],
                data_stats["test_23k"],
                data_stats["test_ape"],
            )
        )

    encoder = GraphFusionEncoder(
        input_size=input_lang.n_words,
        embedding_size=embedding_size,
        hidden_size=hidden_size,
        n_layers=n_layers,
        relation_num=args.graph_relation_num,
        graph_layers=args.graph_layers,
        graph_dropout=args.graph_dropout,
        graph_relation_dropout=args.graph_relation_dropout,
        active_relation_ids=active_relation_ids,
        disable_graph=bool(args.disable_graph),
    )
    predict = Prediction(
        hidden_size=hidden_size,
        op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
        input_size=len(generate_nums),
    )
    generate = GenerateNode(
        hidden_size=hidden_size,
        op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
        embedding_size=embedding_size,
    )
    merge = Merge(hidden_size=hidden_size, embedding_size=embedding_size)

    device = torch.device(f"cuda:{local_rank}" if USE_CUDA else "cpu")
    encoder = encoder.to(device)
    predict = predict.to(device)
    generate = generate.to(device)
    merge = merge.to(device)

    if use_ddp:
        # Encoder may keep BERT pooler unused, so leave this configurable and enabled by default.
        encoder_ddp_kwargs = {"find_unused_parameters": bool(args.ddp_find_unused_encoder)}
        # Decoder heads are called repeatedly inside one training step; unused-parameter graph
        # traversal can trigger "mark a variable ready only once" in this pattern.
        head_ddp_kwargs = {"find_unused_parameters": bool(args.ddp_find_unused_heads)}
        if USE_CUDA:
            encoder = TransparentDDP(encoder, device_ids=[local_rank], output_device=local_rank, **encoder_ddp_kwargs)
            predict = TransparentDDP(predict, device_ids=[local_rank], output_device=local_rank, **head_ddp_kwargs)
            generate = TransparentDDP(generate, device_ids=[local_rank], output_device=local_rank, **head_ddp_kwargs)
            merge = TransparentDDP(merge, device_ids=[local_rank], output_device=local_rank, **head_ddp_kwargs)
        else:
            encoder = TransparentDDP(encoder, **encoder_ddp_kwargs)
            predict = TransparentDDP(predict, **head_ddp_kwargs)
            generate = TransparentDDP(generate, **head_ddp_kwargs)
            merge = TransparentDDP(merge, **head_ddp_kwargs)
        if bool(args.ddp_head_static_graph):
            for module in (predict, generate, merge):
                if hasattr(module, "_set_static_graph"):
                    module._set_static_graph()
        if _is_primary():
            print(
                "[ddp] find_unused_encoder=%s find_unused_heads=%s head_static_graph=%s"
                % (
                    str(bool(args.ddp_find_unused_encoder)),
                    str(bool(args.ddp_find_unused_heads)),
                    str(bool(args.ddp_head_static_graph)),
                )
            )
    else:
        encoder = maybe_wrap_encoder_dp(encoder, use_ddp=False)

    optimizer_cls = torch.optim.Adam if args.optimizer == "adam" else torch.optim.AdamW
    head_lr = learning_rate * args.head_lr_scale
    encoder_optimizer = optimizer_cls(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    predict_optimizer = optimizer_cls(predict.parameters(), lr=head_lr, weight_decay=weight_decay)
    generate_optimizer = optimizer_cls(generate.parameters(), lr=head_lr, weight_decay=weight_decay)
    merge_optimizer = optimizer_cls(merge.parameters(), lr=head_lr, weight_decay=weight_decay)

    encoder_scheduler = torch.optim.lr_scheduler.StepLR(
        encoder_optimizer, step_size=args.step_size, gamma=args.lr_gamma
    )
    predict_scheduler = torch.optim.lr_scheduler.StepLR(
        predict_optimizer, step_size=args.step_size, gamma=args.lr_gamma
    )
    generate_scheduler = torch.optim.lr_scheduler.StepLR(
        generate_optimizer, step_size=args.step_size, gamma=args.lr_gamma
    )
    merge_scheduler = torch.optim.lr_scheduler.StepLR(
        merge_optimizer, step_size=args.step_size, gamma=args.lr_gamma
    )

    generate_num_ids = [output_lang.word2index[num] for num in generate_nums]
    explain_tokenizer = BertTokenizer.from_pretrained(DEFAULT_ZH_BERT_PATH) if (args.dump_explain_dir and _is_primary()) else None
    best_value_acc = -1.0
    best_equation_acc = -1.0
    best_epoch = -1
    best_ape_value_acc = None
    best_ape_equation_acc = None

    for epoch in range(n_epochs):
        random.seed(2000 + epoch)
        loss_total = 0.0
        (
            input_batches,
            input_lengths,
            output_batches,
            output_lengths,
            nums_batches,
            num_stack_batches,
            num_pos_batches,
            num_size_batches,
            bert_batches,
            graph_batches,
        ) = prepare_train_batch_graph(train_pairs, batch_size)

        total_steps = len(input_lengths)
        rank_batch_indices = _build_rank_batch_indices(total_steps, world_size, rank)
        start = time.time()
        if _is_primary():
            print(
                f"[train] epoch={epoch + 1}/{n_epochs} total_steps={total_steps} "
                f"rank_steps={len(rank_batch_indices)} world_size={world_size}"
            )

        for local_step, idx in enumerate(rank_batch_indices, start=1):
            loss = train_tree(
                input_batches[idx],
                input_lengths[idx],
                output_batches[idx],
                output_lengths[idx],
                num_stack_batches[idx],
                num_size_batches[idx],
                generate_num_ids,
                encoder,
                predict,
                generate,
                merge,
                encoder_optimizer,
                predict_optimizer,
                generate_optimizer,
                merge_optimizer,
                output_lang,
                num_pos_batches[idx],
                bert_batches[idx],
                graph_batches[idx],
            )
            loss_total += loss
            if _is_primary() and args.graph_log_every > 0 and local_step % args.graph_log_every == 0:
                encoder_core = unwrap_parallel(encoder)
                if hasattr(encoder_core, "pop_debug_stats"):
                    stats = encoder_core.pop_debug_stats()
                    stats_text = format_graph_stats(stats)
                    if stats_text:
                        print(
                            "[graph] epoch=%d step=%d/%d %s"
                            % (epoch + 1, local_step, len(rank_batch_indices), stats_text)
                        )

        encoder_scheduler.step()
        predict_scheduler.step()
        generate_scheduler.step()
        merge_scheduler.step()

        loss_stats = torch.tensor([loss_total, float(len(rank_batch_indices))], dtype=torch.float32, device=device)
        if use_ddp:
            dist.all_reduce(loss_stats, op=dist.ReduceOp.SUM)
        global_avg_loss = (loss_stats[0] / loss_stats[1]).item() if loss_stats[1].item() > 0 else 0.0
        if _is_primary():
            encoder_core = unwrap_parallel(encoder)
            if hasattr(encoder_core, "pop_debug_stats"):
                epoch_stats = encoder_core.pop_debug_stats()
                epoch_stats_text = format_graph_stats(epoch_stats)
                if epoch_stats_text:
                    print("[graph] epoch=%d summary %s" % (epoch + 1, epoch_stats_text))
            print(
                f"epoch={epoch + 1}  avg_loss={global_avg_loss:.4f} "
                f"elapsed={time_since(time.time() - start)}"
            )
            print("--------------------------------")

        ckpt_every = max(1, args.epoch_ckpt_every)
        if args.save_each_epoch and (epoch + 1) % ckpt_every == 0:
            if use_ddp:
                dist.barrier()
            if _is_primary():
                epoch_ckpt_path = os.path.join(epoch_ckpt_root, f"epoch_{epoch + 1:03d}")
                save_graph_model_bundle(epoch_ckpt_path, encoder, predict, generate, merge)
                print(f"[save] epoch_checkpoint={epoch_ckpt_path}")
            if use_ddp:
                dist.barrier()

        eval_every = max(1, args.eval_every)
        current_epoch = epoch + 1
        # Evaluate on epoch 1, then every eval_every epochs, and every epoch in the final eval_every epochs.
        should_eval = (
            current_epoch == 1
            or current_epoch % eval_every == 0
            or current_epoch > n_epochs - eval_every
        )
        if should_eval:
            if use_ddp:
                dist.barrier()
            if _is_primary():
                eval_start = time.time()
                valid_equation_acc = None
                valid_value_acc = None
                if valid_pairs_23k:
                    valid_equation = 0
                    valid_value = 0
                    valid_total = 0
                    for valid_batch in valid_pairs_23k:
                        valid_graph = get_single_example_graph(valid_batch[8])
                        valid_res = evaluate_tree(
                            valid_batch[0],
                            valid_batch[1],
                            generate_num_ids,
                            encoder,
                            predict,
                            generate,
                            merge,
                            output_lang,
                            valid_batch[5],
                            valid_batch[7],
                            valid_graph,
                            beam_size=beam_size,
                        )
                        valid_val_ac, valid_equ_ac, _, _ = compute_prefix_tree_result(
                            valid_res, valid_batch[2], output_lang, valid_batch[4], valid_batch[6]
                        )
                        if valid_val_ac:
                            valid_value += 1
                        if valid_equ_ac:
                            valid_equation += 1
                        valid_total += 1
                    print(valid_equation, valid_value, valid_total)
                    valid_equation_acc = float(valid_equation) / valid_total
                    valid_value_acc = float(valid_value) / valid_total
                    print("valid_answer_acc_23k", valid_equation_acc, valid_value_acc)
                else:
                    print("valid_answer_acc_23k skipped(empty_valid_set)")

                value_ac = 0
                equation_ac = 0
                eval_total = 0
                explain_rows = []
                for test_batch in test_pairs_23k:
                    example_graph = get_single_example_graph(test_batch[8])
                    test_res = evaluate_tree(
                        test_batch[0],
                        test_batch[1],
                        generate_num_ids,
                        encoder,
                        predict,
                        generate,
                        merge,
                        output_lang,
                        test_batch[5],
                        test_batch[7],
                        example_graph,
                        beam_size=beam_size,
                    )
                    val_ac, equ_ac, pred_prefix, tar_prefix = compute_prefix_tree_result(
                        test_res, test_batch[2], output_lang, test_batch[4], test_batch[6]
                    )
                    if val_ac:
                        value_ac += 1
                    if equ_ac:
                        equation_ac += 1
                    eval_total += 1
                    if args.dump_explain_dir:
                        explain_rows.append(
                            {
                                "sample_index": eval_total - 1,
                                "num_list": test_batch[4],
                                "num_pos_bert": test_batch[5],
                                "input_tokens_bert": _to_token_list(explain_tokenizer, test_batch[7]),
                                "graph_relations": example_graph[0].tolist() if isinstance(example_graph, np.ndarray) else [],
                                "predicted_ids": [int(i) for i in test_res],
                                "target_ids": [int(i) for i in test_batch[2]],
                                "predicted_prefix": pred_prefix,
                                "target_prefix": tar_prefix,
                                "predicted_value": _safe_prefix_value(pred_prefix),
                                "target_value": _safe_prefix_value(tar_prefix),
                                "value_correct": bool(val_ac),
                                "equation_correct": bool(equ_ac),
                            }
                        )
                print(equation_ac, value_ac, eval_total)
                equation_acc = float(equation_ac) / eval_total
                value_acc = float(value_ac) / eval_total
                print("test_answer_acc_23k", equation_acc, value_acc)
                ape_equation_acc = None
                ape_value_acc = None
                if test_pairs_ape:
                    ape_equation = 0
                    ape_value = 0
                    ape_total = 0
                    for test_batch in test_pairs_ape:
                        example_graph = get_single_example_graph(test_batch[8])
                        test_res = evaluate_tree(
                            test_batch[0],
                            test_batch[1],
                            generate_num_ids,
                            encoder,
                            predict,
                            generate,
                            merge,
                            output_lang,
                            test_batch[5],
                            test_batch[7],
                            example_graph,
                            beam_size=beam_size,
                        )
                        val_ac, equ_ac, _, _ = compute_prefix_tree_result(
                            test_res, test_batch[2], output_lang, test_batch[4], test_batch[6]
                        )
                        if val_ac:
                            ape_value += 1
                        if equ_ac:
                            ape_equation += 1
                        ape_total += 1
                    print(ape_equation, ape_value, ape_total)
                    ape_equation_acc = float(ape_equation) / ape_total
                    ape_value_acc = float(ape_value) / ape_total
                    print("test_answer_acc_ape", ape_equation_acc, ape_value_acc)
                print("testing time", time_since(time.time() - eval_start))
                print("------------------------------------------------------")
                if args.dump_explain_dir:
                    explain_path = os.path.join(args.dump_explain_dir, "eval_epoch_%d.jsonl" % (epoch + 1))
                    with open(explain_path, "w", encoding="utf-8") as fout:
                        for row in explain_rows:
                            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    print("explain_dump:", explain_path)
                is_better = (
                    value_acc > best_value_acc or
                    (abs(value_acc - best_value_acc) < 1e-12 and equation_acc > best_equation_acc)
                )
                if is_better:
                    best_value_acc = value_acc
                    best_equation_acc = equation_acc
                    best_epoch = epoch + 1
                    best_ape_equation_acc = ape_equation_acc
                    best_ape_value_acc = ape_value_acc
                    save_graph_model_bundle(args.save_dir, encoder, predict, generate, merge)
                    best_meta = {
                        "best_epoch": best_epoch,
                        "valid_equation_acc_23k": valid_equation_acc,
                        "valid_value_acc_23k": valid_value_acc,
                        "best_equation_acc_23k": best_equation_acc,
                        "best_value_acc_23k": best_value_acc,
                        "best_equation_acc_ape": best_ape_equation_acc,
                        "best_value_acc_ape": best_ape_value_acc,
                        "ape_train_ratio": args.ape_train_ratio,
                        # Keep compatibility with existing tooling.
                        "best_equation_acc": best_equation_acc,
                        "best_value_acc": best_value_acc,
                    }
                    with open(os.path.join(args.save_dir, "best_metrics.json"), "w", encoding="utf-8") as f:
                        json.dump(best_meta, f, ensure_ascii=False, indent=2)
                    print(
                        "[save] new_best epoch=%d 23k_equation=%.6f 23k_value=%.6f -> %s"
                        % (best_epoch, best_equation_acc, best_value_acc, args.save_dir)
                    )
                    if best_ape_value_acc is not None:
                        print(
                            "[save] linked_ape_metrics epoch=%d ape_equation=%.6f ape_value=%.6f"
                            % (best_epoch, best_ape_equation_acc, best_ape_value_acc)
                        )
                else:
                    print(
                        "[save] keep_best epoch=%d 23k_equation=%.6f 23k_value=%.6f"
                        % (best_epoch, best_equation_acc, best_value_acc)
                    )
                    if best_ape_value_acc is not None:
                        print(
                            "[save] keep_best_linked_ape epoch=%d ape_equation=%.6f ape_value=%.6f"
                            % (best_epoch, best_ape_equation_acc, best_ape_value_acc)
                        )
            if use_ddp:
                dist.barrier()

    if _is_primary():
        if best_epoch > 0:
            print(
                "[best] epoch=%d 23k_equation=%.6f 23k_value=%.6f"
                % (best_epoch, best_equation_acc, best_value_acc)
            )
            if best_ape_value_acc is not None:
                print(
                    "[best] linked_ape_equation=%.6f linked_ape_value=%.6f"
                    % (best_ape_equation_acc, best_ape_value_acc)
                )
        else:
            print("[best] no evaluation was executed.")

    if use_ddp and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
