import torch
import torch.nn as nn
import os
from collections.abc import Mapping
from transformers import BertModel

DEFAULT_ZH_BERT_PATH = os.environ.get("MWP_ZH_BERT_PATH", "hfl/chinese-bert-wwm-ext")
DEFAULT_ZH_ROBERTA_PATH = os.environ.get("MWP_ZH_ROBERTA_PATH", "hfl/chinese-roberta-wwm-ext")


class TransparentDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


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

    return {
        "input_ids": torch.stack(input_ids, dim=0).long(),
        "attention_mask": torch.stack(attention_mask, dim=0).long(),
    }


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


class RelationGraphLayer(nn.Module):
    def __init__(self, hidden_size, relation_num=5, dropout=0.1, relation_dropout=0.0):
        super(RelationGraphLayer, self).__init__()
        self.relation_num = relation_num
        self.relation_dropout = relation_dropout
        self.relation_linears = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(relation_num)])
        self.relation_gate = nn.Linear(hidden_size, relation_num)
        self.out_linear = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.norm_1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.norm_2 = nn.LayerNorm(hidden_size)

    def forward(self, node_embeddings, relation_graph, relation_mask=None):
        relation_graph = relation_graph.float()
        relation_messages = []
        for ridx, linear in enumerate(self.relation_linears):
            adj = relation_graph[:, ridx]
            degree = adj.sum(-1, keepdim=True).clamp(min=1.0)
            agg = torch.bmm(adj, node_embeddings) / degree
            relation_messages.append(torch.relu(linear(agg)))
        relation_messages = torch.stack(relation_messages, dim=1)

        pooled_nodes = node_embeddings.mean(dim=1)
        relation_logits = self.relation_gate(pooled_nodes)
        if relation_mask is not None:
            inactive_mask = (~relation_mask).unsqueeze(0).to(relation_logits.device)
            relation_logits = relation_logits.masked_fill(inactive_mask, -1e4)
        relation_weights = torch.softmax(relation_logits, dim=-1)

        if self.training and self.relation_dropout > 0:
            keep_prob = 1.0 - self.relation_dropout
            keep_mask = (torch.rand_like(relation_weights) < keep_prob).float()
            all_dropped = keep_mask.sum(dim=-1, keepdim=True).eq(0)
            if all_dropped.any():
                top_idx = relation_weights.argmax(dim=-1, keepdim=True)
                keep_mask = keep_mask.scatter(1, top_idx, 1.0)
            relation_weights = relation_weights * keep_mask
            relation_weights = relation_weights / relation_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        merged = (relation_messages * relation_weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
        merged = self.dropout(self.out_linear(merged))
        hidden = self.norm_1(node_embeddings + merged)
        out = self.norm_2(hidden + self.dropout(self.ffn(hidden)))
        return out, relation_weights


class RelationGraphModule(nn.Module):
    def __init__(
        self,
        hidden_size,
        relation_num=5,
        num_layers=2,
        dropout=0.1,
        relation_dropout=0.0,
        active_relation_ids=None,
    ):
        super(RelationGraphModule, self).__init__()
        self.relation_num = relation_num
        self.layers = nn.ModuleList(
            [
                RelationGraphLayer(
                    hidden_size=hidden_size,
                    relation_num=relation_num,
                    dropout=dropout,
                    relation_dropout=relation_dropout,
                )
                for _ in range(max(1, num_layers))
            ]
        )
        relation_mask = torch.ones(relation_num, dtype=torch.bool)
        if active_relation_ids is not None:
            relation_mask.zero_()
            for ridx in active_relation_ids:
                if 0 <= ridx < relation_num:
                    relation_mask[ridx] = True
        self.register_buffer("relation_mask", relation_mask, persistent=False)

    def forward(self, node_embeddings, relation_graph):
        if relation_graph.size(1) != self.relation_num:
            raise ValueError(
                "relation_graph relation dim mismatch: expected %d, got %d"
                % (self.relation_num, relation_graph.size(1))
            )
        hidden = node_embeddings
        relation_weights_collect = []
        for layer in self.layers:
            hidden, relation_weights = layer(hidden, relation_graph, relation_mask=self.relation_mask)
            relation_weights_collect.append(relation_weights.mean(dim=0))
        stats = {}
        if relation_weights_collect:
            stats["relation_weights"] = torch.stack(relation_weights_collect, dim=0).mean(dim=0)
        return hidden, stats


class GraphFusionEncoder(nn.Module):
    def __init__(
        self,
        input_size,
        embedding_size,
        hidden_size,
        n_layers=2,
        dropout=0.5,
        relation_num=5,
        graph_layers=2,
        graph_dropout=0.1,
        graph_relation_dropout=0.0,
        active_relation_ids=None,
        disable_graph=False,
    ):
        super(GraphFusionEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.relation_num = relation_num
        self.disable_graph = disable_graph
        self.bert_rnn = BertModel.from_pretrained(DEFAULT_ZH_BERT_PATH)
        self.graph_module = RelationGraphModule(
            hidden_size=hidden_size,
            relation_num=relation_num,
            num_layers=graph_layers,
            dropout=graph_dropout,
            relation_dropout=graph_relation_dropout,
            active_relation_ids=active_relation_ids,
        )
        self.fusion_gate = nn.Linear(hidden_size * 2, hidden_size)
        self.problem_gate = nn.Linear(hidden_size * 2, hidden_size)
        self.problem_norm = nn.LayerNorm(hidden_size)
        self._reset_debug_stats()

    def _reset_debug_stats(self):
        self._debug_steps = 0
        self._debug_fusion_gate_sum = 0.0
        self._debug_problem_gate_sum = 0.0
        self._debug_valid_nodes = 0.0
        self._debug_total_nodes = 0.0
        self._debug_relation_steps = 0
        self._debug_relation_sum = torch.zeros(self.relation_num, dtype=torch.float32)

    def _record_debug_stats(self, fusion_gate_mean, problem_gate_mean, valid_nodes, total_nodes, relation_weights):
        self._debug_steps += 1
        self._debug_fusion_gate_sum += float(fusion_gate_mean)
        self._debug_problem_gate_sum += float(problem_gate_mean)
        self._debug_valid_nodes += float(valid_nodes)
        self._debug_total_nodes += float(total_nodes)
        if relation_weights is not None:
            relation_cpu = relation_weights.detach().float().cpu().view(-1)
            if relation_cpu.numel() != self._debug_relation_sum.numel():
                aligned = torch.zeros_like(self._debug_relation_sum)
                keep = min(aligned.numel(), relation_cpu.numel())
                aligned[:keep] = relation_cpu[:keep]
                relation_cpu = aligned
            self._debug_relation_sum += relation_cpu
            self._debug_relation_steps += 1

    def pop_debug_stats(self):
        if self._debug_steps == 0:
            return {}
        result = {
            "fusion_gate_mean": self._debug_fusion_gate_sum / self._debug_steps,
            "problem_gate_mean": self._debug_problem_gate_sum / self._debug_steps,
            "valid_node_ratio": self._debug_valid_nodes / max(self._debug_total_nodes, 1.0),
        }
        if self._debug_relation_steps > 0:
            relation_avg = self._debug_relation_sum / self._debug_relation_steps
            result["relation_weights"] = relation_avg.tolist()
        self._reset_debug_stats()
        return result

    def _fuse_numbers_back(self, bert_output, enhanced_numbers, num_pos):
        seq_len = bert_output.size(0)
        batch_size = bert_output.size(1)
        hidden_size = bert_output.size(2)
        batch_outputs = []
        gate_values = []
        valid_node_counts = []
        for bidx in range(batch_size):
            token_output = bert_output[:, bidx, :]
            valid_positions = [p for p in num_pos[bidx] if 0 <= p < seq_len]
            node_count = min(len(valid_positions), enhanced_numbers.size(1))
            valid_node_counts.append(node_count)
            if node_count == 0:
                batch_outputs.append(token_output)
                continue

            pos_tensor = torch.tensor(valid_positions[:node_count], dtype=torch.long, device=bert_output.device)
            bert_slice = token_output.index_select(0, pos_tensor)
            graph_slice = enhanced_numbers[bidx, :node_count, :]
            gate = torch.sigmoid(self.fusion_gate(torch.cat([bert_slice, graph_slice], dim=-1)))
            gate_values.append(gate.mean())
            fused_slice = gate * graph_slice + (1 - gate) * bert_slice
            token_output = token_output.scatter(0, pos_tensor.unsqueeze(1).expand(-1, hidden_size), fused_slice)
            batch_outputs.append(token_output)

        if gate_values:
            fusion_gate_mean = torch.stack(gate_values).mean()
        else:
            fusion_gate_mean = torch.tensor(0.0, device=bert_output.device)
        valid_node_counts = torch.tensor(valid_node_counts, dtype=torch.float32, device=bert_output.device)
        return torch.stack(batch_outputs, dim=1), fusion_gate_mean, valid_node_counts

    def _pool_enhanced_numbers(self, enhanced_numbers, valid_node_counts):
        batch_size, num_size, hidden_size = enhanced_numbers.size()
        pooled = []
        for bidx in range(batch_size):
            node_count = int(valid_node_counts[bidx].item())
            if node_count <= 0:
                pooled.append(torch.zeros(hidden_size, device=enhanced_numbers.device, dtype=enhanced_numbers.dtype))
            else:
                pooled.append(enhanced_numbers[bidx, :node_count, :].mean(dim=0))
        return torch.stack(pooled, dim=0)

    def forward(self, input_seqs, input_lengths, bert_encoding, batch_graph=None, num_pos=None, hidden=None):
        device = next(self.bert_rnn.parameters()).device
        input_ids, attention_mask = _prepare_bert_inputs(bert_encoding, device)
        bert_output = self.bert_rnn(input_ids, attention_mask=attention_mask)[0].transpose(0, 1)
        if self.disable_graph or batch_graph is None or num_pos is None:
            problem_output = bert_output.mean(0)
            return bert_output, problem_output

        if not torch.is_tensor(batch_graph):
            batch_graph = torch.tensor(batch_graph, dtype=torch.float32, device=device)
        else:
            batch_graph = batch_graph.to(device=device, dtype=torch.float32)

        batch_size = len(num_pos)
        num_size = batch_graph.size(-1)
        num_encoder_outputs = get_all_number_encoder_outputs(
            bert_output, num_pos, batch_size, num_size, self.hidden_size
        )
        enhanced_numbers, graph_stats = self.graph_module(num_encoder_outputs, batch_graph)
        fused_output, fusion_gate_mean, valid_node_counts = self._fuse_numbers_back(bert_output, enhanced_numbers, num_pos)
        problem_output = fused_output.mean(0)

        number_summary = self._pool_enhanced_numbers(enhanced_numbers, valid_node_counts)
        problem_gate = torch.sigmoid(self.problem_gate(torch.cat([problem_output, number_summary], dim=-1)))
        mixed_problem = problem_gate * number_summary + (1 - problem_gate) * problem_output
        has_number = valid_node_counts.gt(0).float().unsqueeze(-1)
        problem_output = self.problem_norm(has_number * mixed_problem + (1 - has_number) * problem_output)

        relation_weights = graph_stats.get("relation_weights")
        if has_number.any():
            per_sample_problem_gate = problem_gate.mean(dim=-1)
            problem_gate_mean = per_sample_problem_gate[has_number.squeeze(-1).bool()].mean()
        else:
            problem_gate_mean = torch.tensor(0.0, device=problem_output.device)
        self._record_debug_stats(
            fusion_gate_mean=fusion_gate_mean.detach().item(),
            problem_gate_mean=problem_gate_mean.detach().item(),
            valid_nodes=valid_node_counts.sum().detach().item(),
            total_nodes=float(batch_size * num_size),
            relation_weights=relation_weights,
        )
        return fused_output, problem_output

    def evaluate(self, input_seqs, input_lengths, bert_encoding, batch_graph=None, num_pos=None):
        return self.forward(
            input_seqs, input_lengths, bert_encoding, batch_graph=batch_graph, num_pos=num_pos, hidden=None
        )

class GTS_Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=2, dropout=0.5):
        super(GTS_Encoder, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        self.em_dropout = nn.Dropout(dropout)
        self.gru_pade = nn.GRU(embedding_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)

    def forward(self, input_seqs, input_lengths, batch_graph, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)  # S x B x E
        embedded = self.em_dropout(embedded)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        pade_hidden = hidden
        pade_outputs, pade_hidden = self.gru_pade(packed, pade_hidden)
        pade_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs)

        problem_output = pade_outputs[-1, :, :self.hidden_size] + pade_outputs[0, :, self.hidden_size:]
        pade_outputs = pade_outputs[:, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]  # S x B x H

        return pade_outputs, problem_output


class EncoderRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=2, dropout=0.5):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        self.em_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embedding_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)  # S x B x E
        embedded = self.em_dropout(embedded)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        # S x B x H
        return outputs, hidden


class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, hidden, encoder_outputs, seq_mask=None):
        max_len = encoder_outputs.size(0)
        repeat_dims = [1] * hidden.dim()
        repeat_dims[0] = max_len
        hidden = hidden.repeat(*repeat_dims)  # S x B x H
        # For each position of encoder outputs
        this_batch_size = encoder_outputs.size(1)
        energy_in = torch.cat((hidden, encoder_outputs), 2).view(-1, 2 * self.hidden_size)
        attn_energies = self.score(torch.tanh(self.attn(energy_in)))  # (S x B) x 1
        attn_energies = attn_energies.squeeze(1)
        attn_energies = attn_energies.view(max_len, this_batch_size).transpose(0, 1)  # B x S
        if seq_mask is not None:
            attn_energies = attn_energies.masked_fill_(seq_mask.bool(), -1e12)
        attn_energies = self.softmax(attn_energies)
        # Normalize energies to weights in range 0 to 1, resize to B x 1 x S
        return attn_energies.unsqueeze(1)


class AttnDecoderRNN(nn.Module):
    def __init__(
            self, hidden_size, embedding_size, input_size, output_size, n_layers=2, dropout=0.5):
        super(AttnDecoderRNN, self).__init__()

        # Keep for reference
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.em_dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        self.gru = nn.GRU(hidden_size + embedding_size, hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        # Choose attention model
        self.attn = Attn(hidden_size)

    def forward(self, input_seq, last_hidden, encoder_outputs, seq_mask):
        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq)
        embedded = self.em_dropout(embedded)
        embedded = embedded.view(1, batch_size, self.embedding_size)  # S=1 x B x N

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(last_hidden[-1].unsqueeze(0), encoder_outputs, seq_mask)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # B x S=1 x N

        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.gru(torch.cat((embedded, context.transpose(0, 1)), 2), last_hidden)

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        output = self.out(torch.tanh(self.concat(torch.cat((rnn_output.squeeze(0), context.squeeze(1)), 1))))

        # Return final output, hidden state
        return output, hidden


class TreeNode:  # the class save the tree node
    def __init__(self, embedding, left_flag=False):
        self.embedding = embedding
        self.left_flag = left_flag


class Score(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Score, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, num_embeddings, num_mask=None):
        max_len = num_embeddings.size(1)
        repeat_dims = [1] * hidden.dim()
        repeat_dims[1] = max_len
        hidden = hidden.repeat(*repeat_dims)  # B x O x H
        # For each position of encoder outputs
        this_batch_size = num_embeddings.size(0)
        energy_in = torch.cat((hidden, num_embeddings), 2).view(-1, self.input_size + self.hidden_size)
        score = self.score(torch.tanh(self.attn(energy_in)))  # (B x O) x 1
        score = score.squeeze(1)
        score = score.view(this_batch_size, -1)  # B x O
        if num_mask is not None:
            score = score.masked_fill_(num_mask.bool(), -1e12)
        return score


class TreeAttn(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TreeAttn, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, hidden, encoder_outputs, seq_mask=None):
        max_len = encoder_outputs.size(0)

        repeat_dims = [1] * hidden.dim()
        repeat_dims[0] = max_len
        hidden = hidden.repeat(*repeat_dims)  # S x B x H
        this_batch_size = encoder_outputs.size(1)

        energy_in = torch.cat((hidden, encoder_outputs), 2).view(-1, self.input_size + self.hidden_size)

        score_feature = torch.tanh(self.attn(energy_in))
        attn_energies = self.score(score_feature)  # (S x B) x 1
        attn_energies = attn_energies.squeeze(1)
        attn_energies = attn_energies.view(max_len, this_batch_size).transpose(0, 1)  # B x S
        if seq_mask is not None:
            attn_energies = attn_energies.masked_fill_(seq_mask.bool(), -1e12)
        attn_energies = nn.functional.softmax(attn_energies, dim=1)  # B x S

        return attn_energies.unsqueeze(1)

class Encoder_rbt(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=2, dropout=0.5):
        super(Encoder_rbt, self).__init__()

        self.hidden_size = hidden_size
        #self.bert_rnn = BertModel.from_pretrained("hfl/chinese-bert-wwm-ext")
        #self.bert_rnn = BertModel.from_pretrained("/ibex/scratch/lianz0a/bert2tree-master/MWPbert-retrained/checkpoint-10000")
        #self.bert_rnn = BertModel.from_pretrained("/ibex/scratch/lianz0a/bert2tree-master/models/self_epoch_173")
        #self.bert_rnn = BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext")
        self.bert_rnn = BertModel.from_pretrained(DEFAULT_ZH_ROBERTA_PATH)
    def forward(self, input_seqs, input_lengths, bert_encoding, hidden=None, batch_graph=None, num_pos=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        device = next(self.bert_rnn.parameters()).device
        input_ids, attention_mask = _prepare_bert_inputs(bert_encoding, device)
        bert_output = self.bert_rnn(input_ids, attention_mask=attention_mask)[0].transpose(0,1)        

        problem_output = bert_output.mean(0)
        #pade_outputs = pade_outputs[:, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]  # S x B x H
        return bert_output, problem_output
    def evaluate(self, input_seqs, input_lengths, bert_encoding, batch_graph=None, num_pos=None):
        device = next(self.bert_rnn.parameters()).device
        input_ids, attention_mask = _prepare_bert_inputs(bert_encoding, device)
        bert_output = self.bert_rnn(input_ids, attention_mask=attention_mask)[0].transpose(0,1) # S x B x E
        problem_output = bert_output.mean(0)
        
        return bert_output, problem_output #seq_len, batch_size, H(768)

class EncoderSeq(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=2, dropout=0.5):
        super(EncoderSeq, self).__init__()

        self.hidden_size = hidden_size
        self.bert_rnn = BertModel.from_pretrained(DEFAULT_ZH_BERT_PATH)
        #self.bert_rnn = BertModel.from_pretrained("Your Path")
        
    def forward(self, input_seqs, input_lengths, bert_encoding, hidden=None, batch_graph=None, num_pos=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        device = next(self.bert_rnn.parameters()).device
        input_ids, attention_mask = _prepare_bert_inputs(bert_encoding, device)
        bert_output = self.bert_rnn(input_ids, attention_mask=attention_mask)[0].transpose(0,1)        

        problem_output = bert_output.mean(0)
        #pade_outputs = pade_outputs[:, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]  # S x B x H
        return bert_output, problem_output
    def evaluate(self, input_seqs, input_lengths, bert_encoding, batch_graph=None, num_pos=None):
        device = next(self.bert_rnn.parameters()).device
        input_ids, attention_mask = _prepare_bert_inputs(bert_encoding, device)
        bert_output = self.bert_rnn(input_ids, attention_mask=attention_mask)[0].transpose(0,1) # S x B x E
        problem_output = bert_output.mean(0)
        
        return bert_output, problem_output #seq_len, batch_size, H(768)

class Prediction(nn.Module):
    # a seq2tree decoder with Problem aware dynamic encoding

    def __init__(self, hidden_size, op_nums, input_size, dropout=0.5):
        super(Prediction, self).__init__()

        # Keep for reference
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.op_nums = op_nums

        # Define layers
        self.dropout = nn.Dropout(dropout)

        self.embedding_weight = nn.Parameter(torch.randn(1, input_size, hidden_size))

        # for Computational symbols and Generated numbers
        self.concat_l = nn.Linear(hidden_size, hidden_size)
        self.concat_r = nn.Linear(hidden_size * 2, hidden_size)
        self.concat_lg = nn.Linear(hidden_size, hidden_size)
        self.concat_rg = nn.Linear(hidden_size * 2, hidden_size)

        self.ops = nn.Linear(hidden_size * 2, op_nums)

        self.attn = TreeAttn(hidden_size, hidden_size)
        self.score = Score(hidden_size * 2, hidden_size)

    def forward(self, node_stacks, left_childs, encoder_outputs, num_pades, padding_hidden, seq_mask, mask_nums):
        current_embeddings = []

        for st in node_stacks:
            if len(st) == 0:
                current_embeddings.append(padding_hidden)
            else:
                current_node = st[-1]
                current_embeddings.append(current_node.embedding)

        current_node_temp = []
        for l, c in zip(left_childs, current_embeddings):
            if l is None:
                c = self.dropout(c)
                g = torch.tanh(self.concat_l(c))
                t = torch.sigmoid(self.concat_lg(c))
                current_node_temp.append(g * t)
            else:
                ld = self.dropout(l)
                c = self.dropout(c)
                g = torch.tanh(self.concat_r(torch.cat((ld, c), 1)))
                t = torch.sigmoid(self.concat_rg(torch.cat((ld, c), 1)))
                current_node_temp.append(g * t)

        current_node = torch.stack(current_node_temp)

        current_embeddings = self.dropout(current_node)

        current_attn = self.attn(current_embeddings.transpose(0, 1), encoder_outputs, seq_mask) #seq_len, batch_size, 768
        current_context = current_attn.bmm(encoder_outputs.transpose(0, 1))  # B x 1 x N

        # the information to get the current quantity
        batch_size = current_embeddings.size(0)
        # predict the output (this node corresponding to output(number or operator)) with PADE

        repeat_dims = [1] * self.embedding_weight.dim()
        repeat_dims[0] = batch_size
        embedding_weight = self.embedding_weight.repeat(*repeat_dims)  # B x input_size x N
        embedding_weight = torch.cat((embedding_weight, num_pades), dim=1)  # B x O x N

        leaf_input = torch.cat((current_node, current_context), 2)
        leaf_input = leaf_input.squeeze(1)
        leaf_input = self.dropout(leaf_input)

        # p_leaf = nn.functional.softmax(self.is_leaf(leaf_input), 1)
        # max pooling the embedding_weight
        embedding_weight_ = self.dropout(embedding_weight)
        num_score = self.score(leaf_input.unsqueeze(1), embedding_weight_, mask_nums)

        # num_score = nn.functional.softmax(num_score, 1)

        op = self.ops(leaf_input)

        # return p_leaf, num_score, op, current_embeddings, current_attn

        return num_score, op, current_node, current_context, embedding_weight


class GenerateNode(nn.Module):
    def __init__(self, hidden_size, op_nums, embedding_size, dropout=0.5):
        super(GenerateNode, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.embeddings = nn.Embedding(op_nums, embedding_size)
        self.em_dropout = nn.Dropout(dropout)
        self.generate_l = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_r = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_lg = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_rg = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, node_label, current_context):
        node_label_ = self.embeddings(node_label)
        node_label = self.em_dropout(node_label_)
        node_embedding = node_embedding.squeeze(1)
        current_context = current_context.squeeze(1)
        node_embedding = self.em_dropout(node_embedding)
        current_context = self.em_dropout(current_context)

        l_child = torch.tanh(self.generate_l(torch.cat((node_embedding, current_context, node_label), 1)))
        l_child_g = torch.sigmoid(self.generate_lg(torch.cat((node_embedding, current_context, node_label), 1)))
        r_child = torch.tanh(self.generate_r(torch.cat((node_embedding, current_context, node_label), 1)))
        r_child_g = torch.sigmoid(self.generate_rg(torch.cat((node_embedding, current_context, node_label), 1)))
        l_child = l_child * l_child_g
        r_child = r_child * r_child_g
        return l_child, r_child, node_label_


class Merge(nn.Module):
    def __init__(self, hidden_size, embedding_size, dropout=0.5):
        super(Merge, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.em_dropout = nn.Dropout(dropout)
        self.merge = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.merge_g = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, sub_tree_1, sub_tree_2):
        sub_tree_1 = self.em_dropout(sub_tree_1)
        sub_tree_2 = self.em_dropout(sub_tree_2)
        node_embedding = self.em_dropout(node_embedding)

        sub_tree = torch.tanh(self.merge(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree_g = torch.sigmoid(self.merge_g(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree = sub_tree * sub_tree_g
        return sub_tree
