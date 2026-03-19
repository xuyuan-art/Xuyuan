# coding: utf-8
from src.train_and_evaluate import *
from src.models import *
import time
import torch.optim
from src.expressions_transfer import *
import json
import torch
import os
import argparse

if torch.cuda.is_available():
    torch.cuda.set_device(int(os.environ.get("MWP_CUDA_DEVICE", "0")))
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--embedding_size", type=int, default=128)
parser.add_argument("--hidden_size", type=int, default=768)
parser.add_argument("--n_epochs", type=int, default=100)
parser.add_argument("--learning_rate", type=float, default=3e-5)
parser.add_argument("--weight_decay", type=float, default=1e-5)
parser.add_argument("--beam_size", type=int, default=5)
parser.add_argument("--n_layers", type=int, default=2)
parser.add_argument("--step_size", type=int, default=25)
parser.add_argument("--eval_every", type=int, default=5)
parser.add_argument("--save_dir", type=str, default="models")
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
args = parser.parse_args()


def maybe_wrap_encoder_dp(model):
    use_encoder_dp = os.environ.get("MWP_ENCODER_DP", "1") != "0"
    if USE_CUDA and use_encoder_dp and torch.cuda.device_count() > 1:
        return TransparentDataParallel(model)
    return model


def unwrap_parallel(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(args.data_dir, path)


def ensure_dir_path(path):
    if path.endswith(os.sep):
        return path
    return path + os.sep

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
math23k_processed_path = resolve_path(args.math23k_processed_path)
ape_path = resolve_path(args.ape_path)
ape_test_path = resolve_path(args.ape_test_path)
ape_id = resolve_path(args.ape_id)
ape_test_id = resolve_path(args.ape_test_id)
overlap_id = resolve_path(args.overlap_id)
test_overlap_id = resolve_path(args.test_overlap_id)
os.makedirs(args.save_dir, exist_ok=True)

id_list = open(ape_id, 'r').read().split()
id_list = [str(i) for i in id_list]
test_id_list = open(ape_test_id, 'r').read().split()
test_id_list = [str(i) for i in test_id_list]
overlap_list = open(overlap_id, 'r').read().split()
overlap_list = [str(i) for i in overlap_list]
test_overlap_list = open(test_overlap_id, 'r').read().split()
test_overlap_list = [str(i) for i in test_overlap_list]


data = load_raw_data(math23k_path) + raw_data_new(ape_path, id_list, overlap_list) + raw_data_new(ape_test_path, test_id_list, test_overlap_list)
#data = raw_data_new("data/ape_simple_train.json", id_list, overlap_list) + raw_data_new("data/ape_simple_test.json", test_id_list, test_overlap_list)
group_data = read_json(math23k_processed_path)


pairs, generate_nums, copy_nums = transfer_num(data)

temp_pairs = []
for p in pairs:
    temp_pairs.append((p[0], from_infix_to_prefix(p[1]), p[2], p[3]))
pairs = temp_pairs

train_fold, test_fold, valid_fold = get_train_test_fold(ori_path,prefix,data,pairs,group_data,ape_path,ape_id, ape_test_id)


best_acc_fold = []

pairs_tested = test_fold
pairs_tested_ape = valid_fold
pairs_trained = train_fold

input_lang, output_lang, train_pairs, test_pairs, test_pairs_ape = prepare_data(pairs_trained, pairs_tested, pairs_tested_ape, 5, generate_nums,
                                                                copy_nums, tree=True)

encoder = EncoderSeq(input_size=input_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size,
                     n_layers=n_layers)
predict = Prediction(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                     input_size=len(generate_nums))
generate = GenerateNode(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                        embedding_size=embedding_size)
merge = Merge(hidden_size=hidden_size, embedding_size=embedding_size)
# the embedding layer is  only for generated number embeddings, operators, and paddings


encoder_optimizer = torch.optim.AdamW(encoder.parameters(), lr=learning_rate, weight_decay=1e-2)
predict_optimizer = torch.optim.AdamW(predict.parameters(), lr=learning_rate * 10, weight_decay=weight_decay)
generate_optimizer = torch.optim.AdamW(generate.parameters(), lr=learning_rate* 10, weight_decay=weight_decay)
merge_optimizer = torch.optim.AdamW(merge.parameters(), lr=learning_rate* 10, weight_decay=weight_decay)

encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=args.step_size, gamma=0.5)
predict_scheduler = torch.optim.lr_scheduler.StepLR(predict_optimizer, step_size=args.step_size, gamma=0.5)
generate_scheduler = torch.optim.lr_scheduler.StepLR(generate_optimizer, step_size=args.step_size, gamma=0.5)
merge_scheduler = torch.optim.lr_scheduler.StepLR(merge_optimizer, step_size=args.step_size, gamma=0.5)

# Move models to GPU
if USE_CUDA:
    encoder.cuda()
    predict.cuda()
    generate.cuda()
    merge.cuda()

encoder = maybe_wrap_encoder_dp(encoder)

generate_num_ids = []
for num in generate_nums:
    generate_num_ids.append(output_lang.word2index[num])

for epoch in range(n_epochs):

    loss_total = 0
    input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches, bert_batches = prepare_train_batch(train_pairs, batch_size)
    print("epoch:", epoch + 1)
    start = time.time()
    for idx in range(len(input_lengths)):

        loss = train_tree(
            input_batches[idx], input_lengths[idx], output_batches[idx], output_lengths[idx],
            num_stack_batches[idx], num_size_batches[idx], generate_num_ids, encoder, predict, generate, merge,
            encoder_optimizer, predict_optimizer, generate_optimizer, merge_optimizer, output_lang, num_pos_batches[idx], bert_batches[idx])
        loss_total += loss
    encoder_scheduler.step()
    predict_scheduler.step()
    generate_scheduler.step()
    merge_scheduler.step()
    print("loss:", loss_total / len(input_lengths))
    print("training time", time_since(time.time() - start))
    print("--------------------------------")
    if epoch % args.eval_every == 0 or epoch > n_epochs - (args.eval_every * 2):
        value_ac = 0
        equation_ac = 0
        eval_total = 0
        start = time.time()
        for test_batch in test_pairs:
            #print(test_batch)
            #batch_graph = get_single_example_graph(test_batch[0], test_batch[1], test_batch[7], test_batch[4], test_batch[5])
            test_res = evaluate_tree(test_batch[0], test_batch[1], generate_num_ids, encoder, predict, generate,
                                         merge, output_lang, test_batch[5], test_batch[7], beam_size=beam_size)
            val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[6])
            if val_ac:
                value_ac += 1
            if equ_ac:
                equation_ac += 1
            eval_total += 1
        print(equation_ac, value_ac, eval_total)
        print("test_answer_acc_23k", float(equation_ac) / eval_total, float(value_ac) / eval_total)
        print("testing time", time_since(time.time() - start))
        print("------------------------------------------------------")
        value_ac = 0
        equation_ac = 0
        eval_total = 0
        start = time.time()
        for test_batch in test_pairs_ape:
            #print(test_batch)
            #batch_graph = get_single_example_graph(test_batch[0], test_batch[1], test_batch[7], test_batch[4], test_batch[5])
            test_res = evaluate_tree(test_batch[0], test_batch[1], generate_num_ids, encoder, predict, generate,
                                         merge, output_lang, test_batch[5], test_batch[7], beam_size=beam_size)
            val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[6])
            if val_ac:
                value_ac += 1
            if equ_ac:
                equation_ac += 1
            eval_total += 1
        print(equation_ac, value_ac, eval_total)
        print("test_answer_acc_ape", float(equation_ac) / eval_total, float(value_ac) / eval_total)
        print("testing time", time_since(time.time() - start))
        print("------------------------------------------------------")
        torch.save(unwrap_parallel(encoder).state_dict(), os.path.join(args.save_dir, "encoder_new"))
        torch.save(predict.state_dict(), os.path.join(args.save_dir, "predict_new"))
        torch.save(generate.state_dict(), os.path.join(args.save_dir, "generate_new"))
        torch.save(merge.state_dict(), os.path.join(args.save_dir, "merge_new"))
