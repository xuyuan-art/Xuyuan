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
import random
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


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
parser.add_argument("--n_epochs", type=int, default=85)
parser.add_argument("--learning_rate", type=float, default=3e-5)
parser.add_argument("--weight_decay", type=float, default=1e-5)
parser.add_argument("--beam_size", type=int, default=5)
parser.add_argument("--n_layers", type=int, default=2)
parser.add_argument("--step_size", type=int, default=30)
parser.add_argument("--eval_every", type=int, default=10)
parser.add_argument("--log_every", type=int, default=100)
parser.add_argument("--save_dir", type=str, default="models")
parser.add_argument("--save_each_epoch", type=int, default=1)
parser.add_argument("--epoch_ckpt_dir", type=str, default="epoch_checkpoints")
parser.add_argument("--data_dir", type=str, default="./data")
parser.add_argument("--ori_path", type=str, default=None)
parser.add_argument("--prefix", type=str, default="23k_processed.json")
parser.add_argument("--math23k_path", type=str, default="Math_23K.json")
parser.add_argument("--math23k_processed_path", type=str, default="Math_23K_processed.json")
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


def save_model_bundle(save_root, encoder, predict, generate, merge):
    os.makedirs(save_root, exist_ok=True)
    torch.save(unwrap_parallel(encoder).state_dict(), os.path.join(save_root, "encoder_"))
    torch.save(unwrap_parallel(predict).state_dict(), os.path.join(save_root, "predict_"))
    torch.save(unwrap_parallel(generate).state_dict(), os.path.join(save_root, "generate_"))
    torch.save(unwrap_parallel(merge).state_dict(), os.path.join(save_root, "merge_"))


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


def get_train_test_fold(ori_path, prefix, data, pairs, group):
    mode_train = 'train'
    mode_valid = 'valid'
    mode_test = 'test'
    train_path = os.path.join(ori_path, mode_train + prefix)
    valid_path = os.path.join(ori_path, mode_valid + prefix)
    test_path = os.path.join(ori_path, mode_test + prefix)
    train = read_json(train_path)
    train_id = [item['id'] for item in train]
    valid = read_json(valid_path)
    valid_id = [item['id'] for item in valid]
    test = read_json(test_path)
    test_id = [item['id'] for item in test]
    train_fold = []
    valid_fold = []
    test_fold = []
    for item, pair, g in zip(data, pairs, group):
        pair = list(pair)
        pair.append(g['group_num'])
        pair = tuple(pair)
        if item['id'] in train_id:
            train_fold.append(pair)
        elif item['id'] in test_id:
            test_fold.append(pair)
        else:
            valid_fold.append(pair)
    return train_fold, test_fold, valid_fold


def main():
    use_ddp = _is_distributed()
    if use_ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            backend = "nccl"
        else:
            backend = "gloo"
        dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        local_rank = int(os.environ.get("MWP_CUDA_DEVICE", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        rank = 0
        world_size = 1

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
    os.makedirs(args.save_dir, exist_ok=True)
    epoch_ckpt_root = os.path.join(args.save_dir, args.epoch_ckpt_dir)
    if args.save_each_epoch and _is_primary():
        os.makedirs(epoch_ckpt_root, exist_ok=True)

    data = load_raw_data(math23k_path)
    group_data = read_json(math23k_processed_path)

    pairs, generate_nums, copy_nums = transfer_num(data)
    temp_pairs = []
    for p in pairs:
        temp_pairs.append((p[0], from_infix_to_prefix(p[1]), p[2], p[3]))
    pairs = temp_pairs

    train_fold, test_fold, valid_fold = get_train_test_fold(ori_path, prefix, data, pairs, group_data)
    pairs_tested = test_fold
    pairs_trained = train_fold

    input_lang, output_lang, train_pairs, test_pairs = prepare_data_23k(
        pairs_trained, pairs_tested, 5, generate_nums, copy_nums, tree=True
    )

    encoder = EncoderSeq(
        input_size=input_lang.n_words,
        embedding_size=embedding_size,
        hidden_size=hidden_size,
        n_layers=n_layers,
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
        # BERT pooler parameters can stay unused in this training graph; enable unused-param handling for DDP.
        ddp_kwargs = {"find_unused_parameters": True}
        if USE_CUDA:
            encoder = TransparentDDP(encoder, device_ids=[local_rank], output_device=local_rank, **ddp_kwargs)
            predict = TransparentDDP(predict, device_ids=[local_rank], output_device=local_rank, **ddp_kwargs)
            generate = TransparentDDP(generate, device_ids=[local_rank], output_device=local_rank, **ddp_kwargs)
            merge = TransparentDDP(merge, device_ids=[local_rank], output_device=local_rank, **ddp_kwargs)
        else:
            encoder = TransparentDDP(encoder, **ddp_kwargs)
            predict = TransparentDDP(predict, **ddp_kwargs)
            generate = TransparentDDP(generate, **ddp_kwargs)
            merge = TransparentDDP(merge, **ddp_kwargs)
    else:
        encoder = maybe_wrap_encoder_dp(encoder, use_ddp=False)

    encoder_optimizer = torch.optim.AdamW(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    predict_optimizer = torch.optim.AdamW(predict.parameters(), lr=learning_rate * 10, weight_decay=weight_decay)
    generate_optimizer = torch.optim.AdamW(generate.parameters(), lr=learning_rate * 10, weight_decay=weight_decay)
    merge_optimizer = torch.optim.AdamW(merge.parameters(), lr=learning_rate * 10, weight_decay=weight_decay)

    encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=args.step_size, gamma=0.5)
    predict_scheduler = torch.optim.lr_scheduler.StepLR(predict_optimizer, step_size=args.step_size, gamma=0.5)
    generate_scheduler = torch.optim.lr_scheduler.StepLR(generate_optimizer, step_size=args.step_size, gamma=0.5)
    merge_scheduler = torch.optim.lr_scheduler.StepLR(merge_optimizer, step_size=args.step_size, gamma=0.5)

    generate_num_ids = [output_lang.word2index[num] for num in generate_nums]

    for epoch in range(n_epochs):
        random.seed(1000 + epoch)
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
        ) = prepare_train_batch(train_pairs, batch_size)

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
            )
            loss_total += loss
            if _is_primary() and (local_step % args.log_every == 0 or local_step == len(rank_batch_indices)):
                avg_loss = loss_total / max(local_step, 1)
                elapsed = time_since(time.time() - start)
                current_lr = encoder_optimizer.param_groups[0]["lr"]
                print(
                    f"[train] epoch={epoch + 1}/{n_epochs} step={local_step}/{len(rank_batch_indices)} "
                    f"loss={loss:.4f} avg_loss={avg_loss:.4f} lr={current_lr:.2e} elapsed={elapsed}"
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
            print(
                f"epoch={epoch + 1}  avg_loss={global_avg_loss:.4f} "
                f"elapsed={time_since(time.time() - start)}"
            )
            print("--------------------------------")

        if args.save_each_epoch:
            if use_ddp:
                dist.barrier()
            if _is_primary():
                epoch_ckpt_path = os.path.join(epoch_ckpt_root, f"epoch_{epoch + 1:03d}")
                save_model_bundle(epoch_ckpt_path, encoder, predict, generate, merge)
                print(f"[save] epoch_checkpoint={epoch_ckpt_path}")
            if use_ddp:
                dist.barrier()

        if epoch % args.eval_every == 0 or epoch > n_epochs - args.eval_every:
            if use_ddp:
                dist.barrier()
            if _is_primary():
                value_ac = 0
                equation_ac = 0
                eval_total = 0
                eval_start = time.time()
                for test_batch in test_pairs:
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
                        beam_size=beam_size,
                    )
                    val_ac, equ_ac, _, _ = compute_prefix_tree_result(
                        test_res, test_batch[2], output_lang, test_batch[4], test_batch[6]
                    )
                    if val_ac:
                        value_ac += 1
                    if equ_ac:
                        equation_ac += 1
                    eval_total += 1
                print(equation_ac, value_ac, eval_total)
                print("test_answer_acc", float(equation_ac) / eval_total, float(value_ac) / eval_total)
                print("testing time", time_since(time.time() - eval_start))
                print("------------------------------------------------------")
                save_model_bundle(args.save_dir, encoder, predict, generate, merge)
            if use_ddp:
                dist.barrier()

    if use_ddp and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
