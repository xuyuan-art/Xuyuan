#!/usr/bin/env bash
set -euo pipefail

WATCH_SAVE_DIR="/root/autodl-tmp/finetune_models-zh"
PROJECT_DIR="/root/lunwen/finetune/Math23k_graph_v2"
RUN_SAVE_DIR="/root/autodl-tmp/finetune_models-zh-v2"
PYTHON_BIN="/root/miniconda3/envs/mwp-bert-5090/bin/python"
BERT_PATH="/root/MWP-BERT_zh_hf"
POLL_SECONDS=120

is_watch_running() {
  while IFS= read -r line; do
    comm="$(echo "$line" | awk '{print $1}')"
    case "$comm" in
      python|python3|python3.*)
        if [[ "$line" == *"math23k_graph.py"* ]] && [[ "$line" =~ --save_dir[[:space:]]+${WATCH_SAVE_DIR}([[:space:]]|$) ]]; then
          return 0
        fi
        ;;
    esac
  done < <(ps -eo comm=,args=)
  return 1
}

ts() {
  date -u '+%Y-%m-%d %H:%M:%S UTC'
}

mkdir -p "${RUN_SAVE_DIR}"

echo "[$(ts)] queue watcher started"
echo "[$(ts)] waiting for run with --save_dir ${WATCH_SAVE_DIR} to finish"

while is_watch_running; do
  echo "[$(ts)] target still running, sleep ${POLL_SECONDS}s"
  sleep "${POLL_SECONDS}"
done

echo "[$(ts)] target finished, launching queued training"
cd "${PROJECT_DIR}"

export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1
export MWP_ZH_BERT_PATH="${BERT_PATH}"
export PYTHONUNBUFFERED=1

"${PYTHON_BIN}" -m torch.distributed.run --nproc_per_node=2 math23k_graph.py \
  --data_dir ./data \
  --save_dir "${RUN_SAVE_DIR}" \
  --batch_size 32 \
  --learning_rate 3e-5 \
  --eval_every 5 \
  --graph_log_every 100 \
  2>&1 | tee "${RUN_SAVE_DIR}/train.log"

echo "[$(ts)] queued training finished"
