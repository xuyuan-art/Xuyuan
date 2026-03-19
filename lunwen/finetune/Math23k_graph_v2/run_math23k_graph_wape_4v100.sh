#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# If you need fixed visible devices, uncomment:
# export CUDA_VISIBLE_DEVICES=0,1,2,3

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29501}"
DEFAULT_PYTHON_BIN="/root/miniconda3/envs/mwp-bert-5090/bin/python"
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON_BIN}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python)"
fi
# Force a safe default for multi-process CPU preprocessing.
# Override by setting OMP_NUM_THREADS_OVERRIDE explicitly when needed.
export OMP_NUM_THREADS="${OMP_NUM_THREADS_OVERRIDE:-1}"

echo "[launch] python=$PYTHON_BIN nproc_per_node=$NPROC_PER_NODE master_port=$MASTER_PORT omp_threads=$OMP_NUM_THREADS"
"$PYTHON_BIN" -m torch.distributed.run \
  --nproc_per_node "$NPROC_PER_NODE" \
  --master_port "$MASTER_PORT" \
  math23k_graph_wape_4v100.py "$@"
