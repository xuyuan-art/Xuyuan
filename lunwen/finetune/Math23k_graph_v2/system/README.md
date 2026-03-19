# MWP Solver System

Use the trained bundle (for example `finetune_models-zh-v2`) as an inference system.

## 1) One-shot CLI solve

```bash
cd /root/lunwen/finetune/Math23k_graph_v2
MWP_ZH_BERT_PATH=/root/MWP-BERT_zh_hf \
/root/miniconda3/envs/mwp-bert-5090/bin/python system/mwp_solver_system.py \
  --model_dir /root/autodl-tmp/finetune_models-zh-v2 \
  --device cpu \
  --text "小明每本书2元，买11本一共多少钱？"
```

## 2) HTTP service

```bash
cd /root/lunwen/finetune/Math23k_graph_v2
MWP_ZH_BERT_PATH=/root/MWP-BERT_zh_hf \
/root/miniconda3/envs/mwp-bert-5090/bin/python system/mwp_solver_system.py \
  --model_dir /root/autodl-tmp/finetune_models-zh-v2 \
  --device cpu \
  --serve \
  --host 0.0.0.0 \
  --port 18081
```

Health check:

```bash
curl http://127.0.0.1:18081/health
```

Solve:

```bash
curl -X POST http://127.0.0.1:18081/solve \
  -H "Content-Type: application/json" \
  -d '{"text":"镇海雅乐学校二年级的小朋友到一条小路的一边植树。小朋友们每隔2米种一棵树（马路两头都种了树），最后发现一共种了11棵，这条小路长多少米？"}'
```

## Notes

- Defaults to CPU to avoid interfering with ongoing training.
- If `--model_dir` has `epoch_checkpoints` only, latest epoch is picked automatically.
- A metadata cache file (`runtime_meta_23k_graph.pkl`) is generated in `model_dir` to speed up subsequent starts.

## 3) Full Web App (Frontend MPA + Backend API)

Includes:

- Static frontend pages in `system/frontend/` (MPA)
- Backend API in `system/web_app.py` (`/api/*`)
- Smart solving + smart question generation
- Stats center with history CRUD + heatmap + user portrait
- Admin backend (`/admin`) with user management and error logs

Run backend API:

```bash
cd /root/lunwen/finetune/Math23k_graph_v2
MWP_ZH_BERT_PATH=/root/MWP-BERT_zh_hf \
/root/miniconda3/envs/mwp-bert-5090/bin/python system/web_app.py \
  --host 0.0.0.0 \
  --port 6008 \
  --db_path /root/autodl-tmp/mwp_web_app/webapp.db \
  --model_dir /root/autodl-tmp/finetune_models-zh-v2 \
  --device cpu
```

Run frontend static server:

```bash
cd /root/lunwen/finetune/Math23k_graph_v2/system/frontend
/root/miniconda3/envs/mwp-bert-5090/bin/python -m http.server 6011
```

Then open:

```bash
http://127.0.0.1:6011
```

Production/long-running suggestion (`tmux`):

```bash
tmux new-session -d -s mwp_web_app "\
cd /root/lunwen/finetune/Math23k_graph_v2/system && \
/root/miniconda3/envs/mwp-bert-5090/bin/python web_app.py \
  --host 0.0.0.0 \
  --port 6008 \
  --db_path /root/autodl-tmp/mwp_web_app/webapp.db \
  --model_dir /root/autodl-tmp/finetune_models-zh-v2 \
  --data_dir /root/lunwen/finetune/Math23k_graph_v2/data \
  --device cpu \
  --beam_size 5 \
  --admin_username admin \
  >> /root/autodl-tmp/mwp_web_app/web.log 2>&1"
```

Stop/restart:

```bash
tmux kill-session -t mwp_web_app
```

Detailed web app doc:

- `WEB_APP_GUIDE.md`
- `RESTART_GUIDE.md`（服务器关闭后如何快速重启）
