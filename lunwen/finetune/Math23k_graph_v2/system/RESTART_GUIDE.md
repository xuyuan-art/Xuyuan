# 服务器关闭后如何重启系统（前后端分离版）

适用环境：当前这台服务器 + 当前项目目录。

## 固定信息（按当前部署）

- 项目目录：`/root/lunwen/finetune/Math23k_graph_v2/system`
- 前端目录：`/root/lunwen/finetune/Math23k_graph_v2/system/frontend`
- Python：`/root/miniconda3/envs/mwp-bert-5090/bin/python`
- 模型目录：`/root/autodl-tmp/finetune_models-zh-v2`
- 数据目录：`/root/lunwen/finetune/Math23k_graph_v2/data`
- 数据库：`/root/autodl-tmp/mwp_web_app/webapp.db`
- 后端日志：`/root/autodl-tmp/mwp_web_app/web.log`
- 前端日志：`/root/autodl-tmp/mwp_web_app/frontend.log`
- 后端会话名：`mwp_web_app`
- 前端会话名：`mwp_frontend`
- 后端端口：`6008`
- 前端端口：`6011`

## 一、重启系统（推荐）

### 1) 清理旧进程

```bash
tmux kill-session -t mwp_web_app 2>/dev/null || true
tmux kill-session -t mwp_frontend 2>/dev/null || true
pkill -f "python web_app.py --host 0.0.0.0 --port 6008" 2>/dev/null || true
pkill -f "python -m http.server 6011" 2>/dev/null || true
```

### 2) 启动后端（API）

```bash
mkdir -p /root/autodl-tmp/mwp_web_app
tmux new-session -d -s mwp_web_app "\
cd /root/lunwen/finetune/Math23k_graph_v2/system && \
MWP_APP_TIMEZONE=Asia/Shanghai \
MWP_ZH_BERT_PATH=/root/MWP-BERT_zh_hf \
/root/miniconda3/envs/mwp-bert-5090/bin/python web_app.py \
  --host 0.0.0.0 \
  --port 6008 \
  --db_path /root/autodl-tmp/mwp_web_app/webapp.db \
  --model_dir /root/autodl-tmp/finetune_models-zh-v2 \
  --data_dir /root/lunwen/finetune/Math23k_graph_v2/data \
  --device cpu \
  --beam_size 5 \
  --admin_username admin \
  --cors_allowed_origins '*' \
  >> /root/autodl-tmp/mwp_web_app/web.log 2>&1"
```

### 3) 启动前端静态服务

```bash
tmux new-session -d -s mwp_frontend "\
cd /root/lunwen/finetune/Math23k_graph_v2/system/frontend && \
/root/miniconda3/envs/mwp-bert-5090/bin/python -m http.server 6011 \
  >> /root/autodl-tmp/mwp_web_app/frontend.log 2>&1"
```

### 4) 检查是否启动成功

```bash
tmux ls | grep -E "mwp_web_app|mwp_frontend"
curl -s http://127.0.0.1:6008/api/health
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:6011/
```

期望：

- `api/health` 返回 `frontend_mode: api_only`
- 前端首页返回 `200`

### 5) 浏览器访问

- 前端入口：`http://127.0.0.1:6011/`
- 后端接口健康检查：`http://127.0.0.1:6008/api/health`

## 二、停止系统

```bash
tmux kill-session -t mwp_web_app 2>/dev/null || true
tmux kill-session -t mwp_frontend 2>/dev/null || true
```

## 三、查看实时日志

```bash
tail -f /root/autodl-tmp/mwp_web_app/web.log
tail -f /root/autodl-tmp/mwp_web_app/frontend.log
```

## 四、一条命令重启（复制即用）

```bash
tmux kill-session -t mwp_web_app 2>/dev/null || true; tmux kill-session -t mwp_frontend 2>/dev/null || true; pkill -f "python web_app.py --host 0.0.0.0 --port 6008" 2>/dev/null || true; pkill -f "python -m http.server 6011" 2>/dev/null || true; mkdir -p /root/autodl-tmp/mwp_web_app; tmux new-session -d -s mwp_web_app "cd /root/lunwen/finetune/Math23k_graph_v2/system && MWP_APP_TIMEZONE=Asia/Shanghai MWP_ZH_BERT_PATH=/root/MWP-BERT_zh_hf /root/miniconda3/envs/mwp-bert-5090/bin/python web_app.py --host 0.0.0.0 --port 6008 --db_path /root/autodl-tmp/mwp_web_app/webapp.db --model_dir /root/autodl-tmp/finetune_models-zh-v2 --data_dir /root/lunwen/finetune/Math23k_graph_v2/data --device cpu --beam_size 5 --admin_username admin --cors_allowed_origins '*' >> /root/autodl-tmp/mwp_web_app/web.log 2>&1"; tmux new-session -d -s mwp_frontend "cd /root/lunwen/finetune/Math23k_graph_v2/system/frontend && /root/miniconda3/envs/mwp-bert-5090/bin/python -m http.server 6011 >> /root/autodl-tmp/mwp_web_app/frontend.log 2>&1"
```

## 五、常见问题

### 1) 端口占用（Address already in use）

先执行“清理旧进程”，再重新启动。

### 2) 页面 404

- 先确认你访问的是前端地址 `http://127.0.0.1:6011/`。
- `http://127.0.0.1:6008/` 会返回 JSON，不是前端页面。

### 3) 前端能打开但接口失败

- 检查 `http://127.0.0.1:6008/api/health` 是否正常。
- 检查 `frontend/config.js` 中 `apiBase` 是否正确。
