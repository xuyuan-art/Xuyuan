# MWP Web App 文档说明

本文档对应 `system/web_app.py` + `system/frontend/` 当前实现（2026-03-18）。

## 1. 架构模式

当前仅支持前后端分离：

- 前端：静态多页（MPA，非 SPA）
- 后端：`/api/*` JSON 接口

说明：

- 前端目录：`system/frontend/`
- 前端页面：`index.html`、`login.html`、`home.html`、`solve.html`、`stats.html`、`profile.html`、`admin.html`、`docs.html`
- 前端脚本：`js/common.js`、`js/login.js`、`js/app-pages.js`
- 样式：`css/common.css`（并保留 `app.css` 兼容入口）

## 2. 后端路由

### 2.1 API 路由

#### 系统与会话

- `GET /api/health`
- `GET /api/auth/me`
- `POST /api/auth/register`
- `POST /api/auth/login`
- `POST /api/auth/logout`
- `POST /api/online/heartbeat`

#### 普通用户

- `GET /api/docs`
- `GET /api/home/overview`
- `GET /api/stats/overview`
- `GET /api/profile`
- `POST /api/profile/verify`
- `POST /api/profile/update`

#### 解题与历史

- `POST /api/solve`
- `GET /api/history?page=&page_size=`
- `GET /api/history/{id}`
- `POST /api/history`
- `PUT /api/history/{id}`
- `DELETE /api/history/{id}`

#### 管理后台

- `GET /api/admin/overview?page=&page_size=`
- `POST /api/admin/users`
- `POST /api/admin/users/{id}/toggle-admin`
- `POST /api/admin/users/{id}/toggle-active`
- `POST /api/admin/users/{id}/reset-password`
- `DELETE /api/admin/users/{id}`
- `DELETE /api/admin/errors`

## 3. 启动方式（前后端分离）

1) 启动后端（仅 API）：

```bash
cd /root/lunwen/finetune/Math23k_graph_v2/system
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
  --cors_allowed_origins "*"
```

2) 启动前端静态服务：

```bash
cd /root/lunwen/finetune/Math23k_graph_v2/system/frontend
/root/miniconda3/envs/mwp-bert-5090/bin/python -m http.server 6011
```

3) 浏览器访问前端：

```text
http://127.0.0.1:6011/
```

`config.js` 会优先使用：

- `?apiBase=https://...` 查询参数
- `localStorage.__MWP_API_BASE__`
- 自动推断 `当前主机:6008`

## 4. 备注

- 当前后端恒为 API-only：`/` 返回服务信息 JSON，页面入口应从前端静态服务访问。
- 若前端页面正常但接口报跨域，请检查后端 `--cors_allowed_origins`。
- 会话 Cookie 默认：`HttpOnly`、`SameSite=Lax`、有效期 7 天。
