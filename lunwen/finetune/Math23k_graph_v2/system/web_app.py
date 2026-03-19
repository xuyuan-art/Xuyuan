#!/usr/bin/env python3
# coding: utf-8

import argparse
import html
import json
import math
import os
import re
import secrets
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from hashlib import pbkdf2_hmac
from hmac import compare_digest
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlparse
try:
    from zoneinfo import ZoneInfo
except Exception:
    try:
        from backports.zoneinfo import ZoneInfo  # type: ignore
    except Exception:
        class ZoneInfo:  # type: ignore[override]
            _FALLBACKS = {
                "UTC": timezone.utc,
                "Asia/Shanghai": timezone(timedelta(hours=8)),
            }

            def __new__(cls, key: str):
                if key in cls._FALLBACKS:
                    return cls._FALLBACKS[key]
                raise ValueError(f"Unsupported timezone without zoneinfo support: {key}")

from mwp_solver_system import Math23kSolverSystem


ROOT = Path(__file__).resolve().parent
DEFAULT_DB_PATH = ROOT / "webapp.db"
SESSION_TTL_SECONDS = 7 * 24 * 3600
ONLINE_HEARTBEAT_INTERVAL_SECONDS = 30
ONLINE_HEARTBEAT_MAX_GAP_SECONDS = 120
PROFILE_VERIFY_TTL_SECONDS = 10 * 60
DEFAULT_APP_TIMEZONE = "Asia/Shanghai"
APP_TIMEZONE_NAME = os.getenv("MWP_APP_TIMEZONE", DEFAULT_APP_TIMEZONE)
DEFAULT_CORS_ALLOW_HEADERS = "Content-Type, Accept"
DEFAULT_CORS_ALLOW_METHODS = "GET, POST, PUT, DELETE, OPTIONS"
try:
    APP_TZ = ZoneInfo(APP_TIMEZONE_NAME)
except Exception:
    APP_TIMEZONE_NAME = "UTC"
    APP_TZ = ZoneInfo(APP_TIMEZONE_NAME)


def parse_origin_list(*raw_values: Optional[str]) -> List[str]:
    seen = set()
    origins: List[str] = []
    for raw in raw_values:
        if not raw:
            continue
        for item in str(raw).split(","):
            origin = item.strip().rstrip("/")
            if not origin or origin in seen:
                continue
            seen.add(origin)
            origins.append(origin)
    return origins


def now_ts() -> int:
    return int(time.time())


def local_now() -> datetime:
    return datetime.now(APP_TZ)


def local_day_start_ts(day_obj) -> int:
    return int(datetime(day_obj.year, day_obj.month, day_obj.day, tzinfo=APP_TZ).timestamp())


def local_week_start_day(day_obj):
    return day_obj - timedelta(days=day_obj.weekday())


def local_week_end_day(day_obj):
    return local_week_start_day(day_obj) + timedelta(days=6)


def local_day_key(ts: int) -> str:
    return datetime.fromtimestamp(int(ts), APP_TZ).date().isoformat()


def format_local_ts(ts: int) -> str:
    return datetime.fromtimestamp(int(ts), APP_TZ).strftime("%Y-%m-%d %H:%M:%S")


def split_local_day_seconds(start_ts: int, end_ts: int) -> Dict[str, int]:
    start = int(start_ts)
    end = int(end_ts)
    if end <= start:
        return {}

    start_dt = datetime.fromtimestamp(start, APP_TZ)
    end_dt = datetime.fromtimestamp(end, APP_TZ)
    result: Dict[str, int] = {}
    cursor = start_dt
    while cursor < end_dt:
        next_day = datetime(cursor.year, cursor.month, cursor.day, tzinfo=APP_TZ) + timedelta(days=1)
        segment_end = next_day if next_day < end_dt else end_dt
        seconds = int((segment_end - cursor).total_seconds())
        if seconds > 0:
            day_key = cursor.date().isoformat()
            result[day_key] = result.get(day_key, 0) + seconds
        cursor = segment_end
    return result


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return f"{salt.hex()}${digest.hex()}"


def verify_password(password: str, password_hash: str) -> bool:
    try:
        salt_hex, digest_hex = password_hash.split("$", 1)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(digest_hex)
    except Exception:
        return False
    got = pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return compare_digest(got, expected)


def validate_password_strength(password: str) -> Optional[str]:
    weak_passwords = {
        "123456",
        "12345678",
        "11111111",
        "00000000",
        "password",
        "qwerty",
        "abcdef",
        "admin123456",
    }
    if len(password) < 8:
        return "密码至少 8 位。"
    if password.lower() in weak_passwords:
        return "密码过于简单，请更换为更强密码。"
    if password.isdigit():
        return "密码不能为纯数字。"
    return None


def _coerce_prefix_tokens(prefix_tokens: Optional[object]) -> List[str]:
    if not isinstance(prefix_tokens, list):
        return []
    return [str(item).strip() for item in prefix_tokens if str(item).strip()]


def _tokenize_prefix_text(prefix_text: Optional[object]) -> List[str]:
    if prefix_text is None:
        return []
    raw = str(prefix_text).strip()
    if not raw:
        return []
    tokens = [tok for tok in raw.split() if tok]
    if not tokens:
        return []

    merged: List[str] = []
    idx = 0
    while idx < len(tokens):
        cur = tokens[idx]
        if (
            idx + 1 < len(tokens)
            and re.fullmatch(r"\d+", cur)
            and re.fullmatch(r"\(\d+\s*/\s*\d+\)", tokens[idx + 1])
        ):
            merged.append(cur + tokens[idx + 1].replace(" ", ""))
            idx += 2
            continue
        merged.append(cur)
        idx += 1
    return merged


def _readable_expression(prefix_tokens: Optional[object], prefix_text: Optional[object]) -> Optional[str]:
    tokens = _coerce_prefix_tokens(prefix_tokens)
    if not tokens:
        tokens = _tokenize_prefix_text(prefix_text)
    if not tokens:
        return None
    return Math23kSolverSystem.prefix_tokens_to_infix_text(tokens)


def _render_explainability_html(explainability: Optional[object]) -> str:
    if not isinstance(explainability, dict):
        return (
            "<div class='solve-explain'>"
            "<h4>推理过程（可解释性）</h4>"
            "<p class='sub'>当前记录未包含可解释性数据。</p>"
            "</div>"
        )

    if str(explainability.get("type") or "") == "model_trace":
        preprocess = explainability.get("preprocess")
        preprocess = preprocess if isinstance(preprocess, dict) else {}
        graph_view = explainability.get("graph_view")
        graph_view = graph_view if isinstance(graph_view, dict) else {}
        decoder_trace = explainability.get("decoder_trace")
        decoder_trace = decoder_trace if isinstance(decoder_trace, dict) else {}
        expression_postprocess = explainability.get("expression_postprocess")
        expression_postprocess = expression_postprocess if isinstance(expression_postprocess, dict) else {}
        final = explainability.get("final")
        final = final if isinstance(final, dict) else {}

        note = str(explainability.get("source_note") or "").strip()
        note_html = f"<p class='sub'>{html.escape(note)}</p>" if note else ""

        original_text = str(preprocess.get("original_text") or "").strip()
        normalized_text = str(preprocess.get("normalized_text") or "").strip()
        matched_numbers_raw = preprocess.get("matched_numbers")
        matched_numbers = matched_numbers_raw if isinstance(matched_numbers_raw, list) else []
        matched_display = []
        for m in matched_numbers[:12]:
            if not isinstance(m, dict):
                continue
            mt = str(m.get("text") or "").strip()
            st = m.get("start")
            ed = m.get("end")
            if mt:
                matched_display.append(f"{mt}[{st},{ed}]")
        num_pos_raw = preprocess.get("num_pos_bert")
        num_pos = num_pos_raw if isinstance(num_pos_raw, list) else []
        bert_tokens_raw = preprocess.get("bert_tokens")
        bert_tokens = bert_tokens_raw if isinstance(bert_tokens_raw, list) else []

        preprocess_html = (
            "<ul class='kv'>"
            f"<li><b>原始输入</b>：{html.escape(original_text or '未知')}</li>"
            f"<li><b>规范化输入</b>：{html.escape(normalized_text or '未知')}</li>"
            f"<li><b>识别数字</b>：<span class='mono'>{html.escape(', '.join(matched_display) if matched_display else '[]')}</span></li>"
            f"<li><b>数字在BERT位置</b>：<span class='mono'>{html.escape(str(num_pos))}</span></li>"
            f"<li><b>BERT分词序列</b>：<span class='mono'>{html.escape(' '.join(str(x) for x in bert_tokens[:80]))}</span></li>"
            "</ul>"
        )

        nodes_raw = graph_view.get("nodes")
        nodes = nodes_raw if isinstance(nodes_raw, list) else []
        node_tags = []
        for idx, node in enumerate(nodes[:10]):
            if not isinstance(node, dict):
                continue
            token = str(node.get("token") or "").strip()
            value = node.get("value")
            label = f"N{idx}: {token}" if token else f"N{idx}"
            if value is not None and str(value).strip() != "":
                label += f" ({value})"
            node_tags.append(f"<span class='solve-explain-tag'>{html.escape(label)}</span>")
        nodes_html = "<div class='solve-explain-tags'>" + "".join(node_tags) + "</div>" if node_tags else "<p class='sub'>无数字节点。</p>"

        edges_raw = graph_view.get("edges")
        edges = edges_raw if isinstance(edges_raw, list) else []
        edge_items = []
        edge_seen = set()
        for rel in edges:
            if not isinstance(rel, dict):
                continue
            desc = str(rel.get("description") or "").strip()
            if not desc or desc in edge_seen:
                continue
            edge_seen.add(desc)
            edge_items.append(f"<li>{html.escape(desc)}</li>")
            if len(edge_items) >= 12:
                break
        if not edge_items:
            edge_items.append("<li>无可展示关系。</li>")
        edges_html = "<ul class='solve-explain-rels'>" + "".join(edge_items) + "</ul>"

        trace_steps_raw = decoder_trace.get("steps")
        trace_steps = trace_steps_raw if isinstance(trace_steps_raw, list) else []
        reasoning_lines = []
        detailed_items = []
        for st in trace_steps[:12]:
            if not isinstance(st, dict):
                continue
            t = int(st.get("step_index", 0))
            topk = st.get("topk_candidates_from_best_beam")
            topk = topk if isinstance(topk, list) else []
            cand_tokens = []
            for cand in topk[:5]:
                if not isinstance(cand, dict):
                    continue
                token_obj = cand.get("token")
                token_obj = token_obj if isinstance(token_obj, dict) else {}
                resolved = str(token_obj.get("resolved_token") or token_obj.get("vocab_token") or "").strip()
                if resolved:
                    cand_tokens.append(resolved)
            chosen_obj = st.get("chosen_token")
            chosen_obj = chosen_obj if isinstance(chosen_obj, dict) else {}
            chosen = str(chosen_obj.get("resolved_token") or chosen_obj.get("vocab_token") or "").strip()
            line = f"t={t}: 候选[{', '.join(cand_tokens) if cand_tokens else '-'}], 选择[{chosen or '-'}]"
            reasoning_lines.append(line)
            detailed_items.append(f"<li>{html.escape(line)}</li>")
        if not reasoning_lines:
            reasoning_lines.append("暂无可展示的解码轨迹。")
            detailed_items.append("<li>暂无可展示的解码轨迹。</li>")
        reasoning_box_html = (
            "<div class='solve-reasoning-box'>"
            + "".join(f"<div class='solve-reasoning-line'>{html.escape(line)}</div>" for line in reasoning_lines[:6])
            + "</div>"
        )
        steps_html = "<ol class='solve-explain-steps'>" + "".join(detailed_items) + "</ol>"

        postprocess_html = ""
        if expression_postprocess:
            post_summary = str(expression_postprocess.get("summary") or "").strip()
            post_infix = str(expression_postprocess.get("infix_expression") or "").strip()
            post_final = expression_postprocess.get("final_result_text")
            post_steps_raw = expression_postprocess.get("steps")
            post_steps = post_steps_raw if isinstance(post_steps_raw, list) else []
            post_items = []
            for st in post_steps[:12]:
                if not isinstance(st, dict):
                    continue
                formula = str(st.get("formula") or "").strip()
                if formula:
                    post_items.append(f"<li>{html.escape(formula)}</li>")
            if not post_items:
                post_items.append("<li>表达式未生成可展示的逐步计算。</li>")
            postprocess_html = (
                "<div class='solve-explain-block'>"
                "<div class='solve-explain-title'>表达式后处理解释（符号执行）</div>"
                "<ul class='kv'>"
                f"<li><b>摘要</b>：{html.escape(post_summary or '无')}</li>"
                f"<li><b>中缀表达式</b>：<span class='mono'>{html.escape(post_infix or '未知')}</span></li>"
                f"<li><b>后处理结果</b>：<span class='mono'>{html.escape(str(post_final))}</span></li>"
                "</ul>"
                "<ol class='solve-explain-steps'>"
                + "".join(post_items)
                + "</ol>"
                "</div>"
            )

        prefix_tokens = final.get("predicted_prefix_tokens")
        prefix_tokens = prefix_tokens if isinstance(prefix_tokens, list) else []
        infix = str(final.get("predicted_infix_text") or "").strip()
        pred_val = final.get("predicted_value")
        summary_line = (
            f"预处理识别 {len(matched_display)} 个数字，解码轨迹 {len(trace_steps)} 步，关系边展示 {len(edge_seen)} 条。"
        )
        return (
            "<div class='solve-explain'>"
            "<h4>推理过程（模型真实轨迹）</h4>"
            f"{note_html}"
            f"<p class='sub'>{html.escape(summary_line)}</p>"
            f"{reasoning_box_html}"
            "<div class='solve-explain-block'>"
            "<div class='solve-explain-title'>输入预处理</div>"
            f"{preprocess_html}"
            "</div>"
            "<div class='solve-explain-block'>"
            "<div class='solve-explain-title'>数字关系图（由预处理关系矩阵生成）</div>"
            f"{nodes_html}"
            f"{edges_html}"
            "</div>"
            "<div class='solve-explain-block'>"
            "<div class='solve-explain-title'>解码轨迹（Beam Search）</div>"
            f"{steps_html}"
            "</div>"
            f"{postprocess_html}"
            "<div class='solve-explain-block'>"
            "<div class='solve-explain-title'>模型输出</div>"
            "<ul class='kv'>"
            f"<li><b>前缀表达式</b>：<span class='mono'>{html.escape(' '.join(str(x) for x in prefix_tokens) if prefix_tokens else '未知')}</span></li>"
            f"<li><b>中缀表达式</b>：<span class='mono'>{html.escape(infix or '未知')}</span></li>"
            f"<li><b>数值答案</b>：<span class='mono'>{html.escape(str(pred_val))}</span></li>"
            "</ul>"
            "</div>"
            "</div>"
        )

    # Backward compatibility for older explainability format
    return (
        "<div class='solve-explain'>"
        "<h4>推理过程（可解释性）</h4>"
        "<p class='sub'>该记录使用旧版解释格式，建议重新解题生成新版“模型真实轨迹”。</p>"
        "<pre style='white-space:pre-wrap; margin:0; background:#f6f8fc; border:1px solid var(--line); border-radius:10px; padding:10px;'>"
        f"{html.escape(json.dumps(explainability, ensure_ascii=False, indent=2)[:2500])}"
        "</pre>"
        "</div>"
    )


class AppState:
    def __init__(
        self,
        db_path: Path,
        model_dir: str,
        data_dir: str,
        device: str,
        beam_size: int,
        admin_username: str,
        admin_password: str,
    ):
        self.db_path = db_path
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.device = device
        self.beam_size = beam_size
        self.admin_username = admin_username
        self.admin_password = admin_password
        self.db_lock = threading.Lock()
        self.solver_lock = threading.Lock()
        self.profile_verify_lock = threading.Lock()
        self.profile_verify_tokens: Dict[int, Dict[str, object]] = {}
        self._solver: Optional[Math23kSolverSystem] = None
        self.init_db()
        self.ensure_default_admin()

    def write_bootstrap_admin_info(self, username: str, password: str):
        info_path = self.db_path.parent / "admin_bootstrap.txt"
        lines = [
            f"time={format_local_ts(now_ts())}",
            f"username={username}",
            f"password={password}",
            "",
        ]
        try:
            info_path.parent.mkdir(parents=True, exist_ok=True)
            info_path.write_text("\n".join(lines), encoding="utf-8")
            print(f"[web] admin bootstrap credentials written to: {info_path}")
        except Exception as exc:
            print(f"[web] warning: failed to write admin bootstrap file: {exc}")

    def db_connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        with self.db_connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    is_admin INTEGER NOT NULL DEFAULT 0,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    created_at INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    session_token TEXT NOT NULL UNIQUE,
                    expires_at INTEGER NOT NULL,
                    created_at INTEGER NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                );

                CREATE TABLE IF NOT EXISTS solve_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    question TEXT NOT NULL,
                    predicted_prefix_text TEXT,
                    predicted_value REAL,
                    raw_json TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                );

                CREATE TABLE IF NOT EXISTS app_errors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    scope TEXT NOT NULL,
                    message TEXT NOT NULL,
                    context_json TEXT,
                    created_at INTEGER NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                );

                CREATE TABLE IF NOT EXISTS user_online_daily (
                    user_id INTEGER NOT NULL,
                    day_key TEXT NOT NULL,
                    duration_sec INTEGER NOT NULL DEFAULT 0,
                    updated_at INTEGER NOT NULL,
                    PRIMARY KEY (user_id, day_key),
                    FOREIGN KEY(user_id) REFERENCES users(id)
                );

                CREATE TABLE IF NOT EXISTS user_online_state (
                    user_id INTEGER PRIMARY KEY,
                    last_seen_ts INTEGER NOT NULL,
                    last_day_key TEXT NOT NULL,
                    updated_at INTEGER NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                );
                """
            )
            user_cols = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(users)").fetchall()
            }
            if "is_admin" not in user_cols:
                conn.execute("ALTER TABLE users ADD COLUMN is_admin INTEGER NOT NULL DEFAULT 0")
            if "is_active" not in user_cols:
                conn.execute("ALTER TABLE users ADD COLUMN is_active INTEGER NOT NULL DEFAULT 1")
            conn.commit()

    def ensure_default_admin(self):
        username = (self.admin_username or "").strip()
        password = self.admin_password or ""
        if not username:
            return
        with self.db_connect() as conn:
            row = conn.execute("SELECT id, is_admin FROM users WHERE username = ?", (username,)).fetchone()
            if row is None:
                if not password:
                    password = secrets.token_urlsafe(12)
                    self.write_bootstrap_admin_info(username=username, password=password)
                conn.execute(
                    """
                    INSERT INTO users (username, password_hash, is_admin, is_active, created_at)
                    VALUES (?, ?, 1, 1, ?)
                    """,
                    (username, hash_password(password), now_ts()),
                )
                conn.commit()
                return
            if int(row["is_admin"]) != 1:
                conn.execute("UPDATE users SET is_admin = 1, is_active = 1 WHERE id = ?", (int(row["id"]),))
                conn.commit()

    def log_error(self, scope: str, message: str, user_id: Optional[int] = None, context: Optional[Dict[str, object]] = None):
        context_json = json.dumps(context, ensure_ascii=False) if context else None
        with self.db_connect() as conn:
            conn.execute(
                """
                INSERT INTO app_errors (user_id, scope, message, context_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user_id, scope, message, context_json, now_ts()),
            )
            conn.commit()

    def record_online_heartbeat(self, user_id: int, heartbeat_ts: Optional[int] = None) -> Dict[str, int]:
        ts = int(heartbeat_ts) if heartbeat_ts is not None else now_ts()
        day_key = local_day_key(ts)
        added_sec = 0
        with self.db_connect() as conn:
            state = conn.execute(
                "SELECT last_seen_ts FROM user_online_state WHERE user_id = ?",
                (int(user_id),),
            ).fetchone()
            if state is not None:
                try:
                    last_seen_ts = int(state["last_seen_ts"])
                except Exception:
                    last_seen_ts = ts
                delta = ts - last_seen_ts
                if 1 <= delta <= ONLINE_HEARTBEAT_MAX_GAP_SECONDS:
                    by_day = split_local_day_seconds(last_seen_ts, ts)
                    for seg_day_key, seg_sec in by_day.items():
                        sec_val = int(seg_sec)
                        if sec_val <= 0:
                            continue
                        conn.execute(
                            """
                            INSERT INTO user_online_daily (user_id, day_key, duration_sec, updated_at)
                            VALUES (?, ?, ?, ?)
                            ON CONFLICT(user_id, day_key)
                            DO UPDATE SET
                              duration_sec = duration_sec + excluded.duration_sec,
                              updated_at = excluded.updated_at
                            """,
                            (int(user_id), seg_day_key, sec_val, ts),
                        )
                        added_sec += sec_val
            conn.execute(
                """
                INSERT INTO user_online_state (user_id, last_seen_ts, last_day_key, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id)
                DO UPDATE SET
                  last_seen_ts = excluded.last_seen_ts,
                  last_day_key = excluded.last_day_key,
                  updated_at = excluded.updated_at
                """,
                (int(user_id), ts, day_key, ts),
            )
            conn.commit()
        return {"added_sec": added_sec, "timestamp": ts}

    def cleanup_sessions(self):
        with self.db_connect() as conn:
            conn.execute("DELETE FROM sessions WHERE expires_at < ?", (now_ts(),))
            conn.commit()

    def _cleanup_profile_verify_tokens_unlocked(self):
        ts = now_ts()
        expired_user_ids: List[int] = []
        for uid, token_data in self.profile_verify_tokens.items():
            try:
                expires_at = int(token_data.get("expires_at", 0))
            except Exception:
                expires_at = 0
            if expires_at <= ts:
                expired_user_ids.append(int(uid))
        for uid in expired_user_ids:
            self.profile_verify_tokens.pop(uid, None)

    def issue_profile_verify_token(self, user_id: int) -> str:
        token = secrets.token_urlsafe(24)
        expires_at = now_ts() + PROFILE_VERIFY_TTL_SECONDS
        with self.profile_verify_lock:
            self._cleanup_profile_verify_tokens_unlocked()
            self.profile_verify_tokens[int(user_id)] = {"token": token, "expires_at": int(expires_at)}
        return token

    def verify_profile_verify_token(self, user_id: int, token: str) -> bool:
        raw_token = (token or "").strip()
        if not raw_token:
            return False
        with self.profile_verify_lock:
            self._cleanup_profile_verify_tokens_unlocked()
            token_data = self.profile_verify_tokens.get(int(user_id))
            if not token_data:
                return False
            expected = str(token_data.get("token", ""))
            return bool(expected) and compare_digest(raw_token, expected)

    def clear_profile_verify_token(self, user_id: int):
        with self.profile_verify_lock:
            self.profile_verify_tokens.pop(int(user_id), None)

    def get_solver(self) -> Math23kSolverSystem:
        with self.solver_lock:
            if self._solver is None:
                self._solver = Math23kSolverSystem(
                    data_dir=self.data_dir,
                    model_dir=self.model_dir,
                    device=self.device,
                    beam_size=self.beam_size,
                )
            return self._solver


class WebHandler(BaseHTTPRequestHandler):
    state: AppState = None
    api_only_mode: bool = True
    cors_allowed_origins: List[str] = []
    session_cookie_samesite: str = "Lax"
    session_cookie_secure: bool = False

    def log_message(self, fmt, *args):
        return

    def _resolve_cors_origin(self) -> Optional[str]:
        origin = (self.headers.get("Origin") or "").strip()
        if not origin:
            return None
        allowed = self.cors_allowed_origins or []
        if not allowed:
            return None
        normalized_origin = origin.rstrip("/")
        if "*" in allowed or normalized_origin in allowed:
            return origin
        return None

    def _apply_cors_headers(self):
        allowed_origin = self._resolve_cors_origin()
        if not allowed_origin:
            return
        self.send_header("Access-Control-Allow-Origin", allowed_origin)
        self.send_header("Access-Control-Allow-Credentials", "true")
        self.send_header("Vary", "Origin")

    def _send_bytes(
        self,
        code: int,
        payload: bytes,
        content_type: str,
        cache_control: Optional[str] = None,
        set_cookie: Optional[str] = None,
    ):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self._apply_cors_headers()
        if cache_control:
            self.send_header("Cache-Control", cache_control)
        if set_cookie:
            self.send_header("Set-Cookie", set_cookie)
        self.send_header("Content-Length", str(len(payload)))
        try:
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            return

    def _send_html(self, code: int, html_text: str, set_cookie: Optional[str] = None):
        self._send_bytes(
            code,
            html_text.encode("utf-8"),
            "text/html; charset=utf-8",
            set_cookie=set_cookie,
        )

    def _send_json(self, code: int, data: Dict[str, object], set_cookie: Optional[str] = None):
        self._send_bytes(
            code,
            json.dumps(data, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
            cache_control="no-store",
            set_cookie=set_cookie,
        )

    def _api_ok(
        self,
        data: Optional[Dict[str, object]] = None,
        message: Optional[str] = None,
        code: int = 200,
        set_cookie: Optional[str] = None,
    ):
        payload: Dict[str, object] = {"ok": True}
        if message:
            payload["message"] = message
        if data is not None:
            payload["data"] = data
        self._send_json(code, payload, set_cookie=set_cookie)

    def _api_error(self, code: int, error: str, message: str):
        self._send_json(code, {"ok": False, "error": error, "message": message})

    def _read_request_body(self) -> bytes:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except Exception:
            length = 0
        if length <= 0:
            return b""
        try:
            return self.rfile.read(length)
        except Exception:
            return b""

    def _parse_json_body(self) -> Dict[str, object]:
        raw = self._read_request_body()
        if not raw:
            return {}
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise ValueError(f"请求体不是合法 JSON：{exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("请求体必须是 JSON 对象。")
        return payload

    def _parse_request_data(self) -> Dict[str, object]:
        raw = self._read_request_body()
        if not raw:
            return {}
        text = raw.decode("utf-8")
        content_type = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
        if content_type == "application/json" or (not content_type and text.lstrip()[:1] in ("{", "[")):
            try:
                payload = json.loads(text)
            except Exception as exc:
                raise ValueError(f"请求体不是合法 JSON：{exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError("请求体必须是 JSON 对象。")
            return payload
        parsed = parse_qs(text, keep_blank_values=True)
        if parsed:
            return {k: (v[0] if v else "") for k, v in parsed.items()}
        raise ValueError("无法解析请求体。")

    def _redirect(self, location: str, set_cookie: Optional[str] = None):
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", location)
        self._apply_cors_headers()
        if set_cookie:
            self.send_header("Set-Cookie", set_cookie)
        self.end_headers()

    def _parse_form(self) -> Dict[str, str]:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        parsed = parse_qs(body, keep_blank_values=True)
        return {k: (v[0] if v else "") for k, v in parsed.items()}

    def _parse_positive_int(self, raw_value: object, default: int = 1) -> int:
        try:
            value = int(str(raw_value).strip())
        except Exception:
            value = int(default)
        return value if value > 0 else int(default)

    def _admin_page_from_request(self, default: int = 1) -> int:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        page_raw = query.get("page", [str(default)])[0]
        return self._parse_positive_int(page_raw, default=default)

    def _get_cookie(self, key: str) -> Optional[str]:
        raw = self.headers.get("Cookie")
        if not raw:
            return None
        cookie = SimpleCookie()
        cookie.load(raw)
        morsel = cookie.get(key)
        return morsel.value if morsel else None

    def _current_user(self) -> Optional[sqlite3.Row]:
        token = self._get_cookie("sid")
        if not token:
            return None
        self.state.cleanup_sessions()
        with self.state.db_connect() as conn:
            row = conn.execute(
                """
                SELECT u.id, u.username, u.is_admin, u.is_active
                FROM sessions s
                JOIN users u ON u.id = s.user_id
                WHERE s.session_token = ? AND s.expires_at > ? AND u.is_active = 1
                """,
                (token, now_ts()),
            ).fetchone()
            return row

    def _session_cookie_suffix(self, max_age: int) -> str:
        attrs = [
            "Path=/",
            "HttpOnly",
            f"SameSite={self.session_cookie_samesite}",
            f"Max-Age={int(max_age)}",
        ]
        if self.session_cookie_secure:
            attrs.append("Secure")
        return "; ".join(attrs)

    def _create_session(self, user_id: int) -> str:
        token = secrets.token_urlsafe(32)
        expires_at = now_ts() + SESSION_TTL_SECONDS
        with self.state.db_connect() as conn:
            conn.execute(
                "INSERT INTO sessions (user_id, session_token, expires_at, created_at) VALUES (?, ?, ?, ?)",
                (user_id, token, expires_at, now_ts()),
            )
            conn.commit()
        return f"sid={token}; {self._session_cookie_suffix(SESSION_TTL_SECONDS)}"

    def _clear_session_cookie(self) -> str:
        token = self._get_cookie("sid")
        if token:
            with self.state.db_connect() as conn:
                conn.execute("DELETE FROM sessions WHERE session_token = ?", (token,))
                conn.commit()
        return f"sid=; {self._session_cookie_suffix(0)}"

    def _path(self) -> str:
        return self.path.split("?", 1)[0]

    def _is_admin(self, user: Optional[sqlite3.Row]) -> bool:
        if user is None:
            return False
        try:
            return int(user["is_admin"]) == 1
        except Exception:
            return False

    def _send_forbidden(self, user: Optional[sqlite3.Row], message: str):
        self._api_error(403, "forbidden", message)

    def _require_admin(self, user: Optional[sqlite3.Row]) -> bool:
        if not user:
            self._redirect("/")
            return False
        if not self._is_admin(user):
            self._send_forbidden(user, "仅管理员可访问该页面。")
            return False
        return True

    def _landing_path_for_user(self, user: sqlite3.Row) -> str:
        return "/admin" if self._is_admin(user) else "/home"

    def _require_user_workspace(self, user: Optional[sqlite3.Row]) -> bool:
        if not user:
            self._redirect("/")
            return False
        if self._is_admin(user):
            self._redirect("/admin")
            return False
        return True

    def _require_api_user(self, user: Optional[sqlite3.Row]) -> Optional[sqlite3.Row]:
        if not user:
            self._api_error(401, "unauthorized", "请先登录。")
            return None
        return user

    def _require_api_user_workspace(self, user: Optional[sqlite3.Row]) -> Optional[sqlite3.Row]:
        user = self._require_api_user(user)
        if not user:
            return None
        if self._is_admin(user):
            self._api_error(403, "forbidden", "管理员账号不使用普通用户工作台接口。")
            return None
        return user

    def _require_api_admin(self, user: Optional[sqlite3.Row]) -> Optional[sqlite3.Row]:
        user = self._require_api_user(user)
        if not user:
            return None
        if not self._is_admin(user):
            self._api_error(403, "forbidden", "仅管理员可访问该接口。")
            return None
        return user

    def _user_payload(self, user: Optional[object]) -> Optional[Dict[str, object]]:
        if user is None:
            return None
        if isinstance(user, sqlite3.Row):
            key_set = set(user.keys())

            def getter(name: str, default=None):
                return user[name] if name in key_set else default
        elif isinstance(user, dict):
            getter = lambda name, default=None: user.get(name, default)
        else:
            return None

        raw_id = getter("id")
        user_id = None
        if raw_id is not None:
            try:
                user_id = int(raw_id)
            except Exception:
                user_id = None
        is_admin = bool(int(getter("is_admin", 0) or 0))
        raw_active = getter("is_active", 1)
        is_active = True if raw_active is None else bool(int(raw_active))
        landing_path = "/admin" if is_admin else "/home"
        return {
            "id": user_id,
            "username": str(getter("username") or ""),
            "is_admin": is_admin,
            "is_active": is_active,
            "role": "admin" if is_admin else "user",
            "landing_path": landing_path,
        }

    def _serialize_history_row(
        self,
        row: sqlite3.Row,
        display_no: Optional[int] = None,
        include_detail: bool = False,
    ) -> Dict[str, object]:
        parsed_raw: Dict[str, object] = {}
        raw_json_text = row["raw_json"]
        if raw_json_text:
            try:
                parsed_candidate = json.loads(raw_json_text)
                if isinstance(parsed_candidate, dict):
                    parsed_raw = parsed_candidate
            except Exception:
                parsed_raw = {}

        prefix_text = str(row["predicted_prefix_text"] or parsed_raw.get("predicted_prefix_text") or "").strip()
        prefix_tokens = parsed_raw.get("predicted_prefix_tokens")
        infix_text = str(parsed_raw.get("predicted_infix_text") or "").strip()
        expression_text = infix_text or _readable_expression(prefix_tokens, prefix_text) or prefix_text
        created_at = int(row["created_at"])
        payload: Dict[str, object] = {
            "id": int(row["id"]),
            "display_no": int(display_no) if display_no is not None else int(row["id"]),
            "question": str(row["question"]),
            "predicted_prefix_text": prefix_text or None,
            "predicted_infix_text": infix_text or None,
            "expression_text": expression_text or "",
            "predicted_value": row["predicted_value"],
            "created_at": created_at,
            "created_at_local": format_local_ts(created_at),
            "source": parsed_raw.get("source"),
        }
        if include_detail:
            numbers = parsed_raw.get("numbers") if isinstance(parsed_raw.get("numbers"), list) else []
            prefix_token_list = prefix_tokens if isinstance(prefix_tokens, list) else []
            payload.update(
                {
                    "numbers": numbers,
                    "predicted_prefix_tokens": prefix_token_list,
                    "explainability": parsed_raw.get("explainability"),
                    "raw_result": parsed_raw,
                }
            )
        return payload

    def _history_list_payload(self, user_id: int, page: int = 1, page_size: int = 5) -> Dict[str, object]:
        page = self._parse_positive_int(page, default=1)
        page_size = max(1, min(self._parse_positive_int(page_size, default=5), 100))
        with self.state.db_connect() as conn:
            total_count = int(
                conn.execute(
                    "SELECT COUNT(*) AS c FROM solve_history WHERE user_id = ?",
                    (int(user_id),),
                ).fetchone()["c"]
                or 0
            )
            total_pages = max(1, (total_count + page_size - 1) // page_size)
            if page > total_pages:
                page = total_pages
            offset = (page - 1) * page_size
            rows = conn.execute(
                """
                SELECT id, question, predicted_prefix_text, predicted_value, raw_json, created_at
                FROM solve_history
                WHERE user_id = ?
                ORDER BY id DESC
                LIMIT ? OFFSET ?
                """,
                (int(user_id), page_size, offset),
            ).fetchall()
        items = [
            self._serialize_history_row(row, display_no=offset + idx + 1)
            for idx, row in enumerate(rows)
        ]
        return {
            "items": items,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_count": total_count,
                "total_pages": total_pages,
            },
        }

    def _history_detail_payload(self, user_id: int, history_id: int) -> Optional[Dict[str, object]]:
        try:
            history_id = int(history_id)
        except Exception:
            return None
        if history_id <= 0:
            return None
        with self.state.db_connect() as conn:
            row = conn.execute(
                """
                SELECT id, question, predicted_prefix_text, predicted_value, raw_json, created_at
                FROM solve_history
                WHERE id = ? AND user_id = ?
                LIMIT 1
                """,
                (history_id, int(user_id)),
            ).fetchone()
            if row is None:
                return None
            display_no = int(
                conn.execute(
                    "SELECT COUNT(*) AS c FROM solve_history WHERE user_id = ? AND id > ?",
                    (int(user_id), history_id),
                ).fetchone()["c"]
                or 0
            ) + 1
        return self._serialize_history_row(row, display_no=display_no, include_detail=True)

    def _build_stats_summary_payload(self, stats: Dict[str, object]) -> Dict[str, object]:
        return {
            "total_cnt": int(stats.get("total_cnt", 0) or 0),
            "today_cnt": int(stats.get("today_cnt", 0) or 0),
            "week_cnt": int(stats.get("week_cnt", 0) or 0),
            "month_cnt": int(stats.get("month_cnt", 0) or 0),
            "active_days_30": int(stats.get("active_days_30", 0) or 0),
            "streak": int(stats.get("streak", 0) or 0),
            "week_range_label": str(stats.get("week_range_label", "") or ""),
            "week_start_day": str(stats.get("week_start_day", "") or ""),
            "week_end_day": str(stats.get("week_end_day", "") or ""),
            "usage_7d_total": float(stats.get("usage_7d_total", 0.0) or 0.0),
        }

    def _build_home_overview_payload(self, user_id: int) -> Dict[str, object]:
        stats = self._user_activity_snapshot(user_id=int(user_id), history_limit=5)
        recent_history = [
            self._serialize_history_row(row, display_no=idx + 1)
            for idx, row in enumerate(stats.get("rows", []))
        ]
        return {
            "summary": self._build_stats_summary_payload(stats),
            "portrait": self._build_user_portrait_data(stats),
            "recent_history": recent_history,
        }

    def _build_stats_overview_payload(self, user_id: int) -> Dict[str, object]:
        try:
            self.state.record_online_heartbeat(user_id=int(user_id))
        except Exception as exc:
            self.state.log_error(scope="online_heartbeat", message=str(exc), user_id=int(user_id))
        stats = self._user_activity_snapshot(user_id=int(user_id), history_limit=50)
        history_items = [
            self._serialize_history_row(row, display_no=idx + 1)
            for idx, row in enumerate(stats.get("rows", []))
        ]
        return {
            "summary": self._build_stats_summary_payload(stats),
            "portrait": self._build_user_portrait_data(stats),
            "heatmap": {"day_count_map": stats.get("day_count_map", {})},
            "usage": {
                "points": stats.get("usage_7d_points", []),
                "total_hours": float(stats.get("usage_7d_total", 0.0) or 0.0),
                "week_range_label": str(stats.get("week_range_label", "") or ""),
            },
            "history": history_items,
        }

    def _build_profile_payload(self, user_id: int) -> Optional[Dict[str, object]]:
        with self.state.db_connect() as conn:
            row = conn.execute(
                "SELECT id, username, is_admin, is_active, created_at FROM users WHERE id = ?",
                (int(user_id),),
            ).fetchone()
        if row is None:
            return None
        is_admin = bool(int(row["is_admin"]))
        is_active = bool(int(row["is_active"]))
        created_at = int(row["created_at"])
        return {
            "user": self._user_payload(row),
            "created_at": created_at,
            "created_at_local": format_local_ts(created_at),
            "role_text": "管理员" if is_admin else "普通用户",
            "status_text": "启用" if is_active else "禁用",
            "verify_ttl_seconds": PROFILE_VERIFY_TTL_SECONDS,
        }

    def _load_docs_payload(self) -> Dict[str, object]:
        guide_path = ROOT / "WEB_APP_GUIDE.md"
        try:
            guide_text = guide_path.read_text(encoding="utf-8")
        except Exception as exc:
            guide_text = f"操作文档读取失败：{exc}"
        return {
            "title": "操作文档",
            "content": guide_text,
            "path": str(guide_path),
        }

    def _build_admin_overview_payload(self, page: int = 1, page_size: int = 10) -> Dict[str, object]:
        page = self._parse_positive_int(page, default=1)
        page_size = max(1, min(self._parse_positive_int(page_size, default=10), 50))
        with self.state.db_connect() as conn:
            stats = conn.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM users) AS user_total,
                    (SELECT COUNT(*) FROM users WHERE is_active = 1) AS active_user_total,
                    (SELECT COUNT(*) FROM solve_history) AS solve_total,
                    (SELECT COUNT(*) FROM solve_history WHERE created_at >= ?) AS solve_24h
                """,
                (now_ts() - 24 * 3600,),
            ).fetchone()
            users_total = int(stats["user_total"] or 0)
            total_pages = max(1, math.ceil(users_total / page_size)) if users_total > 0 else 1
            if page > total_pages:
                page = total_pages
            offset = (page - 1) * page_size
            users = conn.execute(
                """
                SELECT id, username, is_admin, is_active, created_at
                FROM users
                ORDER BY id ASC
                LIMIT ? OFFSET ?
                """,
                (page_size, offset),
            ).fetchall()
            errors = conn.execute(
                """
                SELECT e.id, e.scope, e.message, e.context_json, e.created_at, u.username
                FROM app_errors e
                LEFT JOIN users u ON u.id = e.user_id
                ORDER BY e.id DESC
                LIMIT 30
                """
            ).fetchall()
            active_admin_count = int(
                conn.execute(
                    "SELECT COUNT(*) AS c FROM users WHERE is_admin = 1 AND is_active = 1"
                ).fetchone()["c"]
                or 0
            )
            admin_total_count = int(
                conn.execute(
                    "SELECT COUNT(*) AS c FROM users WHERE is_admin = 1"
                ).fetchone()["c"]
                or 0
            )
        user_items = [
            {
                "id": int(row["id"]),
                "username": str(row["username"]),
                "is_admin": bool(int(row["is_admin"])),
                "is_active": bool(int(row["is_active"])),
                "created_at": int(row["created_at"]),
                "created_at_local": format_local_ts(int(row["created_at"])),
            }
            for row in users
        ]
        error_items = [
            {
                "id": int(row["id"]),
                "scope": str(row["scope"]),
                "message": str(row["message"]),
                "context_json": row["context_json"],
                "created_at": int(row["created_at"]),
                "created_at_local": format_local_ts(int(row["created_at"])),
                "username": row["username"] if row["username"] else None,
            }
            for row in errors
        ]
        return {
            "summary": {
                "user_total": users_total,
                "active_user_total": int(stats["active_user_total"] or 0),
                "solve_total": int(stats["solve_total"] or 0),
                "solve_24h": int(stats["solve_24h"] or 0),
            },
            "admin_counts": {
                "admin_total_count": admin_total_count,
                "active_admin_count": active_admin_count,
            },
            "users": {
                "items": user_items,
                "pagination": {
                    "page": page,
                    "page_size": page_size,
                    "total_count": users_total,
                    "total_pages": total_pages,
                },
            },
            "errors": error_items,
        }

    def _handle_api_get(self, path: str):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        user = self._current_user()

        if path == "/api/health":
            frontend_mode = "api_only"
            self._api_ok(
                {
                    "service": "mwp-web-app",
                    "frontend_mode": frontend_mode,
                    "app_timezone": APP_TIMEZONE_NAME,
                }
            )
            return

        if path == "/api/auth/me":
            frontend_mode = "api_only"
            landing_path = self._landing_path_for_user(user) if user else "/login"
            self._api_ok(
                {
                    "authenticated": bool(user),
                    "user": self._user_payload(user),
                    "landing_path": landing_path,
                    "frontend_mode": frontend_mode,
                }
            )
            return

        if path == "/api/docs":
            user = self._require_api_user(user)
            if not user:
                return
            self._api_ok(self._load_docs_payload())
            return

        if path == "/api/home/overview":
            user = self._require_api_user_workspace(user)
            if not user:
                return
            self._api_ok(self._build_home_overview_payload(int(user["id"])))
            return

        if path == "/api/stats/overview":
            user = self._require_api_user_workspace(user)
            if not user:
                return
            self._api_ok(self._build_stats_overview_payload(int(user["id"])))
            return

        if path == "/api/profile":
            user = self._require_api_user(user)
            if not user:
                return
            profile_payload = self._build_profile_payload(int(user["id"]))
            if profile_payload is None:
                cookie = self._clear_session_cookie()
                self._send_json(401, {"ok": False, "error": "unauthorized", "message": "当前会话已失效，请重新登录。"}, set_cookie=cookie)
                return
            self._api_ok(profile_payload)
            return

        if path == "/api/history":
            user = self._require_api_user_workspace(user)
            if not user:
                return
            page = query.get("page", ["1"])[0]
            page_size = query.get("page_size", ["5"])[0]
            self._api_ok(self._history_list_payload(int(user["id"]), page=page, page_size=page_size))
            return

        history_match = re.fullmatch(r"/api/history/(\d+)", path)
        if history_match:
            user = self._require_api_user_workspace(user)
            if not user:
                return
            item = self._history_detail_payload(int(user["id"]), int(history_match.group(1)))
            if item is None:
                self._api_error(404, "not_found", "记录不存在。")
                return
            self._api_ok({"item": item})
            return

        if path == "/api/admin/overview":
            user = self._require_api_admin(user)
            if not user:
                return
            page = query.get("page", ["1"])[0]
            page_size = query.get("page_size", ["10"])[0]
            self._api_ok(self._build_admin_overview_payload(page=page, page_size=page_size))
            return

        self._api_error(404, "not_found", "接口不存在。")

    def _handle_api_post(self, path: str):
        if path == "/api/online/heartbeat":
            user = self._current_user()
            if not user:
                self._api_error(401, "unauthorized", "请先登录。")
                return
            try:
                result = self.state.record_online_heartbeat(user_id=int(user["id"]))
            except Exception as exc:
                self.state.log_error(scope="online_heartbeat", message=str(exc), user_id=int(user["id"]))
                self._api_error(500, "heartbeat_failed", "在线心跳记录失败。")
                return
            self._api_ok(result)
            return

        if path == "/api/auth/register":
            try:
                form = self._parse_request_data()
            except ValueError as exc:
                self._api_error(400, "invalid_request_body", str(exc))
                return
            username = str(form.get("username", "")).strip()
            password = str(form.get("password", ""))
            if not username or not password:
                self._api_error(400, "invalid_input", "用户名和密码不能为空。")
                return
            if len(password) < 6:
                self._api_error(400, "invalid_input", "密码至少 6 位。")
                return
            try:
                with self.state.db_connect() as conn:
                    conn.execute(
                        "INSERT INTO users (username, password_hash, is_admin, is_active, created_at) VALUES (?, ?, 0, 1, ?)",
                        (username, hash_password(password), now_ts()),
                    )
                    user_id = int(conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()["id"])
                    conn.commit()
            except sqlite3.IntegrityError:
                self._api_error(400, "username_exists", "用户名已存在。")
                return
            cookie = self._create_session(user_id)
            self._api_ok(
                {
                    "user": self._user_payload({"id": user_id, "username": username, "is_admin": 0, "is_active": 1}),
                    "landing_path": "/home",
                },
                message="注册成功。",
                set_cookie=cookie,
            )
            return

        if path == "/api/auth/login":
            try:
                form = self._parse_request_data()
            except ValueError as exc:
                self._api_error(400, "invalid_request_body", str(exc))
                return
            username = str(form.get("username", "")).strip()
            password = str(form.get("password", ""))
            with self.state.db_connect() as conn:
                target = conn.execute(
                    "SELECT id, username, password_hash, is_admin, is_active FROM users WHERE username = ?",
                    (username,),
                ).fetchone()
            if not target or not verify_password(password, target["password_hash"]):
                self._api_error(401, "invalid_credentials", "用户名或密码错误。")
                return
            if int(target["is_active"]) != 1:
                self._api_error(403, "account_disabled", "账号已被禁用，请联系管理员。")
                return
            cookie = self._create_session(int(target["id"]))
            self._api_ok(
                {
                    "user": self._user_payload(target),
                    "landing_path": self._landing_path_for_user(target),
                },
                message="登录成功。",
                set_cookie=cookie,
            )
            return

        if path == "/api/auth/logout":
            cookie = self._clear_session_cookie()
            self._api_ok({"logged_out": True}, message="已退出登录。", set_cookie=cookie)
            return

        if path == "/api/profile/verify":
            user = self._require_api_user(self._current_user())
            if not user:
                return
            try:
                form = self._parse_request_data()
            except ValueError as exc:
                self._api_error(400, "invalid_request_body", str(exc))
                return
            current_username = str(form.get("current_username", "")).strip()
            current_password = str(form.get("current_password", ""))
            if not current_username or not current_password:
                self.state.clear_profile_verify_token(int(user["id"]))
                self._api_error(400, "invalid_input", "请填写当前登录账号与密码。")
                return
            with self.state.db_connect() as conn:
                target = conn.execute(
                    "SELECT id, username, password_hash FROM users WHERE id = ?",
                    (int(user["id"]),),
                ).fetchone()
            if target is None:
                cookie = self._clear_session_cookie()
                self._send_json(401, {"ok": False, "error": "unauthorized", "message": "当前会话已失效，请重新登录。"}, set_cookie=cookie)
                return
            current_username_db = str(target["username"])
            if current_username != current_username_db or (not verify_password(current_password, target["password_hash"])):
                self.state.clear_profile_verify_token(int(user["id"]))
                self._api_error(401, "invalid_credentials", "当前登录账号或密码错误。")
                return
            verify_token = self.state.issue_profile_verify_token(int(user["id"]))
            self._api_ok(
                {
                    "verify_token": verify_token,
                    "profile": self._build_profile_payload(int(user["id"])),
                },
                message="身份验证通过，请继续修改账户信息。",
            )
            return

        if path == "/api/profile/update":
            user = self._require_api_user(self._current_user())
            if not user:
                return
            try:
                form = self._parse_request_data()
            except ValueError as exc:
                self._api_error(400, "invalid_request_body", str(exc))
                return
            verify_token = str(form.get("verify_token", "")).strip()
            if not self.state.verify_profile_verify_token(int(user["id"]), verify_token):
                self.state.clear_profile_verify_token(int(user["id"]))
                self._api_error(400, "verify_expired", "身份验证已过期，请重新确认当前登录信息。")
                return

            new_username = str(form.get("new_username", "")).strip()
            new_password = str(form.get("new_password", ""))
            confirm_password = str(form.get("confirm_password", ""))

            if not new_username:
                self._api_error(400, "invalid_input", "用户名不能为空。")
                return
            if len(new_username) < 2:
                self._api_error(400, "invalid_input", "用户名至少 2 位。")
                return
            if len(new_username) > 32:
                self._api_error(400, "invalid_input", "用户名最多 32 位。")
                return
            if any(ch.isspace() for ch in new_username):
                self._api_error(400, "invalid_input", "用户名不能包含空格。")
                return

            password_change_requested = bool(new_password or confirm_password)
            if password_change_requested:
                if not new_password or not confirm_password:
                    self._api_error(400, "invalid_input", "修改密码时，请完整填写新密码与确认新密码。")
                    return
                if len(new_password) < 6:
                    self._api_error(400, "invalid_input", "新密码至少 6 位。")
                    return
                if new_password != confirm_password:
                    self._api_error(400, "invalid_input", "两次输入的新密码不一致。")
                    return

            with self.state.db_connect() as conn:
                target = conn.execute(
                    "SELECT id, username, password_hash FROM users WHERE id = ?",
                    (int(user["id"]),),
                ).fetchone()
                if target is None:
                    cookie = self._clear_session_cookie()
                    self._send_json(401, {"ok": False, "error": "unauthorized", "message": "当前会话已失效，请重新登录。"}, set_cookie=cookie)
                    return

                old_username = str(target["username"])
                if password_change_requested and verify_password(new_password, target["password_hash"]):
                    self._api_error(400, "invalid_input", "新密码不能与当前密码一致。")
                    return

                username_changed = new_username != old_username
                password_changed = password_change_requested
                if (not username_changed) and (not password_changed):
                    self._api_ok({"profile": self._build_profile_payload(int(user["id"]))}, message="未检测到变更。")
                    return

                try:
                    if username_changed:
                        conn.execute("UPDATE users SET username = ? WHERE id = ?", (new_username, int(target["id"])))
                    if password_changed:
                        conn.execute(
                            "UPDATE users SET password_hash = ? WHERE id = ?",
                            (hash_password(new_password), int(target["id"])),
                        )
                    conn.commit()
                except sqlite3.IntegrityError:
                    self._api_error(400, "username_exists", "用户名已存在，请更换一个。")
                    return

            self.state.clear_profile_verify_token(int(user["id"]))
            if password_changed and username_changed:
                message = "登录账号和密码已更新。"
            elif username_changed:
                message = f"登录账号已更新为：{new_username}"
            else:
                message = "登录密码已更新。"
            self._api_ok({"profile": self._build_profile_payload(int(user["id"]))}, message=message)
            return

        if path == "/api/solve":
            user = self._require_api_user_workspace(self._current_user())
            if not user:
                return
            try:
                form = self._parse_request_data()
            except ValueError as exc:
                self._api_error(400, "invalid_request_body", str(exc))
                return
            question = str(form.get("question", "")).strip()
            if not question:
                self._api_error(400, "invalid_input", "题目不能为空。")
                return
            try:
                solver = self.state.get_solver()
                result = solver.solve(question)
                with self.state.db_connect() as conn:
                    cur = conn.execute(
                        """
                        INSERT INTO solve_history (user_id, question, predicted_prefix_text, predicted_value, raw_json, created_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            int(user["id"]),
                            question,
                            result.get("predicted_prefix_text"),
                            result.get("predicted_value"),
                            json.dumps(result, ensure_ascii=False),
                            now_ts(),
                        ),
                    )
                    history_id = int(cur.lastrowid)
                    conn.commit()
            except Exception as exc:
                self.state.log_error(scope="solve", message=str(exc), user_id=int(user["id"]), context={"question": question})
                self._api_error(500, "solve_failed", f"解题失败：{str(exc)}")
                return
            history_item = self._history_detail_payload(int(user["id"]), history_id)
            self._api_ok({"result": result, "history_item": history_item}, message="解题成功。")
            return

        if path == "/api/history":
            user = self._require_api_user_workspace(self._current_user())
            if not user:
                return
            try:
                form = self._parse_request_data()
            except ValueError as exc:
                self._api_error(400, "invalid_request_body", str(exc))
                return
            question = str(form.get("question", "")).strip()
            prefix = str(form.get("predicted_prefix_text", "")).strip()
            value_raw = str(form.get("predicted_value", "")).strip()
            if not question:
                self._api_error(400, "invalid_input", "新增记录失败：题目不能为空。")
                return
            predicted_value = None
            if value_raw:
                try:
                    predicted_value = float(value_raw)
                except Exception:
                    self._api_error(400, "invalid_input", "新增记录失败：答案必须是数字。")
                    return
            raw_json = json.dumps(
                {
                    "source": "manual_record",
                    "question": question,
                    "predicted_prefix_text": prefix or None,
                    "predicted_value": predicted_value,
                },
                ensure_ascii=False,
            )
            with self.state.db_connect() as conn:
                cur = conn.execute(
                    """
                    INSERT INTO solve_history (user_id, question, predicted_prefix_text, predicted_value, raw_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (int(user["id"]), question, prefix or None, predicted_value, raw_json, now_ts()),
                )
                history_id = int(cur.lastrowid)
                conn.commit()
            history_item = self._history_detail_payload(int(user["id"]), history_id)
            self._api_ok({"history_item": history_item}, message="统计记录新增成功。", code=201)
            return

        if path == "/api/admin/users":
            user = self._require_api_admin(self._current_user())
            if not user:
                return
            try:
                form = self._parse_request_data()
            except ValueError as exc:
                self._api_error(400, "invalid_request_body", str(exc))
                return
            username = str(form.get("username", "")).strip()
            password = str(form.get("password", ""))
            is_admin = 1 if str(form.get("is_admin", "0")).lower() in {"1", "true", "yes", "on"} else 0
            is_active = 1 if str(form.get("is_active", "1")).lower() in {"1", "true", "yes", "on"} else 0
            if not username:
                self._api_error(400, "invalid_input", "用户名不能为空。")
                return
            if len(password) < 6:
                self._api_error(400, "invalid_input", "新建用户密码至少 6 位。")
                return
            try:
                with self.state.db_connect() as conn:
                    conn.execute(
                        """
                        INSERT INTO users (username, password_hash, is_admin, is_active, created_at)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (username, hash_password(password), is_admin, is_active, now_ts()),
                    )
                    conn.commit()
            except sqlite3.IntegrityError:
                self._api_error(400, "username_exists", f"用户名 {username} 已存在。")
                return
            self._api_ok({"username": username}, message=f"用户 {username} 创建成功。", code=201)
            return

        admin_action_match = re.fullmatch(r"/api/admin/users/(\d+)/(toggle-admin|toggle-active|reset-password)", path)
        if admin_action_match:
            user = self._require_api_admin(self._current_user())
            if not user:
                return
            target_user_id = int(admin_action_match.group(1))
            action = admin_action_match.group(2)
            form: Dict[str, object] = {}
            if action == "reset-password":
                try:
                    form = self._parse_request_data()
                except ValueError as exc:
                    self._api_error(400, "invalid_request_body", str(exc))
                    return
            with self.state.db_connect() as conn:
                target = conn.execute(
                    "SELECT id, username, is_admin, is_active FROM users WHERE id = ?",
                    (target_user_id,),
                ).fetchone()
                if target is None:
                    self._api_error(404, "not_found", "目标用户不存在。")
                    return
                active_admin_count = int(
                    conn.execute("SELECT COUNT(*) AS c FROM users WHERE is_admin = 1 AND is_active = 1").fetchone()["c"]
                    or 0
                )
                admin_total_count = int(
                    conn.execute("SELECT COUNT(*) AS c FROM users WHERE is_admin = 1").fetchone()["c"]
                    or 0
                )

                if action == "toggle-admin":
                    is_target_admin = int(target["is_admin"]) == 1
                    new_flag = 0 if is_target_admin else 1
                    if int(target["id"]) == int(user["id"]) and new_flag == 0:
                        self._api_error(400, "invalid_action", "不能取消自己管理员权限。")
                        return
                    if is_target_admin and admin_total_count <= 1:
                        self._api_error(400, "invalid_action", "系统至少需要保留 1 个管理员账号。")
                        return
                    if is_target_admin and int(target["is_active"]) == 1 and active_admin_count <= 1:
                        self._api_error(400, "invalid_action", "系统至少需要保留 1 个活跃管理员账号。")
                        return
                    conn.execute("UPDATE users SET is_admin = ? WHERE id = ?", (new_flag, int(target["id"])))
                    conn.commit()
                    self._api_ok({"user_id": int(target["id"]), "is_admin": bool(new_flag)}, message=f"已更新 {target['username']} 的管理员状态。")
                    return

                if action == "toggle-active":
                    new_flag = 0 if int(target["is_active"]) == 1 else 1
                    if int(target["id"]) == int(user["id"]) and new_flag == 0:
                        self._api_error(400, "invalid_action", "不能禁用当前登录账号。")
                        return
                    if int(target["is_admin"]) == 1 and int(target["is_active"]) == 1 and new_flag == 0 and active_admin_count <= 1:
                        self._api_error(400, "invalid_action", "不能禁用唯一活跃管理员。")
                        return
                    conn.execute("UPDATE users SET is_active = ? WHERE id = ?", (new_flag, int(target["id"])))
                    conn.commit()
                    self._api_ok({"user_id": int(target["id"]), "is_active": bool(new_flag)}, message=f"已更新 {target['username']} 的启用状态。")
                    return

                new_password = str(form.get("new_password", ""))
                confirm_password = str(form.get("confirm_password", ""))
                if not new_password or not confirm_password:
                    self._api_error(400, "invalid_input", "请完整填写新密码和确认新密码。")
                    return
                if len(new_password) < 6:
                    self._api_error(400, "invalid_input", "新密码至少 6 位。")
                    return
                if new_password != confirm_password:
                    self._api_error(400, "invalid_input", "两次输入的新密码不一致。")
                    return
                conn.execute(
                    "UPDATE users SET password_hash = ? WHERE id = ?",
                    (hash_password(new_password), int(target["id"])),
                )
                conn.commit()
            self._api_ok({"user_id": target_user_id}, message=f"已重置 {target['username']} 的密码。")
            return

        self._api_error(404, "not_found", "接口不存在。")

    def _handle_api_put(self, path: str):
        history_match = re.fullmatch(r"/api/history/(\d+)", path)
        if not history_match:
            self._api_error(404, "not_found", "接口不存在。")
            return
        user = self._require_api_user_workspace(self._current_user())
        if not user:
            return
        try:
            form = self._parse_request_data()
        except ValueError as exc:
            self._api_error(400, "invalid_request_body", str(exc))
            return
        history_id = int(history_match.group(1))
        question = str(form.get("question", "")).strip()
        prefix = str(form.get("predicted_prefix_text", "")).strip()
        value_raw = str(form.get("predicted_value", "")).strip()
        if not question:
            self._api_error(400, "invalid_input", "更新失败：题目不能为空。")
            return
        predicted_value = None
        if value_raw:
            try:
                predicted_value = float(value_raw)
            except Exception:
                self._api_error(400, "invalid_input", "更新失败：答案必须是数字。")
                return
        raw_json = json.dumps(
            {
                "source": "manual_edit",
                "question": question,
                "predicted_prefix_text": prefix or None,
                "predicted_value": predicted_value,
            },
            ensure_ascii=False,
        )
        with self.state.db_connect() as conn:
            cur = conn.execute(
                """
                UPDATE solve_history
                SET question = ?, predicted_prefix_text = ?, predicted_value = ?, raw_json = ?
                WHERE id = ? AND user_id = ?
                """,
                (question, prefix or None, predicted_value, raw_json, history_id, int(user["id"])),
            )
            conn.commit()
        if int(cur.rowcount or 0) <= 0:
            self._api_error(404, "not_found", "更新失败：记录不存在或无权限。")
            return
        history_item = self._history_detail_payload(int(user["id"]), history_id)
        self._api_ok({"history_item": history_item}, message=f"记录 #{history_id} 更新成功。")

    def _handle_api_delete(self, path: str):
        history_match = re.fullmatch(r"/api/history/(\d+)", path)
        if history_match:
            user = self._require_api_user_workspace(self._current_user())
            if not user:
                return
            history_id = int(history_match.group(1))
            with self.state.db_connect() as conn:
                cur = conn.execute(
                    "DELETE FROM solve_history WHERE id = ? AND user_id = ?",
                    (history_id, int(user["id"])),
                )
                conn.commit()
            if int(cur.rowcount or 0) <= 0:
                self._api_error(404, "not_found", "删除失败：记录不存在或无权限。")
                return
            self._api_ok({"history_id": history_id}, message=f"记录 #{history_id} 已删除。")
            return

        admin_user_match = re.fullmatch(r"/api/admin/users/(\d+)", path)
        if admin_user_match:
            user = self._require_api_admin(self._current_user())
            if not user:
                return
            target_user_id = int(admin_user_match.group(1))
            with self.state.db_connect() as conn:
                target = conn.execute(
                    "SELECT id, username, is_admin, is_active FROM users WHERE id = ?",
                    (target_user_id,),
                ).fetchone()
                if target is None:
                    self._api_error(404, "not_found", "目标用户不存在。")
                    return
                if int(target["id"]) == int(user["id"]):
                    self._api_error(400, "invalid_action", "不能删除当前登录账号。")
                    return
                active_admin_count = int(
                    conn.execute("SELECT COUNT(*) AS c FROM users WHERE is_admin = 1 AND is_active = 1").fetchone()["c"]
                    or 0
                )
                admin_total_count = int(
                    conn.execute("SELECT COUNT(*) AS c FROM users WHERE is_admin = 1").fetchone()["c"]
                    or 0
                )
                if int(target["is_admin"]) == 1 and admin_total_count <= 1:
                    self._api_error(400, "invalid_action", "不能删除唯一管理员账号。")
                    return
                if int(target["is_admin"]) == 1 and int(target["is_active"]) == 1 and active_admin_count <= 1:
                    self._api_error(400, "invalid_action", "不能删除唯一活跃管理员账号。")
                    return
                conn.execute("DELETE FROM sessions WHERE user_id = ?", (int(target["id"]),))
                conn.execute("DELETE FROM solve_history WHERE user_id = ?", (int(target["id"]),))
                conn.execute("DELETE FROM app_errors WHERE user_id = ?", (int(target["id"]),))
                conn.execute("DELETE FROM user_online_daily WHERE user_id = ?", (int(target["id"]),))
                conn.execute("DELETE FROM user_online_state WHERE user_id = ?", (int(target["id"]),))
                conn.execute("DELETE FROM users WHERE id = ?", (int(target["id"]),))
                conn.commit()
            self._api_ok({"user_id": target_user_id}, message=f"用户 {target['username']} 已删除。")
            return

        if path == "/api/admin/errors":
            user = self._require_api_admin(self._current_user())
            if not user:
                return
            with self.state.db_connect() as conn:
                conn.execute("DELETE FROM app_errors")
                conn.commit()
            self._api_ok({"cleared": True}, message="错误日志已清空。")
            return

        self._api_error(404, "not_found", "接口不存在。")

    def _user_activity_snapshot(self, user_id: int, history_limit: int = 20) -> Dict[str, object]:
        today = local_now().date()
        today_start = local_day_start_ts(today)
        week_start_day = local_week_start_day(today)
        week_end_day = local_week_end_day(today)
        week_start = local_day_start_ts(week_start_day)
        week_next_start = local_day_start_ts(week_end_day + timedelta(days=1))
        month_start = local_day_start_ts(today - timedelta(days=29))
        usage_window_start_day = week_start_day.isoformat()
        usage_window_end_day = week_end_day.isoformat()

        with self.state.db_connect() as conn:
            rows = conn.execute(
                """
                SELECT id, question, predicted_prefix_text, predicted_value, raw_json, created_at
                FROM solve_history
                WHERE user_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (user_id, history_limit),
            ).fetchall()
            stat_row = conn.execute(
                """
                SELECT
                  COUNT(*) AS total_cnt,
                  SUM(CASE WHEN created_at >= ? THEN 1 ELSE 0 END) AS today_cnt,
                  SUM(CASE WHEN created_at >= ? AND created_at < ? THEN 1 ELSE 0 END) AS week_cnt,
                  SUM(CASE WHEN created_at >= ? THEN 1 ELSE 0 END) AS month_cnt
                FROM solve_history
                WHERE user_id = ?
                """,
                (today_start, week_start, week_next_start, month_start, user_id),
            ).fetchone()
            ts_rows = conn.execute(
                """
                SELECT created_at
                FROM solve_history
                WHERE user_id = ?
                ORDER BY created_at DESC
                """,
                (user_id,),
            ).fetchall()
            usage_rows = conn.execute(
                """
                SELECT day_key, duration_sec
                FROM user_online_daily
                WHERE user_id = ? AND day_key >= ? AND day_key <= ?
                ORDER BY day_key ASC
                """,
                (user_id, usage_window_start_day, usage_window_end_day),
            ).fetchall()

        total_cnt = int(stat_row["total_cnt"] or 0)
        today_cnt = int(stat_row["today_cnt"] or 0)
        week_cnt = int(stat_row["week_cnt"] or 0)
        month_cnt = int(stat_row["month_cnt"] or 0)

        day_count_map: Dict[str, int] = {}
        solved_days = set()
        active_days_30_set = set()
        for row in ts_rows:
            try:
                ts = int(row["created_at"])
            except Exception:
                continue
            day_key = local_day_key(ts)
            day_count_map[day_key] = day_count_map.get(day_key, 0) + 1
            solved_days.add(day_key)
            if ts >= month_start:
                active_days_30_set.add(day_key)
        active_days_30 = len(active_days_30_set)

        streak = 0
        cursor = today
        while cursor.isoformat() in solved_days:
            streak += 1
            cursor = cursor - timedelta(days=1)

        usage_sec_map: Dict[str, int] = {}
        for row in usage_rows:
            day_key = str(row["day_key"])
            try:
                sec = int(row["duration_sec"] or 0)
            except Exception:
                sec = 0
            if sec <= 0:
                continue
            usage_sec_map[day_key] = usage_sec_map.get(day_key, 0) + sec

        usage_7d_points: List[Dict[str, object]] = []
        for day_offset in range(7):
            day_obj = week_start_day + timedelta(days=day_offset)
            day_key = day_obj.isoformat()
            day_sec = max(0, int(usage_sec_map.get(day_key, 0)))
            hours = 0.0 if day_sec <= 0 else max(0.01, round(day_sec / 3600.0, 2))
            usage_7d_points.append(
                {
                    "day_key": day_key,
                    "label": f"{day_obj.month}/{day_obj.day}",
                    "hours": hours,
                }
            )
        usage_7d_total = round(sum(float(item["hours"]) for item in usage_7d_points), 2)

        return {
            "rows": rows,
            "total_cnt": total_cnt,
            "today_cnt": today_cnt,
            "week_cnt": week_cnt,
            "month_cnt": month_cnt,
            "active_days_30": active_days_30,
            "day_count_map": day_count_map,
            "streak": streak,
            "usage_7d_points": usage_7d_points,
            "usage_7d_total": usage_7d_total,
            "week_start_day": week_start_day.isoformat(),
            "week_end_day": week_end_day.isoformat(),
            "week_range_label": f"{week_start_day.month}/{week_start_day.day}-{week_end_day.month}/{week_end_day.day}",
        }

    def _build_user_portrait_data(self, stats: Dict[str, object]) -> Dict[str, object]:
        total_cnt = int(stats.get("total_cnt", 0))
        week_cnt = int(stats.get("week_cnt", 0))
        month_cnt = int(stats.get("month_cnt", 0))
        streak = int(stats.get("streak", 0))
        active_days_30 = int(stats.get("active_days_30", 0))
        avg_per_active_day = round(month_cnt / max(active_days_30, 1), 2) if month_cnt > 0 else 0

        tags = []
        if streak >= 14:
            tags.append("连续打卡型")
        if week_cnt >= 21:
            tags.append("高频训练型")
        if active_days_30 >= 20:
            tags.append("稳定学习型")
        if total_cnt <= 10:
            tags.append("新手探索型")
        if not tags:
            tags.append("稳步提升型")

        return {
            "tags": tags[:3],
            "active_days_30": active_days_30,
            "month_cnt": month_cnt,
            "avg_per_active_day": avg_per_active_day,
            "streak": streak,
            "week_cnt": week_cnt,
            "total_cnt": total_cnt,
        }

    def do_GET(self):
        path = self._path()
        if path.startswith("/api/"):
            self._handle_api_get(path)
            return
        if path in {"/", "/health"}:
            self._api_ok(
                {
                    "service": "mwp-web-app",
                    "mode": "api_only",
                    "frontend_mode": "api_only",
                    "app_timezone": APP_TIMEZONE_NAME,
                }
            )
            return
        self._api_error(404, "not_found", "当前服务仅提供 /api/* 接口。")

    def do_POST(self):
        path = self._path()
        if path.startswith("/api/"):
            self._handle_api_post(path)
            return
        self._api_error(404, "not_found", "当前服务仅提供 /api/* 接口。")

    def do_PUT(self):
        path = self._path()
        if path.startswith("/api/"):
            self._handle_api_put(path)
            return
        self._send_html(405, "<h1>405 Method Not Allowed</h1>")

    def do_DELETE(self):
        path = self._path()
        if path.startswith("/api/"):
            self._handle_api_delete(path)
            return
        self._send_html(405, "<h1>405 Method Not Allowed</h1>")

    def do_OPTIONS(self):
        path = self._path()
        self.send_response(HTTPStatus.NO_CONTENT)
        self._apply_cors_headers()
        if path.startswith("/api/"):
            request_headers = (self.headers.get("Access-Control-Request-Headers") or "").strip()
            self.send_header("Access-Control-Allow-Headers", request_headers or DEFAULT_CORS_ALLOW_HEADERS)
            self.send_header("Access-Control-Allow-Methods", DEFAULT_CORS_ALLOW_METHODS)
            self.send_header("Access-Control-Max-Age", "86400")
        else:
            self.send_header("Allow", DEFAULT_CORS_ALLOW_METHODS)
        self.send_header("Content-Length", "0")
        self.end_headers()

def main():
    default_same_site = str(os.environ.get("MWP_SESSION_COOKIE_SAMESITE", "Lax")).strip().title()
    if default_same_site not in {"Lax", "Strict", "None"}:
        default_same_site = "Lax"
    default_cookie_secure = str(os.environ.get("MWP_SESSION_COOKIE_SECURE", "0")).strip().lower() in {"1", "true", "yes", "on"}

    parser = argparse.ArgumentParser(description="MWP Web App")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18082)
    parser.add_argument("--db_path", type=str, default=str(DEFAULT_DB_PATH))
    parser.add_argument("--model_dir", type=str, default="/root/autodl-tmp/finetune_models-zh-v2")
    parser.add_argument("--data_dir", type=str, default=str(ROOT.parent / "data"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--admin_username", type=str, default=os.environ.get("MWP_ADMIN_USERNAME", "admin"))
    parser.add_argument("--admin_password", type=str, default=os.environ.get("MWP_ADMIN_PASSWORD", ""))
    parser.add_argument("--frontend_origin", type=str, default=os.environ.get("MWP_FRONTEND_ORIGIN", ""))
    parser.add_argument("--cors_allowed_origins", type=str, default=os.environ.get("MWP_CORS_ALLOWED_ORIGINS", ""))
    parser.add_argument("--session_cookie_samesite", type=str, choices=["Lax", "Strict", "None"], default=default_same_site)
    parser.add_argument("--session_cookie_secure", action="store_true", default=default_cookie_secure)
    args = parser.parse_args()

    state = AppState(
        db_path=Path(args.db_path),
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        device=args.device,
        beam_size=args.beam_size,
        admin_username=args.admin_username,
        admin_password=args.admin_password,
    )
    cors_allowed_origins = parse_origin_list(args.frontend_origin, args.cors_allowed_origins)
    WebHandler.state = state
    WebHandler.api_only_mode = True
    WebHandler.cors_allowed_origins = cors_allowed_origins
    WebHandler.session_cookie_samesite = str(args.session_cookie_samesite)
    WebHandler.session_cookie_secure = bool(args.session_cookie_secure)
    server = ThreadingHTTPServer((args.host, args.port), WebHandler)
    print(f"[web] serving on http://{args.host}:{args.port}")
    print(f"[web] db={args.db_path}")
    print(f"[web] model_dir={args.model_dir}")
    print(f"[web] solver_device={args.device}")
    print(f"[web] app_timezone={APP_TIMEZONE_NAME}")
    print(f"[web] api_only={WebHandler.api_only_mode}")
    print(
        "[web] cors_allowed_origins="
        + (",".join(cors_allowed_origins) if cors_allowed_origins else "(disabled)")
    )
    print(f"[web] session_cookie_samesite={WebHandler.session_cookie_samesite}")
    print(f"[web] session_cookie_secure={WebHandler.session_cookie_secure}")
    if WebHandler.session_cookie_samesite == "None" and not WebHandler.session_cookie_secure:
        print("[web] warning: SameSite=None without Secure may be blocked by modern browsers.")
    server.serve_forever()


if __name__ == "__main__":
    main()
