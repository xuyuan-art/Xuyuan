(function () {
  "use strict";

  function normalizeApiBase(raw) {
    if (raw === null || raw === undefined) return "";
    var txt = String(raw).trim();
    if (!txt) return "";
    return txt.replace(/\/+$/, "");
  }

  var cfg = window.__MWP_RUNTIME_CONFIG__ || {};
  var API_BASE = normalizeApiBase(cfg.apiBase || "");

  function buildApiUrl(path) {
    var p = String(path || "");
    if (!p) return API_BASE || "/";
    if (/^https?:\/\//i.test(p)) return p;
    if (p.charAt(0) !== "/") p = "/" + p;
    return API_BASE ? API_BASE + p : p;
  }

  function escapeHtml(value) {
    return String(value === null || value === undefined ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/\"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  async function requestJSON(path, options) {
    var opts = options || {};
    var method = opts.method || "GET";
    var fetchOptions = {
      method: method,
      credentials: "include",
      headers: {
        Accept: "application/json",
      },
    };
    if (opts.data !== undefined) {
      fetchOptions.headers["Content-Type"] = "application/json";
      fetchOptions.body = JSON.stringify(opts.data);
    }

    var response;
    try {
      response = await fetch(buildApiUrl(path), fetchOptions);
    } catch (err) {
      var e = new Error("网络请求失败，请稍后重试。");
      e.cause = err;
      throw e;
    }

    var payload = {};
    var raw = await response.text();
    if (raw) {
      try {
        payload = JSON.parse(raw);
      } catch (err2) {
        var e2 = new Error("服务返回了无法解析的响应。");
        e2.cause = err2;
        e2.status = response.status;
        throw e2;
      }
    }

    if (!response.ok || payload.ok === false) {
      var msg = payload.message || ("请求失败（" + response.status + "）");
      var e3 = new Error(msg);
      e3.status = response.status;
      e3.payload = payload;
      throw e3;
    }

    return payload;
  }

  function landingPath(user) {
    if (!user) return "/login.html";
    return user.is_admin ? "/admin.html" : "/home.html";
  }

  function showFlash(type, message) {
    var box = document.getElementById("flash");
    if (!box) return;
    if (!message) {
      box.innerHTML = "";
      return;
    }
    box.innerHTML =
      '<div class="toast ' + escapeHtml(type || "") + '">' +
      '<div style="display:flex;justify-content:space-between;gap:12px;align-items:flex-start;">' +
      '<div>' + escapeHtml(message) + '</div>' +
      '<button class="btn ghost small" type="button" id="flash-close">关闭</button>' +
      '</div></div>';
    var btn = document.getElementById("flash-close");
    if (btn) {
      btn.onclick = function () {
        showFlash("", "");
      };
    }
  }

  async function getSession() {
    var payload = await requestJSON("/api/auth/me");
    return payload.data || { authenticated: false, user: null, landing_path: "/login" };
  }

  async function ensureSession(options) {
    var opts = options || {};
    var sess = await getSession();
    if (!sess.authenticated) {
      if (opts.redirect !== false) window.location.href = "/login.html";
      return null;
    }
    if (opts.adminOnly && !(sess.user && sess.user.is_admin)) {
      if (opts.redirect !== false) window.location.href = "/home.html";
      return null;
    }
    if (opts.userOnly && sess.user && sess.user.is_admin) {
      if (opts.redirect !== false) window.location.href = "/admin.html";
      return null;
    }
    return sess;
  }

  function renderTopbar(params) {
    var p = params || {};
    var user = p.user || { username: "访客", is_admin: false };
    var active = p.active || "";
    var topbar = document.getElementById("topbar");
    if (!topbar) return;

    var onProfile = active === "/profile.html";
    var items = user.is_admin
      ? (onProfile
          ? [["/profile.html", "账户设置"]]
          : [
              ["/admin.html", "数据后台"],
              ["/docs.html", "操作文档"],
            ])
      : (onProfile
          ? [["/profile.html", "账户设置"]]
          : [
              ["/home.html", "首页"],
              ["/solve.html", "智能解题"],
              ["/stats.html", "统计记录"],
            ]);

    var nav = "";
    for (var i = 0; i < items.length; i += 1) {
      var href = items[i][0];
      var label = items[i][1];
      nav +=
        '<a class="top-link' + (active === href ? " active" : "") + '" href="' + escapeHtml(href) + '">' +
        escapeHtml(label) +
        "</a>";
    }

    topbar.innerHTML =
      '<div class="top-accent"></div>' +
      '<header class="topbar"><div class="topbar-inner">' +
      '<a class="brand" href="' + escapeHtml(landingPath(user)) + '"><span class="brand-text">基于数学语义理解的数学应用题求解系统</span></a>' +
      '<nav class="top-nav">' + nav + '</nav>' +
      '<div class="top-right">' +
      '<div class="user-menu">' +
      '<button class="avatar-btn" type="button" aria-label="用户菜单">' +
      escapeHtml(String(user.username || "U").slice(0, 1).toUpperCase()) +
      '</button>' +
      '<div class="user-dropdown">' +
      '<div class="dropdown-user">' + escapeHtml(user.username || "") + "</div>" +
      '<a href="/profile.html">账户设置</a>' +
      '<button type="button" id="btn-logout">退出登录</button>' +
      "</div>" +
      "</div>" +
      "</div></div></header>";

    var logoutBtn = document.getElementById("btn-logout");
    if (logoutBtn) {
      logoutBtn.onclick = async function () {
        try {
          await requestJSON("/api/auth/logout", { method: "POST" });
        } catch (err) {
          // ignore
        }
        window.location.href = "/login.html";
      };
    }
  }

  window.MWP = {
    API_BASE: API_BASE,
    buildApiUrl: buildApiUrl,
    requestJSON: requestJSON,
    escapeHtml: escapeHtml,
    showFlash: showFlash,
    getSession: getSession,
    ensureSession: ensureSession,
    renderTopbar: renderTopbar,
    landingPath: landingPath,
  };
})();
