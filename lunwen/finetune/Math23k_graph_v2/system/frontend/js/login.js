(function () {
  "use strict";

  var qs = new URLSearchParams(window.location.search || "");
  var tab = qs.get("tab") === "register" ? "register" : "login";

  function formDataToObj(form) {
    var data = {};
    var fd = new FormData(form);
    fd.forEach(function (value, key) { data[key] = value; });
    return data;
  }

  function showAuthError(message) {
    var box = document.getElementById("auth-error");
    if (!box) return;
    if (!message) {
      box.innerHTML = "";
      return;
    }
    box.innerHTML = '<div class="auth-alert">' + window.MWP.escapeHtml(message) + '</div>';
  }

  function setTab(nextTab) {
    tab = nextTab === "register" ? "register" : "login";
    var loginTab = document.getElementById("tab-login");
    var registerTab = document.getElementById("tab-register");
    var loginForm = document.getElementById("form-login");
    var registerForm = document.getElementById("form-register");
    if (loginTab) loginTab.classList.toggle("active", tab === "login");
    if (registerTab) registerTab.classList.toggle("active", tab === "register");
    if (loginForm) loginForm.classList.toggle("hidden", tab !== "login");
    if (registerForm) registerForm.classList.toggle("hidden", tab !== "register");
    showAuthError("");
  }

  async function boot() {
    try {
      var sess = await window.MWP.getSession();
      if (sess && sess.authenticated) {
        window.location.replace(sess.user && sess.user.is_admin ? "./admin.html" : "./home.html");
        return;
      }
    } catch (e) {
      // ignore
    }

    setTab(tab);

    var loginForm = document.getElementById("form-login");
    if (loginForm) {
      loginForm.addEventListener("submit", async function (event) {
        event.preventDefault();
        showAuthError("");
        try {
          var payload = await window.MWP.requestJSON("/api/auth/login", {
            method: "POST",
            data: formDataToObj(loginForm),
          });
          var landing = payload.data && payload.data.landing_path ? payload.data.landing_path : "/home";
          window.location.href = landing === "/admin" ? "./admin.html" : "./home.html";
        } catch (err) {
          showAuthError(err.message || "登录失败，请稍后重试。");
        }
      });
    }

    var registerForm = document.getElementById("form-register");
    if (registerForm) {
      registerForm.addEventListener("submit", async function (event) {
        event.preventDefault();
        showAuthError("");
        try {
          var payload = await window.MWP.requestJSON("/api/auth/register", {
            method: "POST",
            data: formDataToObj(registerForm),
          });
          var landing = payload.data && payload.data.landing_path ? payload.data.landing_path : "/home";
          window.location.href = landing === "/admin" ? "./admin.html" : "./home.html";
        } catch (err) {
          showAuthError(err.message || "注册失败，请稍后重试。");
        }
      });
    }
  }

  boot();
})();
