(function () {
  "use strict";

  function guessApiBase() {
    try {
      var params = new URLSearchParams(window.location.search || "");
      var fromQuery = (params.get("apiBase") || "").trim();
      if (fromQuery) {
        try {
          window.localStorage.setItem("__MWP_API_BASE__", fromQuery);
        } catch (e0) {
          // ignore
        }
        return fromQuery;
      }
    } catch (e) {
      // ignore
    }

    try {
      var fromStorage = (window.localStorage.getItem("__MWP_API_BASE__") || "").trim();
      if (fromStorage) return fromStorage;
    } catch (e2) {
      // ignore
    }

    var host = window.location.hostname || "127.0.0.1";
    var protocol = window.location.protocol === "https:" ? "https:" : "http:";
    var currentPort = window.location.port || (protocol === "https:" ? "443" : "80");
    var isLocalHost = host === "127.0.0.1" || host === "localhost";

    // Classic split-port local mode: frontend :6011 + api :6008.
    if (isLocalHost || currentPort === "6011") {
      return protocol + "//" + host + ":6008";
    }

    // Public entry mode: same-origin reverse proxy routes /api.
    return window.location.origin;
  }

  window.__MWP_RUNTIME_CONFIG__ = window.__MWP_RUNTIME_CONFIG__ || {
    // Override with ?apiBase=https://your-backend or localStorage.__MWP_API_BASE__
    apiBase: guessApiBase(),
  };
})();
