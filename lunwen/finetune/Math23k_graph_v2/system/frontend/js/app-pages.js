(function () {
  "use strict";

  function formatVal(v, fallback) {
    if (v === null || v === undefined || v === "") return fallback || "-";
    return String(v);
  }

  function tableHtml(headers, rows, emptyText) {
    if (!rows || !rows.length) {
      return '<div class="empty">' + window.MWP.escapeHtml(emptyText || "暂无数据") + '</div>';
    }
    var head = headers.map(function (h) { return "<th>" + window.MWP.escapeHtml(h) + "</th>"; }).join("");
    var body = rows.map(function (cols) {
      return "<tr>" + cols.map(function (c) { return "<td>" + c + "</td>"; }).join("") + "</tr>";
    }).join("");
    return '<div class="table-wrap"><table><thead><tr>' + head + '</tr></thead><tbody>' + body + '</tbody></table></div>';
  }

  function solveOutputPlaceholder() {
    return "<h3>输出结果</h3><p class='sub'>提交题目后将在这里展示表达式与答案。</p><p class='sub'>也可以在下方“最近解题记录”中点击“查看”回看历史输出。</p>";
  }

  function renderSolveExplainability(explainability) {
    if (!explainability || typeof explainability !== "object") {
      return "<div class='solve-explain'><h4>推理过程（可解释性）</h4><p class='sub'>当前记录未包含可解释性数据。</p></div>";
    }
    function asObject(v) {
      return v && typeof v === "object" ? v : {};
    }
    function asArray(v) {
      return Array.isArray(v) ? v : [];
    }
    function inferIntent(text) {
      var q = String(text || "");
      if (!q) return "未识别到明确题意关键词。";
      if (/还剩|剩下|剩余/.test(q)) return "根据“还剩/剩下”等词，题目意图可能是求剩余量。";
      if (/一共|总共|合计|共/.test(q)) return "根据“一共/总共/合计”等词，题目意图可能是求总量。";
      if (/平均|每/.test(q)) return "根据“平均/每”相关词，题目意图可能与单位量或平均量有关。";
      if (/相差|多|少/.test(q)) return "根据“多/少/相差”等词，题目意图可能是比较差值。";
      if (/倍/.test(q)) return "根据“倍”相关词，题目意图可能是倍数关系计算。";
      return "未识别到明确题意关键词。";
    }

    var note = window.MWP.escapeHtml(formatVal(explainability.source_note, ""));
    var preprocess = asObject(explainability.preprocess);
    var graphView = asObject(explainability.graph_view);
    var decoderTrace = asObject(explainability.decoder_trace);
    var expressionPostprocess = asObject(explainability.expression_postprocess);
    var final = asObject(explainability.final);

    var originalText = formatVal(preprocess.original_text, "");
    var matchedNumbers = asArray(preprocess.matched_numbers);
    var numberTags = matchedNumbers
      .map(function (m) {
        var item = asObject(m);
        var txt = formatVal(item.text, "");
        var s = item.start;
        var e = item.end;
        if (!txt) return "";
        if (s === undefined || e === undefined) return txt;
        return txt + "[" + s + "," + e + "]";
      })
      .filter(function (x) { return x; });
    var numPos = asArray(preprocess.num_pos_bert);

    var relationEdges = asArray(graphView.edges);
    var relationList = relationEdges
      .map(function (e) { return formatVal(asObject(e).description, ""); })
      .filter(function (x) { return x; })
      .slice(0, 8);

    var traceSteps = asArray(decoderTrace.steps);
    var traceLines = traceSteps.slice(0, 6).map(function (st) {
      var step = asObject(st);
      var stepIndex = Number(step.step_index || 0);
      var candidates = asArray(step.topk_candidates_from_best_beam).slice(0, 5).map(function (cand) {
        var token = asObject(asObject(cand).token);
        return formatVal(token.resolved_token, formatVal(token.vocab_token, ""));
      }).filter(function (x) { return x; });
      var chosenObj = asObject(step.chosen_token);
      var chosen = formatVal(chosenObj.resolved_token, formatVal(chosenObj.vocab_token, "-"));
      return "第 " + stepIndex + " 步：候选 [" + (candidates.length ? candidates.join("，") : "-") + "]，选择 [" + chosen + "]";
    });
    if (!traceLines.length) traceLines = ["暂无可展示的解码轨迹。"];

    var calcSteps = asArray(expressionPostprocess.steps);
    var calcLines = calcSteps.slice(0, 10).map(function (st) {
      var item = asObject(st);
      return formatVal(item.formula, "");
    }).filter(function (x) { return x; });
    if (!calcLines.length) {
      var infix = formatVal(final.predicted_infix_text, "");
      if (infix) calcLines = [infix];
    }

    var pretty = "";
    try {
      pretty = JSON.stringify(explainability, null, 2);
    } catch (e) {
      pretty = "{}";
    }

    return (
      "<div class='solve-explain'>" +
      "<h4>推理过程（可解释性）</h4>" +
      (note && note !== "-" ? "<p class='sub'>" + note + "</p>" : "") +

      "<div class='solve-explain-block'>" +
      "<div class='solve-explain-title'>1) 题意理解（输入层）</div>" +
      "<ul class='kv'>" +
      "<li><b>题目文本</b>：" + window.MWP.escapeHtml(originalText || "未知") + "</li>" +
      "<li><b>识别数字</b>：<span class='mono'>" + window.MWP.escapeHtml(numberTags.length ? numberTags.join("，") : "[]") + "</span></li>" +
      "<li><b>数字位置</b>：<span class='mono'>" + window.MWP.escapeHtml(numPos.length ? JSON.stringify(numPos) : "[]") + "</span></li>" +
      "<li><b>题意提示</b>：" + window.MWP.escapeHtml(inferIntent(originalText)) + "</li>" +
      "</ul>" +
      "</div>" +

      "<div class='solve-explain-block'>" +
      "<div class='solve-explain-title'>2) 关系线索（图结构）</div>" +
      (relationList.length
        ? ("<ul class='solve-explain-rels'>" + relationList.map(function (line) {
            return "<li>" + window.MWP.escapeHtml(line) + "</li>";
          }).join("") + "</ul>")
        : "<p class='sub'>未提取到可展示的数字关系。</p>") +
      "</div>" +

      "<div class='solve-explain-block'>" +
      "<div class='solve-explain-title'>3) 解题轨迹（模型决策）</div>" +
      "<div class='solve-reasoning-box'>" +
      traceLines.map(function (line) {
        return "<div class='solve-reasoning-line'>" + window.MWP.escapeHtml(line) + "</div>";
      }).join("") +
      "</div>" +
      "</div>" +

      "<div class='solve-explain-block'>" +
      "<div class='solve-explain-title'>4) 计算过程（符号执行）</div>" +
      (calcLines.length
        ? ("<ol class='solve-explain-steps'>" + calcLines.map(function (line) {
            return "<li>" + window.MWP.escapeHtml(line) + "</li>";
          }).join("") + "</ol>")
        : "<p class='sub'>暂无可展示的逐步计算。</p>") +
      "</div>" +

      "<details class='solve-explain-raw'>" +
      "<summary>查看原始轨迹（JSON）</summary>" +
      "<pre style='white-space:pre-wrap; margin:8px 0 0; background:#f6f8fc; border:1px solid var(--line); border-radius:10px; padding:10px;'>" +
      window.MWP.escapeHtml(pretty) +
      "</pre>" +
      "</details>" +
      "</div>"
    );
  }

  function renderSolveOutput(item, options) {
    if (!item) return solveOutputPlaceholder();
    var opts = options || {};
    var titleText = opts.title || "输出结果";
    var showClose = opts.showClose !== false;
    var showTime = opts.timeText && String(opts.timeText).trim();
    var numbersText = "[]";
    try {
      numbersText = JSON.stringify(Array.isArray(item.numbers) ? item.numbers : [], null, 0);
    } catch (e) {
      numbersText = "[]";
    }
    var expressionLabel = "表达式";
    var rawPrefix = formatVal(item.predicted_prefix_text, "");
    var expressionText = formatVal(item.expression_text, rawPrefix);
    if (!formatVal(item.predicted_infix_text, "") && expressionText === rawPrefix) {
      expressionLabel = "表达式（原始）";
    }
    var explainHtml = renderSolveExplainability(item.explainability);
    return (
      "<div class='solve-output-head'>" +
      "<h3>" + window.MWP.escapeHtml(titleText) + "</h3>" +
      (showClose ? "<button class='solve-view-close' type='button' id='solve-view-close' title='关闭查看' aria-label='关闭查看'>×</button>" : "") +
      "</div>" +
      (showTime ? "<p class='sub'>时间：" + window.MWP.escapeHtml(String(opts.timeText)) + "</p>" : "") +
      explainHtml +
      "<div class='solve-final-box'>" +
      "<div class='solve-final-title'>最终结果</div>" +
      "<ul class='kv'>" +
      "<li><b>题目</b>：" + window.MWP.escapeHtml(formatVal(item.question, "")) + "</li>" +
      "<li><b>" + window.MWP.escapeHtml(expressionLabel) + "</b>：<span class='mono'>" + window.MWP.escapeHtml(expressionText) + "</span></li>" +
      "<li><b>数值答案</b>：<span class='mono'>" + window.MWP.escapeHtml(formatVal(item.predicted_value, "-")) + "</span></li>" +
      "<li><b>识别数字</b>：<span class='mono'>" + window.MWP.escapeHtml(numbersText) + "</span></li>" +
      "</ul>" +
      "</div>"
    );
  }

  function solvePaginationHtml(page, totalPages, totalCount) {
    if (totalPages <= 1) return "";
    var nodes = [];
    function pageNode(pageNum, label, current) {
      var txt = window.MWP.escapeHtml(String(label || pageNum));
      if (current) return "<span class='page-current'>" + txt + "</span>";
      return "<button class='btn ghost page-btn btn-solve-page' type='button' data-page='" + pageNum + "'>" + txt + "</button>";
    }
    if (page > 1) nodes.push(pageNode(page - 1, "上一页", false));

    var startPage = Math.max(1, page - 2);
    var endPage = Math.min(totalPages, page + 2);

    if (startPage > 1) {
      nodes.push(pageNode(1, "1", false));
      if (startPage > 2) nodes.push("<span class='page-dots'>...</span>");
    }
    for (var p = startPage; p <= endPage; p += 1) {
      nodes.push(pageNode(p, String(p), p === page));
    }
    if (endPage < totalPages) {
      if (endPage < totalPages - 1) nodes.push("<span class='page-dots'>...</span>");
      nodes.push(pageNode(totalPages, String(totalPages), false));
    }
    if (page < totalPages) nodes.push(pageNode(page + 1, "下一页", false));
    nodes.push("<span class='page-meta'>第 " + page + "/" + totalPages + " 页 · 共 " + totalCount + " 条</span>");
    return "<div class='solve-pagination'>" + nodes.join("") + "</div>";
  }

  function usageChartHtml(usage) {
    var points = usage && Array.isArray(usage.points) ? usage.points : [];
    if (!points.length) {
      return "<div class='usage-chart'><div class='usage-head'><div class='usage-metric'>本周总在线 <b>0.0</b> 小时</div></div><div class='empty'>暂无在线时长数据</div></div>";
    }
    var values = points.map(function (p) {
      var n = Number(p && p.hours);
      return Number.isFinite(n) ? n : 0;
    });
    var maxVal = Math.max.apply(Math, values.concat([1]));
    var chartMax = Math.max(1, Math.ceil(maxVal));
    var width = 920;
    var height = 280;
    var left = 46;
    var right = 18;
    var top = 18;
    var bottom = 42;
    var innerW = width - left - right;
    var innerH = height - top - bottom;
    var ticks = 4;
    var areaPoints = [];
    var linePoints = [];
    var dots = [];
    var labels = [];
    var grids = [];
    var yLabels = [];

    for (var i = 0; i <= ticks; i += 1) {
      var y = top + (innerH / ticks) * i;
      var tickVal = ((ticks - i) * chartMax) / ticks;
      grids.push("<line class='usage-grid' x1='" + left + "' y1='" + y.toFixed(2) + "' x2='" + (width - right) + "' y2='" + y.toFixed(2) + "'></line>");
      yLabels.push("<text class='usage-y' x='" + (left - 8) + "' y='" + (y + 4).toFixed(2) + "' text-anchor='end'>" + window.MWP.escapeHtml(tickVal.toFixed(tickVal % 1 === 0 ? 0 : 1)) + "</text>");
    }

    for (var j = 0; j < points.length; j += 1) {
      var x = left + (innerW * j) / Math.max(points.length - 1, 1);
      var value = values[j];
      var y2 = top + innerH - (value / chartMax) * innerH;
      areaPoints.push((j === 0 ? "M" : "L") + x.toFixed(2) + "," + y2.toFixed(2));
      linePoints.push((j === 0 ? "M" : "L") + x.toFixed(2) + "," + y2.toFixed(2));
      dots.push("<circle class='usage-dot' cx='" + x.toFixed(2) + "' cy='" + y2.toFixed(2) + "' r='4'></circle>");
      labels.push("<text class='usage-x' x='" + x.toFixed(2) + "' y='" + (height - 14) + "' text-anchor='middle'>" + window.MWP.escapeHtml(formatVal(points[j].label, "")) + "</text>");
    }

    var areaPath = areaPoints.join(" ") +
      " L" + (left + innerW).toFixed(2) + "," + (top + innerH).toFixed(2) +
      " L" + left.toFixed(2) + "," + (top + innerH).toFixed(2) + " Z";
    var linePath = linePoints.join(" ");
    var totalHours = usage && usage.total_hours !== undefined ? usage.total_hours : 0;

    return (
      "<div class='usage-chart'>" +
      "<div class='usage-head'><div class='usage-metric'>本周总在线 <b>" + window.MWP.escapeHtml(formatVal(totalHours, "0")) + "</b> 小时</div></div>" +
      "<div class='usage-canvas'>" +
      "<svg viewBox='0 0 " + width + " " + height + "' preserveAspectRatio='xMidYMid meet' class='usage-svg'>" +
      "<defs>" +
      "<linearGradient id='usageAreaGrad' x1='0' y1='0' x2='0' y2='1'>" +
      "<stop offset='0%' stop-color='#4f67f7' stop-opacity='0.44'></stop>" +
      "<stop offset='100%' stop-color='#4f67f7' stop-opacity='0.05'></stop>" +
      "</linearGradient>" +
      "</defs>" +
      grids.join("") +
      "<path class='usage-area' d='" + areaPath + "'></path>" +
      "<path class='usage-line' d='" + linePath + "'></path>" +
      dots.join("") +
      yLabels.join("") +
      labels.join("") +
      "<text class='usage-axis-unit' x='14' y='16'>小时</text>" +
      "</svg>" +
      "</div>" +
      "</div>"
    );
  }

  function monthTitle(monthKey) {
    var parts = String(monthKey || "").split("-");
    if (parts.length !== 2) return monthKey;
    return parts[0] + " 年 " + Number(parts[1]) + " 月";
  }

  function heatLevel(count, maxCount) {
    if (count <= 0) return "heat-l0";
    if (maxCount <= 1) return "heat-l1";
    var ratio = count / maxCount;
    if (ratio <= 0.25) return "heat-l1";
    if (ratio <= 0.5) return "heat-l2";
    if (ratio <= 0.75) return "heat-l3";
    return "heat-l4";
  }

  function buildHeatmapData(dayCountMap) {
    var today = new Date();
    var keys = Object.keys(dayCountMap || {});
    var minDate = new Date(today.getFullYear(), today.getMonth() - 11, 1);
    if (keys.length) {
      var first = keys.slice().sort()[0].split("-").map(Number);
      if (first.length === 3 && first[0]) {
        var maybe = new Date(first[0], first[1] - 1, 1);
        if (!isNaN(maybe.getTime()) && maybe < minDate) minDate = maybe;
      }
    }
    var maxDate = new Date(today.getFullYear(), today.getMonth(), 1);
    var months = [];
    var cursor = new Date(minDate.getFullYear(), minDate.getMonth(), 1);
    while (cursor <= maxDate) {
      months.push(new Date(cursor.getFullYear(), cursor.getMonth(), 1));
      cursor = new Date(cursor.getFullYear(), cursor.getMonth() + 1, 1);
    }
    return months;
  }

  function heatmapHtml(dayCountMap) {
    var map = dayCountMap || {};
    var months = buildHeatmapData(map);
    var maxCount = 0;
    Object.keys(map).forEach(function (k) {
      var val = Number(map[k]) || 0;
      if (val > maxCount) maxCount = val;
    });
    var today = new Date();
    var latestMonth = months.length ? months[months.length - 1] : new Date(today.getFullYear(), today.getMonth(), 1);
    var latestMonthKey = latestMonth.getFullYear() + "-" + String(latestMonth.getMonth() + 1).padStart(2, "0");
    var yearMap = {};
    var monthCards = months.map(function (monthDate) {
      var year = monthDate.getFullYear();
      var month = monthDate.getMonth() + 1;
      var monthKey = year + "-" + String(month).padStart(2, "0");
      if (!yearMap[year]) yearMap[year] = [];
      yearMap[year].push(String(month).padStart(2, "0"));
      var firstWeekday = new Date(year, month - 1, 1).getDay();
      var daysInMonth = new Date(year, month, 0).getDate();
      var cells = [];
      var weekLabels = ["日", "一", "二", "三", "四", "五", "六"].map(function (d) {
        return "<div class='heat-weekday'>" + d + "</div>";
      }).join("");
      for (var i = 0; i < firstWeekday; i += 1) {
        cells.push("<div class='heat-pad'></div>");
      }
      for (var day = 1; day <= daysInMonth; day += 1) {
        var dayKey = monthKey + "-" + String(day).padStart(2, "0");
        var count = Number(map[dayKey]) || 0;
        var cellDate = new Date(year, month - 1, day);
        var isFuture = cellDate > today;
        var classes = "heat-day " + (isFuture ? "heat-out" : heatLevel(count, maxCount));
        var countCls = count > 0 ? "heat-count" : "heat-count heat-count-empty";
        var countText = isFuture ? "-" : String(count || 0);
        var title = isFuture ? "未来日期" : (dayKey + " 解题 " + count + " 次");
        cells.push(
          "<div class='heat-item' title='" + window.MWP.escapeHtml(title) + "'>" +
          "<div class='" + classes + "'><span class='" + countCls + "'>" + window.MWP.escapeHtml(countText) + "</span></div>" +
          "<div class='heat-daynum'>" + day + "</div>" +
          "</div>"
        );
      }
      return (
        "<div class='heat-month" + (monthKey === latestMonthKey ? " is-active" : "") + "' data-month='" + monthKey + "'>" +
        "<div class='heat-month-title'>" + monthTitle(monthKey) + "</div>" +
        "<div class='heat-month-grid'>" + weekLabels + cells.join("") + "</div>" +
        "</div>"
      );
    }).join("");

    var yearOptions = Object.keys(yearMap).sort().map(function (year) {
      return "<option value='" + year + "' data-months='" + yearMap[year].join(",") + "'" + (String(latestMonth.getFullYear()) === String(year) ? " selected" : "") + ">" + year + "</option>";
    }).join("");

    var monthOptions = (yearMap[latestMonth.getFullYear()] || []).map(function (month) {
      return "<option value='" + month + "'" + (month === String(latestMonth.getMonth() + 1).padStart(2, "0") ? " selected" : "") + ">" + month + "</option>";
    }).join("");

    return (
      "<div class='heatmap-wrap' data-selected-month='" + latestMonthKey + "'>" +
      "<div class='heatmap-head'>" +
      "<div class='heatmap-controls'>" +
      "<span class='heat-ym-label'>年月</span>" +
      "<select class='heat-year-select heat-ym-select'>" + yearOptions + "</select>" +
      "<span class='heat-ym-sep'>年</span>" +
      "<select class='heat-month-select heat-ym-select'>" + monthOptions + "</select>" +
      "<span class='heat-ym-sep'>月</span>" +
      "</div>" +
      "<div class='heat-toggle-wrap'><button class='btn ghost heat-toggle-btn' type='button'>隐藏题量</button></div>" +
      "</div>" +
      "<div class='heatmap-months'>" + monthCards + "</div>" +
      "</div>"
    );
  }

  function bindHeatmap(root) {
    var wrap = root.querySelector(".heatmap-wrap");
    if (!wrap) return;
    var yearSel = wrap.querySelector(".heat-year-select");
    var monthSel = wrap.querySelector(".heat-month-select");
    var toggleBtn = wrap.querySelector(".heat-toggle-btn");

    function applyMonth(monthKey) {
      var cards = wrap.querySelectorAll(".heat-month");
      for (var i = 0; i < cards.length; i += 1) {
        cards[i].classList.toggle("is-active", cards[i].getAttribute("data-month") === monthKey);
      }
      wrap.setAttribute("data-selected-month", monthKey);
    }

    function rebuildMonthOptions(yearValue, preferred) {
      var opt = yearSel && yearSel.querySelector("option[value='" + yearValue + "']");
      if (!opt || !monthSel) return;
      var months = String(opt.getAttribute("data-months") || "").split(",").filter(Boolean);
      monthSel.innerHTML = months.map(function (m) {
        return "<option value='" + m + "'" + (m === preferred ? " selected" : "") + ">" + m + "</option>";
      }).join("");
      if (!monthSel.value && months.length) monthSel.value = months[months.length - 1];
    }

    if (yearSel) {
      yearSel.addEventListener("change", function () {
        rebuildMonthOptions(yearSel.value, "");
        if (monthSel && monthSel.value) applyMonth(yearSel.value + "-" + monthSel.value);
      });
    }
    if (monthSel) {
      monthSel.addEventListener("change", function () {
        if (yearSel) applyMonth(yearSel.value + "-" + monthSel.value);
      });
    }
    if (toggleBtn) {
      toggleBtn.addEventListener("click", function () {
        var hidden = wrap.classList.toggle("hide-count");
        toggleBtn.textContent = hidden ? "显示题量" : "隐藏题量";
      });
    }
  }

  async function loadHome(session) {
    var root = document.getElementById("page");
    root.innerHTML =
      "<div class='card home-banner'>" +
      "<section class='home-hero'>" +
      "<div class='home-hero-inner'>" +
      "<div>" +
      "<h1 class='home-title'>基于数学语义理解的数学应用题求解系统</h1>" +
      "<p class='home-sub'>面向中文数学应用题的建模与推理，支持表达式生成与结果验证，保持清晰、稳定、可复现的解题体验。</p>" +
      "</div>" +
      "<aside class='home-art' aria-hidden='true'>" +
      "<span class='home-orbit a'></span>" +
      "<span class='home-orbit b'></span>" +
      "<span class='home-orbit c'></span>" +
      "</aside>" +
      "</div>" +
      "<div class='home-actions'>" +
      "<a class='btn' href='/solve.html'>进入智能解题</a>" +
      "<a class='btn ghost' href='/docs.html'>操作文档</a>" +
      "</div>" +
      "</section>" +
      "</div>";
  }

  async function loadDocs() {
    var root = document.getElementById("page");
    root.innerHTML = '<div class="card"><h2>操作文档</h2><p class="sub">正在加载...</p></div>';
    var payload = await window.MWP.requestJSON("/api/docs");
    var data = payload.data || {};
    root.innerHTML =
      '<div class="card">' +
      '<div class="row" style="justify-content:space-between;align-items:center;margin-bottom:12px;">' +
      '<h2 style="margin:0;">' + window.MWP.escapeHtml(formatVal(data.title, "操作文档")) + '</h2>' +
      '<a class="btn ghost" href="/home.html">返回首页</a>' +
      '</div>' +
      '<pre style="white-space:pre-wrap;line-height:1.7;background:#f6f8fc;border:1px solid var(--line);border-radius:10px;padding:14px;">' +
      window.MWP.escapeHtml(formatVal(data.content, "")) + '</pre></div>';
  }

  async function loadSolve() {
    var root = document.getElementById("page");
    root.innerHTML =
      "<div class='solve-shell'>" +
      "<div class='solve-grid'>" +
      "<div class='card solve-main'>" +
      "<form id='solve-form'>" +
      "<h2>智能解题</h2>" +
      "<p class='sub'>左侧输入题目，点击中间按钮后在右侧查看输出结果。</p>" +
      "<label>题目</label>" +
      "<textarea name='question' placeholder='例如：每本书2元，买11本一共多少钱？'></textarea>" +
      "</form>" +
      "</div>" +
      "<div class='solve-cta-col'>" +
      "<button class='btn solve-submit' type='submit' form='solve-form'>开始解题</button>" +
      "<p class='sub'>提交后右侧更新</p>" +
      "</div>" +
      "<div class='card solve-output' id='solve-result'>" + solveOutputPlaceholder() + "</div>" +
      "</div>" +
      "<div class='card solve-history'>" +
      "<div class='row solve-history-head'>" +
      "<h3 style='margin:0;'>最近解题记录</h3>" +
      "<a class='btn ghost' href='/stats.html'>进入统计记录</a>" +
      "</div>" +
      "<div id='solve-history'></div>" +
      "<div id='solve-pagination'></div>" +
      "</div>" +
      "</div>";

    var state = {
      page: 1,
      pageSize: 5,
      selectedId: null,
    };

    function bindCloseView() {
      var closeBtn = document.getElementById("solve-view-close");
      if (closeBtn) {
        closeBtn.onclick = function () {
          state.selectedId = null;
          document.getElementById("solve-result").innerHTML = solveOutputPlaceholder();
        };
      }
    }

    async function showDetail(historyId) {
      var payload = await window.MWP.requestJSON("/api/history/" + historyId);
      var item = payload.data && payload.data.item ? payload.data.item : null;
      state.selectedId = item && item.id ? Number(item.id) : Number(historyId);
      var title = "输出结果";
      if (item && item.display_no !== undefined && item.display_no !== null) {
        title = "记录 #" + String(item.display_no) + " 输出";
      }
      document.getElementById("solve-result").innerHTML = renderSolveOutput(item, {
        title: title,
        showClose: true,
        timeText: item ? item.created_at_local : "",
      });
      bindCloseView();
    }

    async function refreshHistory() {
      var box = document.getElementById("solve-history");
      var pager = document.getElementById("solve-pagination");
      var payload = await window.MWP.requestJSON("/api/history?page=" + state.page + "&page_size=" + state.pageSize);
      var data = payload.data || {};
      var items = Array.isArray(data.items) ? data.items : [];

      var historyRowsHtml = items.map(function (item) {
        return (
          "<tr>" +
          "<td class='mono'>" + window.MWP.escapeHtml(formatVal(item.display_no, item.id)) + "</td>" +
          "<td>" + window.MWP.escapeHtml(formatVal(item.created_at_local, "-")) + "</td>" +
          "<td>" + window.MWP.escapeHtml(formatVal(item.question, "")) + "</td>" +
          "<td class='mono'>" + window.MWP.escapeHtml(formatVal(item.expression_text, "")) + "</td>" +
          "<td class='mono'>" + window.MWP.escapeHtml(formatVal(item.predicted_value, "-")) + "</td>" +
          "<td class='ops'>" +
          "<button class='btn ghost small btn-view-history' type='button' data-history-id='" + item.id + "'>查看</button>" +
          "<button class='btn warn small btn-delete-history' type='button' data-history-id='" + item.id + "'>删除</button>" +
          "</td>" +
          "</tr>"
        );
      }).join("");

      box.innerHTML = historyRowsHtml
        ? (
          "<div class='table-wrap'><table>" +
          "<thead><tr>" +
          "<th style='width:68px;'>序号</th>" +
          "<th style='width:170px;'>时间</th>" +
          "<th>题目</th>" +
          "<th style='width:280px;'>表达式</th>" +
          "<th style='width:120px;'>答案</th>" +
          "<th style='width:165px;text-align:center;'>操作</th>" +
          "</tr></thead>" +
          "<tbody>" + historyRowsHtml + "</tbody></table></div>"
        )
        : "<div class='empty'>暂无记录</div>";

      var pagination = data.pagination || {};
      var totalPages = Number(pagination.total_pages) || 1;
      var page = Number(pagination.page) || 1;
      var totalCount = Number(pagination.total_count) || 0;
      state.page = page;
      pager.innerHTML = solvePaginationHtml(page, totalPages, totalCount);

      var viewBtns = box.querySelectorAll(".btn-view-history");
      for (var i = 0; i < viewBtns.length; i += 1) {
        viewBtns[i].onclick = function () {
          showDetail(this.getAttribute("data-history-id"));
        };
      }
      var delBtns = box.querySelectorAll(".btn-delete-history");
      for (var j = 0; j < delBtns.length; j += 1) {
        delBtns[j].onclick = async function () {
          var historyId = this.getAttribute("data-history-id");
          if (!window.confirm("确认删除这条记录？")) return;
          try {
            await window.MWP.requestJSON("/api/history/" + historyId, { method: "DELETE" });
            if (String(state.selectedId) === String(historyId)) {
              state.selectedId = null;
              document.getElementById("solve-result").innerHTML = solveOutputPlaceholder();
            }
            await refreshHistory();
          } catch (err) {
            window.MWP.showFlash("error", err.message || "删除失败");
          }
        };
      }

      var pageBtns = pager.querySelectorAll(".btn-solve-page");
      for (var k = 0; k < pageBtns.length; k += 1) {
        pageBtns[k].onclick = function () {
          var nextPage = Number(this.getAttribute("data-page")) || 1;
          if (nextPage === state.page) return;
          state.page = nextPage;
          refreshHistory();
        };
      }
    }

    await refreshHistory();

    var form = document.getElementById("solve-form");
    form.addEventListener("submit", async function (event) {
      event.preventDefault();
      var resultBox = document.getElementById("solve-result");
      resultBox.innerHTML = "<h3>输出结果</h3><p class='sub'>正在求解...</p>";
      try {
        var payload = await window.MWP.requestJSON("/api/solve", {
          method: "POST",
          data: { question: String(new FormData(form).get("question") || "") },
        });
        var item = payload.data && payload.data.history_item ? payload.data.history_item : null;
        if (item) {
          state.selectedId = Number(item.id);
          resultBox.innerHTML = renderSolveOutput(item, {
            title: "输出结果",
            showClose: true,
          });
          bindCloseView();
        } else {
          resultBox.innerHTML = "<h3>输出结果</h3><p class='sub'>解题完成。</p>";
        }
        state.page = 1;
        await refreshHistory();
      } catch (err) {
        resultBox.innerHTML = "<h3>输出结果</h3><p style='color:#b63a3a;'>" + window.MWP.escapeHtml(err.message || "解题失败") + "</p>";
      }
    });
  }

  async function loadStats() {
    var root = document.getElementById("page");
    root.innerHTML = '<div class="card"><h2>统计记录</h2><p class="sub">正在加载...</p></div>';
    var payload = await window.MWP.requestJSON("/api/stats/overview");
    var data = payload.data || {};
    var s = data.summary || {};

    root.innerHTML =
      "<div class='stats-shell'>" +
      "<div class='card'>" +
      "<h2>统计记录</h2>" +
      "<p class='sub'>查看累计解题与活跃趋势。</p>" +
      "<div class='row'>" +
      "<div class='stat-box col'><div class='sub'>累计解题</div><div class='stat-num'>" + window.MWP.escapeHtml(formatVal(s.total_cnt, "0")) + "</div></div>" +
      "<div class='stat-box col'><div class='sub'>今日</div><div class='stat-num'>" + window.MWP.escapeHtml(formatVal(s.today_cnt, "0")) + "</div></div>" +
      "<div class='stat-box col'><div class='sub'>本周（" + window.MWP.escapeHtml(formatVal(s.week_range_label, "")) + "）</div><div class='stat-num'>" + window.MWP.escapeHtml(formatVal(s.week_cnt, "0")) + "</div></div>" +
      "<div class='stat-box col'><div class='sub'>近30天</div><div class='stat-num'>" + window.MWP.escapeHtml(formatVal(s.month_cnt, "0")) + "</div></div>" +
      "<div class='stat-box col'><div class='sub'>连续打卡</div><div class='stat-num'>" + window.MWP.escapeHtml(formatVal(s.streak, "0")) + "</div></div>" +
      "<div class='stat-box col'><div class='sub'>30天活跃</div><div class='stat-num'>" + window.MWP.escapeHtml(formatVal(s.active_days_30, "0")) + "</div></div>" +
      "</div>" +
      "</div>" +
      "<div class='card'><h3>本周在线时长</h3>" + usageChartHtml(data.usage || {}) + "</div>" +
      "<div class='card'><h3>活跃热力图</h3>" + heatmapHtml((data.heatmap && data.heatmap.day_count_map) || {}) + "</div>" +
      "</div>";
    bindHeatmap(root);
  }

  async function loadProfile(session) {
    var root = document.getElementById("page");
    root.innerHTML = '<div class="card"><h2>账户设置</h2><p class="sub">正在加载...</p></div>';
    var payload = await window.MWP.requestJSON("/api/profile");
    var data = payload.data || {};
    var profile = {
      user: data.user || session.user || {},
      role_text: data.role_text || "普通用户",
      status_text: data.status_text || "启用",
      created_at_local: data.created_at_local || "-",
    };
    var verifyToken = "";
    var alertType = "";
    var alertMsg = "";

    function setAlert(type, msg) {
      alertType = type || "";
      alertMsg = msg || "";
    }

    function applyProfilePayload(next) {
      if (!next || typeof next !== "object") return;
      if (next.user) profile.user = next.user;
      if (next.role_text) profile.role_text = next.role_text;
      if (next.status_text) profile.status_text = next.status_text;
      if (next.created_at_local) profile.created_at_local = next.created_at_local;
    }

    function renderProfile() {
      var user = profile.user || {};
      var isAdmin = !!user.is_admin;
      var sideTitleHtml = isAdmin ? "" : "<h2>账户设置</h2>";
      var backLinkHtml = isAdmin
        ? "<a class='btn ghost' href='/admin.html'>↩ 返回数据后台</a>"
        : "<a class='btn ghost' href='/home.html'>↩ 返回首页</a>";
      var avatar = window.MWP.escapeHtml(String(formatVal(user.username, "U")).slice(0, 1).toUpperCase());
      var alerts = "";
      if (alertMsg) {
        alerts = "<div class='alert " + (alertType === "ok" ? "ok" : "err") + "'>" + window.MWP.escapeHtml(alertMsg) + "</div>";
      }
      var stepHtml = "";
      if (verifyToken) {
        stepHtml =
          "<h3 style='margin-bottom:8px;'>修改账户信息</h3>" +
          "<p class='sub'>用户名可直接修改。若修改密码，请填写并确认新密码。</p>" +
          "<form class='profile-form' id='profile-update-form'>" +
          "<input type='hidden' name='verify_token' value='" + window.MWP.escapeHtml(verifyToken) + "' />" +
          "<label>登录账号（用户名）</label>" +
          "<input name='new_username' value='" + window.MWP.escapeHtml(formatVal(user.username, "")) + "' minlength='2' maxlength='32' required />" +
          "<label>新密码（留空表示不修改）</label>" +
          "<input name='new_password' type='password' minlength='6' />" +
          "<label>确认新密码</label>" +
          "<input name='confirm_password' type='password' minlength='6' />" +
          "<div class='profile-actions'>" +
          "<button class='btn' type='submit'>保存修改</button>" +
          "<button class='btn ghost' type='button' id='profile-back-verify'>返回身份验证</button>" +
          "</div></form>";
      } else {
        stepHtml =
          "<h3 style='margin-bottom:8px;'>身份验证</h3>" +
          "<p class='sub'>请输入当前登录账号与密码。</p>" +
          "<form class='profile-form' id='profile-verify-form'>" +
          "<label>当前登录账号</label>" +
          "<input name='current_username' value='" + window.MWP.escapeHtml(formatVal(user.username, "")) + "' required />" +
          "<label>当前登录密码</label>" +
          "<input name='current_password' type='password' required />" +
          "<div class='profile-actions'><button class='btn' type='submit'>确认</button></div></form>";
      }

      root.innerHTML =
        "<div class='profile-shell'>" +
        "<div class='profile-grid'>" +
        "<div class='card profile-side'>" +
        sideTitleHtml +
        backLinkHtml +
        "<div style='height:1px;background:#e8edf7;margin:10px 0 12px;'></div>" +
        "<div class='profile-summary'>" +
        "<div style='display:flex;align-items:center;gap:10px;'>" +
        "<button class='avatar-btn' type='button' style='color:#2f447d;border-color:#ccd6f0;background:#eef3ff;'>" + avatar + "</button>" +
        "<div><div><b>" + window.MWP.escapeHtml(formatVal(user.username, "")) + "</b></div>" +
        "<div class='sub'>UID: " + window.MWP.escapeHtml(formatVal(user.id, "-")) + "</div></div></div>" +
        "<ul class='kv' style='margin-top:8px;'>" +
        "<li>角色：<span class='mono'>" + window.MWP.escapeHtml(formatVal(profile.role_text, "-")) + "</span></li>" +
        "<li>状态：<span class='mono'>" + window.MWP.escapeHtml(formatVal(profile.status_text, "-")) + "</span></li>" +
        "<li>注册时间：<span class='mono'>" + window.MWP.escapeHtml(formatVal(profile.created_at_local, "-")) + "</span></li>" +
        "</ul></div></div>" +
        "<div class='profile-main'><div class='card profile-form-card'>" +
        alerts +
        stepHtml +
        "</div></div></div></div>";

      var verifyForm = document.getElementById("profile-verify-form");
      if (verifyForm) {
        verifyForm.addEventListener("submit", async function (event) {
          event.preventDefault();
          var fd = new FormData(verifyForm);
          try {
            var resp = await window.MWP.requestJSON("/api/profile/verify", {
              method: "POST",
              data: {
                current_username: String(fd.get("current_username") || ""),
                current_password: String(fd.get("current_password") || ""),
              },
            });
            var respData = resp.data || {};
            verifyToken = String(respData.verify_token || "");
            applyProfilePayload(respData.profile || {});
            setAlert("ok", resp.message || "身份验证通过，请继续修改账户信息。");
          } catch (err) {
            verifyToken = "";
            setAlert("err", err.message || "身份验证失败");
          }
          renderProfile();
        });
      }

      var updateForm = document.getElementById("profile-update-form");
      if (updateForm) {
        updateForm.addEventListener("submit", async function (event) {
          event.preventDefault();
          var fd = new FormData(updateForm);
          try {
            var resp = await window.MWP.requestJSON("/api/profile/update", {
              method: "POST",
              data: {
                verify_token: String(fd.get("verify_token") || ""),
                new_username: String(fd.get("new_username") || ""),
                new_password: String(fd.get("new_password") || ""),
                confirm_password: String(fd.get("confirm_password") || ""),
              },
            });
            var respData = resp.data || {};
            applyProfilePayload(respData.profile || {});
            verifyToken = "";
            setAlert("ok", resp.message || "账户信息已更新。");
            window.MWP.renderTopbar({ user: profile.user, active: "/profile.html" });
          } catch (err) {
            setAlert("err", err.message || "保存失败");
          }
          renderProfile();
        });
      }

      var backVerifyBtn = document.getElementById("profile-back-verify");
      if (backVerifyBtn) {
        backVerifyBtn.onclick = function () {
          verifyToken = "";
          setAlert("", "");
          renderProfile();
        };
      }
    }

    renderProfile();
  }

  async function loadAdmin() {
    var root = document.getElementById("page");
    root.innerHTML = '<div class="card"><h2>数据后台</h2><p class="sub">正在加载...</p></div>';
    var state = {
      page: 1,
      pageSize: 10,
      noticeType: "",
      noticeMsg: "",
    };
    var latestData = null;

    function setNotice(type, msg) {
      state.noticeType = type || "";
      state.noticeMsg = msg || "";
    }

    function adminPaginationHtml(page, totalPages, totalCount) {
      if (totalPages <= 1) return "";
      var prevHtml = page > 1
        ? "<button class='btn ghost page-btn btn-admin-page' type='button' data-page='" + (page - 1) + "'>上一页</button>"
        : "<span class='btn ghost page-btn disabled'>上一页</span>";
      var nextHtml = page < totalPages
        ? "<button class='btn ghost page-btn btn-admin-page' type='button' data-page='" + (page + 1) + "'>下一页</button>"
        : "<span class='btn ghost page-btn disabled'>下一页</span>";
      var windowSize = 5;
      var startPage = 1;
      var endPage = totalPages;
      if (totalPages > windowSize) {
        var half = Math.floor(windowSize / 2);
        startPage = page - half;
        if (startPage < 1) startPage = 1;
        var maxStart = totalPages - windowSize + 1;
        if (startPage > maxStart) startPage = maxStart;
        endPage = startPage + windowSize - 1;
      }
      var pageNodes = [];
      for (var p = startPage; p <= endPage; p += 1) {
        if (p === page) {
          pageNodes.push("<span class='page-current'>" + p + "</span>");
        } else {
          pageNodes.push("<button class='btn ghost page-btn btn-admin-page' type='button' data-page='" + p + "'>" + p + "</button>");
        }
      }
      return (
        "<div class='admin-pagination'>" +
        prevHtml +
        pageNodes.join("") +
        nextHtml +
        "<span class='page-meta'>第 " + page + "/" + totalPages + " 页，共 " + totalCount + " 条</span>" +
        "</div>"
      );
    }

    function renderAdmin(data) {
      latestData = data || {};
      var summary = latestData.summary || {};
      var adminCounts = latestData.admin_counts || {};
      var users = latestData.users && Array.isArray(latestData.users.items) ? latestData.users.items : [];
      var pagination = latestData.users && latestData.users.pagination ? latestData.users.pagination : {};
      var errors = Array.isArray(latestData.errors) ? latestData.errors : [];
      var activeAdminCount = Number(adminCounts.active_admin_count) || 0;
      var adminTotalCount = Number(adminCounts.admin_total_count) || 0;

      var alerts = "";
      if (state.noticeMsg) {
        alerts = "<div class='alert " + (state.noticeType === "ok" ? "ok" : "err") + "'>" + window.MWP.escapeHtml(state.noticeMsg) + "</div>";
      }

      var userRowsHtml = users.map(function (u) {
        var userId = Number(u.id) || 0;
        var resetFormId = "reset-pwd-" + userId;
        var adminBtn = u.is_admin ? "取消管理员" : "设为管理员";
        var activeBtn = u.is_active ? "禁用账号" : "启用账号";
        var hint = "";
        if (u.is_admin && u.is_active && activeAdminCount <= 1) {
          hint = "<div class='sub'>当前为唯一活跃管理员，受保护。</div>";
        } else if (u.is_admin && adminTotalCount <= 1) {
          hint = "<div class='sub'>当前为唯一管理员，受保护。</div>";
        }
        return (
          "<tr>" +
          "<td>" + window.MWP.escapeHtml(formatVal(u.id, "")) + "</td>" +
          "<td>" + window.MWP.escapeHtml(formatVal(u.username, "")) + "</td>" +
          "<td>" + window.MWP.escapeHtml(u.is_admin ? "是" : "否") + "</td>" +
          "<td>" + window.MWP.escapeHtml(u.is_active ? "启用" : "禁用") + "</td>" +
          "<td>" + window.MWP.escapeHtml(formatVal(u.created_at_local, "-")) + "</td>" +
          "<td>" +
          "<div class='admin-ops'>" +
          "<button class='btn ghost btn-admin-action' type='button' data-action='toggle-admin' data-user-id='" + userId + "'>" + window.MWP.escapeHtml(adminBtn) + "</button>" +
          "<button class='btn ghost btn-admin-action' type='button' data-action='toggle-active' data-user-id='" + userId + "'>" + window.MWP.escapeHtml(activeBtn) + "</button>" +
          "<button class='btn ghost btn-toggle-reset' type='button' data-form-id='" + resetFormId + "'>重置密码</button>" +
          "<button class='btn warn btn-admin-delete' type='button' data-user-id='" + userId + "'>删除用户</button>" +
          "</div>" +
          "<form id='" + resetFormId + "' class='reset-pwd-form' data-user-id='" + userId + "'>" +
          "<input name='new_password' type='password' minlength='6' placeholder='新密码(>=6位)' required />" +
          "<input name='confirm_password' type='password' minlength='6' placeholder='确认新密码' required />" +
          "<button class='btn ghost' type='submit'>确认重置</button>" +
          "<button class='btn ghost btn-cancel-reset' type='button' data-form-id='" + resetFormId + "'>取消</button>" +
          "</form>" +
          hint +
          "</td>" +
          "</tr>"
        );
      }).join("");

      var errorsHtml = errors.map(function (e) {
        var contextPreview = formatVal(e.context_json, "");
        if (contextPreview !== "-" && contextPreview.length > 180) contextPreview = contextPreview.slice(0, 180) + "...";
        return (
          "<tr>" +
          "<td>" + window.MWP.escapeHtml(formatVal(e.id, "")) + "</td>" +
          "<td>" + window.MWP.escapeHtml(formatVal(e.created_at_local, "-")) + "</td>" +
          "<td>" + window.MWP.escapeHtml(formatVal(e.username, "-")) + "</td>" +
          "<td>" + window.MWP.escapeHtml(formatVal(e.scope, "-")) + "</td>" +
          "<td>" + window.MWP.escapeHtml(formatVal(e.message, "-")) + "</td>" +
          "<td class='mono'>" + window.MWP.escapeHtml(contextPreview === "-" ? "" : contextPreview) + "</td>" +
          "</tr>"
        );
      }).join("");

      root.innerHTML =
        "<div class='admin-shell'>" +
        "<div class='card'>" +
        "<h3>系统统计</h3>" +
        alerts +
        "<div class='row admin-kpis'>" +
        "<div class='col card'><h3>用户总数</h3><p class='mono'>" + window.MWP.escapeHtml(formatVal(summary.user_total, "0")) + "</p></div>" +
        "<div class='col card'><h3>启用用户</h3><p class='mono'>" + window.MWP.escapeHtml(formatVal(summary.active_user_total, "0")) + "</p></div>" +
        "<div class='col card'><h3>累计解题</h3><p class='mono'>" + window.MWP.escapeHtml(formatVal(summary.solve_total, "0")) + "</p></div>" +
        "<div class='col card'><h3>24h 解题</h3><p class='mono'>" + window.MWP.escapeHtml(formatVal(summary.solve_24h, "0")) + "</p></div>" +
        "</div>" +
        "<p class='sub admin-kpi-note'>当前管理员数：" + window.MWP.escapeHtml(formatVal(adminTotalCount, "0")) + "，活跃管理员数：" + window.MWP.escapeHtml(formatVal(activeAdminCount, "0")) + "</p>" +
        "</div>" +

        "<div class='card'>" +
        "<h3>新增用户</h3>" +
        "<form id='admin-create-form'>" +
        "<div class='row'>" +
        "<div class='col'><label>用户名</label><input name='username' required /></div>" +
        "<div class='col'><label>初始密码（至少 6 位）</label><input name='password' type='password' minlength='6' required /></div>" +
        "</div>" +
        "<div class='row admin-create-options'>" +
        "<label><input type='checkbox' name='is_admin' value='1' /> 管理员</label>" +
        "<label><input type='checkbox' name='is_active' value='1' checked /> 启用</label>" +
        "</div>" +
        "<div class='admin-create-submit'><button class='btn' type='submit'>创建用户</button></div>" +
        "</form>" +
        "</div>" +

        "<div class='card'>" +
        "<h3>用户列表</h3>" +
        "<div class='table-wrap'><table>" +
        "<thead><tr>" +
        "<th style='width:60px;'>ID</th><th style='width:120px;'>用户名</th><th style='width:80px;'>管理员</th><th style='width:80px;'>状态</th><th style='width:160px;'>创建时间</th><th style='width:430px;'>操作</th>" +
        "</tr></thead>" +
        "<tbody>" + (userRowsHtml || "<tr><td colspan='6' class='sub'>暂无用户</td></tr>") + "</tbody>" +
        "</table></div>" +
        adminPaginationHtml(Number(pagination.page) || state.page, Number(pagination.total_pages) || 1, Number(pagination.total_count) || 0) +
        "</div>" +

        "<div class='card'>" +
        "<div class='row' style='justify-content:space-between;align-items:center;'>" +
        "<h3 style='margin:0;'>最近错误日志</h3>" +
        "<button class='btn warn' type='button' id='btn-clear-errors'>清空错误日志</button>" +
        "</div>" +
        "<div class='table-wrap'><table>" +
        "<thead><tr><th style='width:60px;'>ID</th><th style='width:170px;'>时间</th><th style='width:120px;'>用户</th><th style='width:90px;'>范围</th><th>错误信息</th><th style='width:230px;'>上下文</th></tr></thead>" +
        "<tbody>" + (errorsHtml || "<tr><td colspan='6' class='sub'>暂无错误日志</td></tr>") + "</tbody>" +
        "</table></div>" +
        "</div>" +
        "</div>";

      bindAdminEvents();
    }

    function rerenderCurrent() {
      if (latestData) renderAdmin(latestData);
    }

    async function refreshAdmin() {
      var payload = await window.MWP.requestJSON(
        "/api/admin/overview?page=" + state.page + "&page_size=" + state.pageSize
      );
      renderAdmin(payload.data || {});
    }

    function bindAdminEvents() {
      var createForm = document.getElementById("admin-create-form");
      if (createForm) {
        createForm.addEventListener("submit", async function (event) {
          event.preventDefault();
          var fd = new FormData(createForm);
          try {
            var resp = await window.MWP.requestJSON("/api/admin/users", {
              method: "POST",
              data: {
                username: String(fd.get("username") || ""),
                password: String(fd.get("password") || ""),
                is_admin: fd.get("is_admin") ? "1" : "0",
                is_active: fd.get("is_active") ? "1" : "0",
              },
            });
            state.page = 1;
            setNotice("ok", resp.message || "创建成功。");
            await refreshAdmin();
          } catch (err) {
            setNotice("err", err.message || "创建失败");
            rerenderCurrent();
          }
        });
      }

      var adminActionBtns = root.querySelectorAll(".btn-admin-action");
      for (var i = 0; i < adminActionBtns.length; i += 1) {
        adminActionBtns[i].onclick = async function () {
          var userId = this.getAttribute("data-user-id");
          var action = this.getAttribute("data-action");
          var endpoint = "/api/admin/users/" + userId + "/" + (action === "toggle-admin" ? "toggle-admin" : "toggle-active");
          try {
            var resp = await window.MWP.requestJSON(endpoint, { method: "POST" });
            setNotice("ok", resp.message || "更新成功。");
            await refreshAdmin();
          } catch (err) {
            setNotice("err", err.message || "更新失败");
            rerenderCurrent();
          }
        };
      }

      var deleteBtns = root.querySelectorAll(".btn-admin-delete");
      for (var j = 0; j < deleteBtns.length; j += 1) {
        deleteBtns[j].onclick = async function () {
          var userId = this.getAttribute("data-user-id");
          if (!window.confirm("确认删除该用户及其会话和历史记录？")) return;
          try {
            var resp = await window.MWP.requestJSON("/api/admin/users/" + userId, { method: "DELETE" });
            setNotice("ok", resp.message || "删除成功。");
            await refreshAdmin();
          } catch (err) {
            setNotice("err", err.message || "删除失败");
            rerenderCurrent();
          }
        };
      }

      function closeAllResetForms() {
        var opened = root.querySelectorAll(".reset-pwd-form.is-open");
        for (var c = 0; c < opened.length; c += 1) opened[c].classList.remove("is-open");
      }

      var toggleResetBtns = root.querySelectorAll(".btn-toggle-reset");
      for (var k = 0; k < toggleResetBtns.length; k += 1) {
        toggleResetBtns[k].onclick = function () {
          var formId = this.getAttribute("data-form-id");
          var form = document.getElementById(formId);
          if (!form) return;
          var shouldClose = form.classList.contains("is-open");
          closeAllResetForms();
          if (!shouldClose) form.classList.add("is-open");
        };
      }

      var cancelResetBtns = root.querySelectorAll(".btn-cancel-reset");
      for (var m = 0; m < cancelResetBtns.length; m += 1) {
        cancelResetBtns[m].onclick = function () {
          var formId = this.getAttribute("data-form-id");
          var form = document.getElementById(formId);
          if (form) form.classList.remove("is-open");
        };
      }

      var resetForms = root.querySelectorAll(".reset-pwd-form");
      for (var n = 0; n < resetForms.length; n += 1) {
        resetForms[n].addEventListener("submit", async function (event) {
          event.preventDefault();
          var userId = this.getAttribute("data-user-id");
          var fd = new FormData(this);
          try {
            var resp = await window.MWP.requestJSON("/api/admin/users/" + userId + "/reset-password", {
              method: "POST",
              data: {
                new_password: String(fd.get("new_password") || ""),
                confirm_password: String(fd.get("confirm_password") || ""),
              },
            });
            setNotice("ok", resp.message || "密码已重置。");
            await refreshAdmin();
          } catch (err) {
            setNotice("err", err.message || "重置密码失败");
            rerenderCurrent();
          }
        });
      }

      var clearErrorsBtn = document.getElementById("btn-clear-errors");
      if (clearErrorsBtn) {
        clearErrorsBtn.onclick = async function () {
          if (!window.confirm("确认清空错误日志？")) return;
          try {
            var resp = await window.MWP.requestJSON("/api/admin/errors", { method: "DELETE" });
            setNotice("ok", resp.message || "错误日志已清空。");
            await refreshAdmin();
          } catch (err) {
            setNotice("err", err.message || "清空错误日志失败");
            rerenderCurrent();
          }
        };
      }

      var pageBtns = root.querySelectorAll(".btn-admin-page");
      for (var p = 0; p < pageBtns.length; p += 1) {
        pageBtns[p].onclick = function () {
          var nextPage = Number(this.getAttribute("data-page")) || 1;
          if (nextPage === state.page) return;
          state.page = nextPage;
          refreshAdmin();
        };
      }
    }

    await refreshAdmin();
  }

  function startHeartbeat(session) {
    if (!session || !session.user || session.user.is_admin) return;
    function ping() {
      window.MWP.requestJSON("/api/online/heartbeat", { method: "POST" }).catch(function () {});
    }
    setTimeout(ping, 1000);
    setInterval(ping, 30000);
  }

  async function boot() {
    var page = document.body.getAttribute("data-page");
    var session = await window.MWP.ensureSession({
      adminOnly: page === "admin",
      userOnly: page === "home" || page === "solve" || page === "stats",
    });
    if (!session) return;

    var active = "/" + page + ".html";
    if (page === "docs") active = "/docs.html";
    if (page === "profile") active = "/profile.html";
    window.MWP.renderTopbar({ user: session.user, active: active });
    startHeartbeat(session);

    try {
      if (page === "home") await loadHome(session);
      else if (page === "solve") await loadSolve();
      else if (page === "stats") await loadStats();
      else if (page === "profile") await loadProfile(session);
      else if (page === "admin") await loadAdmin();
      else if (page === "docs") await loadDocs();
      else document.getElementById("page").innerHTML = '<div class="card"><h2>页面不存在</h2></div>';
    } catch (err) {
      document.getElementById("page").innerHTML =
        '<div class="card"><h2>加载失败</h2><p style="color:#b63a3a;">' +
        window.MWP.escapeHtml(err.message || "请求失败") +
        '</p></div>';
    }
  }

  boot();
})();
