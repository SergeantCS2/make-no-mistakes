/* makenomistakes web UI: the terminal's small behaviours. No framework, no loops: everything
   runs on load, on a click or key, or when htmx swaps the status bar in (every 10 s).
   Preferences live in localStorage (per browser), the VRAM history and "seen" states in
   sessionStorage; each access is guarded, the page works without them. */
(function () {
  "use strict";
  var root = document.documentElement;
  var reduced = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function load(store, key, fallback) {
    try { var v = window[store].getItem(key); return v === null ? fallback : JSON.parse(v); } catch (e) { return fallback; }
  }
  function save(store, key, value) {
    try { window[store].setItem(key, JSON.stringify(value)); } catch (e) { /* private window */ }
  }

  /* ---- preferences: phosphor colour, effects, sound ---- */
  var prefs = load("localStorage", "mnm.prefs", {});
  prefs = { phosphor: prefs.phosphor || "amber", fx: prefs.fx || "on", sound: prefs.sound || "off", keys: prefs.keys || "on" };
  function applyPrefs() {
    root.dataset.phosphor = prefs.phosphor;
    root.dataset.fx = prefs.fx;
    document.querySelectorAll("[data-pref]").forEach(function (b) {
      var key = b.dataset.pref, on = b.dataset.on;
      b.setAttribute("aria-pressed", String(prefs[key] === on));
    });
  }
  document.addEventListener("click", function (e) {
    var b = e.target.closest("[data-pref]");
    if (!b) return;
    var key = b.dataset.pref;
    prefs[key] = prefs[key] === b.dataset.on ? b.dataset.off : b.dataset.on;
    save("localStorage", "mnm.prefs", prefs);
    applyPrefs();
    if (key === "sound" && prefs.sound === "on") beep(880, 0.06);
  });

  /* ---- sound: synthesised clicks, only after the user turns it on ---- */
  var audio = null;
  function beep(freq, length, gain) {
    if (prefs.sound !== "on") return;
    try {
      audio = audio || new (window.AudioContext || window.webkitAudioContext)();
      var osc = audio.createOscillator(), amp = audio.createGain();
      osc.type = "square"; osc.frequency.value = freq;
      amp.gain.value = gain || 0.025;
      osc.connect(amp); amp.connect(audio.destination);
      osc.start(); osc.stop(audio.currentTime + length);
    } catch (e) { /* no audio here */ }
  }
  document.addEventListener("click", function (e) {
    if (e.target.closest(".btn, .tabs a, .subtabs a")) beep(1400 + Math.random() * 300, 0.018);
  });

  /* ---- copy buttons (data-copy) ---- */
  window.copyText = function (button) {
    var text = button.dataset.copy, old = button.textContent;
    function done(label) { button.textContent = label; setTimeout(function () { button.textContent = old; }, 1200); }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(function () { done("Copied"); }, function () { done("Select it"); });
    } else { done("Select it"); }
    beep(660, 0.04);
  };

  /* ---- the clock ---- */
  function tick() {
    var el = document.getElementById("clock");
    if (!el) return;
    var d = new Date();
    el.textContent = String(d.getHours()).padStart(2, "0") + ":" + String(d.getMinutes()).padStart(2, "0");
  }

  /* ---- boot sequence: once a browser session, any key or click skips it ---- */
  function boot() {
    var el = document.getElementById("boot");
    if (!el) return;
    if (reduced || prefs.fx === "off" || load("sessionStorage", "mnm.booted", false)) { el.remove(); return; }
    save("sessionStorage", "mnm.booted", true);
    el.hidden = false;
    var lines = JSON.parse(el.dataset.lines || "[]"), out = el.querySelector(".lines"), i = 0, timer = null;
    function finish() { clearTimeout(timer); el.remove(); document.removeEventListener("keydown", finish); }
    function next() {
      if (i >= lines.length) { timer = setTimeout(finish, 280); return; }
      var p = document.createElement("p"); p.textContent = lines[i++]; out.appendChild(p);
      beep(520 + i * 60, 0.03);
      timer = setTimeout(next, 150);
    }
    el.addEventListener("click", finish);
    document.addEventListener("keydown", finish);
    next();
  }

  /* ---- typed text: [data-type] spells itself out once ---- */
  function typeOut(scope) {
    (scope || document).querySelectorAll("[data-type]:not([data-typed])").forEach(function (el) {
      el.dataset.typed = "1";
      if (reduced || prefs.fx === "off") return;
      // The untyped rest stays in the line, invisible, so the layout is final from the first frame;
      // a live region around it is busy meanwhile, so a screen reader hears the whole line once.
      var text = el.textContent, n = 0, step = Math.max(1, Math.ceil(text.length / 60));
      var shown = document.createElement("span"), rest = document.createElement("span");
      var live = el.closest("[aria-live]:not([aria-live='off'])");
      shown.className = "typed"; rest.className = "untyped"; rest.setAttribute("aria-hidden", "true");
      if (live) live.setAttribute("aria-busy", "true");
      el.textContent = ""; el.appendChild(shown); el.appendChild(rest);
      (function more() {
        n += step; shown.textContent = text.slice(0, n); rest.textContent = text.slice(n);
        if (n < text.length) { setTimeout(more, 16); return; }
        el.textContent = text;
        if (live) live.removeAttribute("aria-busy");
      })();
    });
  }

  /* ---- split-flap states: flip a cell whose value changed since this browser last saw it ---- */
  function flaps(scope) {
    var seen = load("sessionStorage", "mnm.seen", {});
    (scope || document).querySelectorAll(".flap[data-key]").forEach(function (el) {
      var key = el.dataset.key, value = el.textContent.trim();
      if (!el.dataset.split) {
        el.dataset.split = "1";
        el.innerHTML = "";
        Array.prototype.forEach.call(value, function (ch, i) {
          var c = document.createElement("span"); c.className = "c"; c.textContent = ch; c.style.setProperty("--i", i);
          c.setAttribute("aria-hidden", "true"); el.appendChild(c);
        });
      }
      if (seen[key] !== undefined && seen[key] !== value && !reduced) {
        el.classList.add("flipping");
        beep(300, 0.05, 0.02);
      }
      seen[key] = value;
    });
    save("sessionStorage", "mnm.seen", seen);
  }

  /* ---- VRAM history: one reading per status-bar poll, the last 60 (10 minutes) ---- */
  function remember() {
    var bar = document.getElementById("bar");
    if (!bar || !bar.dataset.total) return;
    var history = load("sessionStorage", "mnm.vram", []), now = Date.now();
    var last = load("sessionStorage", "mnm.vram.at", 0), share = Number(bar.dataset.used) / Number(bar.dataset.total);
    if (now - last < 8000 && history.length) history[history.length - 1] = share; // a page change, not a new poll
    else { history.push(share); save("sessionStorage", "mnm.vram.at", now); }
    history = history.slice(-60);
    save("sessionStorage", "mnm.vram", history);
    document.querySelectorAll("[data-spark]").forEach(function (el) {
      el.innerHTML = "";
      history.forEach(function (share) {
        var i = document.createElement("i"); i.style.height = Math.max(4, Math.round(share * 100)) + "%"; el.appendChild(i);
      });
      el.setAttribute("aria-label", "VRAM in use over the last " + history.length + " readings, 10 s apart; now " + Math.round(history[history.length - 1] * 100) + "%");
    });
  }

  /* ---- a box that scrolls (a code block, a wide table) takes the keyboard's focus ---- */
  function scrollables(scope) {
    (scope || document).querySelectorAll("pre, .table-scroll").forEach(function (el) {
      var scrolls = el.scrollWidth > el.clientWidth + 1 || el.scrollHeight > el.clientHeight + 1;
      if (scrolls && !el.hasAttribute("tabindex")) el.tabIndex = 0;
    });
  }

  function hashTarget() {
    var hash = location.hash.slice(1);
    return hash && /^[a-z-]+$/.test(hash) ? document.getElementById(hash) : null;
  }
  function plainClick(e) { return e.button === 0 && !(e.ctrlKey || e.metaKey || e.shiftKey || e.altKey); }

  /* ---- in-page tabs (an instance's sections): one shown at a time, the hash names it.
     Without this script every section shows, one after another. ---- */
  function tabs() {
    var list = document.querySelector("[data-tablist]");
    if (!list) return;
    var links = Array.prototype.slice.call(list.querySelectorAll("a[data-tab]"));
    // Panels are looked up at every switch, never kept: a swap may have replaced one.
    function panel(a) { return document.getElementById(a.dataset.tab); }
    list.setAttribute("role", "tablist");
    links.forEach(function (a) {
      a.setAttribute("role", "tab"); a.id = "tab-" + a.dataset.tab;
      a.setAttribute("aria-controls", a.dataset.tab);
    });
    function show(name, focus) {
      var found = false;
      links.forEach(function (a) {
        var on = a.dataset.tab === name, p = panel(a);
        found = found || on;
        a.setAttribute("aria-selected", String(on)); a.tabIndex = on ? 0 : -1;
        if (p) { p.setAttribute("role", "tabpanel"); p.setAttribute("aria-labelledby", a.id); p.hidden = !on; if (on) scrollables(p); }
        if (on && focus) a.focus();
      });
      return found;
    }
    function fromHash(e) {
      var target = hashTarget(), owner = target && target.closest("[data-tabpanel]");
      // A hash outside the panels (the skip link's #main) keeps the tab that is open.
      if (e && target && !owner) return;
      if (!show(owner ? owner.id : links[0].dataset.tab)) show(links[0].dataset.tab);
      if (target && owner && target !== owner) target.scrollIntoView({ block: "start" });
      else if (target === owner && owner && list.getBoundingClientRect().top > innerHeight * 0.4) list.scrollIntoView({ block: "start" });
    }
    // A tab chosen by hand opens at its start: the strip comes up when the panel starts low
    // (a phone under a tall header); with the strip stuck at the top, the panel scrolls back
    // to just under it rather than opening where the last one was scrolled to.
    function choose(name, focus) {
      show(name, focus); history.replaceState(null, "", "#" + name);
      var top = list.getBoundingClientRect().top, how = { block: "start", behavior: reduced ? "auto" : "smooth" };
      var p = document.getElementById(name);
      if (top > innerHeight * 0.4) list.scrollIntoView(how);
      else if (p && top <= 1 && p.getBoundingClientRect().top < list.getBoundingClientRect().bottom) p.scrollIntoView(how);
    }
    document.documentElement.classList.add("tabs-ready");
    links.forEach(function (a, i) {
      a.addEventListener("click", function (e) {
        if (!plainClick(e)) return; // a new tab or window gets the link as it is
        e.preventDefault(); choose(a.dataset.tab);
      });
      a.addEventListener("keydown", function (e) {
        var step = { ArrowRight: 1, ArrowLeft: -1, Home: -i, End: links.length - 1 - i }[e.key];
        if (step === undefined) return;
        e.preventDefault();
        var next = links[(i + step + links.length) % links.length];
        choose(next.dataset.tab, true);
      });
    });
    document.addEventListener("click", function (e) {
      var go = e.target.closest("[data-tab-link]");
      if (!go || !plainClick(e)) return;
      e.preventDefault(); choose(go.dataset.tabLink, true);
    });
    window.addEventListener("hashchange", fromHash);
    fromHash();
  }

  /* ---- a waveform (the studio's): a click plays its file from there. Without this
     script the player beside it does the same. ---- */
  document.addEventListener("click", function (e) {
    var wave = e.target.closest("svg.wave[data-audio]");
    if (!wave || !plainClick(e)) return;
    var audio = document.getElementById(wave.dataset.audio), box = wave.getBoundingClientRect();
    var seconds = parseFloat(wave.dataset.seconds);
    if (!audio || !box.width || !(seconds > 0)) return;
    var at = Math.max(0, Math.min(1, (e.clientX - box.left) / box.width)) * seconds;
    audio.currentTime = at;
    audio.play();
    if (!wave.dataset.ranges) return;  /* the edits' range: a start, then an end */
    var field = wave.dataset.next === "end" ? "end" : "start";
    wave.dataset.next = field === "start" ? "end" : "start";
    wave.dataset.ranges.split(" ").forEach(function (id) {
      var form = document.getElementById(id);
      if (!form || !form.elements[field]) return;
      form.elements[field].value = at.toFixed(1);
      if (field === "start" && form.elements.end) form.elements.end.value = "";
      if (form.elements.part) form.elements.part.value = "";  /* a range, not both */
    });
  });

  /* ---- the command line: the CLI's verbs, done the web's way ---- */
  var COMMANDS = [
    ["ls", "list the instances"], ["add <link>", "plan a model"], ["start <name> [--stop-others|--anyway]", "start one"],
    ["stop <name>… | stop --all", "stop"], ["run <name>", "Run tab"], ["try <name>", "Test it panel"], ["logs <name> [--review|--tail 0]", "engine log"],
    ["url <name> [--for <app>]", "the cards"], ["show|set <name>", "settings"], ["replan <name>", "plan again"],
    ["sets [ls|start|stop|show] <name>", "sets"], ["workflows", "ComfyUI workflows"], ["outputs | clean", "runs and clean up"], ["starter", "starter models"],
    ["doctor | engines | calibration | comfyui | system", "system pages"], ["help", "this list"]
  ];
  function post(action, fields) {
    var form = document.createElement("form");
    form.method = "post"; form.action = action;
    Object.keys(fields || {}).forEach(function (k) {
      var input = document.createElement("input"); input.type = "hidden"; input.name = k; input.value = fields[k]; form.appendChild(input);
    });
    document.body.appendChild(form); form.submit();
  }
  function enc(s) { return encodeURIComponent(s); }
  function run(line) {
    var words = line.trim().replace(/^mnm\s+/, "").split(/\s+/).filter(Boolean);
    var verb = words.shift(), arg = words[0], flags = words.filter(function (w) { return w.indexOf("--") === 0; });
    var here = location.pathname, named = words.filter(function (w) { return w.indexOf("--") !== 0; });
    function need() { if (!arg || arg.indexOf("--") === 0) { say("which instance? `" + verb + " <name>`"); return false; } return true; }
    switch (verb) {
      case undefined: return;
      case "help": case "?": say(COMMANDS.map(function (c) { return c[0]; }).join(" · ")); return;
      case "ls": case "status": location.href = "/"; return;
      case "add": if (!arg) { location.href = "/add"; return; } location.href = "/add?plan=1&reference=" + enc(arg); return;
      case "start":
        if (!need()) return;
        var others = flags.indexOf("--stop-others") >= 0, anyway = flags.indexOf("--anyway") >= 0;
        if (others && anyway) { say("Pass --stop-others or --anyway, not both."); return; }
        post("/instances/" + enc(arg) + "/start", { next: "/instances/" + enc(arg), anyway: anyway ? "on" : "", make_room: others ? "on" : "" });
        return;
      case "stop":
        if (flags.indexOf("--all") >= 0) { if (confirm("Stop every instance? They stay stopped until started.")) post("/instances/stop-all", { next: here }); return; }
        if (!need()) return;
        if (named.length > 1) { post("/instances/stop-all", { next: here, names: named.join(" ") }); return; }
        post("/instances/" + enc(arg) + "/stop", { next: here }); return;
      case "rm": if (!need()) return; location.href = "/instances/" + enc(arg) + "#remove"; say("Remove it from its page (it asks first)."); return;
      case "try": if (!need()) return; location.href = "/instances/" + enc(arg) + "#test"; return;
      case "run": if (!need()) return; location.href = "/instances/" + enc(arg) + "#run"; return;
      case "outputs": case "runs": location.href = "/runs"; return;
      case "comfy": case "workflows": location.href = "/comfy"; return;
      case "clean": location.href = "/runs#clean"; return;
      case "logs": if (!need()) return;
        location.href = "/instances/" + enc(arg) + "?log=" + (flags.indexOf("--review") >= 0 ? "issues" : words.indexOf("0") >= 0 ? "all" : "tail") + "#log"; return;
      case "url": if (!need()) return;
        var i = words.indexOf("--for"); location.href = "/instances/" + enc(arg) + (i >= 0 && words[i + 1] ? "?consumer=" + enc(words[i + 1]) : "") + "#cards"; return;
      case "show": case "set": if (!need()) return; location.href = "/instances/" + enc(arg) + "#settings"; return;
      case "replan": if (!need()) return; post("/instances/" + enc(arg) + "/replan", { any_quant: flags.indexOf("--any-quant") >= 0 ? "on" : "" }); return;
      case "workflow": case "sets":
        var sub = words[0], name = words[1];
        if (!sub || sub === "ls") { location.href = "/workflows"; return; }
        if (sub === "start" && name) { location.href = "/workflows/" + enc(name); say("Start all is on its page, with the fit-together verdict."); return; }
        if (sub === "stop" && name) { post("/workflows/" + enc(name) + "/stop"); return; }
        location.href = "/workflows/" + enc(sub === "show" || sub === "url" ? name || "" : sub); return;
      case "starter": location.href = "/starter"; return;
      case "doctor": location.href = "/doctor"; return;
      case "engines": location.href = "/system/engines"; return;
      case "calibration": location.href = "/system/calibration"; return;
      case "comfyui": location.href = "/system/comfyui"; return;
      case "system": case "router": case "service": location.href = "/system"; return;
      case "vram": location.href = "/#machine"; return;
      default: say("unknown command `" + verb + "`. Type help.");
    }
  }
  function say(text) { var out = document.getElementById("cmd-out"); if (out) { out.textContent = text; out.title = text; } }

  /* ---- keys: 1-6 switch tabs, / or Ctrl+K focus the command line ---- */
  document.addEventListener("keydown", function (e) {
    var cmd = document.getElementById("cmd");
    var typing = /^(INPUT|TEXTAREA|SELECT)$/.test((e.target.tagName || "")) || e.target.isContentEditable;
    var single = prefs.keys === "on"; // WCAG 2.1.4: plain-key shortcuts can be turned off
    if ((e.key === "k" && (e.ctrlKey || e.metaKey)) || (e.key === "/" && !typing && single)) {
      if (cmd) { e.preventDefault(); cmd.focus(); cmd.select(); }
      return;
    }
    if (typing || !single || e.ctrlKey || e.metaKey || e.altKey) return;
    var tab = document.querySelector('.tabs a[data-key="' + e.key + '"]');
    if (tab) { location.href = tab.href; }
  });

  function init() {
    applyPrefs(); tick(); setInterval(tick, 30000);
    boot(); tabs(); typeOut(); flaps(); remember(); scrollables();
    var resized; window.addEventListener("resize", function () { clearTimeout(resized); resized = setTimeout(scrollables, 200); });
    var form = document.getElementById("cmd-form");
    if (form) {
      form.addEventListener("submit", function (e) { e.preventDefault(); var input = document.getElementById("cmd"); run(input.value); input.value = ""; });
      document.getElementById("cmd").addEventListener("keydown", function (e) { if (e.key.length === 1) beep(1800 + Math.random() * 600, 0.012, 0.015); });
    }
    document.addEventListener("visibilitychange", function () { root.classList.toggle("paused", document.hidden); });
    document.body.addEventListener("htmx:afterSettle", function (e) {
      // An outerHTML swap replaces its target, so read the page again rather than the event.
      if (e.detail.target && e.detail.target.id === "bar") remember();
      typeOut(); flaps(); scrollables();
    });
    var target = hashTarget();
    if (target && target.id !== "main") target.classList.add("flicker");
    // A table in a closed <details> has no size until it opens.
    document.addEventListener("toggle", function (e) { if (e.target.open) scrollables(e.target); }, true);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
