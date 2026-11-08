// BrainRot Meter content script.
// The model is PIXELS ONLY: we never read captions or hashtags. For each reel: capture a frame every ~0.5 s,
// measure cuts/sec in-page, get an early prediction at 4 frames and a final one at 8 (meter updates live),
// then when the reel changes, log the view (watch time, skip, loops) for the dashboard.
(() => {
  const CFG = BRAINROT_CONFIG;
  const platform = location.hostname.includes("tiktok") ? "tiktok" : "instagram";

  // ---------------------------------------------------------------- overlay
  const meter = document.createElement("div");
  meter.id = "brainrot-meter";
  meter.innerHTML = `
    <div class="br-title">🧠 BRAIN ROT</div>
    <div class="br-bar"><div class="br-fill"></div></div>
    <div class="br-pct">--</div>
    <div class="br-state"></div>
    <div class="br-flag" hidden>🚩 ROT DETECTED</div>
    <div class="br-reasons"></div>
    <div class="br-why"></div>`;
  document.body.appendChild(meter);
  const $ = (s) => meter.querySelector(s);

  function render(pred) {
    const pct = Math.round(pred.prob * 100);
    $(".br-fill").style.width = pct + "%";
    $(".br-fill").style.background = `hsl(${120 - pct * 1.2}, 85%, 50%)`;
    $(".br-pct").textContent = `${pct}%`;
    $(".br-state").textContent = pred.final ? "" : "analyzing… (live)";
    $(".br-flag").hidden = !pred.flagged;
    $(".br-reasons").textContent = pred.reasons.join(" · ");
    meter.classList.toggle("br-flagged", pred.flagged);
  }

  function reset() {
    $(".br-fill").style.width = "0%";
    $(".br-pct").textContent = "--";
    $(".br-state").textContent = "watching…";
    $(".br-flag").hidden = true;
    $(".br-reasons").textContent = "";
    $(".br-why").textContent = "";
    meter.classList.remove("br-flagged");
  }

  // ---------------------------------------------------------------- helpers
  const api = (path, body) => chrome.runtime.sendMessage({ type: "API", url: CFG.API_URL + path, body });

  async function ids() {
    const now = Date.now();
    let { uid, sid, lastSeen } = await chrome.storage.local.get(["uid", "sid", "lastSeen"]);
    if (!uid) uid = crypto.randomUUID();
    if (!sid || !lastSeen || now - lastSeen > CFG.SESSION_GAP_MS) sid = crypto.randomUUID();
    await chrome.storage.local.set({ uid, sid, lastSeen: now });
    return { uid, sid };
  }

  function activeVideo() {
    let best = null, bestArea = 0;
    for (const v of document.querySelectorAll("video")) {
      const r = v.getBoundingClientRect();
      const w = Math.max(0, Math.min(r.right, innerWidth) - Math.max(r.left, 0));
      const h = Math.max(0, Math.min(r.bottom, innerHeight) - Math.max(r.top, 0));
      if (w * h > bestArea && !v.paused) { best = v; bestArea = w * h; }
    }
    return best;
  }

  function containerOf(video) {
    let el = video;
    for (let i = 0; i < 12 && el; i++, el = el.parentElement) {
      if (el.innerText && el.innerText.length > 30) return el;
    }
    return video.parentElement;
  }

  // Only an ID (for caching + the dashboard). No text is read: the model judges pixels alone.
  function reelId(video) {
    const pat = /\/video\/(\d+)|\/reels?\/([\w-]+)/;
    const m = location.pathname.match(pat);
    if (m) return m[1] || m[2];
    const link = containerOf(video).querySelector('a[href*="/video/"], a[href*="/reel"]');
    const lm = link && link.getAttribute("href").match(pat);
    if (lm) return lm[1] || lm[2];
    return "v" + [...(video.currentSrc || String(Date.now()))].reduce((h, c) => (h * 31 + c.charCodeAt(0)) | 0, 0);
  }

  // In-page cut detector: tiny 32x56 thumbnails at ~10 fps, colour-histogram distance between frames.
  // If the site blocks reading video pixels (tainted canvas) we give up and the server uses frames only.
  function makeCutDetector(video) {
    const c = document.createElement("canvas");
    c.width = 32; c.height = 56;
    const ctx = c.getContext("2d", { willReadFrequently: true });
    let prev = null, cuts = 0, samples = 0, dead = false;
    const timer = setInterval(() => {
      if (dead || video.paused) return;
      try {
        ctx.drawImage(video, 0, 0, c.width, c.height);
        const px = ctx.getImageData(0, 0, c.width, c.height).data;
        const hist = new Float32Array(64);
        for (let i = 0; i < px.length; i += 4) hist[((px[i] >> 6) << 4) | ((px[i + 1] >> 6) << 2) | (px[i + 2] >> 6)]++;
        const n = px.length / 4;
        if (prev) {
          let d = 0;
          for (let k = 0; k < 64; k++) d += Math.abs(hist[k] - prev[k]) / n;
          if (d > 0.6) cuts++;
        }
        prev = hist; samples++;
      } catch (e) { dead = true; }
    }, 100);
    return {
      stop: () => clearInterval(timer),
      rate: () => (dead || samples < 30 ? null : cuts / (samples / 10)), // need >= 3 s of samples
    };
  }

  // ---------------------------------------------------------------- per-reel state
  let cur = null; // { id, video, frames, startedAt, watchMs, lastTick, loops, lastTime, pred, cut }
  let starting = false; // guards against two ticks starting the same reel

  async function finishView() {
    if (!cur) return;
    const c = cur; cur = null;
    c.cut.stop();
    if (c.watchMs < 500) return; // ignore flicks past
    const dur = Number.isFinite(c.video.duration) ? c.video.duration * 1000 : null;
    const { uid, sid } = await ids();
    api("/events/view", {
      user_id: uid, session_id: sid, video_id: c.id, platform,
      started_at: new Date(c.startedAt).toISOString(), watch_ms: Math.round(c.watchMs),
      video_duration_ms: dur ? Math.round(dur) : null, loops: c.loops,
      skipped: c.watchMs < 2000 || (dur ? c.watchMs < 0.25 * dur : false),
      prob: c.pred ? c.pred.prob : null, flagged: c.pred ? c.pred.flagged : null,
    });
  }

  async function startReel(video) {
    if (starting) return;
    starting = true;
    try { await finishView(); } finally { starting = false; }
    cur = { id: reelId(video), video, frames: [], startedAt: Date.now(), watchMs: 0, lastTick: Date.now(),
            loops: 0, lastTime: video.currentTime, pred: null, cut: makeCutDetector(video) };
    reset();
    captureLoop(cur);
  }

  async function predict(c) {
    const res = await api("/predict", {
      video_id: c.id, platform, frames_b64: c.frames, cuts_per_sec: c.cut.rate(),
    });
    if (c !== cur) return;
    if (!res || !res.ok) { $(".br-state").textContent = "API offline"; return; }
    const wasFlagged = c.pred && c.pred.flagged;
    c.pred = res.data;
    render(res.data);
    if (res.data.flagged && !wasFlagged && CFG.EXPLAIN) {
      api("/explain", { prob: res.data.prob, reasons: res.data.reasons, frames_b64: c.frames.slice(-2) }).then((e) => {
        if (c === cur && e && e.ok) $(".br-why").textContent = e.data.explanation;
      });
    }
  }

  async function captureLoop(c) {
    while (c === cur && c.frames.length < CFG.MAX_FRAMES) {
      const r = c.video.getBoundingClientRect();
      const shot = await chrome.runtime.sendMessage({
        type: "CAPTURE_FRAME", rect: { x: r.x, y: r.y, width: r.width, height: r.height }, dpr: devicePixelRatio,
      });
      if (c !== cur) return;
      if (shot && shot.ok) c.frames.push(shot.frame);
      if (c.frames.length === CFG.EARLY_FRAMES || c.frames.length === CFG.MAX_FRAMES) predict(c); // live update
      await new Promise((res) => setTimeout(res, CFG.FRAME_INTERVAL_MS));
    }
  }

  // ---------------------------------------------------------------- main tick
  setInterval(() => {
    const v = activeVideo();
    if (cur) {
      const now = Date.now();
      if (!document.hidden && !cur.video.paused) cur.watchMs += now - cur.lastTick;
      cur.lastTick = now;
      if (cur.video.currentTime + 0.5 < cur.lastTime) cur.loops++; // video restarted = loop
      cur.lastTime = cur.video.currentTime;
    }
    if (!v || v.currentTime < 0.3) return;
    const id = reelId(v);
    if (!cur || id !== cur.id) startReel(v);
  }, 250);

  addEventListener("pagehide", finishView);
  document.addEventListener("visibilitychange", () => { if (document.hidden) finishView(); });
})();
