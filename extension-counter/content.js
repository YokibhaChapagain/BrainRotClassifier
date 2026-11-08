(() => {
  let on = false, pill = null, timer = null, cur = null;
  const platform = location.hostname.includes("tiktok") ? "tiktok" : "instagram";

  const today = () => new Date().toLocaleDateString("en-CA");
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const nextPaint = () => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
  const send = (msg) => chrome.runtime.sendMessage(msg).catch(() => null); // extension reloaded = null

  function buildPill() {
    pill = document.createElement("div");
    pill.id = "brc-pill";
    pill.innerHTML = `
      <div class="brc-main"><span class="brc-rot">0</span> brain rot</div>
      <div class="brc-sub"><span class="brc-reels">0</span> reels today · <span class="brc-pct">0%</span></div>
      <div class="brc-now">waiting for a video</div>`;
    document.body.appendChild(pill);
    chrome.storage.local.get("pos").then(({ pos }) => {
      if (pos) Object.assign(pill.style, { left: pos.x + "px", top: pos.y + "px", right: "auto" });
    });

    // drag to move, position is remembered
    let dx = 0, dy = 0, dragging = false;
    pill.addEventListener("mousedown", (e) => {
      dragging = true;
      const r = pill.getBoundingClientRect();
      dx = e.clientX - r.left; dy = e.clientY - r.top;
      e.preventDefault();
    });
    addEventListener("mousemove", (e) => {
      if (!dragging) return;
      const x = Math.max(0, Math.min(innerWidth - pill.offsetWidth, e.clientX - dx));
      const y = Math.max(0, Math.min(innerHeight - pill.offsetHeight, e.clientY - dy));
      Object.assign(pill.style, { left: x + "px", top: y + "px", right: "auto" });
    });
    addEventListener("mouseup", () => {
      if (!dragging) return;
      dragging = false;
      const r = pill.getBoundingClientRect();
      chrome.storage.local.set({ pos: { x: Math.round(r.left), y: Math.round(r.top) } });
    });
  }

  async function refresh() {
    if (!pill) return;
    const { days = {} } = await chrome.storage.local.get("days");
    const vals = Object.values(days[today()] || {});
    const rot = vals.filter(Boolean).length;
    pill.querySelector(".brc-rot").textContent = rot;
    pill.querySelector(".brc-reels").textContent = vals.length;
    pill.querySelector(".brc-pct").textContent = vals.length ? Math.round((rot * 100) / vals.length) + "%" : "0%";
  }

  function setNow(text, flagged = false) {
    if (!pill) return;
    pill.querySelector(".brc-now").textContent = text;
    pill.classList.toggle("brc-flagged", flagged);
  }

  // biggest playing video on screen
  function activeVideo() {
    let best = null, bestArea = 0;
    for (const v of document.querySelectorAll("video")) {
      if (v.paused) continue;
      const r = v.getBoundingClientRect();
      const w = Math.max(0, Math.min(r.right, innerWidth) - Math.max(r.left, 0));
      const h = Math.max(0, Math.min(r.bottom, innerHeight) - Math.max(r.top, 0));
      if (w * h > bestArea) { best = v; bestArea = w * h; }
    }
    return best;
  }

  // instagram: /reels/<id>, /reel/<id>, /p/<id>   tiktok: /@user/video/<id>
  const ID_PAT = platform === "tiktok" ? /\/video\/(\d+)/ : /\/(?:reels?|p)\/([\w-]+)/;
  const LINK_SEL = platform === "tiktok" ? 'a[href*="/video/"]' : 'a[href*="/p/"], a[href*="/reel/"]';

  function reelId(video) {
    const prefix = platform === "tiktok" ? "tt:" : "ig:";
    // reels pages keep the current id in the url. feeds (home, For You) don't, so look for the post's link
    const m = location.pathname.match(ID_PAT);
    if (m) return prefix + m[1];
    let el = video;
    for (let i = 0; i < 15 && el; i++, el = el.parentElement) {
      if (el !== video && el.querySelectorAll("video").length > 1) break; // reached the next post
      const a = el.querySelector && el.querySelector(LINK_SEL);
      const lm = a && a.getAttribute("href").match(ID_PAT);
      if (lm) return prefix + lm[1];
      if (el.tagName === "ARTICLE") break;
    }
    const s = video.currentSrc || video.poster || "";
    return prefix + "v" + [...s].reduce((h, ch) => (h * 31 + ch.charCodeAt(0)) | 0, 0);
  }

  async function grab(c) {
    const r = c.video.getBoundingClientRect();
    const p = pill.getBoundingClientRect();
    // keep the pill out of the frames the model sees
    const overlap = !(p.right < r.left || p.left > r.right || p.bottom < r.top || p.top > r.bottom);
    if (overlap) { pill.style.visibility = "hidden"; await nextPaint(); }
    try {
      return await send({ type: "CAPTURE", rect: { x: r.x, y: r.y, width: r.width, height: r.height }, dpr: devicePixelRatio });
    } finally {
      if (pill) pill.style.visibility = "";
    }
  }

  const blobToB64 = (blob) => new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onload = () => resolve(String(fr.result).split(",")[1]);
    fr.onerror = () => reject(fr.error);
    fr.readAsDataURL(blob);
  });

  // Record what the <video> element is already playing (picture + sound). This makes no extra
  // requests to Instagram/TikTok: it reuses the bytes the browser downloaded to show you the video.
  // Returns a Blob, null (not watched long enough), or "unsupported" (fall back to screenshots).
  async function recordClip(c) {
    let stream, rec, failed = null;
    try {
      stream = c.video.captureStream();
      const mime = ["video/webm;codecs=vp9,opus", "video/webm;codecs=vp8,opus", "video/webm"]
        .find((m) => MediaRecorder.isTypeSupported(m));
      rec = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: CONFIG.CLIP_BITRATE });
      rec.onerror = (e) => { failed = e.error || e; };
      rec.start(1000);
    } catch (e) {
      if (stream) stream.getTracks().forEach((t) => t.stop());
      console.warn("[brainrot] can't record this video, using screenshots:", e);
      return "unsupported";
    }
    const chunks = [];
    rec.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
    const stopped = new Promise((r) => { rec.onstop = r; });
    setNow("recording this reel…");

    // count only seconds actually played on screen; pause the recorder while the video is paused
    let watched = 0, last = performance.now();
    while (on && c === cur && !failed && watched < CONFIG.CLIP_SECONDS) {
      await sleep(250);
      const now = performance.now();
      const playing = !c.video.paused && !document.hidden;
      if (playing) watched += (now - last) / 1000;
      if (!playing && rec.state === "recording") rec.pause();
      if (playing && rec.state === "paused") rec.resume();
      last = now;
    }
    if (rec.state !== "inactive") rec.stop();
    await stopped;
    stream.getTracks().forEach((t) => t.stop()); // stops our copy only, the video keeps playing
    if (failed) {
      console.warn("[brainrot] recording failed, using screenshots:", failed);
      return c === cur ? "unsupported" : null;
    }
    if (!on || watched < CONFIG.CLIP_MIN_SECONDS || !chunks.length) return null;
    return new Blob(chunks, { type: rec.mimeType || "video/webm" });
  }

  async function scoreClip(c, clip) {
    let clip_b64;
    try { clip_b64 = await blobToB64(clip); } catch (e) { console.warn("[brainrot] clip read failed", e); return; }
    if (c === cur) setNow("asking the model…");
    handleAnswer(c, await send({ type: "PREDICT_CLIP", id: c.id, platform, clip_b64, mime: clip.type }));
  }

  async function score(c) {
    handleAnswer(c, await send({ type: "PREDICT", id: c.id, platform, frames: c.frames.slice() }));
  }

  function handleAnswer(c, res) {
    if (!on) return;
    const showing = c === cur; // the model can answer after the user has already swiped on
    if (!res || !res.ok) {
      const err = (res && res.error) || "";
      console.warn("[brainrot] predict failed:", err);
      const offline = !res || /failed to fetch|timeout|aborted/i.test(err);
      if (showing) setNow(offline ? "model offline, not counted" : "model error, not counted");
      return;
    }
    const { prob, flagged } = res.data;
    // same id twice just overwrites, so a reel is only counted once
    send({ type: "RECORD", id: c.id, flagged });
    if (!showing) return;
    const pct = Math.round(prob * 100);
    setNow(flagged ? `this reel: ${pct}% · brain rot` : `this reel: ${pct}%`, flagged);
  }

  async function run(c) {
    if (CONFIG.MODE === "api" && CONFIG.SEND === "clip") {
      const clip = await recordClip(c);
      if (clip !== "unsupported") {
        if (clip) scoreClip(c, clip);
        return;
      }
      // this video can't be recorded (e.g. cross-origin source): screenshots below
    }
    while (on && c === cur && c.frames.length < CONFIG.MAX_FRAMES) {
      if (!document.hidden) {
        const shot = await grab(c);
        if (c !== cur) return;
        if (shot && shot.ok) {
          c.frames.push(shot.frame);
          const n = c.frames.length;
          // the model costs a gemini call, so it's asked once per video
          if (CONFIG.MODE === "demo" && n === CONFIG.EARLY_FRAMES) score(c);
          if (n === CONFIG.MAX_FRAMES) {
            if (CONFIG.MODE !== "demo") setNow("asking the model…");
            score(c);
          }
        }
      }
      await sleep(CONFIG.FRAME_INTERVAL_MS);
    }
  }

  function tick() {
    const v = activeVideo();
    if (!v || v.currentTime < 0.3) return;
    const id = reelId(v);
    if (cur && cur.id === id) return;
    cur = { id, video: v, frames: [] };
    setNow("scoring this reel…");
    run(cur);
  }

  function enable() {
    if (on) return;
    on = true;
    if (!pill) buildPill();
    refresh();
    timer = setInterval(tick, 250);
  }

  function disable() {
    on = false;
    cur = null;
    clearInterval(timer);
    if (pill) { pill.remove(); pill = null; }
  }

  chrome.storage.onChanged.addListener((changes) => {
    if (changes.enabled) changes.enabled.newValue === false ? disable() : enable();
    if (changes.days) refresh();
  });

  chrome.storage.local.get("enabled").then(({ enabled = true }) => { if (enabled) enable(); });
})();
