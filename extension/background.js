// Service worker: captures one frame of the visible tab per request (avoids cross-origin canvas taint on
// <video>), crops it to the video's on-screen rect, and proxies API calls (avoids page CSP / mixed content).

async function toB64(blob) {
  const buf = new Uint8Array(await blob.arrayBuffer());
  let bin = "";
  for (let i = 0; i < buf.length; i += 0x8000) bin += String.fromCharCode(...buf.subarray(i, i + 0x8000));
  return btoa(bin);
}

async function cropToRect(dataUrl, rect, dpr) {
  const bmp = await createImageBitmap(await (await fetch(dataUrl)).blob());
  const sx = Math.max(0, rect.x * dpr), sy = Math.max(0, rect.y * dpr);
  const sw = Math.min(bmp.width - sx, rect.width * dpr), sh = Math.min(bmp.height - sy, rect.height * dpr);
  const scale = Math.min(1, 384 / sw); // downscale to keep payloads small
  const canvas = new OffscreenCanvas(Math.round(sw * scale), Math.round(sh * scale));
  canvas.getContext("2d").drawImage(bmp, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);
  return toB64(await canvas.convertToBlob({ type: "image/jpeg", quality: 0.7 }));
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  (async () => {
    try {
      if (msg.type === "CAPTURE_FRAME") {
        const url = await chrome.tabs.captureVisibleTab(sender.tab.windowId, { format: "jpeg", quality: 70 });
        sendResponse({ ok: true, frame: await cropToRect(url, msg.rect, msg.dpr) });
      } else if (msg.type === "API") {
        const res = await fetch(msg.url, {
          method: msg.body ? "POST" : "GET",
          headers: { "Content-Type": "application/json" },
          body: msg.body ? JSON.stringify(msg.body) : undefined,
        });
        sendResponse({ ok: res.ok, data: await res.json() });
      }
    } catch (e) {
      sendResponse({ ok: false, error: String(e) });
    }
  })();
  return true; // async response
});
