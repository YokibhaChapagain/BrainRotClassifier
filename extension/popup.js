(async () => {
  const { uid } = await chrome.storage.local.get("uid");
  const note = document.getElementById("note");
  document.getElementById("open").onclick = () =>
    chrome.tabs.create({ url: `${BRAINROT_CONFIG.API_URL}/dashboard?user=${encodeURIComponent(uid || "anon")}` });
  if (!uid) { note.textContent = "Watch some reels first."; return; }
  try {
    const s = await (await fetch(`${BRAINROT_CONFIG.API_URL}/stats/summary?user_id=${uid}&days=1`)).json();
    document.getElementById("share").textContent = `${Math.round(s.rot_share * 100)}%`;
    document.getElementById("min").textContent = Math.round(s.rot_watch_min);
    document.getElementById("views").textContent = s.views;
    if (!s.views) note.textContent = "No data yet (is DATABASE_URL set on the API?)";
  } catch (e) {
    note.textContent = "API offline";
  }
})();
