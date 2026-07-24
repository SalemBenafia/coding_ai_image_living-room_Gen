/* AI Living Room Generator — frontend logic (vanilla JS).
   nginx reverse-proxies /api and /health to the backend, so all calls are
   same-origin relative URLs. */
"use strict";

const API = ""; // same origin (nginx proxy)

const $ = (id) => document.getElementById(id);
let lastResult = null;

/* ---------------- init ---------------- */
document.addEventListener("DOMContentLoaded", () => {
  bindEvents();
  loadStyles();
  checkHealth();
  loadHistory();
});

function bindEvents() {
  $("generate").addEventListener("click", onGenerate);
  $("refresh-history").addEventListener("click", loadHistory);
  $("reuse").addEventListener("click", reuseSettings);
  $("steps").addEventListener("input", (e) => ($("steps-val").textContent = e.target.value));
  $("cfg").addEventListener("input", (e) => ($("cfg-val").textContent = e.target.value));
}

/* ---------------- API helpers ---------------- */
async function apiGet(path) {
  const r = await fetch(API + path);
  if (!r.ok) throw new Error(`GET ${path} → ${r.status}`);
  return r.json();
}

/* ---------------- styles ---------------- */
async function loadStyles() {
  try {
    const styles = await apiGet("/api/styles");
    const sel = $("style");
    for (const s of styles) {
      const opt = document.createElement("option");
      opt.value = s.key;
      opt.textContent = `${s.label} — ${s.description}`;
      sel.appendChild(opt);
    }
  } catch (e) {
    console.warn("Could not load styles:", e);
  }
}

/* ---------------- health badge ---------------- */
async function checkHealth() {
  const dot = $("status-dot");
  const text = $("status-text");
  try {
    const h = await apiGet("/health");
    const ok = h.status === "ok";
    dot.className = "dot " + (ok ? "ok" : "bad");
    text.textContent = ok
      ? `ready · ${h.enhancer}`
      : `degraded (minio:${h.minio ? "up" : "down"}, ai:${h.ai_service ? "up" : "down"})`;
  } catch {
    dot.className = "dot bad";
    text.textContent = "backend offline";
  }
}

/* ---------------- generate ---------------- */
async function onGenerate() {
  const errEl = $("form-error");
  errEl.textContent = "";

  const prompt = $("prompt").value.trim();
  if (prompt.length < 3) {
    errEl.textContent = "Please enter a longer prompt (min 3 characters).";
    return;
  }

  const [w, h] = $("aspect").value.split("x").map(Number);
  const seedRaw = $("seed").value.trim();
  const body = {
    prompt,
    negative_prompt: $("negative").value.trim() || null,
    style: $("style").value || null,
    width: w,
    height: h,
    steps: Number($("steps").value),
    guidance_scale: Number($("cfg").value),
    seed: seedRaw === "" ? null : Number(seedRaw),
    enhance_prompt: $("enhance").checked,
  };

  setLoading(true);
  try {
    const r = await fetch(API + "/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await r.json();
    if (!r.ok) {
      throw new Error(formatApiError(data));
    }
    lastResult = data;
    renderResult(data);
    loadHistory();
  } catch (e) {
    errEl.textContent = e.message || "Generation failed.";
    showState("empty");
  } finally {
    setLoading(false);
  }
}

function formatApiError(data) {
  if (!data) return "Unknown error.";
  if (typeof data.detail === "string") return data.detail;
  if (Array.isArray(data.detail)) {
    // FastAPI/Pydantic validation error format
    return data.detail.map((d) => d.msg || JSON.stringify(d)).join("; ");
  }
  return "Generation failed.";
}

function setLoading(on) {
  $("generate").disabled = on;
  $("generate").querySelector(".btn-label").textContent = on ? "Generating…" : "Generate";
  showState(on ? "loading" : lastResult ? "figure" : "empty");
}

function showState(which) {
  $("result-empty").classList.toggle("hidden", which !== "empty");
  $("result-loading").classList.toggle("hidden", which !== "loading");
  $("result-figure").classList.toggle("hidden", which !== "figure");
}

function renderResult(item) {
  const img = $("result-img");
  img.src = API + item.image_url + `?t=${Date.now()}`; // cache-bust the freshly created image
  $("download").href = API + item.image_url;
  $("download").download = `living_room_${item.id.slice(0, 8)}.png`;
  $("result-meta").innerHTML = metaHtml(item);
  showState("figure");
}

function metaHtml(item) {
  return `
    <div><b>Prompt:</b> ${escapeHtml(item.enhanced_prompt)}</div>
    <div><b>Style:</b> ${item.style || "none"} &nbsp; <b>Seed:</b> ${item.seed}</div>
    <div><b>Size:</b> ${item.width}×${item.height} &nbsp; <b>Steps:</b> ${item.steps} &nbsp; <b>CFG:</b> ${item.guidance_scale}</div>`;
}

function reuseSettings() {
  if (!lastResult) return;
  $("prompt").value = lastResult.prompt;
  $("style").value = lastResult.style || "";
  $("steps").value = lastResult.steps;
  $("steps-val").textContent = lastResult.steps;
  $("cfg").value = lastResult.guidance_scale;
  $("cfg-val").textContent = lastResult.guidance_scale;
  $("seed").value = lastResult.seed;
  $("aspect").value = `${lastResult.width}x${lastResult.height}`;
  window.scrollTo({ top: 0, behavior: "smooth" });
}

/* ---------------- history ---------------- */
async function loadHistory() {
  const gallery = $("gallery");
  try {
    const page = await apiGet("/api/history?limit=24");
    gallery.innerHTML = "";
    $("history-empty").classList.toggle("hidden", page.total > 0);
    for (const item of page.items) {
      gallery.appendChild(makeCard(item));
    }
  } catch (e) {
    console.warn("History load failed:", e);
  }
}

function makeCard(item) {
  const card = document.createElement("div");
  card.className = "card";
  card.title = item.prompt;

  const img = document.createElement("img");
  img.src = API + item.image_url;
  img.alt = item.prompt;
  img.loading = "lazy";
  img.addEventListener("click", () => {
    lastResult = item;
    renderResult(item);
    window.scrollTo({ top: 0, behavior: "smooth" });
  });

  const del = document.createElement("button");
  del.className = "del";
  del.textContent = "✕";
  del.title = "Delete";
  del.addEventListener("click", (ev) => {
    ev.stopPropagation();
    deleteItem(item.id);
  });

  card.appendChild(img);
  card.appendChild(del);
  return card;
}

async function deleteItem(id) {
  if (!confirm("Delete this generation?")) return;
  try {
    const r = await fetch(API + `/api/history/${id}`, { method: "DELETE" });
    if (!r.ok && r.status !== 204) throw new Error(`status ${r.status}`);
    if (lastResult && lastResult.id === id) {
      lastResult = null;
      showState("empty");
    }
    loadHistory();
  } catch (e) {
    alert("Could not delete: " + e.message);
  }
}

/* ---------------- utils ---------------- */
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
}
