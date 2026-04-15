import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse

from app.api.v1.router import api_router
from app.core.config import MODEL_DOWNLOAD_ROOT


_UI_TEST_PAGE_HTML = r"""<!DOCTYPE html>
<html lang="th">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LiteRT — Chat</title>
  <style>
    :root {
      --bg: #0d0d0f;
      --sidebar: #1a1a1f;
      --panel: #222228;
      --border: #2e2e36;
      --text: #ececf1;
      --muted: #8e8ea0;
      --accent: #10a37f;
      --accent-dim: #1a7f64;
      --warn: #f59e0b;
      --err: #ef4444;
    }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: "Sarabun", "Segoe UI", system-ui, sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; display: flex; flex-direction: column; }
    .app { display: flex; flex: 1; min-height: 0; }
    aside.sidebar {
      width: 280px; min-width: 260px; background: var(--sidebar);
      border-right: 1px solid var(--border); display: flex; flex-direction: column; padding: 0.85rem;
    }
    .brand { font-weight: 700; font-size: 1rem; padding-bottom: 0.75rem; border-bottom: 1px solid var(--border); margin-bottom: 0.85rem; letter-spacing: 0.02em; }
    .brand span { color: var(--accent); }
    label.lbl { display: block; font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.35rem; }
    select#modelSelect {
      width: 100%; padding: 0.55rem 0.6rem; border-radius: 8px; border: 1px solid var(--border);
      background: var(--panel); color: var(--text); font-size: 0.88rem; cursor: pointer;
    }
    select#modelSelect:focus { outline: none; border-color: var(--accent); }
    .model-row { margin-top: 0.65rem; display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; }
    .pill {
      display: inline-flex; align-items: center; gap: 0.35rem; font-size: 0.75rem; padding: 0.25rem 0.55rem;
      border-radius: 999px; background: var(--panel); border: 1px solid var(--border);
    }
    .pill .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--muted); }
    .pill.loaded .dot { background: var(--accent); box-shadow: 0 0 8px var(--accent); }
    .pill.warn .dot { background: var(--warn); }
    .pill.err .dot { background: var(--err); }
    .btn-row { display: flex; gap: 0.45rem; margin-top: 0.65rem; flex-wrap: wrap; }
    button {
      cursor: pointer; border: none; border-radius: 8px; padding: 0.45rem 0.75rem; font-size: 0.82rem; font-weight: 600;
      background: var(--accent); color: #fff;
    }
    button:hover:not(:disabled) { filter: brightness(1.08); }
    button.secondary { background: var(--panel); color: var(--text); border: 1px solid var(--border); }
    button.danger { background: #3d2020; color: #fca5a5; border: 1px solid #5c2a2a; }
    button:disabled { opacity: 0.45; cursor: not-allowed; }
    details.adv { margin-top: 1rem; border-top: 1px solid var(--border); padding-top: 0.75rem; font-size: 0.85rem; }
    details.adv summary { cursor: pointer; color: var(--muted); user-select: none; }
    details.adv .inner { margin-top: 0.65rem; }
    input[type="text"], input[type="url"], textarea {
      width: 100%; padding: 0.45rem 0.55rem; border-radius: 6px; border: 1px solid var(--border);
      background: var(--bg); color: var(--text); font-size: 0.85rem;
    }
    textarea { min-height: 72px; resize: vertical; font-family: inherit; }
    main.main {
      flex: 1; display: flex; flex-direction: column; min-width: 0; min-height: 0;
      padding: 0; margin: 0; width: 100%; max-width: none;
    }
    .chat-toolbar {
      flex-shrink: 0; display: flex; flex-wrap: wrap; gap: 0.65rem; align-items: flex-end;
      padding: 0.65rem 1rem; border-bottom: 1px solid var(--border); background: var(--bg);
    }
    .chat-toolbar .tf { flex: 1; min-width: 140px; }
    .chat-toolbar .tf .lbl { margin-bottom: 0.25rem; }
    .chat-toolbar input { width: 100%; font-size: 0.82rem; padding: 0.4rem 0.5rem; }
    .chat-toolbar .tb-actions { display: flex; align-items: center; gap: 0.5rem; padding-bottom: 0.15rem; }
    .chat-toolbar a { color: var(--accent); font-size: 0.85rem; }
    .chat-messages {
      flex: 1; min-height: 0; overflow-y: auto; padding: 1rem 1.15rem;
      display: flex; flex-direction: column; gap: 0.85rem;
    }
    .msg { display: flex; width: 100%; }
    .msg.user { justify-content: flex-end; }
    .msg.assistant { justify-content: flex-start; }
    .bubble {
      max-width: min(90%, 720px); padding: 0.65rem 0.95rem; border-radius: 14px;
      line-height: 1.55; font-size: 0.9rem; white-space: pre-wrap; word-break: break-word;
    }
    .msg.user .bubble {
      background: linear-gradient(145deg, var(--accent-dim), #145a47); color: #f4fffb;
      border: 1px solid var(--accent);
    }
    .msg.assistant .bubble {
      background: var(--panel); border: 1px solid var(--border); color: var(--text);
    }
    .msg.assistant .bubble.streaming { border-color: var(--accent); box-shadow: 0 0 0 1px rgba(16, 163, 127, 0.25); }
    .bubble.err-bubble { border-color: var(--err) !important; color: #fecaca !important; background: #2a1515 !important; }
    .chat-welcome {
      margin: auto; text-align: center; color: var(--muted); font-size: 0.88rem; max-width: 360px; line-height: 1.5; padding: 2rem 1rem;
    }
    .chat-composer {
      flex-shrink: 0; border-top: 1px solid var(--border); padding: 0.75rem 1rem 1rem;
      background: var(--sidebar);
    }
    .composer-row { display: flex; gap: 0.65rem; align-items: flex-end; max-width: 960px; margin: 0 auto; width: 100%; }
    #chatInput {
      flex: 1; min-height: 48px; max-height: 200px; resize: none; padding: 0.65rem 0.85rem;
      border-radius: 12px; font-family: inherit; font-size: 0.9rem; line-height: 1.45;
    }
    .composer-actions { display: flex; flex-direction: column; gap: 0.4rem; align-items: stretch; }
    #btnSend { min-width: 92px; }
    .composer-meta { font-size: 0.75rem; color: var(--muted); margin-top: 0.5rem; max-width: 960px; margin-left: auto; margin-right: auto; min-height: 1.2em; }
    .row { display: flex; gap: 0.65rem; flex-wrap: wrap; align-items: flex-end; margin-bottom: 0.5rem; }
    .row > div { flex: 1; min-width: 140px; }
    .check { display: flex; align-items: center; gap: 0.4rem; padding-bottom: 0.2rem; }
    .check label, .check span { margin: 0; font-size: 0.8rem; cursor: pointer; user-select: none; }
    .err { color: var(--err); }
    #modelLoadNote { font-size: 0.72rem; color: var(--muted); margin-top: 0.35rem; line-height: 1.35; }
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <div class="brand">Lite<span>RT</span> · LM</div>
      <label class="lbl" for="modelSelect">โมเดลจาก <code>GET /v1/models</code></label>
      <select id="modelSelect">
        <option value="">— กำลังโหลดรายการโมเดล… —</option>
      </select>
      <div id="modelLoadNote">รายการจาก <code>/v1/models</code> (ไฟล์ใน <code>models/</code>) — เลือกแล้วโหลดเข้าแรมอัตโนมัติ</div>
      <div class="model-row">
        <span class="pill" id="loadPill"><span class="dot"></span><span id="loadPillText">ไม่ทราบสถานะ</span></span>
      </div>
      <div class="btn-row">
        <button type="button" class="danger secondary" id="btnUnloadModel" title="Unload จากแรม">Unload</button>
        <button type="button" class="secondary" id="btnRefreshCatalog">รีเฟรช /v1/models</button>
      </div>
      <label class="lbl" for="systemPrompt" style="margin-top:0.85rem">System prompt</label>
      <textarea id="systemPrompt" rows="3" placeholder="You are a helpful assistant." style="min-height:68px;font-size:0.82rem"></textarea>
      <details class="adv">
        <summary>ดาวน์โหลด / Admin token</summary>
        <div class="inner">
          <label class="lbl" for="adminToken">Bearer (ถ้าเซิร์ฟเวอร์ตั้ง LITERT_ADMIN_TOKEN)</label>
          <input type="text" id="adminToken" placeholder="optional" autocomplete="off" />
          <label class="lbl" for="dlUrl" style="margin-top:0.5rem">Download URL</label>
          <input type="url" id="dlUrl" placeholder="https://..." />
          <label class="lbl" for="dlPath">path ภายใต้ download root</label>
          <input type="text" id="dlPath" placeholder="models/foo.litertlm" />
          <div class="btn-row">
            <button type="button" class="secondary" id="btnDownload">Download</button>
          </div>
          <pre id="adminOut" style="margin-top:0.5rem;font-size:0.75rem;white-space:pre-wrap;max-height:120px;overflow:auto;color:var(--muted)"></pre>
        </div>
      </details>
    </aside>
    <main class="main">
      <div class="chat-toolbar">
        <div class="tf">
          <label class="lbl" for="baseUrl">Base URL</label>
          <input type="text" id="baseUrl" placeholder="ว่าง = โดเมนนี้" />
        </div>
        <div class="tf">
          <label class="lbl" for="modelId">model id</label>
          <input type="text" id="modelId" placeholder="sync จากโมเดลที่โหลด" />
        </div>
        <div class="tb-actions">
          <button type="button" class="secondary" id="btnClearChat" title="ล้างประวัติในหน้าจอ">ล้างแชท</button>
          <a href="/docs">API docs</a>
        </div>
      </div>
      <div class="chat-messages" id="chatMessages">
        <div class="chat-welcome" id="chatWelcome">ส่งข้อความด้านล่าง · ใช้ <code>/v1/chat/completions</code> · โมเดลจาก <code>GET /v1/models</code></div>
      </div>
      <div class="chat-composer">
        <div class="composer-row">
          <textarea id="chatInput" rows="2" placeholder="พิมพ์ข้อความ… (Enter ส่ง · Shift+Enter ขึ้นบรรทัด)"></textarea>
          <div class="composer-actions">
            <button type="button" id="btnSend">ส่ง</button>
            <label class="check">
              <input type="checkbox" id="useStream" checked />
              <span>สตรีม</span>
            </label>
          </div>
        </div>
        <div class="composer-meta" id="chatMeta"></div>
      </div>
    </main>
  </div>
  <script>
(function () {
  const $ = (id) => document.getElementById(id);
  let _catalogActiveRel = "";
  let _loadingModel = false;
  /** @type {{role:string,content:string}[]} */
  let chatHistory = [];

  function apiBase() {
    const u = $("baseUrl").value.trim();
    return u ? u.replace(/\/$/, "") : "";
  }

  function adminHeaders(json) {
    const h = json ? { "Content-Type": "application/json" } : {};
    const t = $("adminToken").value.trim();
    if (t) h["Authorization"] = "Bearer " + t;
    return h;
  }

  const FETCH_TIMEOUT_MS = 20000;

  async function fetchWithTimeout(url, options, ms) {
    const ctrl = new AbortController();
    const id = setTimeout(function () { ctrl.abort(); }, ms);
    try {
      return await fetch(url, Object.assign({}, options || {}, { signal: ctrl.signal }));
    } finally {
      clearTimeout(id);
    }
  }

  async function fetchModelsList() {
    const res = await fetchWithTimeout(apiBase() + "/v1/models", {}, FETCH_TIMEOUT_MS);
    if (!res.ok) throw new Error("models " + res.status);
    return res.json();
  }

  async function fetchStatus() {
    const res = await fetchWithTimeout(apiBase() + "/v1/model/status", {}, FETCH_TIMEOUT_MS);
    if (!res.ok) throw new Error("status " + res.status);
    return res.json();
  }

  function setPill(loaded, explicitUnload, engineInMemory) {
    const pill = $("loadPill");
    const txt = $("loadPillText");
    pill.classList.remove("loaded", "warn", "err");
    if (explicitUnload && !engineInMemory) {
      pill.classList.add("warn");
      txt.textContent = "Unloaded — เลือกโมเดลเพื่อโหลด";
      return;
    }
    if (loaded && engineInMemory) {
      pill.classList.add("loaded");
      txt.textContent = "โหลดในแรมแล้ว";
      return;
    }
    if (engineInMemory) {
      pill.classList.add("loaded");
      txt.textContent = "โหลดแล้ว";
      return;
    }
    txt.textContent = "ยังไม่โหลดในแรม (จะโหลดตอนแชทหรือเลือกโมเดล)";
  }

  function idMatchesActiveModel(id, activeAbs) {
    if (!id || !activeAbs) return false;
    const a = activeAbs.replace(/\\/g, "/");
    const i = id.replace(/\\/g, "/");
    return a === i || a.endsWith("/" + i);
  }

  async function refreshCatalog(selectPath) {
    const sel = $("modelSelect");
    try {
      const [listData, st] = await Promise.all([
        fetchModelsList(),
        fetchStatus(),
      ]);
      const rows = listData.data || [];
      _catalogActiveRel = "";
      const activeAbs = (st.active_model_path || "").replace(/\\/g, "/");
      sel.innerHTML = '<option value="">— เลือกโมเดล —</option>';
      for (let k = 0; k < rows.length; k++) {
        const row = rows[k];
        const id = row.id;
        if (!id) continue;
        const opt = document.createElement("option");
        opt.value = id;
        let label = id;
        if (idMatchesActiveModel(id, activeAbs) && st.engine_in_memory) {
          label += "  ● โหลดอยู่";
          _catalogActiveRel = id;
        }
        opt.textContent = label;
        sel.appendChild(opt);
      }
      const want = selectPath || _catalogActiveRel;
      if (want && [...sel.options].some(function (o) { return o.value === want; })) sel.value = want;
      setPill(
        st.engine_in_memory && !st.must_load_before_chat,
        st.must_load_before_chat,
        st.engine_in_memory
      );
      if (st.model_id && !$("modelId").value) $("modelId").value = st.model_id;
      if (st.model_id) $("modelId").placeholder = st.model_id;
      const note = $("modelLoadNote");
      note.textContent =
        "รายการจาก GET /v1/models — เลือกแล้วโหลดเข้าแรม (path เช่น models/xxx.litertlm)";
      note.style.color = "";
    } catch (e) {
      const msg = (e && e.name === "AbortError") ? "หมดเวลาเชื่อมต่อ — ตรวจ Base URL / เซิร์ฟเวอร์" : String(e.message || e);
      sel.innerHTML = '<option value="">— โหลดรายการไม่ได้ —</option>';
      $("loadPillText").textContent = msg;
      $("loadPill").classList.add("err");
    }
  }

  async function autoLoadSelected() {
    const path = $("modelSelect").value;
    if (!path || _loadingModel) return;
    _loadingModel = true;
    $("modelSelect").disabled = true;
    $("loadPillText").textContent = "กำลังโหลดโมเดล…";
    $("loadPill").classList.remove("loaded", "warn", "err");
    try {
      const res = await fetch(apiBase() + "/v1/model/load", {
        method: "POST",
        headers: adminHeaders(true),
        body: JSON.stringify({ path: path, model_id: null }),
      });
      const text = await res.text();
      let j;
      try { j = JSON.parse(text); } catch { j = {}; }
      if (!res.ok) throw new Error(text.slice(0, 400));
      if (j.model_id) $("modelId").value = j.model_id;
      await refreshCatalog(path);
    } catch (e) {
      $("loadPill").classList.add("err");
      $("loadPillText").textContent = "โหลดไม่สำเร็จ: " + (e.message || e);
    } finally {
      _loadingModel = false;
      $("modelSelect").disabled = false;
    }
  }

  $("modelSelect").addEventListener("change", () => {
    autoLoadSelected();
  });

  $("btnRefreshCatalog").onclick = () => refreshCatalog($("modelSelect").value);
  $("btnUnloadModel").onclick = async () => {
    $("loadPillText").textContent = "กำลัง unload…";
    try {
      const res = await fetch(apiBase() + "/v1/model/unload", {
        method: "POST",
        headers: adminHeaders(false),
      });
      await res.text();
      await refreshCatalog($("modelSelect").value);
    } catch (e) {
      $("loadPill").classList.add("err");
      $("loadPillText").textContent = String(e.message || e);
    }
  };

  $("btnDownload").onclick = async () => {
    const o = $("adminOut");
    const url = $("dlUrl").value.trim();
    const path = $("dlPath").value.trim();
    if (!url || !path) { o.textContent = "ใส่ URL และ path"; return; }
    o.textContent = "กำลังดาวน์โหลด…";
    try {
      const res = await fetch(apiBase() + "/v1/model/download", {
        method: "POST",
        headers: adminHeaders(true),
        body: JSON.stringify({ url, path }),
      });
      const text = await res.text();
      try { o.textContent = JSON.stringify(JSON.parse(text), null, 2); }
      catch { o.textContent = text; }
      await refreshCatalog($("modelSelect").value);
    } catch (e) { o.textContent = String(e); }
  };

  function buildApiMessages() {
    const sys = $("systemPrompt").value.trim();
    const msgs = [];
    if (sys) msgs.push({ role: "system", content: sys });
    for (let i = 0; i < chatHistory.length; i++) msgs.push(chatHistory[i]);
    return msgs;
  }

  function scrollChatToBottom() {
    const el = $("chatMessages");
    el.scrollTop = el.scrollHeight;
  }

  function hideWelcomeIfAny() {
    const w = $("chatWelcome");
    if (w && w.parentNode) w.remove();
  }

  function appendMessageBubble(role, text) {
    hideWelcomeIfAny();
    const wrap = document.createElement("div");
    wrap.className = "msg " + role;
    const b = document.createElement("div");
    b.className = "bubble";
    b.textContent = text;
    wrap.appendChild(b);
    $("chatMessages").appendChild(wrap);
    scrollChatToBottom();
    return b;
  }

  function modelName() {
    const sel = $("modelSelect").value.trim();
    if (sel) return sel;
    const m = $("modelId").value.trim();
    return m || "gemma-litert";
  }

  async function parseSseStream(response, onDelta) {
    const reader = response.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    const t0 = performance.now();
    while (true) {
      const { done, value } = await reader.read();
      buf += dec.decode(value || new Uint8Array(), { stream: !done });
      let sep;
      while ((sep = buf.indexOf("\n\n")) >= 0) {
        const block = buf.slice(0, sep);
        buf = buf.slice(sep + 2);
        for (const line of block.split("\n")) {
          if (!line.startsWith("data:")) continue;
          const d = line.slice(5).trim();
          if (d === "[DONE]") return Math.round(performance.now() - t0);
          try {
            const j = JSON.parse(d);
            const ch = j.choices && j.choices[0] && j.choices[0].delta && j.choices[0].delta.content;
            if (ch) onDelta(ch);
          } catch {}
        }
      }
      if (done) break;
    }
    return Math.round(performance.now() - t0);
  }

  $("btnClearChat").onclick = () => {
    chatHistory = [];
    const box = $("chatMessages");
    box.innerHTML = '<div class="chat-welcome" id="chatWelcome">ส่งข้อความด้านล่าง · ใช้ <code>/v1/chat/completions</code> · โมเดลจาก <code>GET /v1/models</code></div>';
    $("chatMeta").textContent = "";
  };

  $("chatInput").addEventListener("keydown", function (ev) {
    if (ev.key === "Enter" && !ev.shiftKey) {
      ev.preventDefault();
      $("btnSend").click();
    }
  });

  $("btnSend").onclick = async () => {
    const meta = $("chatMeta");
    meta.textContent = "";
    const userText = $("chatInput").value.trim();
    if (!userText) {
      meta.innerHTML = '<span class="err">กรุณาใส่ข้อความ</span>';
      return;
    }
    const stream = $("useStream").checked;
    chatHistory.push({ role: "user", content: userText });
    $("chatInput").value = "";
    appendMessageBubble("user", userText);
    const assistantBubble = appendMessageBubble("assistant", "");
    assistantBubble.classList.add("streaming");
    const url = apiBase() + "/v1/chat/completions";
    const messages = buildApiMessages();
    $("btnSend").disabled = true;
    let assistantText = "";
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: modelName(), messages, stream }),
      });
      if (!res.ok) {
        const t = await res.text();
        throw new Error(res.status + " " + t.slice(0, 500));
      }
      if (stream) {
        const ms = await parseSseStream(res, function (c) {
          assistantText += c;
          assistantBubble.textContent = assistantText;
          scrollChatToBottom();
        });
        assistantBubble.classList.remove("streaming");
        chatHistory.push({ role: "assistant", content: assistantText });
        meta.textContent = "สตรีมเสร็จ · " + Math.round(ms) + " ms";
      } else {
        const data = await res.json();
        assistantText =
          (data.choices && data.choices[0] && data.choices[0].message && data.choices[0].message.content) || "";
        assistantBubble.textContent = assistantText;
        assistantBubble.classList.remove("streaming");
        chatHistory.push({ role: "assistant", content: assistantText });
        meta.textContent = "ไม่สตรีม";
      }
    } catch (e) {
      assistantBubble.classList.remove("streaming");
      assistantBubble.classList.add("err-bubble");
      assistantBubble.textContent = String(e.message || e);
      meta.innerHTML = '<span class="err">' + String(e.message || e) + "</span>";
    } finally {
      $("btnSend").disabled = false;
      scrollChatToBottom();
    }
  };

  refreshCatalog();
})();
  </script>
</body>
</html>
"""


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        os.makedirs(os.path.join(MODEL_DOWNLOAD_ROOT, "models"), exist_ok=True)
    except OSError:
        pass
    yield


app = FastAPI(
    title="OpenAI-compatible LiteRT-LM",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/v1")


@app.get("/", include_in_schema=False)
def root_redirect_ui():
    return RedirectResponse(url="/ui")


@app.get("/ui")
def ui_test_page():
    return HTMLResponse(content=_UI_TEST_PAGE_HTML, media_type="text/html; charset=utf-8")
