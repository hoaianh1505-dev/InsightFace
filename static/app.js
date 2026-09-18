/**
 * app.js — Frontend logic cho Emotion Recognition Dashboard
 * ============================================================
 * - Nhận SSE events từ /events
 * - Cập nhật countdown bar, badge cảm xúc, stats grid
 * - Append row vào bảng records
 * - Xử lý nút Xuất báo cáo & Phiên mới
 */

// ── Cấu hình màu mức độ hài lòng (4 cấp độ) ───────────────────────────────

const EMOTION_COLORS = {
  "Rất hài lòng":   "#22c55e",
  "Hài lòng":        "#38bdf8",
  "Bình thường":    "#94a3b8",
  "Không hài lòng": "#ef4444",
  // Fallback
  "Vui vẻ":     "#22c55e",
  "Ngạc nhiên": "#38bdf8",
  "Trung tính": "#94a3b8",
  "Buồn bã":    "#ef4444",
  "Sợ hãi":     "#ef4444",
  "Tức giận":   "#ef4444",
};

const EMOTION_TEXT_DARK = new Set(["Ngạc nhiên"]);  // Badge dùng text tối

// ── State ───────────────────────────────────────────────────────────────────

let cycleDuration   = 5;   // Sẽ cập nhật từ /api/info
let allRecords      = [];
let emotionCounts   = {};
let lastEmotionBadge = null;

// ── DOM References ───────────────────────────────────────────────────────────

const els = {
  statusText:       document.getElementById("statusText"),
  storageMode:      document.getElementById("storageMode"),
  fpsBadge:         document.getElementById("fpsBadge"),
  faceCountBadge:   document.getElementById("faceCountBadge"),
  phaseDot:         document.getElementById("phaseDot"),
  phaseText:        document.getElementById("phaseText"),
  cycleTimer:       document.getElementById("cycleTimer"),
  countdownBar:     document.getElementById("countdownBar"),
  currentEmotionBadge: document.getElementById("currentEmotionBadge"),
  statsGrid:        document.getElementById("statsGrid"),
  totalRecords:     document.getElementById("totalRecords"),
  recordsBody:      document.getElementById("recordsBody"),
};

// ── Khởi tạo ─────────────────────────────────────────────────────────────────

async function init() {
  // Lấy thông tin cấu hình
  try {
    const info = await fetch("/api/info").then(r => r.json());
    cycleDuration = info.cycle_sec || 5;
    initStatsGrid(info.emotions || []);
    populateModelSelect(info.available_models || [], info.current_model || "");
  } catch (e) {
    console.warn("Không lấy được /api/info:", e);
    initStatsGrid(["Rất hài lòng", "Hài lòng", "Bình thường", "Không hài lòng"]);
  }

  // Kết nối SSE
  connectSSE();
}

function populateModelSelect(models, currentModel) {
  const select = document.getElementById("modelSelect");
  if (!select) return;

  if (!models || models.length === 0) {
    select.innerHTML = `<option value="">Demo Mode (random)</option>`;
    return;
  }

  select.innerHTML = models.map(m => `
    <option value="${m}" ${m === currentModel ? "selected" : ""}>
      ${m}
    </option>
  `).join("");
}

async function onModelChange(selectedModel) {
  if (!selectedModel) return;
  try {
    const resp = await fetch("/api/select_model", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: selectedModel }),
    });
    const res = await resp.json();
    if (resp.ok) {
      showToast(`🤖 Đã đổi sang model: ${res.current_model}`);
    } else {
      showToast(`❌ Lỗi: ${res.error}`);
    }
  } catch (e) {
    showToast(`❌ Không thể đổi model: ${e.message}`);
  }
}

// ── SSE Connection ────────────────────────────────────────────────────────────

function connectSSE() {
  const evtSource = new EventSource("/events");

  evtSource.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data);
      handleSSEEvent(data);
    } catch (err) {
      console.warn("SSE parse error:", err);
    }
  };

  evtSource.onerror = () => {
    els.statusText.textContent = "❌ Mất kết nối — thử lại...";
    setTimeout(connectSSE, 3000);
    evtSource.close();
  };
}

function handleSSEEvent(data) {
  switch (data.type) {
    case "init":
      // Tải bản ghi ban đầu
      allRecords = data.records || [];
      emotionCounts = data.counts || {};
      renderAllRecords();
      updateStatsGrid(emotionCounts);
      break;

    case "status":
      updateStatus(data);
      break;

    case "record":
      // Bản ghi mới từ chu kỳ 5s
      appendRecord(data.record);
      emotionCounts = data.counts || emotionCounts;
      updateStatsGrid(emotionCounts);
      showToast(`✅ Ghi nhận: ${data.record.emotion} (${data.record.stability})`);
      break;

    case "reset":
      allRecords = [];
      emotionCounts = data.counts || {};
      renderAllRecords();
      updateStatsGrid(emotionCounts);
      showToast("🔄 Đã bắt đầu phiên mới");
      break;
  }
}

// ── Cập nhật status (từ SSE) ──────────────────────────────────────────────────

function updateStatus(data) {
  // Sync recognition button state
  if (data.is_active !== undefined && data.is_active !== isRecognitionActive) {
    isRecognitionActive = data.is_active;
    updateRecognitionButton(isRecognitionActive);
  }

  // FPS + face count
  if (data.fps !== undefined) {
    els.fpsBadge.textContent = `${Math.round(data.fps)} FPS`;
  }
  if (data.face_count !== undefined) {
    const n = data.face_count;
    els.faceCountBadge.textContent = `👤 ${n} khuôn mặt`;
  }

  // Storage mode
  if (data.storage_mode) {
    els.storageMode.textContent = data.storage_mode;
  }

  // Phase
  const isObserving = data.phase === "OBSERVING";
  const isPaused    = data.phase === "PAUSED" || data.is_active === false;
  els.phaseDot.classList.toggle("observing", isObserving);

  if (isPaused) {
    els.phaseText.textContent = "⏸️ Tạm dừng nhận diện";
  } else if (isObserving) {
    els.phaseText.textContent = `Đang quan sát  (BG #${data.record_no || 1})`;
  } else {
    els.phaseText.textContent = "Chờ khuôn mặt...";
  }

  // Timer
  const remaining = data.countdown_sec ?? cycleDuration;
  els.cycleTimer.textContent = `${remaining.toFixed(1)}s`;

  // Countdown bar
  const progress = data.progress_pct ?? 0;
  els.countdownBar.style.width = `${progress}%`;
  els.countdownBar.classList.toggle("near-end", progress >= 80);

  // Current emotion badge
  const emotion = data.current_emotion;
  if (emotion && emotion !== lastEmotionBadge) {
    lastEmotionBadge = emotion;
    setEmotionBadge(els.currentEmotionBadge, emotion);
  } else if (!emotion) {
    els.currentEmotionBadge.textContent = "--";
    els.currentEmotionBadge.removeAttribute("data-emotion");
    els.currentEmotionBadge.style.background = "var(--text-muted)";
    els.currentEmotionBadge.style.color = "#fff";
  }

  // System status
  if (isPaused) {
    els.statusText.textContent = `⏸️ Tạm dừng · ${data.record_count ?? 0} bản ghi`;
  } else if (isObserving) {
    els.statusText.textContent = `🟢 Đang ghi nhận · ${data.record_count ?? 0} bản ghi`;
  } else {
    els.statusText.textContent = `🟡 Chờ khuôn mặt · ${data.record_count ?? 0} bản ghi`;
  }
}

// ── Stats Grid ────────────────────────────────────────────────────────────────

function initStatsGrid(emotions) {
  els.statsGrid.innerHTML = emotions.map(em => `
    <div class="stat-item" title="${em}">
      <span class="stat-dot" style="background:${EMOTION_COLORS[em] || '#888'}"></span>
      <span class="stat-emotion">${em}</span>
      <span class="stat-count" id="stat_${sanitizeId(em)}">0</span>
    </div>
  `).join("");
}

function updateStatsGrid(counts) {
  let total = 0;
  for (const [emotion, count] of Object.entries(counts)) {
    const el = document.getElementById(`stat_${sanitizeId(emotion)}`);
    if (el) el.textContent = count;
    total += count;
  }
  els.totalRecords.textContent = total;
}

// ── Records Table ─────────────────────────────────────────────────────────────

function renderAllRecords() {
  if (allRecords.length === 0) {
    els.recordsBody.innerHTML = `
      <tr class="empty-row">
        <td colspan="6">Chưa có bản ghi. Để khuôn mặt trước camera...</td>
      </tr>`;
    return;
  }
  els.recordsBody.innerHTML = "";
  allRecords.forEach(r => appendRecord(r, false));
}

function appendRecord(record, animate = true) {
  // Xóa empty row nếu có
  const emptyRow = els.recordsBody.querySelector(".empty-row");
  if (emptyRow) emptyRow.remove();

  const tr = document.createElement("tr");
  if (animate) tr.classList.add("new-row");

  const color = EMOTION_COLORS[record.emotion] || "#888";
  const textColor = EMOTION_TEXT_DARK.has(record.emotion) ? "#1a1a1a" : "#fff";

  tr.innerHTML = `
    <td style="font-family:monospace;color:var(--text-dim)">${record.no}</td>
    <td>
      <span class="emotion-pill"
            style="background:${color};color:${textColor}"
            data-emotion="${record.emotion}">
        ${record.emotion}
      </span>
    </td>
    <td style="color:var(--text-dim);font-size:11px">${formatTime(record.time)}</td>
    <td style="font-family:monospace">${record.stability}</td>
    <td style="font-family:monospace;color:var(--text-dim)">${record.confidence}</td>
    <td style="font-family:monospace;color:var(--text-dim)">${record.frames}</td>
  `;

  // Chèn lên đầu (record mới nhất ở trên)
  els.recordsBody.insertBefore(tr, els.recordsBody.firstChild);

  // Trigger badge pop animation
  if (animate) {
    els.currentEmotionBadge.classList.remove("new-record");
    void els.currentEmotionBadge.offsetWidth; // reflow
    els.currentEmotionBadge.classList.add("new-record");
  }
}

// ── Nút điều khiển ────────────────────────────────────────────────────────────

function exportReport() {
  const btn = document.getElementById("btnExport");
  btn.textContent = "⏳ Đang tạo báo cáo...";
  btn.disabled = true;

  // Tải file qua redirect
  window.location.href = "/api/export";

  setTimeout(() => {
    btn.innerHTML = '<span class="btn-icon">📥</span> Xuất báo cáo (.xlsx)';
    btn.disabled = false;
  }, 3000);
}

async function resetSession() {
  if (!confirm("Xóa toàn bộ bản ghi và bắt đầu phiên mới?")) return;
  try {
    await fetch("/api/reset", { method: "POST" });
  } catch (e) {
    showToast("❌ Lỗi: " + e.message);
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function setEmotionBadge(el, emotion) {
  el.textContent = emotion;
  el.setAttribute("data-emotion", emotion);
  const color = EMOTION_COLORS[emotion] || "#888";
  const textColor = EMOTION_TEXT_DARK.has(emotion) ? "#1a1a1a" : "#fff";
  el.style.background = color;
  el.style.color = textColor;
}

function formatTime(ts) {
  if (!ts) return "--";
  // Nếu là string "2024-01-15 12:30:45" → lấy phần giờ
  const parts = ts.split(" ");
  return parts.length > 1 ? parts[1] : ts;
}

function sanitizeId(str) {
  return str.replace(/\s+/g, "_").replace(/[^\w]/g, "");
}

let toastTimer;
function showToast(msg) {
  const toast = document.getElementById("toast");
  toast.textContent = msg;
  toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("show"), 3500);
}

// ── WebCam Trình Duyệt ────────────────────────────────────────────────────────

let isWebcamActive  = false;
let webcamStream    = null;
let webcamInterval  = null;
let isProcessing    = false;

async function toggleWebcamSource() {
  const btn = document.getElementById("btnToggleWebcam");

  if (!isWebcamActive) {
    // Bật WebCam trình duyệt
    try {
      webcamStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
        audio: false,
      });

      const videoEl = document.getElementById("webcamVideo");
      videoEl.srcObject = webcamStream;
      await videoEl.play();

      isWebcamActive = true;
      btn.classList.add("btn-active");
      btn.textContent = "💻 Dùng Camera Server";
      showToast("📷 Đã bật WebCam trình duyệt");

      // Khởi động vòng lặp gửi frame (mỗi ~100ms)
      startWebcamLoop();
    } catch (err) {
      console.error("Lỗi mở WebCam trình duyệt:", err);
      showToast("❌ Không thể mở WebCam: " + err.message);
    }
  } else {
    // Tắt WebCam trình duyệt -> quay về Server Camera
    stopWebcamLoop();
    isWebcamActive = false;
    btn.classList.remove("btn-active");
    btn.textContent = "📷 Dùng WebCam trình duyệt";

    const cameraImg = document.getElementById("cameraFeed");
    cameraImg.src = "/video_feed?" + Date.now(); // reload stream MJPEG

    showToast("💻 Đã chuyển sang Camera Server");
  }
}

function startWebcamLoop() {
  stopWebcamLoop();
  const videoEl = document.getElementById("webcamVideo");
  const canvasEl = document.getElementById("webcamCanvas");
  const ctx = canvasEl.getContext("2d");
  const cameraImg = document.getElementById("cameraFeed");

  webcamInterval = setInterval(async () => {
    if (!isWebcamActive || isProcessing) return;

    if (videoEl.videoWidth === 0 || videoEl.videoHeight === 0) return;

    // Set canvas dimensions matching video
    if (canvasEl.width !== videoEl.videoWidth) {
      canvasEl.width = videoEl.videoWidth;
      canvasEl.height = videoEl.videoHeight;
    }

    ctx.drawImage(videoEl, 0, 0, canvasEl.width, canvasEl.height);
    const base64Image = canvasEl.toDataURL("image/jpeg", 0.75);

    isProcessing = true;
    try {
      const resp = await fetch("/api/process_frame", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: base64Image }),
      });

      if (resp.ok) {
        const data = await resp.json();
        if (data.annotated_image) {
          cameraImg.src = data.annotated_image;
        }
        if (data.state) {
          updateStatus(data.state);
        }
        if (data.new_record) {
          appendRecord(data.new_record);
          showToast(`✅ Ghi nhận: ${data.new_record.emotion} (${data.new_record.stability})`);
        }
      }
    } catch (e) {
      console.warn("Lỗi gửi frame webcam:", e);
    } finally {
      isProcessing = false;
    }
  }, 100);
}

function stopWebcamLoop() {
  if (webcamInterval) {
    clearInterval(webcamInterval);
    webcamInterval = null;
  }
  if (webcamStream) {
    webcamStream.getTracks().forEach(t => t.stop());
    webcamStream = null;
  }
  isProcessing = false;
}

// ── Start ─────────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", init);
