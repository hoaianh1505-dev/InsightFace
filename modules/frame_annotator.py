"""
modules/frame_annotator.py — Vẽ annotation lên frame OpenCV
============================================================
Không có alert logic. Chỉ vẽ:
  - Bounding box màu theo cảm xúc
  - Nhãn cảm xúc + confidence
  - Đồng hồ đếm ngược 5 giây (progress bar)
  - Trạng thái OBSERVING / WAITING
"""

import cv2
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import unicodedata

# Map cam xuc → muc hai long ASCII (dung cho cv2.putText tren camera frame)
EMOTION_TO_SAT_ASCII = {
    "Vui vẻ":     "Rat hai long",
    "Ngạc nhiên": "Hai long",
    "Trung tính": "Binh thuong",
    "Buồn bã":    "Khong hai long",
    "Sợ hãi":     "Khong hai long",
    "Tức giận":   "Khong hai long",
}

def _sat_label(emotion: str) -> str:
    """Lay muc hai long ASCII tuong ung voi cam xuc (an toan cho cv2.putText)."""
    result = EMOTION_TO_SAT_ASCII.get(emotion)
    if result:
        return result
    # Fallback: thu NFC normalize
    normalized = unicodedata.normalize("NFC", emotion)
    return EMOTION_TO_SAT_ASCII.get(normalized, emotion)



def annotate_frame(
    frame: np.ndarray,
    face_result: dict | None,
    phase: str,
    countdown_sec: float,
    cycle_sec: float = None,
    record_count: int = 0,
    fps: float = 0.0,
) -> np.ndarray:
    """
    Vẽ toàn bộ annotation lên frame.

    Parameters
    ----------
    frame        : np.ndarray — frame BGR gốc
    face_result  : dict | None — kết quả của khuôn mặt đang quan sát
                   {"bbox": (x,y,w,h), "emotion": str, "confidence": float}
    phase        : str — "WAITING" | "OBSERVING"
    countdown_sec: float — số giây còn lại trong chu kỳ
    cycle_sec    : float — tổng độ dài chu kỳ (mặc định config.CYCLE_DURATION_SEC)
    record_count : int — số bản ghi đã ghi trong phiên
    fps          : float — FPS thực tế

    Returns
    -------
    frame annotated (BGR)
    """
    if cycle_sec is None:
        cycle_sec = config.CYCLE_DURATION_SEC

    h, w = frame.shape[:2]

    # ── Vẽ bounding box khuôn mặt đang quan sát ──────────────────────────────
    if face_result is not None:
        x, y, bw, bh = face_result["bbox"]
        emotion       = face_result["emotion"]
        confidence    = face_result["confidence"]
        
        sat_text = _sat_label(emotion)
        color_bgr = config.SATISFACTION_COLORS_BGR.get(sat_text, (184, 163, 148))

        # Bounding box
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), color_bgr, 2)

        # Nhan muc hai long + confidence (dung ASCII cho cv2.putText)
        label = f"{sat_text}  {confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        label_y = y - 12 if y > 40 else y + bh + 28
        cv2.rectangle(frame, (x, label_y - th - 6), (x + tw + 8, label_y + 4),
                      (0, 0, 0), -1)
        cv2.putText(frame, label, (x + 4, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color_bgr, 2, cv2.LINE_AA)

        # Countdown bar bên dưới bbox
        if phase == "OBSERVING":
            _draw_countdown_bar(frame, x, y + bh + 6, bw, countdown_sec, cycle_sec, color_bgr)

    # ── HUD góc trên trái ─────────────────────────────────────────────────────
    _draw_hud(frame, phase, countdown_sec, record_count, fps)

    return frame


def _draw_countdown_bar(
    frame: np.ndarray,
    x: int, y: int, width: int,
    remaining: float, total: float,
    color: tuple,
):
    """Thanh countdown gắn dưới bounding box."""
    bar_h   = 8
    h_max, w_max = frame.shape[:2]

    # Clamp
    x  = max(0, min(x, w_max - width))
    y  = max(0, min(y, h_max - bar_h - 1))

    filled = int((1.0 - remaining / total) * width)
    filled = max(0, min(filled, width))

    cv2.rectangle(frame, (x, y), (x + width, y + bar_h), (40, 40, 40), -1)
    if filled > 0:
        cv2.rectangle(frame, (x, y), (x + filled, y + bar_h), color, -1)
    cv2.rectangle(frame, (x, y), (x + width, y + bar_h), (100, 100, 100), 1)

    # Số giây
    txt = f"{remaining:.1f}s"
    cv2.putText(frame, txt, (x + width + 6, y + bar_h),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)


def _draw_hud(
    frame: np.ndarray,
    phase: str,
    countdown: float,
    record_count: int,
    fps: float,
):
    """HUD bảng trạng thái góc trên trái."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (310, 90), (10, 12, 20), -1)
    frame[:] = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

    # Phase indicator
    if phase == "OBSERVING":
        phase_color = (34, 197, 100)
        phase_text  = f"DANG QUAN SAT  {countdown:.1f}s"
    elif phase == "PAUSED":
        phase_color = (0, 190, 245)
        phase_text  = "TAM DUNG NHAN DIEN"
    else:
        phase_color = (148, 163, 184)
        phase_text  = "CHO KHUON MAT..."

    cv2.putText(frame, phase_text, (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, phase_color, 2, cv2.LINE_AA)

    info = f"Ban ghi: {record_count}   FPS: {fps:.1f}"
    cv2.putText(frame, info, (8, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    method = "Haar Cascade (Viola-Jones)"
    cv2.putText(frame, f"Detect: {method}", (8, 74),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 140, 200), 1, cv2.LINE_AA)


def draw_all_faces(frame: np.ndarray, results: list[dict]) -> np.ndarray:
    """
    Vẽ tất cả khuôn mặt phát hiện được (màu xám, không đếm ngược).
    Dùng khi phase = WAITING để thấy bounding box dù chưa quan sát.
    """
    for r in results:
        x, y, bw, bh = r["bbox"]
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (100, 100, 100), 1)
    return frame
