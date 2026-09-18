"""
utils/emotion_tracker.py — Logic ghi nhận cảm xúc theo chu kỳ 5 giây
======================================================================
State machine:
  WAITING   → khi không có khuôn mặt nào trong frame
  OBSERVING → khuôn mặt xuất hiện, đang thu thập predictions trong 5s
  COMPLETED → đủ 5s, gọi callback với bản ghi kết quả

Quy tắc:
  - Chỉ quan sát khuôn mặt LỚN NHẤT (gần camera nhất) tại mỗi thời điểm
  - Nếu mặt biến mất giữa chu kỳ → reset, không ghi
  - Khi đủ 5s → tính MODE (cảm xúc xuất hiện nhiều nhất) → ghi bản ghi
  - Nếu mặt vẫn còn sau khi ghi → bắt đầu chu kỳ mới ngay lập tức
"""

import time
import collections
from dataclasses import dataclass, asdict, field
from typing import Callable, Optional
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


@dataclass
class EmotionRecord:
    """Một bản ghi cảm xúc / mức hài lòng sau khi kết thúc chu kỳ 5 giây."""
    record_no:        int            # Số thứ tự bản ghi (bắt đầu từ 1)
    emotion:          str            # Nhãn cảm xúc mode trong chu kỳ (raw)
    satisfaction:     str            # Mức độ hài lòng (tiếng Việt)
    timestamp:        str            # Thời điểm ghi (ISO string)
    timestamp_unix:   float          # Unix timestamp
    stability_pct:    float          # % frame khớp với nhãn mode (0–100)
    avg_confidence:   float          # Confidence trung bình trong chu kỳ
    frame_count:      int            # Số frame được quan sát
    cycle_duration:   float          # Thời gian thực tế của chu kỳ (giây)
    all_counts:       dict = field(default_factory=dict)  # {label: count}

    def to_dict(self) -> dict:
        return asdict(self)

    def to_display_dict(self) -> dict:
        """Dict gọn cho hiển thị bảng web."""
        sat = self.satisfaction or config.EMOTION_TO_SATISFACTION_VI.get(self.emotion, self.emotion)
        return {
            "no":          self.record_no,
            "emotion":     sat,  # Hiển thị mức hài lòng lên Web UI
            "time":        self.timestamp,
            "stability":   f"{self.stability_pct:.0f}%",
            "confidence":  f"{self.avg_confidence:.0%}",
            "frames":      self.frame_count,
        }


def pick_largest_face(results: list[dict]) -> Optional[dict]:
    """
    Từ danh sách khuôn mặt phát hiện được, chọn khuôn mặt LỚN NHẤT
    (diện tích bounding box lớn nhất ≈ gần camera nhất).

    Parameters
    ----------
    results : list of dict
        Kết quả từ AIProcessor.process_frame().
        Mỗi dict: {"bbox": (x,y,w,h), "emotion": str, "confidence": float, ...}

    Returns
    -------
    dict | None — khuôn mặt lớn nhất, hoặc None nếu không có.
    """
    if not results:
        return None
    return max(results, key=lambda r: r["bbox"][2] * r["bbox"][3])


class EmotionCycleTracker:
    """
    Theo dõi và ghi nhận cảm xúc theo chu kỳ cố định (mặc định 5 giây).

    Usage
    -----
    tracker = EmotionCycleTracker(on_record=my_callback)

    # Trong vòng lặp xử lý frame:
    tracker.tick(face_result)   # face_result = None nếu không có mặt

    # Truy vấn trạng thái:
    state = tracker.get_state()
    """

    STATE_PAUSED    = "PAUSED"
    STATE_WAITING   = "WAITING"
    STATE_OBSERVING = "OBSERVING"

    def __init__(
        self,
        cycle_sec: float = None,
        on_record: Callable[[EmotionRecord], None] = None,
    ):
        """
        Parameters
        ----------
        cycle_sec : float
            Độ dài chu kỳ (giây). Mặc định từ config.CYCLE_DURATION_SEC.
        on_record : callable
            Callback được gọi khi hoàn thành một chu kỳ, nhận EmotionRecord.
        """
        self.cycle_sec   = cycle_sec or config.CYCLE_DURATION_SEC
        self._on_record  = on_record
        self._record_no  = 0
        self.is_active   = False   # Mặc định TẮT nhận diện khi mở web

        # State
        self._state       = self.STATE_WAITING
        self._cycle_start = 0.0
        self._predictions: list[tuple[str, float]] = []  # (emotion, confidence)

    # ── API chính ─────────────────────────────────────────────────────────────

    def toggle_active(self, active: Optional[bool] = None) -> bool:
        """Bật/tắt trạng thái nhận diện."""
        if active is None:
            self.is_active = not self.is_active
        else:
            self.is_active = bool(active)

        if not self.is_active:
            self._reset()
        return self.is_active

    def tick(self, face_result: Optional[dict]) -> Optional[EmotionRecord]:
        """
        Gọi mỗi frame với kết quả khuôn mặt lớn nhất.
        """
        if not self.is_active:
            if self._state != self.STATE_WAITING:
                self._reset()
            return None

        now = time.time()

        if self._state == self.STATE_WAITING:
            if face_result is not None:
                # Bắt đầu chu kỳ mới
                self._state       = self.STATE_OBSERVING
                self._cycle_start = now
                self._predictions = []
                self._predictions.append((face_result["emotion"], face_result["confidence"]))
            return None

        elif self._state == self.STATE_OBSERVING:
            elapsed = now - self._cycle_start

            if face_result is None:
                # Khuôn mặt biến mất — reset, không ghi
                self._reset()
                return None

            # Thu thập prediction
            self._predictions.append((face_result["emotion"], face_result["confidence"]))

            if elapsed >= self.cycle_sec:
                # Đủ 5 giây — tính kết quả
                record = self._finalize(elapsed)
                # Tiếp tục chu kỳ mới ngay (mặt vẫn còn)
                self._state       = self.STATE_OBSERVING
                self._cycle_start = now
                self._predictions = [(face_result["emotion"], face_result["confidence"])]

                if self._on_record:
                    self._on_record(record)
                return record

        return None

    def reset_session(self):
        """Reset toàn phiên (phiên mới)."""
        self._reset()
        self._record_no = 0

    # ── Truy vấn trạng thái ───────────────────────────────────────────────────

    def get_state(self) -> dict:
        """
        Trạng thái hiện tại để frontend hiển thị.
        """
        now = time.time()
        if not self.is_active:
            return {
                "phase":           self.STATE_PAUSED,
                "is_active":        False,
                "countdown_sec":   self.cycle_sec,
                "progress_pct":    0.0,
                "current_emotion": None,
                "frame_count":     0,
                "record_no":       self._record_no,
            }
        elif self._state == self.STATE_WAITING:
            return {
                "phase":           self.STATE_WAITING,
                "is_active":        True,
                "countdown_sec":   self.cycle_sec,
                "progress_pct":    0.0,
                "current_emotion": None,
                "frame_count":     0,
                "record_no":       self._record_no,
            }
        else:
            elapsed   = now - self._cycle_start
            remaining = max(0.0, self.cycle_sec - elapsed)
            progress  = min(100.0, elapsed / self.cycle_sec * 100)
            current   = self._predictions[-1][0] if self._predictions else None
            sat_current = config.EMOTION_TO_SATISFACTION_VI.get(current, current) if current else None
            return {
                "phase":           self.STATE_OBSERVING,
                "is_active":        True,
                "countdown_sec":   remaining,
                "progress_pct":    progress,
                "current_emotion": sat_current,
                "frame_count":     len(self._predictions),
                "record_no":       self._record_no + 1,  # bản ghi sắp ghi
            }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _finalize(self, actual_duration: float) -> EmotionRecord:
        """Tính toán bản ghi từ các predictions thu thập được."""
        self._record_no += 1

        # Đếm số lần mỗi cảm xúc
        counts: dict[str, int] = {}
        for emotion, _ in self._predictions:
            counts[emotion] = counts.get(emotion, 0) + 1

        # Mode: cảm xúc xuất hiện nhiều nhất
        dominant_emotion = max(counts, key=counts.get)
        dominant_count   = counts[dominant_emotion]
        total_frames     = len(self._predictions)

        # Stability: % frame khớp với mode
        stability_pct = dominant_count / total_frames * 100

        # Confidence trung bình (của tất cả frames)
        avg_conf = sum(c for _, c in self._predictions) / total_frames

        # Timestamp
        from datetime import datetime
        now_unix = time.time()
        ts       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        sat_label = config.EMOTION_TO_SATISFACTION_VI.get(dominant_emotion, dominant_emotion)

        return EmotionRecord(
            record_no      = self._record_no,
            emotion        = dominant_emotion,
            satisfaction   = sat_label,
            timestamp      = ts,
            timestamp_unix = now_unix,
            stability_pct  = round(stability_pct, 1),
            avg_confidence = round(avg_conf, 3),
            frame_count    = total_frames,
            cycle_duration = round(actual_duration, 2),
            all_counts     = counts,
        )

    def _reset(self):
        """Reset về WAITING state."""
        self._state       = self.STATE_WAITING
        self._cycle_start = 0.0
        self._predictions = []
