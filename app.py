"""
app.py — Flask Entry Point
==========================
Routes:
  GET  /             → index.html
  GET  /video_feed   → MJPEG stream
  GET  /events       → SSE stream (real-time updates cho JS)
  GET  /api/records  → JSON danh sách bản ghi
  GET  /api/status   → JSON trạng thái hiện tại
  GET  /api/export   → Tải file Excel
  POST /api/reset    → Bắt đầu phiên mới

Pipeline chạy trong background thread:
  camera → face detect → emotion classify → 5s tracker → storage → SSE push
"""

import cv2
import base64
import numpy as np
import threading
import time
import json
import queue
import logging
import os
import sys
from datetime import datetime
from flask import Flask, Response, jsonify, send_file, request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from modules.image_capture import CameraCapture
from modules.ai_processor import AIProcessor
from modules.frame_annotator import annotate_frame, draw_all_faces
from utils.emotion_tracker import EmotionCycleTracker, pick_largest_face
from utils.storage import StorageManager

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("app")

# ── Flask app ─────────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder="static", static_url_path="/static")

# ── Global state (thread-safe) ────────────────────────────────────────────────

_lock          = threading.Lock()
_latest_frame  = None          # Frame annotated mới nhất (bytes JPEG)
_current_state = {             # Trạng thái tracker hiện tại
    "phase":           "WAITING",
    "countdown_sec":   5.0,
    "progress_pct":    0.0,
    "current_emotion": None,
    "frame_count":     0,
    "record_no":       0,
    "fps":             0.0,
    "face_count":      0,
    "storage_mode":    "Đang khởi động...",
    "record_count":    0,
}
_sse_queues: list[queue.Queue] = []  # SSE subscriber queues
_sse_lock   = threading.Lock()       # Lock bảo vệ _sse_queues
_running    = False
_camera     = None
_ai         = None
_tracker    = None
_storage    = None
_last_sse_push = 0.0                 # Timestamp lần push SSE cuối


# ══════════════════════════════════════════════════════════════════════════════
# BACKGROUND PROCESSING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def _on_record_complete(record):
    """Callback khi tracker ghi xong 1 bản ghi."""
    _storage.write_record(record)
    logger.info(f"📝 Record #{record.record_no}: {record.emotion} "
                f"(ổn định {record.stability_pct:.0f}%)")

    # Push SSE event "record" cho tất cả subscribers
    event_data = json.dumps({
        "type":   "record",
        "record": record.to_display_dict(),
        "counts": _storage.get_summary_counts(),
    })
    _push_sse(event_data)


def _push_sse(data: str):
    """Đẩy message vào tất cả SSE queues đang mở (thread-safe)."""
    global _sse_queues
    with _sse_lock:
        live = []
        for q in _sse_queues:
            try:
                q.put_nowait(data)
                live.append(q)
            except queue.Full:
                pass  # Bỏ queue chậm, không thêm lại
        _sse_queues = live


def _processing_loop():
    """
    Vòng lặp xử lý chính chạy trong background thread.
    camera → detect → classify → tracker → annotate → JPEG encode
    """
    global _latest_frame, _current_state, _running, _last_sse_push

    fps_times    = []
    frame_time   = 1.0 / config.STREAM_FPS
    sse_interval = 0.3   # Push SSE status mỗi 300ms

    while _running:
        t0    = time.time()
        frame = _camera.get_frame(timeout=0.05)
        if frame is None:
            time.sleep(0.01)
            continue

        # FPS
        fps_times.append(t0)
        fps_times = [t for t in fps_times if t > t0 - 1.0]
        fps = len(fps_times)

        # AI: detect + classify
        try:
            results = _ai.process_frame(frame)
        except Exception as e:
            logger.error(f"AI error: {e}")
            results = []
        largest_face = pick_largest_face(results)

        # Tracker tick
        _tracker.tick(largest_face)   # callback _on_record_complete handles writing

        # Trạng thái tracker
        state = _tracker.get_state()
        state["fps"]          = fps
        state["face_count"]   = len(results)
        state["storage_mode"] = _storage.storage_mode
        state["record_count"] = _storage.record_count

        with _lock:
            _current_state = state

        # Vẽ annotation
        try:
            if largest_face is not None:
                frame = annotate_frame(
                    frame,
                    face_result   = largest_face,
                    phase         = state["phase"],
                    countdown_sec = state["countdown_sec"],
                    cycle_sec     = config.CYCLE_DURATION_SEC,
                    record_count  = _storage.record_count,
                    fps           = fps,
                )
            else:
                frame = draw_all_faces(frame, results)
                frame = annotate_frame(
                    frame,
                    face_result   = None,
                    phase         = "WAITING",
                    countdown_sec = config.CYCLE_DURATION_SEC,
                    record_count  = _storage.record_count,
                    fps           = fps,
                )
        except Exception as e:
            logger.error(f"Annotate error: {e}")

        # Encode JPEG
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, config.STREAM_QUALITY]
        ret, buffer   = cv2.imencode(".jpg", frame, encode_params)
        if ret:
            with _lock:
                _latest_frame = buffer.tobytes()

        # Push trạng thái SSE theo interval cố định (tránh float arithmetic flaky)
        now = time.time()
        if now - _last_sse_push >= sse_interval:
            _last_sse_push = now
            try:
                _push_sse(json.dumps({"type": "status", **state}))
            except Exception as e:
                logger.debug(f"SSE push error: {e}")

        # Throttle
        elapsed = time.time() - t0
        sleep_t = max(0, frame_time - elapsed)
        if sleep_t > 0:
            time.sleep(sleep_t)


def _generate_mjpeg():
    """Generator MJPEG stream (multipart/x-mixed-replace)."""
    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    while _running:
        with _lock:
            frame = _latest_frame
        if frame is None:
            time.sleep(0.05)
            continue
        yield boundary + frame + b"\r\n"
        time.sleep(1.0 / config.STREAM_FPS)


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Trang chính."""
    return app.send_static_file("index.html")


@app.route("/video_feed")
def video_feed():
    """MJPEG camera stream."""
    return Response(
        _generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/events")
def events():
    """
    SSE (Server-Sent Events) stream.
    Frontend subscribe để nhận record mới và status update.
    """
    def stream():
        q = queue.Queue(maxsize=30)
        with _sse_lock:
            _sse_queues.append(q)
        try:
            # Gửi trạng thái ban đầu
            init_data = json.dumps({
                "type":    "init",
                "records": _storage.get_records_as_dicts() if _storage else [],
                "counts":  _storage.get_summary_counts() if _storage else {},
            })
            yield f"data: {init_data}\n\n"

            while _running:
                try:
                    data = q.get(timeout=20)
                    yield f"data: {data}\n\n"
                except queue.Empty:
                    # Heartbeat để giữ kết nối
                    yield ": heartbeat\n\n"
        except GeneratorExit:
            pass
        finally:
            with _sse_lock:
                try:
                    _sse_queues.remove(q)
                except ValueError:
                    pass

    return Response(
        stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.route("/api/records")
def api_records():
    """Danh sách tất cả bản ghi trong phiên."""
    return jsonify({
        "records":      _storage.get_records_as_dicts(),
        "total":        _storage.record_count,
        "summary":      _storage.get_summary_counts(),
        "storage_mode": _storage.storage_mode,
    })


@app.route("/api/status")
def api_status():
    """Trạng thái hệ thống hiện tại."""
    with _lock:
        state = dict(_current_state)
    return jsonify(state)


@app.route("/api/export")
def api_export():
    """Tạo và tải file Excel báo cáo."""
    if _storage is None:
        return jsonify({"error": "Hệ thống chưa khởi động"}), 503
    try:
        path = _storage.export_excel()
        # send_file cần absolute path dạng string
        abs_path = os.path.abspath(path)
        return send_file(
            abs_path,
            as_attachment=True,
            download_name=f"EmotionReport_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception as e:
        logger.error(f"Lỗi xuất Excel: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/reset", methods=["POST"])
def api_reset():
    """Xóa bản ghi, bắt đầu phiên mới."""
    _storage.clear()
    _tracker.reset_session()
    logger.info("🔄 Phiên mới đã bắt đầu.")
    _push_sse(json.dumps({
        "type":    "reset",
        "records": [],
        "counts":  _storage.get_summary_counts(),
    }))
    return jsonify({"status": "ok", "message": "Đã bắt đầu phiên mới"})


@app.route("/api/toggle_recognition", methods=["POST"])
def api_toggle_recognition():
    """Bật / Tắt trạng thái nhận diện cảm xúc."""
    if _tracker is None:
        return jsonify({"error": "Tracker chưa sẵn sàng"}), 503

    data = request.get_json(silent=True) or {}
    active = data.get("active")
    is_active = _tracker.toggle_active(active)

    state = _tracker.get_state()
    state["fps"]          = 0.0
    state["storage_mode"] = _storage.storage_mode if _storage else "N/A"
    state["record_count"] = _storage.record_count if _storage else 0

    _push_sse(json.dumps({"type": "status", **state}))

    return jsonify({
        "status": "ok",
        "is_active": is_active,
        "message": "Đã bật nhận diện" if is_active else "Đã tạm dừng nhận diện",
    })


@app.route("/api/info")
def api_info():
    """Thông tin cấu hình hệ thống."""
    return jsonify({
        "cycle_sec":     config.CYCLE_DURATION_SEC,
        "emotions":      config.SATISFACTION_LEVELS,
        "colors":        config.SATISFACTION_COLORS_HEX,
        "storage_mode":  _storage.storage_mode if _storage else "N/A",
        "model_ready":   _ai.classifier.is_ready if _ai else False,
        "available_models": config.get_available_models(),
        "current_model": _ai.current_model_name if _ai else "",
    })


@app.route("/api/select_model", methods=["POST"])
def api_select_model():
    """Đổi mô hình AI đang chạy."""
    if _ai is None:
        return jsonify({"error": "AIProcessor chưa sẵn sàng"}), 503

    data = request.get_json(silent=True) or {}
    model_name = data.get("model")
    if not model_name:
        return jsonify({"error": "Thiếu tên file model"}), 400

    success = _ai.switch_model(model_name)
    if success:
        return jsonify({
            "status": "ok",
            "message": f"Đã đổi sang model {model_name}",
            "current_model": _ai.current_model_name,
        })
    else:
        return jsonify({"error": f"Không thể tải model {model_name}"}), 400


@app.route("/api/process_frame", methods=["POST"])
def api_process_frame():
    """
    Nhận frame base64 từ WebCam trình duyệt -> AI detect -> tracker tick -> return annotated frame base64 + status.
    """
    if _ai is None or _tracker is None:
        return jsonify({"error": "Pipeline chưa sẵn sàng"}), 503

    data = request.get_json(silent=True) or {}
    image_data = data.get("image")
    if not image_data:
        return jsonify({"error": "Thiếu dữ liệu image base64"}), 400

    try:
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Không thể decode frame"}), 400

        # AI processing
        results = _ai.process_frame(frame)
        largest_face = pick_largest_face(results)

        # Tracker tick
        new_record = _tracker.tick(largest_face)
        state = _tracker.get_state()
        state["face_count"]   = len(results)
        state["storage_mode"] = _storage.storage_mode
        state["record_count"] = _storage.record_count

        # Annotate frame
        if largest_face is not None:
            frame = annotate_frame(
                frame,
                face_result   = largest_face,
                phase         = state["phase"],
                countdown_sec = state["countdown_sec"],
                cycle_sec     = config.CYCLE_DURATION_SEC,
                record_count  = _storage.record_count,
                fps           = 0.0,
            )
        else:
            frame = draw_all_faces(frame, results)
            frame = annotate_frame(
                frame,
                face_result   = None,
                phase         = "WAITING",
                countdown_sec = config.CYCLE_DURATION_SEC,
                record_count  = _storage.record_count,
                fps           = 0.0,
            )

        # Encode frame back to JPEG base64
        ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, config.STREAM_QUALITY])
        annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

        res = {
            "status": "ok",
            "annotated_image": annotated_b64,
            "state": state,
        }
        if new_record:
            res["new_record"] = new_record.to_display_dict()

        return jsonify(res)

    except Exception as e:
        logger.error(f"Lỗi process_frame: {e}")
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
# STARTUP / SHUTDOWN
# ══════════════════════════════════════════════════════════════════════════════

def start_pipeline():
    """Khởi động camera + AI + tracker + storage."""
    global _camera, _ai, _tracker, _storage, _running

    logger.info("=" * 50)
    logger.info("🚀 Khởi động hệ thống nhận diện cảm xúc...")
    logger.info("=" * 50)

    # Khởi tạo modules
    _storage = StorageManager()
    logger.info(f"✅ Storage: {_storage.storage_mode}")

    _tracker = EmotionCycleTracker(
        cycle_sec=config.CYCLE_DURATION_SEC,
        on_record=_on_record_complete,
    )

    _ai = AIProcessor()
    logger.info(f"✅ AI: {'Model sẵn sàng' if _ai.classifier.is_ready else '⚠ Demo mode (random)'}")

    _camera = CameraCapture(
        source=config.CAMERA_INDEX,
        width=config.FRAME_WIDTH,
        height=config.FRAME_HEIGHT,
        fps=config.CAMERA_FPS,
    )
    if not _camera.start():
        logger.warning("⚠ Không thể mở camera Server (có thể đang bận). Hệ thống sẵn sàng cho WebCam Trình duyệt!")
    else:
        logger.info(f"✅ Camera Server: {config.CAMERA_INDEX} ({config.FRAME_WIDTH}×{config.FRAME_HEIGHT})")

    # Tạo thư mục exports
    os.makedirs(config.EXPORTS_DIR, exist_ok=True)

    _running = True
    t = threading.Thread(target=_processing_loop, daemon=True)
    t.start()

    logger.info(f"✅ Camera: {config.CAMERA_INDEX} ({config.FRAME_WIDTH}×{config.FRAME_HEIGHT})")
    logger.info(f"✅ Chu kỳ ghi: {config.CYCLE_DURATION_SEC}s")
    logger.info(f"🌐 Mở trình duyệt: http://localhost:{config.FLASK_PORT}")
    logger.info("=" * 50)
    return True


def stop_pipeline():
    """Dừng hệ thống."""
    global _running
    _running = False
    if _camera:
        _camera.stop()
    logger.info("Hệ thống đã dừng.")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not start_pipeline():
        sys.exit(1)

    try:
        app.run(
            host=config.FLASK_HOST,
            port=config.FLASK_PORT,
            debug=False,
            threaded=True,
            use_reloader=False,  # Tắt reloader để tránh khởi động camera 2 lần
        )
    except KeyboardInterrupt:
        logger.info("\nDừng bởi người dùng.")
    finally:
        stop_pipeline()
