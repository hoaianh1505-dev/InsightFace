"""
modules/image_capture.py -- Mo-dun 1: Thu nhan hinh anh
========================================================
Thread-safe camera capture voi queue-based frame delivery.
Ho tro: webcam index, video file, RTSP URL.
"""

import cv2
import threading
import queue
import time
import logging

logger = logging.getLogger(__name__)


class CameraCapture:
    """
    Lop thu nhan hinh anh tu camera theo thoi gian thuc.
    Chay trong thread rieng, dua frame vao queue de thread chinh lay.
    """

    def __init__(self, source=0, width=1280, height=720, fps=30, queue_size=2):
        self.source     = source
        self.width      = width
        self.height     = height
        self.fps        = fps
        self._queue     = queue.Queue(maxsize=queue_size)
        self._cap       = None
        self._thread    = None
        self._running   = False
        self._lock      = threading.Lock()
        self.actual_fps   = 0.0
        self.frame_count  = 0
        self.is_connected = False

    def start(self) -> bool:
        with self._lock:
            if self._running:
                return True
            self._cap = cv2.VideoCapture(self.source)
            if not self._cap.isOpened():
                logger.error(f"Khong the mo camera: {self.source}")
                return False
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS,          self.fps)
            self.is_connected = True
            self._running     = True
            self._thread      = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()
            logger.info(f"Camera started: {self.source}")
            return True

    def stop(self):
        with self._lock:
            self._running     = False
            self.is_connected = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.info("Camera stopped.")

    def get_frame(self, timeout=0.1):
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _capture_loop(self):
        fps_counter = 0
        t_start     = time.time()
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                if isinstance(self.source, str):
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    logger.warning("Lost camera connection.")
                    self.is_connected = False
                    break
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
            self._queue.put(frame)
            self.frame_count += 1
            fps_counter      += 1
            elapsed = time.time() - t_start
            if elapsed >= 1.0:
                self.actual_fps = fps_counter / elapsed
                fps_counter     = 0
                t_start         = time.time()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()
