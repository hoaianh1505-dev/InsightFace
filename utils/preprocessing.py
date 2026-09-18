"""
utils/preprocessing.py -- Tien xu ly anh khuon mat
====================================================
Pipeline: crop -> resize 48x48 -> grayscale -> CLAHE -> normalize [0,1] -> tensor
"""

import cv2
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def crop_face(frame: np.ndarray, bbox: tuple) -> np.ndarray:
    x, y, w, h = bbox
    pad   = int(min(w, h) * 0.1)
    h_max, w_max = frame.shape[:2]
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w_max, x + w + pad)
    y2 = min(h_max, y + h + pad)
    return frame[y1:y2, x1:x2]


def resize_face(face_img: np.ndarray, size: int = None) -> np.ndarray:
    if size is None:
        size = config.FACE_INPUT_SIZE
    return cv2.resize(face_img, (size, size), interpolation=cv2.INTER_AREA)


def to_grayscale(face_img: np.ndarray) -> np.ndarray:
    if len(face_img.shape) == 3 and face_img.shape[2] == 3:
        return cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    return face_img


def apply_clahe(gray_img: np.ndarray) -> np.ndarray:
    if not config.USE_CLAHE:
        return cv2.equalizeHist(gray_img)
    clahe = cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP,
        tileGridSize=config.CLAHE_TILE_GRID
    )
    return clahe.apply(gray_img)


def normalize_pixels(gray_img: np.ndarray) -> np.ndarray:
    return gray_img.astype(np.float32) / 255.0


def preprocess_face(frame: np.ndarray, bbox: tuple) -> np.ndarray | None:
    """
    Full pipeline: crop -> resize -> grayscale -> CLAHE -> normalize.
    Returns tensor shape (1, 48, 48, 1) for CNN input, or None on error.
    """
    try:
        face = crop_face(frame, bbox)
        if face.size == 0:
            return None
        face = resize_face(face)
        face = to_grayscale(face)
        face = apply_clahe(face)
        face = normalize_pixels(face)
        return face.reshape(1, config.FACE_INPUT_SIZE, config.FACE_INPUT_SIZE, 1)
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Preprocessing error: {e}")
        return None
