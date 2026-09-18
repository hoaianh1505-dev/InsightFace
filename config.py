"""
config.py — Cấu hình trung tâm hệ thống nhận diện cảm xúc (Web Edition)
=========================================================================
Thay đổi các tham số tại đây mà không cần sửa code logic.
"""

import os

# ─── Đường dẫn ──────────────────────────────────────────────────────────────

BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR       = os.path.join(BASE_DIR, "models")
EXPORTS_DIR      = os.path.join(BASE_DIR, "exports")
CREDENTIALS_PATH = os.path.join(BASE_DIR, "credentials.json")

EMOTION_MODEL_PATH = os.path.join(MODELS_DIR, "emotion_model", "emotion_model.h5")

# URL tải model FER2013 pre-trained (nếu chưa có)
EMOTION_MODEL_URL = (
    "https://github.com/oarriaga/face_classification/raw/master/"
    "trained_models/fer2013_mini_XCEPTION.102-0.66.hdf5"
)

def get_available_models() -> list[str]:
    """Quét và trả về danh sách tên các file model (.h5, .hdf5, .keras) có sẵn."""
    model_dir = os.path.join(MODELS_DIR, "emotion_model")
    if not os.path.exists(model_dir):
        return []
    return [
        f for f in os.listdir(model_dir)
        if f.endswith((".h5", ".hdf5", ".keras"))
    ]

# ─── Camera ──────────────────────────────────────────────────────────────────

CAMERA_INDEX  = 0           # 0 = webcam mặc định
FRAME_WIDTH   = 1280
FRAME_HEIGHT  = 720
CAMERA_FPS    = 30

# ─── Tiền xử lý ảnh khuôn mặt ────────────────────────────────────────────────

FACE_INPUT_SIZE = 48        # Model CNN nhận ảnh 48×48 grayscale
USE_CLAHE       = True
CLAHE_CLIP      = 2.0
CLAHE_TILE_GRID = (8, 8)

# ─── Face Detection (Haar Cascade) ───────────────────────────────────────────

HAAR_SCALE_FACTOR  = 1.1
HAAR_MIN_NEIGHBORS = 5
HAAR_MIN_SIZE      = (30, 30)

# ─── Phân loại cảm xúc ────────────────────────────────────────────────────────

# 6 nhãn theo thứ tự output của FER2013 mini-XCEPTION
# angry=0, fear=1, happy=2, sad=3, surprise=4, neutral=5
EMOTION_LABELS = ["Tức giận", "Sợ hãi", "Vui vẻ", "Buồn bã", "Ngạc nhiên", "Trung tính"]

# ─── Mức độ hài lòng (4 cấp độ) ─────────────────────────────────────────────

# Map cảm xúc → mức hài lòng ASCII (cho cv2.putText trên camera frame)
EMOTION_TO_SATISFACTION_ASCII = {
    "Vui vẻ":     "Rat hai long",
    "Ngạc nhiên": "Hai long",
    "Trung tính":  "Binh thuong",
    "Buồn bã":    "Khong hai long",
    "Sợ hãi":     "Khong hai long",
    "Tức giận":  "Khong hai long",
}

# Nhãn hiển thị tiếng Việt đầy đủ (cho Web UI & Excel Báo cáo)
EMOTION_TO_SATISFACTION_VI = {
    "Vui vẻ":     "Rất hài lòng",
    "Ngạc nhiên": "Hài lòng",
    "Trung tính":  "Bình thường",
    "Buồn bã":    "Không hài lòng",
    "Sợ hãi":     "Không hài lòng",
    "Tức giận":  "Không hài lòng",
}

# Danh sách 4 mức hài lòng (dùng cho Web UI & thống kê)
SATISFACTION_LEVELS = [
    "Rất hài lòng",
    "Hài lòng",
    "Bình thường",
    "Không hài lòng",
]

# Màu hex cho 4 mức hài lòng (web UI)
SATISFACTION_COLORS_HEX = {
    "Rất hài lòng":   "#22c55e",
    "Hài lòng":        "#38bdf8",
    "Bình thường":    "#94a3b8",
    "Không hài lòng": "#ef4444",
}

# Màu BGR cho OpenCV (camera frame)
SATISFACTION_COLORS_BGR = {
    "Rat hai long":   (34,  197, 34 ),
    "Hai long":       (248, 189, 56 ),
    "Binh thuong":    (184, 163, 148),
    "Khong hai long": (68,  68,  239),
}

# Legacy Emotion color map fallback
EMOTION_COLORS_HEX = SATISFACTION_COLORS_HEX
EMOTION_COLORS_BGR = {
    "Tức giận":   (0,   0,   239),
    "Sợ hãi":     (0,   115, 249),
    "Vui vẻ":     (34,  197, 34 ),
    "Buồn bã":    (250, 165, 96 ),
    "Ngạc nhiên": (172, 239, 134),
    "Trung tính": (184, 163, 148),
}

# Nhóm cảm xúc (để hiển thị màu frame viền)
POSITIVE_EMOTIONS = {"Vui vẻ", "Ngạc nhiên"}
NEGATIVE_EMOTIONS = {"Tức giận", "Sợ hãi", "Buồn bã"}

# ─── Logic ghi nhận 5 giây ────────────────────────────────────────────────────

CYCLE_DURATION_SEC   = 5    # Độ dài mỗi chu kỳ quan sát (giây)
MIN_FRAMES_TO_RECORD = 10   # Số frame tối thiểu để ghi nhận (tránh ghi khi vừa xuất hiện)

# ─── Google Sheets ────────────────────────────────────────────────────────────

GOOGLE_SHEET_NAME    = "EmotionRecognitionDemo"  # Tên Google Sheet
GOOGLE_WORKSHEET     = "Records"                  # Tên worksheet

# ─── Flask Server ────────────────────────────────────────────────────────────

FLASK_HOST  = "0.0.0.0"
FLASK_PORT  = 5000
FLASK_DEBUG = False

# ─── MJPEG Stream ────────────────────────────────────────────────────────────

STREAM_QUALITY = 85     # JPEG quality (0–100)
STREAM_FPS     = 25     # Max FPS cho MJPEG stream
