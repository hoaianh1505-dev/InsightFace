"""
modules/ai_processor.py — Mô-đun 2: Xử lý AI
=============================================
Gồm 2 bước:
  A. Phát hiện khuôn mặt bằng Haar Cascade (Viola-Jones)
  B. Phân loại cảm xúc bằng CNN (FER2013 mini-XCEPTION)
     Input: (1, 48, 48, 1) — ảnh xám chuẩn hóa
     Output: xác suất 6 nhãn cảm xúc

Tham khảo phương pháp (Mục 2.1 báo cáo):
  - Haar Cascade: Viola & Jones (2001) — tốc độ cao, phù hợp real-time
  - (có thể mở rộng): HOG+SVM Dalal & Triggs (2005), MTCNN Zhang et al. (2016)
"""

import cv2
import numpy as np
import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.preprocessing import preprocess_face

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN A: PHÁT HIỆN KHUÔN MẶT — Haar Cascade
# ══════════════════════════════════════════════════════════════════════════════

class HaarCascadeDetector:
    """
    Phát hiện khuôn mặt bằng Haar Cascade (Viola-Jones Algorithm, 2001).

    Ưu điểm:
      - Tốc độ rất nhanh, phù hợp real-time (30+ FPS)
      - Không cần GPU, tích hợp sẵn trong OpenCV
      - Không cần cài thêm thư viện

    Nhược điểm so với deep learning:
      - Tỉ lệ false positive cao hơn
      - Kém hiệu quả với khuôn mặt nghiêng góc lớn (>45°) hoặc bị che khuất
    """

    def __init__(self):
        # Ưu tiên file XML từ config, nếu không có dùng OpenCV built-in
        builtin = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(builtin)

        if self._cascade.empty():
            raise RuntimeError("Không thể tải Haar Cascade classifier!")

        logger.info("Haar Cascade (Viola-Jones) khởi động thành công.")

    def detect(self, frame: np.ndarray) -> list[tuple]:
        """
        Phát hiện khuôn mặt trong frame (Tối ưu hóa luồng HD với Auto-Scaling).

        Parameters
        ----------
        frame : np.ndarray — Frame BGR từ camera (HD 720p / 1080p)

        Returns
        -------
        list of (x, y, w, h) — Danh sách bounding box trên kích thước gốc
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Tối ưu luồng HD: Nếu ảnh có chiều rộng > 640px, thu nhỏ ảnh xám để detect tốc độ cao (tăng 3-4x FPS)
        max_dim = 640
        if w > max_dim:
            scale = max_dim / float(w)
            small_w = max_dim
            small_h = int(h * scale)
            small_gray = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA)

            min_size = (max(10, int(config.HAAR_MIN_SIZE[0] * scale)),
                        max(10, int(config.HAAR_MIN_SIZE[1] * scale)))

            faces = self._cascade.detectMultiScale(
                small_gray,
                scaleFactor=config.HAAR_SCALE_FACTOR,
                minNeighbors=config.HAAR_MIN_NEIGHBORS,
                minSize=min_size,
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
            if len(faces) == 0:
                return []

            # Ánh xạ tọa độ bounding box ngược về kích thước HD gốc
            inv_scale = 1.0 / scale
            return [
                (int(x * inv_scale), int(y * inv_scale), int(w_b * inv_scale), int(h_b * inv_scale))
                for x, y, w_b, h_b in faces
            ]
        else:
            faces = self._cascade.detectMultiScale(
                gray,
                scaleFactor=config.HAAR_SCALE_FACTOR,
                minNeighbors=config.HAAR_MIN_NEIGHBORS,
                minSize=config.HAAR_MIN_SIZE,
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
            if len(faces) == 0:
                return []
            return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN B: PHÂN LOẠI CẢM XÚC — CNN FER2013
# ══════════════════════════════════════════════════════════════════════════════

class EmotionClassifier:
    """
    Phân loại cảm xúc bằng mô hình CNN mini-XCEPTION huấn luyện trên FER2013.

    Kiến trúc mini-XCEPTION (Arriaga et al.):
      - Input:  (48, 48, 1) — ảnh xám chuẩn hóa về [0, 1]
      - Blocks: Depthwise Separable Conv (XCEPTION-style)
      - Output: Softmax 6 lớp
        [0] Tức giận, [1] Sợ hãi, [2] Vui vẻ,
        [3] Buồn bã,  [4] Ngạc nhiên, [5] Trung tính
    """

    def __init__(self, model_path: str = None):
        self._model  = None
        self._labels = config.EMOTION_LABELS
        self.current_model_name = "DEMO (random)"
        self._load_or_download(model_path or config.EMOTION_MODEL_PATH)

    def _load_or_download(self, model_path: str):
        """Tải model từ file, hoặc tải tự động từ internet."""
        model_dir = os.path.join(config.MODELS_DIR, "emotion_model")

        # Các đường dẫn thử theo thứ tự (mặc định fer2013_mini_XCEPTION.hdf5)
        candidates = [
            os.path.join(model_dir, "fer2013_mini_XCEPTION.hdf5"),
            os.path.join(model_dir, "my_custom_model.h5"),
            model_path,
            os.path.join(model_dir, "emotion_model.h5"),
            os.path.join(model_dir, "fer2013_mini_XCEPTION.102-0.66.hdf5"),
        ]

        if os.path.exists(model_dir):
            for f in os.listdir(model_dir):
                if f.endswith((".h5", ".hdf5", ".keras")):
                    full_path = os.path.join(model_dir, f)
                    if full_path not in candidates:
                        candidates.append(full_path)

        for path in candidates:
            if path and os.path.exists(path):
                self._load_model(path)
                if self._model is not None:
                    return

        # Không tìm thấy → tải tự động
        logger.info("Model chưa có local. Đang tải FER2013 pre-trained...")
        self._download_model()

    def _load_model(self, path: str):
        try:
            import tensorflow as tf
            self._model = tf.keras.models.load_model(path, compile=False)
            self.current_model_name = os.path.basename(path)
            logger.info(f"✅ Model CNN tải thành công: {self.current_model_name}")
        except Exception as e:
            logger.error(f"Lỗi tải model từ {path}: {e}")
            self._model = None

    def switch_model(self, model_filename: str) -> bool:
        """Đổi sang mô hình khác trong thư mục models/emotion_model/."""
        if not model_filename.endswith((".h5", ".hdf5", ".keras")):
            logger.error("Định dạng file model không hợp lệ (cần .h5 / .hdf5 / .keras)")
            return False

        full_path = os.path.join(config.MODELS_DIR, "emotion_model", model_filename)
        if not os.path.exists(full_path):
            if os.path.exists(model_filename):
                full_path = model_filename
            else:
                logger.error(f"File model không tồn tại: {full_path}")
                return False

        try:
            import tensorflow as tf
            logger.info(f"Đang chuyển sang model: {model_filename}...")
            new_model = tf.keras.models.load_model(full_path, compile=False)
            self._model = new_model
            self.current_model_name = os.path.basename(full_path)
            logger.info(f"✅ Đã chuyển đổi model thành công sang: {self.current_model_name}")
            return True
        except Exception as e:
            logger.error(f"Không thể tải model {full_path}: {e}")
            return False

    def _download_model(self):
        """Tai model FER2013 pre-trained. Thu nhieu URL."""
        save_dir  = os.path.join(config.MODELS_DIR, "emotion_model")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "fer2013_mini_XCEPTION.102-0.66.hdf5")

        # Xoa file corrupt neu co (nho hon 1MB thi suspect)
        if os.path.exists(save_path):
            if os.path.getsize(save_path) < 1_000_000:
                os.remove(save_path)
                logger.warning("Xoa file model bi corrupt, tai lai...")
            else:
                self._load_model(save_path)
                return

        # Danh sach URL thu theo thu tu
        urls = [
            config.EMOTION_MODEL_URL,
            "https://huggingface.co/spaces/ElSheikh/Facial-Emotion-Recognition/resolve/main/model.h5",
        ]

        for url in urls:
            try:
                import requests
                from tqdm import tqdm
                logger.info(f"Dang tai model tu: {url}")
                r = requests.get(url, stream=True, timeout=60,
                                 headers={"User-Agent": "Mozilla/5.0"})
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                with open(save_path, "wb") as f, tqdm(
                    total=total, unit="B", unit_scale=True, desc="Tai model"
                ) as bar:
                    for chunk in r.iter_content(8192):
                        f.write(chunk)
                        bar.update(len(chunk))

                # Kiem tra file size
                if os.path.getsize(save_path) > 1_000_000:
                    self._load_model(save_path)
                    if self._model is not None:
                        return
                else:
                    os.remove(save_path)
                    logger.warning(f"File tai ve qua nho tu {url}, thu URL khac.")
            except Exception as e:
                logger.warning(f"Khong tai duoc tu {url}: {e}")
                if os.path.exists(save_path):
                    os.remove(save_path)

        logger.warning(
            "Khong the tai model. Chay o che do DEMO (predictions ngau nhien).\n"
            "De dung model that, tai thu cong va dat vao:\n"
            f"  {save_path}"
        )

    def predict(self, input_tensor: np.ndarray) -> tuple[str, float, np.ndarray]:
        """
        Dự đoán cảm xúc.

        Parameters
        ----------
        input_tensor : np.ndarray shape (1, 48, 48, 1)

        Returns
        -------
        (label, confidence, probabilities)
        """
        if self._model is None:
            # Demo mode: random (seed để ổn định trong demo)
            np.random.seed(int(abs(np.sum(input_tensor)) * 1000) % 2**31)
            probs = np.random.dirichlet(np.ones(6) * 0.5)
            idx   = int(np.argmax(probs))
            return self._labels[idx], float(probs[idx]), probs

        try:
            # Tự động điều chỉnh kích thước & chuẩn hóa theo input_shape của model đang nạp
            model_tensor = input_tensor
            if hasattr(self._model, "input_shape") and self._model.input_shape:
                req_shape = self._model.input_shape
                if len(req_shape) == 4 and req_shape[1] is not None and req_shape[2] is not None:
                    target_h, target_w = req_shape[1], req_shape[2]
                    # Nếu model yêu cầu 64x64 (ví dụ fer2013_mini_XCEPTION)
                    if input_tensor.shape[1] != target_h or input_tensor.shape[2] != target_w:
                        sq_img = input_tensor[0, :, :, 0]
                        resized = cv2.resize(sq_img, (target_w, target_h), interpolation=cv2.INTER_AREA)
                        model_tensor = resized.reshape(1, target_h, target_w, 1)

                    # Chuẩn hóa về [-1, 1] nếu là model pre-trained 64x64
                    if target_h == 64:
                        model_tensor = (model_tensor - 0.5) * 2.0

            probs = self._model.predict(model_tensor, verbose=0)[0]
            idx   = int(np.argmax(probs))

            # Tự động phát hiện số lớp output (4 lớp hài lòng, 6 lớp cảm xúc, 7 lớp FER2013)
            if len(probs) == 4:
                # Keras image_dataset_from_directory tự động sắp xếp tên thư mục theo ABC:
                # 0: binh_thuong     -> Bình thường
                # 1: hai_long        -> Hài lòng
                # 2: khong_hai_long  -> Không hài lòng
                # 3: rat_hai_long    -> Rất hài lòng
                labels_4 = ["Bình thường", "Hài lòng", "Không hài lòng", "Rất hài lòng"]
                return labels_4[idx], float(probs[idx]), probs
            elif len(probs) == 7:
                # FER2013 mini-XCEPTION 7 classes: 0:angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
                labels_7 = ["Tức giận", "Tức giận", "Sợ hãi", "Vui vẻ", "Buồn bã", "Ngạc nhiên", "Trung tính"]
                return labels_7[idx], float(probs[idx]), probs
            elif len(probs) == 6:
                labels_6 = ["Tức giận", "Sợ hãi", "Vui vẻ", "Buồn bã", "Ngạc nhiên", "Trung tính"]
                return labels_6[idx], float(probs[idx]), probs
            elif idx < len(self._labels):
                return self._labels[idx], float(probs[idx]), probs
            else:
                return f"Class_{idx}", float(probs[idx]), probs
        except Exception as e:
            logger.error(f"Lỗi predict: {e}")
            probs = np.ones(len(self._labels), dtype=np.float32) / len(self._labels)
            return self._labels[0], 1.0 / len(self._labels), probs

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def labels(self) -> list[str]:
        return self._labels


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE TỔNG HỢP
# ══════════════════════════════════════════════════════════════════════════════

class AIProcessor:
    """
    Kết hợp HaarCascadeDetector + EmotionClassifier.
    Nhận frame thô → trả về danh sách kết quả phân tích mỗi khuôn mặt.
    """

    def __init__(self, model_path: str = None):
        logger.info("Khởi động AIProcessor...")
        self.face_detector = HaarCascadeDetector()
        self.classifier    = EmotionClassifier(model_path)

    def process_frame(self, frame: np.ndarray) -> list[dict]:
        """
        Xử lý một frame: detect → tiền xử lý → phân loại.

        Returns
        -------
        list of dict:
          {
            "bbox":          (x, y, w, h),
            "emotion":       "Vui vẻ",
            "confidence":    0.87,
            "probabilities": np.ndarray(6,),
          }
        """
        bboxes  = self.face_detector.detect(frame)
        results = []

        for bbox in bboxes:
            tensor = preprocess_face(frame, bbox)
            if tensor is None:
                continue
            emotion, confidence, probs = self.classifier.predict(tensor)
            results.append({
                "bbox":          bbox,
                "emotion":       emotion,
                "confidence":    confidence,
                "probabilities": probs,
            })

        return results

    def switch_model(self, model_filename: str) -> bool:
        """Đổi model phân loại."""
        return self.classifier.switch_model(model_filename)

    @property
    def current_model_name(self) -> str:
        return self.classifier.current_model_name
