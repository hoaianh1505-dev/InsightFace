"""
train_custom_model.py — Huấn luyện Mô hình CNN 4 Mức Hài Lòng
========================================================================
Cấu trúc thư mục dữ liệu tự động tạo tại: dataset/

dataset/
├── train/
│   ├── rat_hai_long/     (1: Rất hài lòng)
│   ├── hai_long/         (2: Hài lòng)
│   ├── binh_thuong/      (3: Bình thường)
│   └── khong_hai_long/   (4: Không hài lòng)
└── val/
    ├── rat_hai_long/
    ├── hai_long/
    ├── binh_thuong/
    └── khong_hai_long/
"""

import os
import sys
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
SAVE_DIR = os.path.join(BASE_DIR, "models", "emotion_model")
MODEL_OUTPUT_PATH = os.path.join(SAVE_DIR, "my_custom_model.h5")

OLD_CLASSES = ["angry", "fear", "happy", "sad", "surprise", "neutral"]
CLASSES     = ["rat_hai_long", "hai_long", "binh_thuong", "khong_hai_long"]

def create_dataset_structure():
    """Tạo sẵn cấu trúc 4 thư mục mức độ hài lòng, dọn dẹp các thư mục cũ nếu trống."""
    for split in ["train", "val"]:
        # Dọn dẹp thư mục 6 nhãn cũ nếu trống
        for old_cls in OLD_CLASSES:
            old_folder = os.path.join(DATASET_DIR, split, old_cls)
            if os.path.exists(old_folder):
                try:
                    if len(os.listdir(old_folder)) == 0:
                        os.rmdir(old_folder)
                except Exception:
                    pass

        # Tạo 4 thư mục mới
        for cls in CLASSES:
            folder = os.path.join(DATASET_DIR, split, cls)
            os.makedirs(folder, exist_ok=True)

def build_model(input_shape=(48, 48, 1), num_classes=4):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Rescaling(1./255),

        # Data Augmentation
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),

        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Dense Classifier
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def count_images(directory):
    total = 0
    if not os.path.exists(directory):
        return 0
    for root, _, files in os.walk(directory):
        total += len([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    return total

def auto_crop_faces():
    """Tự động phát hiện và crop khuôn mặt từ các ảnh chưa được crop trong dataset."""
    builtin = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(builtin)
    if face_cascade.empty():
        return

    cropped_count = 0
    print("🔍 Đang tự động quét & crop khuôn mặt từ các ảnh chụp toàn thân/khung hình lớn...")

    for root, _, files in os.walk(DATASET_DIR):
        for fname in files:
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = os.path.join(root, fname)
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue

                    h, w = img.shape[:2]
                    # Nếu ảnh đã là ảnh crop nhỏ (<= 128x128) thì bỏ qua
                    if h <= 128 and w <= 128:
                        continue

                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.1,
                        minNeighbors=4,
                        minSize=(30, 30)
                    )

                    if len(faces) > 0:
                        # Lấy khuôn mặt lớn nhất
                        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
                        pad = int(min(fw, fh) * 0.1)
                        x1 = max(0, x - pad)
                        y1 = max(0, y - pad)
                        x2 = min(w, x + fw + pad)
                        y2 = min(h, y + fh + pad)
                        face_crop = img[y1:y2, x1:x2]
                        if face_crop.size > 0:
                            cv2.imwrite(img_path, face_crop)
                            cropped_count += 1
                except Exception:
                    pass

    if cropped_count > 0:
        print(f"✅ Đã tự động trích xuất và crop thành công {cropped_count} khuôn mặt!")

def main():
    print("=" * 60)
    print("HUAN LUYEN MO HINH CNN TU THU MUC ANH (DATASET)")
    print("=" * 60)

    create_dataset_structure()
    auto_crop_faces()
    train_count = count_images(TRAIN_DIR)
    val_count = count_images(VAL_DIR)

    print("\nThu muc Dataset: ./dataset")
    print(f"- So anh tap Train: {train_count} anh")
    print(f"- So anh tap Val:   {val_count} anh")

    if train_count == 0:
        print("\n[THONG BAO] Thu muc dataset/train hien dang trong!")
        print("Da tao san cau truc thu muc chia theo nhan:")
        print(f"  dataset/train/ ({', '.join(CLASSES)})")
        print(f"  dataset/val/   ({', '.join(CLASSES)})")
        print("\nHuong dan:")
        print("1. Copy anh khuon mat (JPG/PNG) vao cac thu muc tuong ung.")
        print("2. Chay lai lenh: python train_custom_model.py")
        print("=" * 60)
        return

    # Load dataset bang Keras image_dataset_from_directory
    batch_size = 32
    img_size = (48, 48)

    if val_count > 0:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            TRAIN_DIR,
            color_mode="grayscale",
            image_size=img_size,
            batch_size=batch_size,
            label_mode="categorical",
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            VAL_DIR,
            color_mode="grayscale",
            image_size=img_size,
            batch_size=batch_size,
            label_mode="categorical",
        )
    else:
        print("Tự động chia 80% Train / 20% Validation từ tập train...")
        train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
            TRAIN_DIR,
            color_mode="grayscale",
            image_size=img_size,
            batch_size=batch_size,
            label_mode="categorical",
            validation_split=0.2,
            subset="both",
            seed=42,
        )

    # Build & compile
    model = build_model(input_shape=(48, 48, 1), num_classes=len(CLASSES))
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    epochs = 15
    print(f"\nDang huan luyen model trong {epochs} epochs...")

    if val_ds:
        model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    else:
        model.fit(train_ds, epochs=epochs)

    os.makedirs(SAVE_DIR, exist_ok=True)
    model.save(MODEL_OUTPUT_PATH)

    print("\n[OK] Huan luyen thanh cong!")
    print(f"-> Model moi da luu tai: models/emotion_model/my_custom_model.h5")
    print("=" * 60)

if __name__ == "__main__":
    main()
