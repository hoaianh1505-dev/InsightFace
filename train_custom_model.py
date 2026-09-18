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

def main():
    print("=" * 60)
    print("HUAN LUYEN MO HINH CNN TU THU MUC ANH (DATASET)")
    print("=" * 60)

    create_dataset_structure()
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

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        color_mode="grayscale",
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
    )

    val_ds = None
    if val_count > 0:
        val_ds = tf.keras.utils.image_dataset_from_directory(
            VAL_DIR,
            color_mode="grayscale",
            image_size=img_size,
            batch_size=batch_size,
            label_mode="categorical",
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
