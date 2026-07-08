import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==========================
# Project Paths
# ==========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")
TEST_DIR = os.path.join(BASE_DIR, "test")

IMG_SIZE = (48, 48)
BATCH_SIZE = 32

# ==========================
# Training Data Augmentation
# ==========================

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode="nearest"
)

# ==========================
# Validation & Test
# ==========================

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

# ==========================
# Training Generator
# ==========================

train_generator = train_datagen.flow_from_directory(
    directory=TRAIN_DIR,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

# ==========================
# Validation Generator
# ==========================

val_generator = test_datagen.flow_from_directory(
    directory=VAL_DIR,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# ==========================
# Test Generator
# ==========================

test_generator = test_datagen.flow_from_directory(
    directory=TEST_DIR,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# ==========================
# Display Class Information
# ==========================

print("\nDataset Loaded Successfully!")
print("Training Images :", train_generator.samples)
print("Validation Images :", val_generator.samples)
print("Testing Images :", test_generator.samples)

print("\nEmotion Classes")
print(train_generator.class_indices)