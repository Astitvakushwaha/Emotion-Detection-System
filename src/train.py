print("Train.py started")

# ==========================
# Imports
# ==========================

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)

from preprocess import train_generator, val_generator
print("✓ preprocess imported")

from model import build_model
print("✓ model imported")

# ==========================
# Build Model
# ==========================

model = build_model()
print("✓ model built")

# ==========================
# Compile Model
# ==========================

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("✓ model compiled")

# ==========================
# Show Summary
# ==========================

model.summary()

print("✓ summary printed")

# ==========================
# Callbacks
# ==========================

checkpoint = ModelCheckpoint(
    filepath="models/emotion_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

# ==========================
# Train Model
# ==========================

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    callbacks=[
        checkpoint,
        early_stop,
        reduce_lr
    ]
)

# ==========================
# Save Final Model
# ==========================

model.save("models/emotion_model.keras")

print("\n✅ Training Completed Successfully!")