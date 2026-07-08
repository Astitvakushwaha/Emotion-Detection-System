import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

from preprocess import test_generator

# Load trained model
model = load_model("models/emotion_model.keras")

# Predict
predictions = model.predict(test_generator)

predicted_classes = np.argmax(predictions, axis=1)

true_classes = test_generator.classes

print("\nClassification Report\n")

print(
    classification_report(
        true_classes,
        predicted_classes,
        target_names=list(test_generator.class_indices.keys())
    )
)

print("\nConfusion Matrix\n")

print(confusion_matrix(true_classes, predicted_classes))