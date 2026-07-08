# Real-Time Emotion Recognition using CNN

## Overview

This project detects human facial expressions in real time using a Convolutional Neural Network (CNN) trained on the FER2013 dataset. The webcam captures video frames, detects faces using OpenCV Haar Cascade, and predicts one of seven emotions.

## Features

- Real-time emotion recognition
- Face detection using OpenCV Haar Cascade
- CNN model built with TensorFlow/Keras
- Trained on FER2013 dataset
- Live webcam prediction

## Emotions Detected

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

## Technologies Used

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Scikit-learn

## Dataset

FER2013 Dataset

## Project Structure

```
Emotion-Detection-System/
│
├── src/
│   ├── preprocess.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── realtime.py
│
├── models/
│   └── emotion_model.keras
│
├── train/
├── val/
├── test/
│
├── README.md
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Train the Model

```bash
python src/train.py
```

## Evaluate the Model

```bash
python src/evaluate.py
```

## Run Real-Time Detection

```bash
python src/realtime.py
```

## Model

- CNN (Convolutional Neural Network)
- Input Size: 48×48 Grayscale
- Output Classes: 7 Emotions

## Author

**astitva kushwaha**