# Real-Time Expression Detection

## Overview
This project focuses on detecting facial expressions in real-time using a webcam and a deep learning model. The model is trained to recognize expressions such as happiness, disgust, surprise, and more, based on facial images. It leverages OpenCV for real-time face detection and Keras for building and training the Convolutional Neural Network (CNN) model.

## Dataset
The dataset used for this project is from Kaggle:  
[Face Expression Recognition Dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset)

## Features
- Real-time facial expression detection using a webcam.
- Pretrained deep learning model for emotion classification.
- Utilizes OpenCV for face detection and Keras for model training and prediction.

## Tools & Libraries Required
- **Python 3.x:** Core language used for the project.
- **OpenCV:** For real-time face detection and video processing.
- **Keras & TensorFlow:** For building and training the CNN model.
- **NumPy:** For numerical operations and data handling.
- **Matplotlib:** For visualizing data, if needed for debugging.

## Key Concepts

### 1. Convolutional Neural Networks (CNNs)
CNNs are essential for image classification tasks like expression detection. Here’s a brief overview of the key components of a CNN:

- **Convolutional Layer:** Extracts features from the input image, such as edges, textures, or shapes. It uses filters (kernels) that slide over the image to produce feature maps.
- **ReLU Activation:** Applies a non-linear function to the feature maps, ensuring that the model can learn complex patterns.
- **Pooling Layer:** Reduces the spatial dimensions of the feature maps, making computation more efficient while retaining important features.
- **Fully Connected Layer:** Maps the extracted features to the final classification labels (emotions).
- **Softmax Function:** Converts the output into probabilities for each class (expression).

### 2. OpenCV
OpenCV (Open Source Computer Vision Library) is used for image processing and real-time video analysis. Here are the key concepts:

- **Face Detection:** OpenCV’s `CascadeClassifier` is used to detect faces in video frames in real time using pre-trained Haar Cascades.
- **Video Capture:** The `cv2.VideoCapture()` function is used to capture video from a webcam.
- **Image Preprocessing:** OpenCV provides functions to convert images to grayscale, resize, and normalize them before feeding into the CNN model.
- **Drawing on Frames:** OpenCV allows you to draw rectangles around detected faces and display text for predicted expressions.

### 3. Real-Time Detection Workflow
1. Capture video frames using OpenCV.
2. Convert frames to grayscale and use a Haar Cascade to detect faces.
3. Extract the face region, resize it to 48x48 pixels (the input size for the model), and preprocess the image.
4. Feed the image into the trained CNN model to predict the expression.
5. Display the prediction on the video feed by drawing rectangles around faces and adding text with the detected expression.

## Model Architecture
The CNN model used in this project consists of the following layers:

- **Convolutional Layer:** 2 layers with 32 and 64 filters respectively.
- **MaxPooling Layer:** Reduces the size of the feature maps.
- **Flatten Layer:** Converts the 2D feature maps into a 1D vector.
- **Fully Connected Layer:** Dense layer for classification.
- **Output Layer:** Softmax activation to classify the expressions.
