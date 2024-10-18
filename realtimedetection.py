import cv2
import numpy as np
from keras.models import model_from_json
from keras import Sequential  # Explicitly import Sequential

# Load the model architecture and weights
with open("emotiondetector.json", "r") as json_file:
    model_json = json_file.read()

# Make sure to register custom layers if any (not necessary for standard layers)
model = model_from_json(model_json)

# Load model weights
model.load_weights("emotiondetector.h5")

# Load Haar Cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Function to preprocess the input image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)  # Ensure shape matches model input
    return feature / 255.0  # Normalize pixel values

# Open webcam
webcam = cv2.VideoCapture(0)
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

while True:
    i, im = webcam.read()  # Read frame from webcam
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw rectangle around face
        roi_gray = gray[y:y + h, x:x + w]  # Region of interest
        roi_gray = cv2.resize(roi_gray, (48, 48))  # Resize to model input size
        features = extract_features(roi_gray)  # Extract features for prediction
        predictions = model.predict(features)  # Get model predictions
        max_index = np.argmax(predictions[0])  # Get index of highest probability
        emotion = labels[max_index]  # Map index to emotion label

        # Display the emotion label on the frame
        cv2.putText(im, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    cv2.imshow('Facial Emotion Recognition', im)  # Show the video feed

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close windows
webcam.release()
cv2.destroyAllWindows()
