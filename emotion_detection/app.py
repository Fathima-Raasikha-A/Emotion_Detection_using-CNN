import streamlit as st
from PIL import Image
import numpy as np
import cv2
from tensorflow.keras.models import load_model

st.title("Emotion Detection from Uploaded Images")
st.write("Upload a photo of a face, and I'll try to guess the emotion.")

# Load the trained model once when the app starts
model = load_model('emotion_model.h5')

# Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

emotion_labels = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and show the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert to a format OpenCV can work with
    img_array = np.array(image)
    gray_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        st.write("No face detected. Please try a clearer photo.")
    else:
        st.write(f"Found {len(faces)} face(s).")

        for (x, y, w, h) in faces:
            # Crop just the face region
            face_crop = gray_image[y:y+h, x:x+w]

            # Resize to 48x48 to match what the model expects
            face_resized = cv2.resize(face_crop, (48, 48))

            # Scale pixel values and reshape for the model
            face_input = face_resized.astype('float32') / 255.0
            face_input = face_input.reshape(1, 48, 48, 1)

            # Predict emotion
            prediction = model.predict(face_input)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction) * 100

            st.write(f"**Predicted Emotion:** {emotion_labels[predicted_class]} ({confidence:.1f}% confidence)")
else:
    st.write("Please upload an image to get started.")