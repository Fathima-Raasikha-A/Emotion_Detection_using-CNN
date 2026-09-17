import streamlit as st
from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn as nn
from transformers import pipeline

st.title("Emotion Detection from Uploaded Images")
st.write("Upload a photo of a face — this app compares my own CNN (PyTorch) against a pretrained Hugging Face model.")

# --- Define the same CNN architecture used during training ---
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 6 * 6, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

emotion_labels = {
    0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy',
    4: 'Sad', 5: 'Surprise', 6: 'Neutral'
}

@st.cache_resource
def load_pytorch_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionCNN(num_classes=7)
    model.load_state_dict(torch.load('emotion_model_pytorch.pth', map_location=device))
    model.to(device)
    model.eval()
    return model, device

@st.cache_resource
def load_pretrained_classifier():
    return pipeline("image-classification", model="ChristopherLi/vit-fer2013-emotion")

pytorch_model, device = load_pytorch_model()
pretrained_classifier = load_pretrained_classifier()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_array = np.array(image)
    gray_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        st.write("No face detected. Please try a clearer photo.")
    else:
        st.write(f"Found {len(faces)} face(s).")

        for (x, y, w, h) in faces:
            # --- Added padding so the crop isn't too tight ---
            padding = int(0.2 * w)
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(img_array.shape[1], x + w + padding)
            y2 = min(img_array.shape[0], y + h + padding)

            face_crop_color = image.crop((x1, y1, x2, y2))
            face_crop_gray = gray_image[y1:y2, x1:x2]

            st.image(face_crop_color, caption="Detected face", width=150)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("My CNN (PyTorch)")

                face_resized = cv2.resize(face_crop_gray, (48, 48))
                # --- Added contrast normalization ---
                face_resized = cv2.equalizeHist(face_resized)

                face_input = face_resized.astype('float32') / 255.0
                face_tensor = torch.tensor(face_input).unsqueeze(0).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = pytorch_model(face_tensor)
                    probs = torch.softmax(output, dim=1)
                    predicted_class = torch.argmax(probs, dim=1).item()
                    confidence = torch.max(probs).item() * 100

                st.write(f"**{emotion_labels[predicted_class]}** ({confidence:.1f}%)")

            with col2:
                st.subheader("Pretrained ViT model")

                results = pretrained_classifier(face_crop_color)
                top_result = results[0]

                st.write(f"**{top_result['label']}** ({top_result['score']*100:.1f}%)")

            with st.expander("See full breakdown from both models"):
                st.write("**My CNN — all 7 emotions:**")
                for i, score in enumerate(probs[0]):
                    st.write(f"{emotion_labels[i]}: {score.item()*100:.1f}%")

                st.write("**Pretrained model — all emotions:**")
                for r in results:
                    st.write(f"{r['label']}: {r['score']*100:.1f}%")
else:
    st.write("Please upload an image to get started.")