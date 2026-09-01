import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image
import json
import os
import gdown

st.set_page_config(page_title="Clothing Classifier")

WEIGHTS_PATH = "clothing_model.weights.h5"
WEIGHTS_FILE_ID = "1zGuKL4pTq1utHrpNXBpe4_9alOPUyXf6"

def build_original_cnn():
    return models.Sequential([
        layers.Input(shape=(128, 128, 3)),
        layers.Conv2D(16, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

@st.cache_resource
def load_model():
    if not os.path.exists(WEIGHTS_PATH):
        gdown.download(f"https://drive.google.com/uc?id={WEIGHTS_FILE_ID}", WEIGHTS_PATH, quiet=False)
    model = build_original_cnn()
    model.load_weights(WEIGHTS_PATH)
    with open("class_names.json") as f:
        class_names = json.load(f)
    return model, class_names

model, class_names = load_model()

st.title("Clothing Image Classifier")
st.write("Upload an image and the model will predict its clothing category.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_resized = image.resize((128, 128))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]

    top_idx = np.argmax(predictions)
    top_class = class_names[top_idx]
    top_confidence = predictions[top_idx] * 100

    st.subheader("Prediction")
    st.write(f"**Class:** {top_class}")
    st.write(f"**Confidence:** {top_confidence:.2f}%")

    st.subheader("Top-3 Predictions")
    top3_idx = np.argsort(predictions)[-3:][::-1]
    for idx in top3_idx:
        st.write(f"{class_names[idx]}: {predictions[idx]*100:.2f}%")
