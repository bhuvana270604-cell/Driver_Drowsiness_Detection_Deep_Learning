import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗"
)

st.title("🚗 Driver Drowsiness Detection using Deep Learning")

st.write(
    "Upload a driver's facial image to detect drowsiness "
    "using a deep learning model."
)

# Load MobileNetV2 model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mobilenetv2_model.keras")

model = load_model()

# Class names
class_names = [
    "Closed",
    "Open",
    "No Yawn",
    "Yawn"
]

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Preprocess image
    image_resized = image.resize((224, 224))

    image_array = np.array(image_resized)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    prediction = model.predict(image_array)

    predicted_class = np.argmax(prediction[0])
    confidence = np.max(prediction[0]) * 100

    st.success("Image uploaded successfully!")

    st.subheader("Prediction")

    st.write(
        f"**Condition: {class_names[predicted_class]}**"
    )

    st.write(
        f"**Confidence: {confidence:.2f}%**" 
    )