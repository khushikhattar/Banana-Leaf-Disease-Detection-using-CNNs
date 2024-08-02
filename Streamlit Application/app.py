import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array, load_img # type: ignore
from PIL import Image

# Function to load models
def load_model_safely(model_path):
    try:
        model = load_model(model_path)
        print(f"Model '{model_path}' loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model '{model_path}': {e}")
        return None

# Load models
model_lenet = load_model_safely('model_lenet.keras')
model_resnet = load_model_safely('model_resnet.keras')

# Class mapping
class_mapping = {
    0: "Banana Black Sigatoka Disease",
    1: "Banana Bract Mosaic Virus Disease",
    2: "Banana Healthy Leaf",
    3: "Banana Insect Pest Disease",
    4: "Banana Moko Disease",
    5: "Banana Panama Disease",
    6: "Banana Yellow Sigatoka Disease"
}

def preprocess_image(image, target_size):
    img = image.resize(target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def predict_image(image, model):
    if model is None:
        return "Model not loaded properly"
    img_array = preprocess_image(image, (128, 128))
    pred = model.predict(img_array)
    class_idx = np.argmax(pred, axis=1)[0]
    class_label = class_mapping[class_idx]
    return class_label

st.title("Banana Leaf Disease Classification")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

model_option = st.selectbox("Select Model", ("LeNet", "ResNet50"))

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    st.write("")
    st.write("Classifying...")
    
    if model_option == "LeNet":
        model = model_lenet
    else:
        model = model_resnet
    
    label = predict_image(image, model)
    st.write(f"Prediction: {label}")
