import streamlit as st
import numpy as np
from PIL import Image

import os

try:
    from tensorflow.keras.models import load_model
except ImportError:
    os.system("pip install tensorflow")

MODEL_PATH = "modelDeploy/vgg/NASH_vggwDA.keras"


model = load_model(MODEL_PATH)

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((180,320))
    image = np.expand_dims(image, axis=0)
    return image

st.title("Detección de Anomalías en Imágenes de Ecografías")
st.write("Sube una imagen de ecografía para analizar si contiene anomalías.")
class_labels = {0: "Normal", 1: "Anómalo"}

uploaded_file = st.file_uploader("Sube tu imagen aquí (formatos admitidos: .jpg, .png, .jpeg)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        st.image(uploaded_file, caption="Imagen cargada", use_container_width=True)
        image = Image.open(uploaded_file)
        input_image = preprocess_image(image)
        
        prediction = model.predict(input_image)
        predicted_class = np.argmax(prediction)

        st.write("## Resultados")
        st.write(f"Predicción: {class_labels[predicted_class]}")
        st.write(f"Probabilidad: {prediction[0][predicted_class]*100:.2f}%")
        
    except Exception as e:
        st.error(f"Hubo un error procesando la imagen: {e}")