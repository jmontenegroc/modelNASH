import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

MODEL_PATH = "../../models/autoencoderNASH.keras"

model = load_model(MODEL_PATH)
THRESHOLD = 0.0026013339830656557

def calculate_reconstruction_error(original, reconstructed):
    reconstructed = reconstructed.squeeze()
    return np.mean((original - reconstructed) ** 2)

def preprocess_image(image):
    image = image.convert("L")
    image = image.resize((320,180))
    image = np.array(image) 
    image = 1 - (image / 255.0)
    image = np.expand_dims(image, axis=0)
    return image

st.title("Detección de Anomalías en Imágenes de Ecografías")
st.write("Sube una imagen de ecografía para analizar si contiene anomalías.")
class_labels = {0: "Normal", 1: "Anomálo"}

uploaded_file = st.file_uploader("Sube tu imagen aquí (formatos admitidos: .jpg, .png, .jpeg)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        st.image(uploaded_file, caption="Imagen cargada", use_column_width=True)
        image = Image.open(uploaded_file)
        input_image = preprocess_image(image)
        
        prediction = model.predict(input_image)
        predicted_class = np.argmax(prediction)

        st.write("## Resultados")
        reconstruction_error = calculate_reconstruction_error(input_image.squeeze(), prediction)
        
        is_anomalous = reconstruction_error > THRESHOLD
        
        st.write(f"**Error de reconstrucción:** {reconstruction_error:.6f}")
        st.write(f"**Clasificación:** {'Anómala' if is_anomalous else 'Normal'}")
        
        reconstructed_display = (prediction.squeeze() * 255).astype(np.uint8)
        st.image(reconstructed_display, caption="Imagen reconstruida", use_column_width=True)
        
    except Exception as e:
        st.error(f"Hubo un error procesando la imagen: {e}")