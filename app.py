import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Cargar modelo
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("final_mobilenetv2_model.keras")  # o .h5
    return model

model = load_model()

# Etiquetas de clase (ajústalas según tu entrenamiento)
class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic']

# Interfaz
st.title("🗑️ Clasificador de Basura Reciclable")
st.write("Sube una imagen de basura para predecir su clase")

uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Imagen subida", use_column_width=True)

    # Preprocesamiento
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predicción
    prediction = model.predict(img_array)
    pred_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.markdown(f"### 🧠 Predicción: **{pred_class.upper()}** ({confidence:.2%} confianza)")
