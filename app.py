import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Estilo visual para kiosko/aeropuerto
st.set_page_config(page_title="Clasificador de Basura - Aeropuerto", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f0f4f8; }
    h1, h2, h3 { font-family: 'Arial Rounded MT Bold', sans-serif; }
    .stButton>button {
        background-color: #007bff;
        color: white;
        font-size: 20px;
        border-radius: 10px;
        padding: 10px 20px;
    }
    .stRadio > div { justify-content: center; }
    </style>
""", unsafe_allow_html=True)

# Cargar modelo
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("final_mobilenetv2_model.keras")

model = load_model()
class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic']
icon_paths = {
    "cardboard": "icons/cardboard.png",
    "glass": "icons/glass.png",
    "metal": "icons/metal.png",
    "paper": "icons/paper.png",
    "plastic": "icons/plastic.png"
}

# Historial de predicciones
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar institucional
st.sidebar.image("plane.png", width=150)
st.sidebar.title("🌍 Iniciativa Ambiental")
st.sidebar.markdown("""
La propuesta plantea el uso de un sistema inteligente de clasificación de residuos en aeropuerto. 
La idea es que este dispositivo, ubicado en puntos estratégicos como zonas de control o revisión de equipaje,
pueda escanear automáticamente objetos desechados y clasificarlos correctamente. Esto no solo optimizaría la 
separación de residuos en un entorno de alto tránsito, sino que también contribuiría a mantener la limpieza y 
fomentar el reciclaje desde un enfoque automatizado y eficiente
""")
st.sidebar.title("👥 Equipo del Proyecto")
st.sidebar.markdown("""
- Ivan Jhunior Aiza Laime 
- Erika Alejandra Villarroel Zambrana  
- Alan Jesús Cáceres Medrano  
- Fernando Quinteros Gutierrez
""")

# Encabezado
st.markdown("<h1 style='text-align: center; color: green;'>♻️ Clasificador de Basura Inteligente</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Coloca un objeto reciclable frente a la cámara o súbelo como imagen</h3>", unsafe_allow_html=True)

# Modo de entrada
modo = st.radio("Selecciona el modo de entrada:", ["📁 Subir imagen", "📷 Cámara en vivo"])

image = None

# Subir imagen
if modo == "📁 Subir imagen":
    uploaded_file = st.file_uploader("Sube una imagen del residuo", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="📷 Imagen subida", use_column_width=True)

# Captura desde cámara
elif modo == "📷 Cámara en vivo":
    st.info("📡 Cámara encendida. Presiona el botón para capturar la imagen.")

    class VideoProcessor(VideoTransformerBase):
        def __init__(self):
            self.frame = None

        def transform(self, frame):
            self.frame = frame.to_ndarray(format="bgr24")
            return self.frame

    ctx = webrtc_streamer(
        key="live",
        video_transformer_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if ctx.video_transformer:
        if st.button("📸 Tomar foto"):
            frame = ctx.video_transformer.frame
            if frame is not None:
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                st.image(image, caption="📷 Imagen capturada", use_column_width=True)
            else:
                st.error("❌ No se pudo capturar imagen.")

# Clasificación
if image:
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    pred_index = np.argmax(prediction)
    pred_class = class_labels[pred_index]
    confidence = np.max(prediction)

    # Mostrar resultado con ícono grande
    st.markdown(f"<h2 style='text-align: center; color: #004d00;'>🧠 Predicción: {pred_class.upper()}</h2>", unsafe_allow_html=True)
    st.image(icon_paths[pred_class], width=150, caption=pred_class.title(), output_format='PNG')
    st.markdown(f"<h3 style='text-align: center;'>🔍 Confianza del modelo: {confidence:.1%}</h3>", unsafe_allow_html=True)

    # Gráfico de barras de predicción
    fig, ax = plt.subplots()
    bars = ax.bar(class_labels, prediction[0], color='skyblue')
    bars[pred_index].set_color('green')
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probabilidad")
    ax.set_title("Confianza por clase")
    for i, v in enumerate(prediction[0]):
        ax.text(i, v + 0.02, f"{v:.0%}", ha='center', fontsize=10)
    st.pyplot(fig)

    # Historial
    st.session_state.history.append({
        "image": image.copy(),
        "pred": pred_class,
        "conf": confidence
    })

# Mostrar historial
if st.session_state.history:
    st.markdown("## 🧾 Clasificaciones Recientes")
    cols = st.columns(3)
    for i, record in enumerate(reversed(st.session_state.history[-6:])):
        with cols[i % 3]:
            st.image(record["image"], width=200)    
            st.markdown(f"**Predicción:** {record['pred'].title()}")
            st.markdown(f"**Confianza:** {record['conf']:.1%}")
