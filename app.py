# app.py
import streamlit as st
from ultralytics import YOLO
from PIL import Image

# 1️⃣ Cargar modelo entrenado
model = YOLO("best.pt")  # Asegúrate que best.pt esté en la misma carpeta

# 2️⃣ Título y descripción
st.title("Detector de Pisco")
st.write("Sube una imagen y el modelo detectará si hay pisco u otros objetos.")

# 3️⃣ Subida de imagen
uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", width=600)

    # 4️⃣ Inferencia
    results = model(image)

    # 5️⃣ Mostrar imagen con detecciones
    annotated_image = results[0].plot()
    st.image(annotated_image, caption="Resultados", width=600)

    # 6️⃣ Mostrar métricas simples por clase
    metrics = results[0].boxes.data if results[0].boxes is not None else None
    st.write("Número de objetos detectados:", len(metrics) if metrics is not None else 0)