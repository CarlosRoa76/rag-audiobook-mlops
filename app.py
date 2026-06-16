import os
import asyncio
import streamlit as st
from src.components.data_ingestion import DataIngestion
from src.components.text_transformer import TextTransformer
from src.components.audio_generator import AudioGenerator

# 1. Configuración de la página web
st.set_page_config(page_title="RAG Audiobook AI", page_icon="🎧", layout="centered")

st.title("🎧 Generador de Audiolibros con IA")
st.markdown("Sube un documento PDF y nuestra arquitectura MLOps lo convertirá en un audiolibro utilizando **Llama 3.3** y **Edge-TTS**.")

# 2. Definición de rutas del pipeline
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
CLEAN_DIR = "data/clean_text"
AUDIO_DIR = "data/audio_output"

# Asegurarnos de que el directorio raw exista para guardar lo que suba el usuario
os.makedirs(RAW_DIR, exist_ok=True)

# 3. Interfaz de subida de archivos
uploaded_file = st.file_uploader("Sube tu archivo PDF aquí", type=['pdf'])

if uploaded_file is not None:
    st.success(f"Archivo cargado: {uploaded_file.name}")
    
    # Botón para detonar el pipeline
    if st.button("Generar Audiolibro"):
        
        # Guardar físicamente el PDF que subió el usuario en nuestra carpeta raw
        pdf_path = os.path.join(RAW_DIR, uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        # Contenedores visuales para mostrar el progreso
        status_text = st.empty()
        progress_bar = st.progress(0)
        
        try:
            # --- PASO 1: Ingesta ---
            status_text.info("Paso 1/3: Extrayendo texto del PDF (PyMuPDF4LLM)...")
            ingestion = DataIngestion(raw_data_dir=RAW_DIR, processed_data_dir=PROCESSED_DIR)
            md_path = ingestion.extract_text_from_pdf(uploaded_file.name)
            md_filename = os.path.basename(md_path)
            progress_bar.progress(33)
            
            # --- PASO 2: Transformación ---
            status_text.info("Paso 2/3: Limpiando texto semánticamente con Llama 3.3 (Groq)...")
            transformer = TextTransformer(processed_data_dir=PROCESSED_DIR, clean_data_dir=CLEAN_DIR)
            txt_path = transformer.clean_text(md_filename)
            txt_filename = os.path.basename(txt_path)
            progress_bar.progress(66)
            
            # --- PASO 3: Audio ---
            status_text.info("Paso 3/3: Sintetizando voz neuronal (Edge-TTS)...")
            generator = AudioGenerator(clean_data_dir=CLEAN_DIR, audio_output_dir=AUDIO_DIR)
            audio_path = asyncio.run(generator.generate_audio(txt_filename))
            progress_bar.progress(100)
            
            status_text.success("Audiolibro generado")
            
            # --- PASO 4: Resultados (Reproductor y Descarga) ---
            st.audio(audio_path, format="audio/mp3")
            
            with open(audio_path, "rb") as file:
                st.download_button(
                    label="⬇️ Descargar MP3",
                    data=file,
                    file_name=uploaded_file.name.replace('.pdf', '.mp3'),
                    mime="audio/mp3"
                )
                
        except Exception as e:
            st.error(f"Ocurrió un error en el pipeline: {e}")