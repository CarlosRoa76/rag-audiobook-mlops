
import os
import asyncio
from src.components.data_ingestion import DataIngestion
from src.components.text_transformer import TextTransformer
from src.components.audio_generator import AudioGenerator

async def run_pipeline(pdf_filename: str):
    print(f"🚀 Iniciando Pipeline Multimodal para: {pdf_filename}")
    RAW_DIR = "data/raw"
    PROCESSED_DIR = "data/processed"
    CLEAN_DIR = "data/clean_text"
    AUDIO_DIR = "data/audio_output"
    
    try:
        print("\n--- PASO 1: Ingesta de Datos ---")
        ingestion = DataIngestion(raw_data_dir=RAW_DIR, processed_data_dir=PROCESSED_DIR)
        md_path = ingestion.extract_text_from_pdf(pdf_filename)
        md_filename = os.path.basename(md_path)
        
        print("\n--- PASO 2: Transformación Semántica ---")
        transformer = TextTransformer(processed_data_dir=PROCESSED_DIR, clean_data_dir=CLEAN_DIR)
        txt_path = transformer.clean_text(md_filename)
        txt_filename = os.path.basename(txt_path)
        
        print("\n--- PASO 3: Generación de Audio ---")
        generator = AudioGenerator(clean_data_dir=CLEAN_DIR, audio_output_dir=AUDIO_DIR)
        audio_path = await generator.generate_audio(txt_filename)
        
        print(f"\n Audiolibro listo en: {audio_path}")
        
    except Exception as e:
        print(f"\n El pipeline falló: {e}")


if __name__ == "__main__":
    RAW_DIR = "data/raw"
    
    if os.path.exists(RAW_DIR):
        raw_files = [f for f in os.listdir(RAW_DIR) if f.endswith('.pdf')]
        
        if raw_files:
            asyncio.run(run_pipeline(raw_files[0]))
        else:
            print(f"No hay archivos PDF en la carpeta {RAW_DIR}.")
    else:
        print("La estructura de carpetas de datos no existe aún.")
