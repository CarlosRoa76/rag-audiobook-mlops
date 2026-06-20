import os
import pymupdf4llm
from pathlib import Path

class DataIngestion:
    def __init__(self, raw_data_dir: str, processed_data_dir: str):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

    def extract_text_from_pdf(self, pdf_filename: str) -> str:
        """
        Lee un PDF y lo convierte a formato Markdown optimizado para LLMs
        """
        pdf_path = self.raw_data_dir / pdf_filename
        output_filename = pdf_filename.replace('.pdf', '.md')
        output_path = self.processed_data_dir / output_filename

        if not pdf_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {pdf_path}")

        print(f"Iniciando extracción de: {pdf_filename}")
        
        md_text = pymupdf4llm.to_markdown(str(pdf_path))
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md_text)
            
        print(f"Extracción completada. Archivo guardado en: {output_path}")
        return str(output_path)

if __name__ == "__main__":
    # Rutas relativas desde la raíz del proyecto
    RAW_DIR = "data/raw"
    PROCESSED_DIR = "data/processed"
    
    ingestion = DataIngestion(raw_data_dir=RAW_DIR, processed_data_dir=PROCESSED_DIR)
    
    pdfs = [f for f in os.listdir(RAW_DIR) if f.endswith('.pdf')]
    
    if pdfs:
        ingestion.extract_text_from_pdf(pdfs[0])
    else:
        print("No hay PDFs en la carpeta data/raw/. Por favor, agrega uno para probar.")
