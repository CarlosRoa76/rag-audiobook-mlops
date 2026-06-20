import os
import time
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

class TextTransformer:
    def __init__(self, processed_data_dir: str, clean_data_dir: str):
        self.processed_data_dir = Path(processed_data_dir)
        self.clean_data_dir = Path(clean_data_dir)
        self.clean_data_dir.mkdir(parents=True, exist_ok=True)
        
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "Eres un experto editor de audiolibros. Tu única tarea es tomar un documento en formato Markdown y convertirlo en un guion de texto plano, fluido y natural, listo para ser leído por un motor de Text-to-Speech.\n"
                       "Sigue estas reglas ESTRICTAMENTE:\n"
                       "1. Elimina toda la sintaxis Markdown (asteriscos, numerales, etiquetas HTML, etc.).\n"
                       "2. Omite por completo encabezados repetitivos, códigos de documento, versiones y números de página.\n"
                       "3. Si encuentras una tabla, resúmela en una frase en lenguaje natural.\n"
                       "4. No agregues saludos, introducciones ni comentarios tuyos. Devuelve ÚNICAMENTE el texto limpio."),
            ("human", "Transforma el siguiente texto:\n\n{texto_markdown}")
        ])
        
        self.chain = self.prompt | self.llm

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )

    def clean_text(self, filename: str) -> str:
        input_path = self.processed_data_dir / filename
        output_filename = filename.replace('.md', '_clean.txt')
        output_path = self.clean_data_dir / output_filename

        if not input_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {input_path}")

        print(f"Leyendo documento original: {filename}")
        with open(input_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        print("Dividiendo el documento en fragmentos")
        chunks = self.text_splitter.split_text(raw_text)
        print(f"El documento se dividió en {len(chunks)} partes.")

        full_clean_text = ""

        for i, chunk in enumerate(chunks):
            print(f"Procesando fragmento {i+1} de {len(chunks)}")
            try:
                response = self.chain.invoke({"texto_markdown": chunk})
                full_clean_text += response.content + "\n\n"
                
                if i < len(chunks) - 1:
                    # Groq es muy rápido, podemos bajar la pausa a 5 segundos
                    print("Esperando 5 segundos para enfriar la API de Groq")
                    time.sleep(5)
                    
            except Exception as e:
                print(f"Error en el fragmento {i+1}: {e}")
                print("Guardando el texto procesado hasta este punto para no perder el progreso")
                break

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_clean_text)
            
        print(f"Texto transformado exitosamente. Guardado en: {output_path}")
        return str(output_path)

if __name__ == "__main__":
    PROCESSED_DIR = "data/processed"
    CLEAN_DIR = "data/clean_text"
    
    transformer = TextTransformer(processed_data_dir=PROCESSED_DIR, clean_data_dir=CLEAN_DIR)
    md_files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith('.md')]
    
    if md_files:
        transformer.clean_text(md_files[0])
    else:
        print("No hay archivos Markdown para procesar.")
