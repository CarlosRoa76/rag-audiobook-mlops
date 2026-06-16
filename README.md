# 🎧 Multimodal AI Audiobook Pipeline (End-to-End MLOps)

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.34+-FF4B4B.svg)
![LangChain](https://img.shields.io/badge/LangChain-Integration-green.svg)
![Groq](https://img.shields.io/badge/Groq-Llama_3.3-black.svg)
![DVC](https://img.shields.io/badge/DVC-Data_Versioning-orange.svg)

Un pipeline de Machine Learning multimodal y agéntico diseñado para ingerir documentos complejos (PDFs), limpiarlos semánticamente y convertirlos en audiolibros de alta fidelidad. 

Este proyecto está construido con un rigor estricto de **MLOps**, separando el código fuente de los datos pesados, y optimizado para ejecutarse en entornos locales con recursos limitados (ej. 16GB de RAM) delegando el procesamiento pesado de lenguaje natural a inferencia ultrarrápida en la nube.

## 🏗️ Arquitectura del Sistema

El pipeline sigue una estructura modular orientada a objetos:

1. **Ingesta de Datos (`DataIngestion`):** Utiliza `PyMuPDF4LLM` para extraer texto de PDFs pesados conservando la jerarquía estructural en formato Markdown, evitando la pérdida de contexto visual.
2. **Transformación Semántica (`TextTransformer`):** Implementa LangChain y el modelo **Llama 3.3 (70B)** a través de la API de Groq. Aplica *chunking* recursivo con solapamiento para procesar documentos largos sin romper ventanas de contexto ni cuotas de API, traduciendo tablas y limpiando sintaxis Markdown.
3. **Síntesis Neuronal (`AudioGenerator`):** Utiliza `edge-tts` de forma asíncrona para generar un archivo MP3 con voz hiperrealista.
4. **Interfaz de Usuario (`app.py`):** Frontend interactivo levantado en Streamlit.
5. **Control de Versiones:** Código gestionado en GitHub; datos, PDFs y audios versionados con **DVC** y respaldados remotamente en **DagsHub**.

## 📂 Estructura del Proyecto

```text
rag_audiobook_pipeline/
├── .dvc/                   # Configuración de Data Version Control
├── data/                   # Carpeta ignorada por Git, rastreada por DVC
│   ├── raw/                # PDFs originales
│   ├── processed/          # Texto en Markdown
│   ├── clean_text/         # Guiones limpios por la IA
│   └── audio_output/       # Audiolibros finales (MP3)
├── src/
│   └── components/
│       ├── data_ingestion.py
│       ├── text_transformer.py
│       └── audio_generator.py
├── .env                    # Variables de entorno (API Keys)
├── .gitignore
├── app.py                  # Interfaz Streamlit
├── main.py                 # Orquestador del backend CLI
└── requirements.txt        # Dependencias
