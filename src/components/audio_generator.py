import os
import asyncio
import edge_tts
from pathlib import Path

class AudioGenerator:
    def __init__(self, clean_data_dir: str, audio_output_dir: str, voice: str = "es-ES-AlvaroNeural"):
        self.clean_data_dir = Path(clean_data_dir)
        self.audio_output_dir = Path(audio_output_dir)
        self.voice = voice
        self.audio_output_dir.mkdir(parents=True, exist_ok=True)

    async def generate_audio(self, text_filename: str) -> str:
        input_path = self.clean_data_dir / text_filename
        output_filename = text_filename.replace('_clean.txt', '.mp3')
        output_path = self.audio_output_dir / output_filename

        if not input_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {input_path}")

        print(f"Leyendo guion limpio de: {text_filename}...")
        with open(input_path, "r", encoding="utf-8") as f:
            clean_text = f.read()

        print(f"Generando audio final con '{self.voice}'")
        communicate = edge_tts.Communicate(clean_text, self.voice)
        await communicate.save(str(output_path))
        
        print(f"Audio generado exitosamente. Guardado en: {output_path}")
        return str(output_path)

if __name__ == "__main__":
    CLEAN_DIR = "data/clean_text"
    AUDIO_DIR = "data/audio_output"
    
    generator = AudioGenerator(clean_data_dir=CLEAN_DIR, audio_output_dir=AUDIO_DIR)
    txt_files = [f for f in os.listdir(CLEAN_DIR) if f.endswith('.txt')]
    
    if txt_files:
        asyncio.run(generator.generate_audio(txt_files[0]))
    else:
        print("No hay archivos de texto limpio en data/clean_text/.")