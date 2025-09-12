from fastapi import FastAPI, File, UploadFile
from pathlib import Path
import shutil

# Import your existing functions
from audio_module.SpeechtoText import speech_to_text, get_emotion

app = FastAPI()

@app.post("/analyze-audio")
async def analyze_audio(file: UploadFile = File(...)):
    # Save uploaded file
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    file_path = upload_dir / file.filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Call your speech_to_text function
    transcription = speech_to_text(str(file_path))

    # Call your emotion detection function
    emotion = get_emotion(str(file_path))

    return {
        "transcription": transcription,
        "emotion": emotion
    }
