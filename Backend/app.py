from fastapi import FastAPI, File, UploadFile
from pathlib import Path
import shutil
from audio_module.audioEmotion import classify_audio
# Import your existing functions
from audio_module.SpeechtoText import speech_to_text
from facial_module.Facial_Emotions import v_emotion
from FinalInput import Query_llm


app = FastAPI()

@app.post("/analyze-video-audio")
async def analyze_video_audio(file: UploadFile = File(...)):
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    file_path = upload_dir / file.filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 🔹 Extract audio from video (for STT + audio emotion)
    audio_path = str(file_path).replace(".webm", ".wav")
    import moviepy.editor as mp
    clip = mp.VideoFileClip(str(file_path))
    clip.audio.write_audiofile(audio_path)

    # Call your existing modules
    transcription = await speech_to_text(audio_path)   # from audio_module
    audio_emotion = await classify_audio(audio_path)   # from audio_module
    video_emotion, v_confidence = v_emotion(str(file_path))  # from facial_module
    data = {
        'message': f"""
    You are an assistant that combines multimodal emotion and text inputs.

    You are given a dictionary with the following fields:
    •⁠  ⁠transcription: "{transcription}" (speech-to-text result)
    •⁠  ⁠emotion: "{audio_emotion}" (audio emotion with probability weight implied)
    •⁠  ⁠v_emotionc: "{video_emotion}" (video emotion classification)
    •⁠  ⁠v_confidence: {v_confidence}% (confidence score of video emotion)

    Steps:
    1.⁠ ⁠Compare the audio emotion (emotion) and video emotion (v_emotionc with v_confidence).  
       - If v_confidence is higher than the implied audio probability, prefer v_emotionc.  
       - Otherwise, prefer emotion.  
    2.⁠ ⁠Choose the final emotion based on this comparison.  
    3.⁠ ⁠Rewrite or refine the transcription so that it reflects not only what was said but also the emotional tone of the final chosen emotion.  
    4.⁠ ⁠Output a single refined final text sentence that sounds natural, empathetic, and clear.

    Now process the input and give the final text output only.
    """
    }
    respo = await Query_llm(data)
    return {
        "transcription": transcription,
        "audio_emotion": audio_emotion,
        "video_emotion": video_emotion,
        "v_confidence": v_confidence,
        "response_by_llm":respo
    }

