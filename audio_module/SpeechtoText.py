from audio_loader import preprocess_audio
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch

# Load processor and model once
processor = AutoProcessor.from_pretrained("openai/whisper-medium.en")
model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-medium.en")

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def transcribe_file(audio_path: str) -> dict:
    """
    Preprocesses an audio file and transcribes it with Hugging Face Whisper.
    Returns a Python dict with transcript and metadata.
    """
    # Preprocess audio
    processed_file = preprocess_audio(audio_path)

    # Load waveform
    import librosa
    waveform, sr = librosa.load(processed_file, sr=None, mono=True)

    # Convert to input features
    input_features = processor(
        waveform, sampling_rate=sr, return_tensors="pt"
    ).input_features.to(device)

    # Generate transcription
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    # Return dict
    return {
        "transcript": transcription,
        "metadata": {
            "source": audio_path,
            "processed_file": processed_file,
            "sampling_rate": sr
        }
    }

def speech_to_text(audio_path):
    return transcribe_file(audio_path)

