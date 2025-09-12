from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import torch
import librosa

# Load model and processor
model_name = "prithivMLmods/Speech-Emotion-Classification"
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

id2label = model.config.id2label

def classify_audio(audio_path):
    # Load and resample audio to 16kHz
    speech, sample_rate = librosa.load(audio_path, sr=16000)

    # Process audio
    inputs = processor(
        speech,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

    # Use model's built-in label mapping
    id2label = model.config.id2label
    prediction = {id2label[i]: round(probs[i], 3) for i in range(len(probs))}

    return prediction

# Example usage with your existing recorded/preprocessed file
if __name__ == "__main__":
    # Replace this with your audio loader's output file
    print(model.config.id2label)

    audio_path = "audio_module/harvard.wav"  
    prediction = classify_audio(audio_path)
    print("Detected emotion:", prediction)