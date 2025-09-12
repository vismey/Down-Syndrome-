import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 1e-5)  # ensure > 0
    high = min(highcut / nyq, 0.999)  # ensure < 1
    if low >= high:
        raise ValueError(f"Invalid bandpass frequencies: lowcut={lowcut}, highcut={highcut}, fs={fs}")
    b, a = butter(order, [low, high], btype="band")
    return b, a

def bandpass_filter(data, lowcut=50.0, highcut=8000.0, fs=16000, order=5):
    """Filter out noise outside speech frequency range (50–8000 Hz)."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)

def preprocess_audio(input_file, output_file="processed.wav", target_sr=16000):
    """
    Preprocess audio for speech-to-text:
    - Converts to mono
    - Normalizes volume
    - Resamples to 16kHz
    - Applies bandpass filter (speech-focused)
    - Applies light noise reduction
    """

    # Load audio
    y, sr = librosa.load(input_file, sr=None, mono=True)

    # Resample to target sample rate
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

    # Normalize volume (mean=0, max abs=1)
    y = y / np.max(np.abs(y))

    # Bandpass filter (keep speech frequencies)
    y = bandpass_filter(y, fs=target_sr)

    # Simple noise reduction using pre-emphasis
    y = librosa.effects.preemphasis(y)

    # Export as wav
    sf.write(output_file, y, target_sr)
    return output_file

