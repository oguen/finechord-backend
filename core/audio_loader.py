import os
import tempfile
from pathlib import Path
from config import UPLOAD_DIR, AUDIO_SAMPLE_RATE, AUDIO_FORMAT
from core.ffmpeg_utils import run_ffmpeg, run_ffprobe


def extract_audio(input_path: str) -> str:
    """
    Extrait la piste audio d'un fichier (audio ou vidéo) via FFmpeg
    et la convertit en WAV mono 22050Hz.
    Si le fichier est déjà un WAV, retourne le fichier tel quel.
    """
    input_p = Path(input_path)
    if not input_p.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    if input_p.suffix.lower() == '.wav':
        print(f"[audio_loader] File is already WAV: {input_path}")
        return str(input_p)

    output_name = input_p.stem + ".wav"
    output_path = UPLOAD_DIR / output_name

    print(f"[audio_loader] Converting {input_p.suffix} to WAV...")
    run_ffmpeg([
        "-y",
        "-i", str(input_p),
        "-vn",
        "-ar", str(AUDIO_SAMPLE_RATE),
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-f", "wav",
        str(output_path),
    ], timeout=600)

    return str(output_path)


def get_duration(audio_path: str) -> float:
    """Retourne la durée en secondes d'un fichier audio."""
    result = run_ffprobe([
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ], timeout=30)

    return float(result.stdout.strip())
