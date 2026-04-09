import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"

UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
CHECKPOINTS_DIR.mkdir(exist_ok=True)

AUDIO_SAMPLE_RATE = 22050
AUDIO_MONO = True
AUDIO_FORMAT = "wav"

CQT_BINS_PER_OCTAVE = 36
CQT_N_OCTAVES = 7
CHROMA_N_BINS = 12

HOP_LENGTH = 512
N_FFT = 2048

CHORD_CLASSES = [
    "N",
    "A", "Am", "B", "Bm", "C", "Cm", "D", "Dm", "E", "Em", "F", "Fm", "G", "Gm",
    "A#", "A#m", "C#", "C#m", "D#", "D#m", "F#", "F#m", "G#", "G#m",
    "A7", "B7", "C7", "D7", "E7", "F7", "G7",
    "A7M", "B7M", "C7M", "D7M", "E7M", "F7M", "G7M",
    "Asus2", "Asus4", "Csus4", "Dsus4", "Esus4", "Fsus4", "Gsus4",
    "Aø", "Bø", "Cø", "Dø", "Eø", "Fø", "Gø",
    "Adim", "Bdim", "Cdim", "Ddim", "Edim", "Fdim", "Gdim",
]

NUM_CHORD_CLASSES = len(CHORD_CLASSES)

CHORD_TO_INDEX = {chord: i for i, chord in enumerate(CHORD_CLASSES)}
INDEX_TO_CHORD = {i: chord for i, chord in enumerate(CHORD_CLASSES)}

MAJOR_CHORDS = {"A", "B", "C", "D", "E", "F", "G", "A#", "C#", "D#", "F#", "G#"}
MINOR_CHORDS = {"Am", "Bm", "Cm", "Dm", "Em", "Fm", "Gm", "A#m", "C#m", "D#m", "F#m", "G#m"}
SEVENTH_CHORDS = {"A7", "B7", "C7", "D7", "E7", "F7", "G7"}

KEY_ORDER = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# ============================================================================
# KEY DETECTION - Krumhansl-Schmuckler Profiles
# ============================================================================
# Profiles for major and minor keys based on cognitive experiments
# Source: Krumhansl, C. "Cognitive Foundations of Musical Pitch"

KRUMSHANSL_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
KRUMSHANSL_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

# Temperley profiles (from corpus analysis of classical music)
TEMPERLEY_MAJOR = [5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 2.0]
TEMPERLEY_MINOR = [5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.5, 4.5, 3.0, 1.5, 3.5, 1.5]

# EDM profiles (for electronic/pop music)
EDMA_MAJOR = [4.5, 1.5, 3.0, 1.5, 4.0, 3.5, 1.5, 4.0, 2.0, 3.0, 1.5, 2.5]
EDMA_MINOR = [4.5, 1.5, 3.5, 4.0, 2.0, 3.5, 2.0, 4.0, 3.5, 1.5, 3.0, 1.5]

# Default profile for piano music
DEFAULT_KEY_PROFILE = "krumhansl"

# Key profile correlation settings
KEY_CORRELATION_WINDOW = 120  # seconds of audio to analyze
TUNING_STEP = 5.0  # cents for tuning correction
MIN_KEY_CONFIDENCE = 0.3  # minimum correlation to accept key

ROMAN_NUMERALS_MAJOR = {
    "C": "I", "C#": "I", "D": "II", "D#": "III", "E": "III", "F": "IV",
    "F#": "V", "G": "V", "G#": "VI", "A": "VI", "A#": "VII", "B": "VII",
}

MIN_PROB_THRESHOLD = 0.4
MIN_SEGMENT_DURATION = 0.15

RHYTHMIC_RESOLUTIONS = {
    "whole": 4.0,
    "half": 2.0,
    "quarter": 1.0,
    "eighth": 0.5,
    "sixteenth": 0.25,
}
DEFAULT_RHYTHMIC_RESOLUTION = "eighth"

API_TITLE = "FineChord API"
API_VERSION = "1.0.0"
API_DESCRIPTION = "API d'analyse harmonique de fichiers audio et vidéo"

MAX_UPLOAD_SIZE_MB = 150
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".mp4", ".avi", ".mkv", ".webm", ".aac", ".wma"}
