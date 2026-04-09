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
    "A7", "D7", "E7", "G7", "B7",
    "Asus2", "Asus4", "Dsus4", "Esus4", "Gsus4",
    "A7M", "C7M", "D7M", "E7M", "F7M", "G7M",
]

NUM_CHORD_CLASSES = len(CHORD_CLASSES)

CHORD_TO_INDEX = {chord: i for i, chord in enumerate(CHORD_CLASSES)}
INDEX_TO_CHORD = {i: chord for i, chord in enumerate(CHORD_CLASSES)}

MAJOR_CHORDS = {"A", "B", "C", "D", "E", "F", "G", "A#", "C#", "D#", "F#", "G#"}
MINOR_CHORDS = {"Am", "Bm", "Cm", "Dm", "Em", "Fm", "Gm", "A#m", "C#m", "D#m", "F#m", "G#m"}

KEY_ORDER = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

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
