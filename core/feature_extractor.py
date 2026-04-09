import numpy as np
import librosa
from config import (
    AUDIO_SAMPLE_RATE,
    CQT_BINS_PER_OCTAVE,
    CQT_N_OCTAVES,
    CHROMA_N_BINS,
    HOP_LENGTH,
)


def extract_cqt_chroma(audio_path: str) -> np.ndarray:
    """
    Extrait un chromagram basé sur CQT (Constant-Q Transform).
    Meilleure résolution fréquentielle pour la reconnaissance d'accords.
    Inclut une correction de tuning pour corriger les décalages d'octave.
    Retourne: np.ndarray de shape (12, T)
    """
    y, sr = librosa.load(audio_path, sr=AUDIO_SAMPLE_RATE, mono=True)

    tuning_offset = librosa.estimate_tuning(y=y, sr=sr)
    print(f"[FeatureExtractor] Detected tuning offset: {tuning_offset:.2f} semitones")

    cqt = librosa.cqt(
        y=y,
        sr=sr,
        hop_length=HOP_LENGTH,
        n_bins=CQT_BINS_PER_OCTAVE * CQT_N_OCTAVES,
        bins_per_octave=CQT_BINS_PER_OCTAVE,
    )

    chroma = librosa.feature.chroma_cqt(
        y=y,
        sr=sr,
        C=cqt,
        n_chroma=CHROMA_N_BINS,
        bins_per_octave=CQT_BINS_PER_OCTAVE,
    )

    if abs(tuning_offset) > 0.05:
        n_semitones = int(round(tuning_offset))
        print(f"[FeatureExtractor] Applying chroma shift of {n_semitones} semitones")
        chroma = np.roll(chroma, -n_semitones, axis=0)

    return chroma


def extract_cqt_raw(audio_path: str) -> np.ndarray:
    """
    Extrait le CQT brut (252 bins, 7 octaves) sans repliement en chroma.
    Les 36 premiers bins correspondent aux fréquences basses (~30-60 Hz).
    Utilisé pour la détection de la note de basse.
    Retourne: np.ndarray de shape (252, T) - magnitude du CQT
    """
    y, sr = librosa.load(audio_path, sr=AUDIO_SAMPLE_RATE, mono=True)

    cqt = librosa.cqt(
        y=y,
        sr=sr,
        hop_length=HOP_LENGTH,
        n_bins=252,  # 36 bins/octave × 7 octaves
        bins_per_octave=36,
    )

    return np.abs(cqt)


def extract_stft_chroma(audio_path: str) -> np.ndarray:
    """
    Extrait un chromagram basé sur STFT en complément du CQT.
    Retourne: np.ndarray de shape (12, T)
    """
    y, sr = librosa.load(audio_path, sr=AUDIO_SAMPLE_RATE, mono=True)

    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr,
        hop_length=HOP_LENGTH,
    )

    return chroma


def extract_mfcc(audio_path: str, n_mfcc: int = 13) -> np.ndarray:
    """
    Extrait les coefficients MFCC pour les features timbrales.
    Retourne: np.ndarray de shape (n_mfcc, T)
    """
    y, sr = librosa.load(audio_path, sr=AUDIO_SAMPLE_RATE, mono=True)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        hop_length=HOP_LENGTH,
    )

    return mfcc


def extract_combined_features(audio_path: str) -> np.ndarray:
    """
    Combine CQT chroma + STFT chroma + MFCC en un seul tenseur.
    Retourne: np.ndarray de shape (12 + 12 + 13, T) = (37, T)
    """
    cqt_chroma = extract_cqt_chroma(audio_path)
    stft_chroma = extract_stft_chroma(audio_path)
    mfcc = extract_mfcc(audio_path)

    features = np.vstack([cqt_chroma, stft_chroma, mfcc])
    return features
