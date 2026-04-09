"""
Template Matching pour la reconnaissance d'accords.

Approche : comparer le chromagramme extrait à des templates théoriques
d'accords basés sur les intervalles harmoniques.

Précision attendue : ~60-70% (MIREX MajMin)
Référence : Harte et al., "Towards Automatic Extraction of Harmony Information"
"""

import numpy as np
from typing import Tuple


CHORD_TEMPLATES = {
    "N":  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],

    "C":  [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    "C#": [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "D":  [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    "D#": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
    "E":  [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
    "F":  [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    "F#": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    "G":  [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    "G#": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "A":  [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    "A#": [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    "B":  [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],

    "Cm":  [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    "C#m": [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "Dm":  [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    "D#m": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    "Em":  [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    "Fm":  [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "F#m": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    "Gm":  [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
    "G#m": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
    "Am":  [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    "A#m": [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    "Bm":  [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],

    "C7":  [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
    "D7":  [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
    "E7":  [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
    "F7":  [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
    "G7":  [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    "A7":  [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
    "B7":  [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0],

    "C7M":  [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    "D7M":  [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
    "E7M":  [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
    "F7M":  [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
    "G7M":  [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    "A7M":  [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],

    "Asus2": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    "Asus4": [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    "Dsus4": [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    "Esus4": [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
    "Gsus4": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
}

CHORD_NAMES = list(CHORD_TEMPLATES.keys())
MAJOR_CHORDS = [n for n in CHORD_NAMES if len(n) <= 2 and not n.endswith('m') and n != 'N']
MINOR_CHORDS = [n for n in CHORD_NAMES if n.endswith('m') and not n.endswith('7M')]
SEVENTH_CHORDS = [n for n in CHORD_NAMES if '7' in n and '7M' not in n]


def _build_template_matrix() -> np.ndarray:
    """Construit la matrice de templates (12 x N_accords)."""
    templates = []
    for name in CHORD_NAMES:
        templates.append(CHORD_TEMPLATES[name])
    return np.array(templates, dtype=np.float64).T


_TEMPLATE_MATRIX = _build_template_matrix()


def predict_chords_template(
    chroma: np.ndarray,
    hop_length: int = 512,
    sample_rate: int = 22050,
    min_confidence: float = 0.15,
) -> Tuple[np.ndarray, list, np.ndarray]:
    """
    Prédit la séquence d'accords par template matching.

    Pour chaque frame de chroma, on calcule la similarité cosinus
    avec chaque template d'accord et on choisit le meilleur match.

    Args:
        chroma: np.ndarray de shape (12, T) — chromagramme CQT
        hop_length: nombre d'échantillons entre chaque frame
        sample_rate: taux d'échantillonnage en Hz
        min_confidence: seuil minimum de confiance pour valider un accord

    Returns:
        chord_times: timestamps de chaque frame
        chord_labels: liste des noms d'accords prédits
        chord_probs: confiance de chaque prédiction
    """
    T = chroma.shape[1]

    chroma_norm = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-10)

    similarities = chroma_norm.T @ _TEMPLATE_MATRIX

    best_indices = np.argmax(similarities, axis=1)
    best_scores = np.max(similarities, axis=1)

    chord_labels = []
    chord_probs = []

    for i in range(T):
        idx = best_indices[i]
        score = best_scores[i]

        if score < min_confidence:
            chord_labels.append("N")
            chord_probs.append(1.0 - score)
        else:
            chord_labels.append(CHORD_NAMES[idx])
            chord_probs.append(score)

    frame_duration = hop_length / sample_rate
    chord_times = np.arange(T) * frame_duration

    chord_probs = np.array(chord_probs)

    return chord_times, chord_labels, chord_probs
