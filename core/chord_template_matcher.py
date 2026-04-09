import numpy as np
from config import CHORD_CLASSES, CHORD_TO_INDEX

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

CHORD_TEMPLATES = {}

MAJOR_TEMPLATE = [0, 4, 7]
MINOR_TEMPLATE = [0, 3, 7]
SEVENTH_TEMPLATE = [0, 4, 7, 10]
SUS2_TEMPLATE = [0, 2, 7]
SUS4_TEMPLATE = [0, 5, 7]
SEVENTH_MAJ_TEMPLATE = [0, 4, 7, 11]

for root_idx in range(12):
    root = NOTE_NAMES[root_idx]

    major = np.zeros(12)
    for interval in MAJOR_TEMPLATE:
        major[(root_idx + interval) % 12] = 1.0
    CHORD_TEMPLATES[root] = major / np.linalg.norm(major)

    minor = np.zeros(12)
    for interval in MINOR_TEMPLATE:
        minor[(root_idx + interval) % 12] = 1.0
    CHORD_TEMPLATES[f"{root}m"] = minor / np.linalg.norm(minor)

    seventh = np.zeros(12)
    for interval in SEVENTH_TEMPLATE:
        seventh[(root_idx + interval) % 12] = 1.0
    CHORD_TEMPLATES[f"{root}7"] = seventh / np.linalg.norm(seventh)

    sus2 = np.zeros(12)
    for interval in SUS2_TEMPLATE:
        sus2[(root_idx + interval) % 12] = 1.0
    CHORD_TEMPLATES[f"{root}sus2"] = sus2 / np.linalg.norm(sus2)

    sus4 = np.zeros(12)
    for interval in SUS4_TEMPLATE:
        sus4[(root_idx + interval) % 12] = 1.0
    CHORD_TEMPLATES[f"{root}sus4"] = sus4 / np.linalg.norm(sus4)

    seventh_maj = np.zeros(12)
    for interval in SEVENTH_MAJ_TEMPLATE:
        seventh_maj[(root_idx + interval) % 12] = 1.0
    CHORD_TEMPLATES[f"{root}7M"] = seventh_maj / np.linalg.norm(seventh_maj)

NO_CHORD_TEMPLATE = np.zeros(12)


def _normalize_chroma(chroma_frame: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(chroma_frame)
    if norm < 1e-10:
        return np.zeros(12)
    return chroma_frame / norm


def predict_chords_template(
    chroma: np.ndarray,
    hop_length: int = 512,
    sample_rate: int = 22050,
    no_chord_threshold: float = 0.15,
) -> tuple:
    """
    Prédit la séquence d'accords par template matching sur le chromagramme.
    Compare chaque frame chroma à des templates théoriques d'accords
    via la similarité cosinus.
    
    Retourne: (chord_times, chord_labels, chord_probs)
    """
    n_frames = chroma.shape[1]
    chord_labels = []
    chord_probs = []

    for t in range(n_frames):
        frame = chroma[:, t]
        normalized = _normalize_chroma(frame)

        if np.linalg.norm(normalized) < no_chord_threshold:
            chord_labels.append("N")
            chord_probs.append(1.0)
            continue

        best_chord = "N"
        best_score = -1.0

        for chord_name, template in CHORD_TEMPLATES.items():
            score = np.dot(normalized, template)
            if score > best_score:
                best_score = score
                best_chord = chord_name

        chord_labels.append(best_chord)
        chord_probs.append(float(best_score))

    frame_duration = hop_length / sample_rate
    chord_times = np.arange(n_frames) * frame_duration

    return chord_times, chord_labels, np.array(chord_probs)
