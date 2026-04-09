import numpy as np
import librosa
from config import AUDIO_SAMPLE_RATE, HOP_LENGTH


def detect_beats(audio_path: str) -> tuple:
    """
    Détecte les temps (beats) et le BPM avec librosa.
    Utilise une approche multi-candidats pour éviter les détections
    de croches au lieu de noires (BPM 2x trop rapide).
    Retourne: (beat_times, bpm)
    """
    y, sr = librosa.load(audio_path, sr=AUDIO_SAMPLE_RATE, mono=True)

    # Obtenir TOUS les tempos candidats
    all_tempos = librosa.beat.tempo(y=y, sr=sr, hop_length=HOP_LENGTH, aggregate=None)

    # Trier du plus lent au plus rapide
    all_tempos = sorted(all_tempos)

    # Choisir le tempo le plus plausible dans la plage 60-140 BPM
    # (la plupart des morceaux populaires sont dans cette plage)
    plausible_tempos = [t for t in all_tempos if 60 <= t <= 140]

    if plausible_tempos:
        bpm = plausible_tempos[0]  # Prendre le plus lent plausible
    else:
        # Fallback : prendre le premier tempo et normaliser
        bpm = all_tempos[0]
        while bpm > 140:
            bpm /= 2
        while bpm < 60:
            bpm *= 2

    # Re-détecter les beats avec le BPM corrigé
    _, beat_frames = librosa.beat.beat_track(
        y=y,
        sr=sr,
        hop_length=HOP_LENGTH,
        start_bpm=bpm,
    )

    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP_LENGTH)

    return beat_times, bpm


def detect_downbeats(audio_path: str) -> np.ndarray:
    """
    Détecte les temps forts (downbeats) - premier temps de chaque mesure.
    Retourne: np.ndarray des temps de downbeat.
    """
    y, sr = librosa.load(audio_path, sr=AUDIO_SAMPLE_RATE, mono=True)

    pulse = librosa.beat.plp(
        y=y,
        sr=sr,
        hop_length=HOP_LENGTH,
    )

    downbeat_frames = np.where(pulse[0] > 0.5)[0]
    downbeat_times = librosa.frames_to_time(downbeat_frames, sr=sr, hop_length=HOP_LENGTH)

    return downbeat_times


def sync_chords_to_beats(
    chord_times: np.ndarray,
    chord_labels: list,
    beat_times: np.ndarray,
) -> tuple:
    """
    Synchronise les accords détectés avec les beats.
    Chaque beat reçoit l'accord dominant dans son intervalle.
    Retourne: (synced_beat_times, synced_chord_labels)
    """
    if len(beat_times) == 0 or len(chord_times) == 0:
        return beat_times, chord_labels

    synced_labels = []

    for i, beat_time in enumerate(beat_times):
        if i < len(beat_times) - 1:
            beat_end = beat_times[i + 1]
        else:
            beat_end = beat_time + (beat_times[i] - beat_times[i - 1]) if i > 0 else beat_time + 0.5

        mask = (chord_times >= beat_time) & (chord_times < beat_end)
        if mask.any():
            dominant_chord = chord_labels[mask.argmax()]
            synced_labels.append(dominant_chord)
        else:
            if synced_labels:
                synced_labels.append(synced_labels[-1])
            else:
                synced_labels.append("N")

    return beat_times, synced_labels
