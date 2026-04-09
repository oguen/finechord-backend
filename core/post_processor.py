import numpy as np
from config import (
    CHORD_CLASSES,
    CHORD_TO_INDEX,
    INDEX_TO_CHORD,
    MAJOR_CHORDS,
    MINOR_CHORDS,
    KEY_ORDER,
    MIN_PROB_THRESHOLD,
    MIN_SEGMENT_DURATION,
    KRUMSHANSL_MAJOR,
    KRUMSHANSL_MINOR,
    TEMPERLEY_MAJOR,
    TEMPERLEY_MINOR,
    KEY_CORRELATION_WINDOW,
    MIN_KEY_CONFIDENCE,
)


def smooth_with_hmm(
    chord_times: np.ndarray,
    chord_labels: list,
    chord_probs: np.ndarray,
    transition_weight: float = 0.7,
) -> list:
    """
    Lissage temporel avec un modèle HMM simplifié (Viterbi-like).
    Réduit les changements d'accords trop rapides et non justifiés.
    """
    if len(chord_labels) <= 1:
        return chord_labels

    smoothed = chord_labels.copy()

    for i in range(1, len(chord_labels) - 1):
        prev_chord = smoothed[i - 1]
        curr_chord = chord_labels[i]
        next_chord = chord_labels[i + 1]

        curr_prob = chord_probs[i] if i < len(chord_probs) else 0

        if curr_prob < MIN_PROB_THRESHOLD:
            if prev_chord == next_chord:
                smoothed[i] = prev_chord
            else:
                smoothed[i] = prev_chord
        elif curr_chord != prev_chord and curr_chord != next_chord:
            if prev_chord == next_chord:
                smoothed[i] = prev_chord

    return smoothed


def merge_short_segments(
    chord_times: np.ndarray,
    chord_labels: list,
    min_duration: float = 0.3,
) -> tuple:
    """
    Regroupe les frames consécutives avec le même accord en segments.
    Ne supprime aucun segment — préserve tous les changements d'accords détectés.
    """
    if len(chord_labels) <= 1:
        return chord_times, chord_labels

    # Regrouper les frames consécutives identiques en segments
    segment_starts = [chord_times[0]]
    segment_labels = [chord_labels[0]]

    for i in range(1, len(chord_labels)):
        if chord_labels[i] != chord_labels[i - 1]:
            segment_starts.append(chord_times[i])
            segment_labels.append(chord_labels[i])

    print(f"[merge] Grouped {len(chord_labels)} frames into {len(segment_starts)} segments, {len(set(segment_labels) - {'N'})} unique chords")

    return np.array(segment_starts), segment_labels


def detect_key(chord_labels: list) -> str:
    """
    Détecte la tonalité dominante d'un morceau en analysant
    la distribution des accords majeurs et mineurs.
    Retourne: la note fondamentale (ex: "C", "G", "F#")
    """
    key_scores = {key: 0.0 for key in KEY_ORDER}

    for chord in chord_labels:
        if chord == "N":
            continue

        root = chord.rstrip("m7M#sus24")

        for i, key_note in enumerate(KEY_ORDER):
            if root == key_note:
                if chord in MAJOR_CHORDS:
                    key_scores[key_note] += 3.0
                    key_scores[KEY_ORDER[(i + 7) % 12]] += 2.0
                    key_scores[KEY_ORDER[(i + 5) % 12]] += 1.5
                elif chord in MINOR_CHORDS:
                    key_scores[key_note] += 2.0
                    key_scores[KEY_ORDER[(i + 7) % 12]] += 1.5
                    key_scores[KEY_ORDER[(i + 3) % 12]] += 1.0

    if not key_scores or max(key_scores.values()) == 0:
        return "C"

    return max(key_scores, key=key_scores.get)


def detect_key_from_chroma(chroma: np.ndarray, profile_type: str = "krumhansl") -> tuple:
    """
    Détecte la tonalité en utilisant les chroma features et les profiles Krumhansl.
    
    Args:
        chroma: np.ndarray de shape (12, T) - features chroma
        profile_type: "krumhansl", "temperley", ou "edma"
    
    Returns:
        tuple: (key_note, mode, confidence, correlation)
            - key_note: note de la tonalité (ex: "C", "G#")
            - mode: "major" ou "minor"
            - confidence: confiance de la détection (0-1)
            - correlation: correlation maximale avec le profile
    """
    if chroma is None or chroma.size == 0:
        return "C", "major", 0.0, 0.0
    
    # Temporal averaging - moyenne des chroma sur le temps
    chroma_profile = np.mean(chroma, axis=1)
    
    # Normalize
    chroma_profile = chroma_profile / (np.sum(chroma_profile) + 1e-10)
    
    # Select profile
    if profile_type == "krumhansl":
        major_profile = np.array(KRUMSHANSL_MAJOR)
        minor_profile = np.array(KRUMSHANSL_MINOR)
    elif profile_type == "temperley":
        major_profile = np.array(TEMPERLEY_MAJOR)
        minor_profile = np.array(TEMPERLEY_MINOR)
    else:
        major_profile = np.array(KRUMSHANSL_MAJOR)
        minor_profile = np.array(KRUMSHANSL_MINOR)
    
    # Normalize profiles
    major_profile = major_profile / (np.sum(major_profile) + 1e-10)
    minor_profile = minor_profile / (np.sum(minor_profile) + 1e-10)
    
    best_key = "C"
    best_mode = "major"
    best_correlation = -1.0
    
    # Test each possible key (0 = C, 1 = C#, ..., 11 = B)
    for shift in range(12):
        # Circular shift of chroma profile
        shifted_chroma = np.roll(chroma_profile, shift)
        
        # Correlation with major profile
        major_corr = np.corrcoef(shifted_chroma, major_profile)[0, 1]
        # Correlation with minor profile  
        minor_corr = np.corrcoef(shifted_chroma, minor_profile)[0, 1]
        
        if major_corr > best_correlation:
            best_correlation = major_corr
            best_key = KEY_ORDER[shift]
            best_mode = "major"
        
        if minor_corr > best_correlation:
            best_correlation = minor_corr
            best_key = KEY_ORDER[shift]
            best_mode = "minor"
    
    # Ensure correlation is positive
    best_correlation = max(0, best_correlation)
    
    # Calculate confidence based on correlation
    confidence = min(1.0, best_correlation * 2) if best_correlation > 0 else MIN_KEY_CONFIDENCE
    
    return best_key, best_mode, confidence, best_correlation


def detect_key_from_audio(audio_path: str, profile_type: str = "krumhansl") -> tuple:
    """
    Détecte la tonalité directement depuis un fichier audio.
    Utilise librosa pour extraire les chroma features.
    
    Args:
        audio_path: chemin vers le fichier audio
        profile_type: "krumhansl", "temperley", ou "edma"
    
    Returns:
        tuple: (key_note, mode, confidence, correlation)
    """
    try:
        import librosa
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        
        # Extract chroma features
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=12)
        
        return detect_key_from_chroma(chroma, profile_type)
        
    except Exception as e:
        print(f"[KeyDetection] Error: {e}")
        return "C", "major", 0.0, 0.0


def combine_key_detections(
    chord_based_key: str,
    chroma_based_key: str,
    chroma_confidence: float,
    chord_weight: float = 0.4,
    chroma_weight: float = 0.6
) -> str:
    """
    Combine les detections de tonalité basées sur les accords et les chroma features.
    
    Args:
        chord_based_key: tonalité détectée depuis les accords
        chroma_based_key: tonalité détectée depuis les chroma features
        chroma_confidence: confiance de la détection chroma (0-1)
        chord_weight: poids pour la detection accord (défaut: 0.4)
        chroma_weight: poids pour la detection chroma (défaut: 0.6)
    
    Returns:
        str: tonalité finale
    """
    # If chroma detection is confident, trust it more
    if chroma_confidence > 0.7:
        return chroma_based_key
    
    # If both agree, return either
    if chord_based_key == chroma_based_key:
        return chord_based_key
    
    # Use weighted average based on confidence
    if chroma_confidence > 0.5:
        return chroma_based_key
    
    # Low confidence - prefer chord-based (more musically informed)
    return chord_based_key


def chord_to_roman(chord: str, key: str) -> str:
    """
    Convertit un accord en chiffre romain relatif à la tonalité.
    Ex: "G" en tonalité de "C" → "V"
    """
    if chord == "N":
        return "N"

    is_minor = chord.endswith("m") and not chord.endswith("m7") and not chord.endswith("sus4")
    is_seventh = "7" in chord and "7M" not in chord
    is_major_seventh = "7M" in chord
    is_sus = "sus" in chord

    root = chord.replace("m", "").replace("7M", "").replace("7", "").replace("sus2", "").replace("sus4", "")

    if root not in KEY_ORDER:
        return chord

    root_idx = KEY_ORDER.index(root)
    key_idx = KEY_ORDER.index(key)
    degree = (root_idx - key_idx) % 12

    roman_map = {
        0: "I", 1: "I", 2: "II", 3: "III", 4: "III",
        5: "IV", 6: "V", 7: "V", 8: "VI", 9: "VI",
        10: "VII", 11: "VII",
    }

    roman = roman_map.get(degree, "?")

    if is_minor:
        roman = roman.lower()

    if is_seventh:
        roman += "7"
    elif is_major_seventh:
        roman += "M7"
    elif is_sus:
        roman += "sus"

    return roman


def build_analysis_result(
    chord_times: np.ndarray,
    chord_labels: list,
    beat_times: np.ndarray,
    beat_chords: list,
    bpm: float,
    key: str,
    duration: float,
    confidence: float,
    bass_notes: list = None,
) -> dict:
    """
    Construit le résultat d'analyse complet au format JSON.
    """
    if bass_notes is None:
        bass_notes = [''] * len(chord_labels)

    segments = []
    for i in range(len(chord_labels)):
        start = float(chord_times[i])
        end = float(chord_times[i + 1]) if i + 1 < len(chord_times) else duration
        roman = chord_to_roman(chord_labels[i], key)

        chord = chord_labels[i]
        bass = bass_notes[i] if i < len(bass_notes) else ''
        display = _format_chord_display(chord, bass)

        segments.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "chord": chord,
            "bass": bass,
            "display": display,
            "roman": roman,
        })

    beat_segments = []
    for i, bt in enumerate(beat_times):
        beat_segments.append({
            "time": round(float(bt), 3),
            "chord": beat_chords[i] if i < len(beat_chords) else "N",
        })

    return {
        "success": True,
        "duration": round(duration, 2),
        "bpm": round(bpm, 1),
        "key": key,
        "confidence": round(float(confidence), 3),
        "segments": segments,
        "beat_synced": beat_segments,
        "stats": {
            "total_segments": len(segments),
            "unique_chords": len(set(chord_labels) - {"N"}),
            "chord_distribution": _chord_distribution(chord_labels),
        },
    }


def _format_chord_display(chord: str, bass: str) -> str:
    """
    Format chord display with bass note if different from root.
    Example: C with bass G -> C/G, Am with bass E -> Am/E
    """
    if not bass or chord == 'N':
        return chord

    root = chord.rstrip('m7Msus24#')

    if bass == root:
        return chord

    return f"{chord}/{bass}"


def _chord_distribution(chord_labels: list) -> dict:
    """Calcule la distribution des accords."""
    from collections import Counter
    counter = Counter(chord_labels)
    total = len(chord_labels)
    return {
        chord: round(count / total * 100, 1)
        for chord, count in counter.most_common()
        if chord != "N"
    }
