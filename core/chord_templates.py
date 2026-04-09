"""
Template-based chord recognition using chroma feature matching.

This approach compares the extracted chroma vector of each audio frame
to theoretical chord templates (major, minor, 7th, sus, etc.) using
cosine similarity. The chord with the highest similarity is selected.

Based on the approach described in:
- Fujishima, T. (1999). Realtime chord recognition of musical sound
- Sheh, A. & Ellis, D. (2003). Chord segmentation and recognition using EM-trained HMM
"""

import numpy as np


# Chord templates based on semitone intervals
# Each template is a 12-element vector (C, C#, D, D#, E, F, F#, G, G#, A, A#, B)
# Values represent the expected energy for each pitch class

# Enhanced templates with stronger differentiation
# Major: root, major third (4 semitones), perfect fifth (7 semitones)
TEMPLATE_MAJOR = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]

# Minor: root, minor third (3 semitones), perfect fifth (7 semitones)
TEMPLATE_MINOR = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]

# Dominant 7th: root, major third, perfect fifth, minor seventh (10 semitones)
# This makes D7 (D-F#-A-C) very different from D#ø (D#-G#-A#-C#)
TEMPLATE_7TH = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.8]

# Major 7th: root, major third, perfect fifth, major seventh (11 semitones)
TEMPLATE_7M = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.8]

# Suspended templates
TEMPLATE_SUS2 = [1.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
TEMPLATE_SUS4 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0, 0.0]

# Half-diminished (m7b5): root, minor third, diminished fifth (6 semitones), minor seventh
# C Eb Gb Bb - very different from C major (C E G)
TEMPLATE_HALF_DIM = [1.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.8, 0.0]

# Diminished: root, minor third, diminished fifth, diminished seventh (9 semitones)
# C Eb Gb Bbb(A) - symmetric stack
TEMPLATE_DIM = [1.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.8, 0.0, 0.0, 0.8, 0.0, 0.0]


NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

CHORD_DEFINITIONS = {
    'N':   [0.0] * 12,  # No chord / silence
}

# Generate templates for all 12 roots
for root_idx, root_name in enumerate(NOTE_NAMES):
    # Major
    CHORD_DEFINITIONS[root_name] = np.roll(TEMPLATE_MAJOR, root_idx).tolist()
    # Minor
    CHORD_DEFINITIONS[f'{root_name}m'] = np.roll(TEMPLATE_MINOR, root_idx).tolist()
    # 7th (dominant 7th)
    CHORD_DEFINITIONS[f'{root_name}7'] = np.roll(TEMPLATE_7TH, root_idx).tolist()
    # Major 7th
    CHORD_DEFINITIONS[f'{root_name}7M'] = np.roll(TEMPLATE_7M, root_idx).tolist()
    # Sus2
    CHORD_DEFINITIONS[f'{root_name}sus2'] = np.roll(TEMPLATE_SUS2, root_idx).tolist()
    # Sus4
    CHORD_DEFINITIONS[f'{root_name}sus4'] = np.roll(TEMPLATE_SUS4, root_idx).tolist()
    # Half-diminished (m7b5)
    CHORD_DEFINITIONS[f'{root_name}ø'] = np.roll(TEMPLATE_HALF_DIM, root_idx).tolist()
    CHORD_DEFINITIONS[f'{root_name}m7b5'] = np.roll(TEMPLATE_HALF_DIM, root_idx).tolist()
    # Diminished
    CHORD_DEFINITIONS[f'{root_name}dim'] = np.roll(TEMPLATE_DIM, root_idx).tolist()


# Rhythmic resolution mapping: how many subdivisions per beat
RHYTHMIC_DIVISIONS = {
    # Binary
    'breve': 0.25,         # 1 accord toutes les 4 noires (ronde)
    'whole': 0.5,          # 1 accord toutes les 2 noires (blanche)
    'half': 1.0,           # 1 accord par noire
    'quarter': 2.0,        # 1 accord par croche (2 par noire)
    'eighth': 4.0,         # 1 accord par double croche (4 par noire)
    'sixteenth': 8.0,      # 1 accord par triple croche (8 par noire)
    # Ternary
    'dotted_breve': 0.167,     # Ronde pointée (1 accord / 6 noires)
    'dotted_half': 0.333,      # Blanche pointée (1 accord / 3 noires)
    'dotted_quarter': 0.667,   # Noire pointée (2 accords / 3 noires)
    'dotted_eighth': 1.333,    # Croche pointée (4 accords / 3 noires)
}


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def build_template_matrix() -> np.ndarray:
    """
    Build a matrix of all chord templates.
    Shape: (num_chords, 12)
    """
    templates = []
    for chord_name in CHORD_DEFINITIONS:
        templates.append(CHORD_DEFINITIONS[chord_name])
    return np.array(templates, dtype=np.float64)


def predict_chords_template(
    chroma: np.ndarray,
    hop_length: int = 512,
    sample_rate: int = 22050,
    min_confidence: float = 0.3,
) -> tuple:
    """
    Predict chords using template matching (frame-by-frame).
    Legacy method kept for compatibility.
    """
    template_matrix = build_template_matrix()
    chord_names = list(CHORD_DEFINITIONS.keys())

    chroma_norm = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-10)
    similarities = template_matrix @ chroma_norm

    best_indices = np.argmax(similarities, axis=0)
    best_scores = np.max(similarities, axis=0)

    low_confidence = best_scores < min_confidence
    best_indices[low_confidence] = 0
    best_scores[low_confidence] = 0.0

    chord_labels = [chord_names[idx] for idx in best_indices]
    chord_probs = best_scores

    frame_duration = hop_length / sample_rate
    chord_times = np.arange(len(chord_labels)) * frame_duration

    return chord_times, chord_labels, chord_probs


def predict_chords_beat_sync(
    chroma: np.ndarray,
    beat_times: np.ndarray,
    bpm: float,
    hop_length: int = 512,
    sample_rate: int = 22050,
    rhythmic_resolution: str = 'quarter',
    min_confidence: float = 0.4,
) -> tuple:
    """
    Predict chords using beat-synchronized template matching.

    Adaptive aggregation based on rhythmic resolution:
    - Fine resolutions (eighth, sixteenth): max similarity per frame (quasi real-time)
    - Medium resolutions (quarter): mean similarity per sub-interval
    - Coarse resolutions (whole, half): strong vote over full interval (stable)

    Args:
        chroma: np.ndarray of shape (12, T) - chromagram
        beat_times: np.ndarray of beat times in seconds
        bpm: detected tempo
        hop_length: hop length in samples
        sample_rate: audio sample rate
        rhythmic_resolution: 'whole', 'half', 'quarter', 'eighth', 'sixteenth', etc.
        min_confidence: minimum confidence threshold

    Returns:
        chord_times: np.ndarray of chord change times
        chord_labels: list of chord names
        chord_probs: np.ndarray of confidence values
    """
    template_matrix = build_template_matrix()
    chord_names = list(CHORD_DEFINITIONS.keys())

    # Normalize chroma
    chroma_norm = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-10)

    # Compute similarity for all frames
    similarities = template_matrix @ chroma_norm  # (num_chords, T)

    # Generate subdivision times
    subdivisions = _generate_subdivision_times(beat_times, bpm, rhythmic_resolution)

    # Determine aggregation mode based on resolution
    divisions_per_beat = RHYTHMIC_DIVISIONS.get(rhythmic_resolution, 2.0)
    use_max_aggregation = divisions_per_beat >= 4.0  # eighth, sixteenth → quasi real-time

    # Convert times to frame indices
    frame_duration = hop_length / sample_rate
    total_frames = chroma.shape[1]

    chord_times = []
    chord_labels = []
    chord_probs = []

    for i in range(len(subdivisions) - 1):
        start_time = subdivisions[i]
        end_time = subdivisions[i + 1]

        start_frame = int(start_time / frame_duration)
        end_frame = int(end_time / frame_duration)

        if start_frame >= total_frames:
            break

        end_frame = min(end_frame, total_frames)

        if end_frame <= start_frame:
            continue

        # Get similarities for this interval
        interval_similarities = similarities[:, start_frame:end_frame]  # (num_chords, n_frames)
        n_frames = interval_similarities.shape[1]

        if use_max_aggregation and n_frames > 1:
            # Fine resolution: take max per frame → detects rapid changes
            # For each frame, pick the best chord, then pick the most frequent
            frame_best_chords = np.argmax(interval_similarities, axis=0)  # (n_frames,)
            frame_best_scores = np.max(interval_similarities, axis=0)  # (n_frames,)

            # Weighted vote: each frame votes for its best chord, weighted by confidence
            chord_votes = np.zeros(len(chord_names))
            for chord_idx, score in zip(frame_best_chords, frame_best_scores):
                chord_votes[chord_idx] += score

            aggregate_scores = chord_votes
            best_idx = np.argmax(aggregate_scores)
            best_score = aggregate_scores[best_idx] / n_frames
        else:
            # Coarse/medium resolution: aggregate over the full interval
            aggregate_scores = np.sum(interval_similarities, axis=1) / n_frames
            best_idx = np.argmax(aggregate_scores)
            best_score = aggregate_scores[best_idx]

        # Apply confidence threshold
        if best_score < min_confidence:
            best_idx = 0  # 'N'
            best_score = 0.0

        chord_times.append(start_time)
        chord_labels.append(chord_names[best_idx])
        chord_probs.append(best_score)

    return np.array(chord_times), chord_labels, np.array(chord_probs)


def _generate_subdivision_times(
    beat_times: np.ndarray,
    bpm: float,
    rhythmic_resolution: str,
) -> np.ndarray:
    """
    Generate subdivision times based on rhythmic resolution.
    
    For divisions >= 4.0 (eighth, sixteenth): use short fixed windows (~50ms)
    for quasi real-time detection.
    For divisions 1.0-2.0 (half, quarter): subdivide beats evenly.
    For divisions < 1.0 (whole, dotted): group beats together.
    
    Args:
        beat_times: detected beat times
        bpm: tempo
        rhythmic_resolution: 'whole', 'half', 'quarter', 'eighth', 'sixteenth', etc.
    
    Returns:
        Array of subdivision times in seconds
    """
    if len(beat_times) < 2:
        return beat_times
    
    divisions_per_beat = RHYTHMIC_DIVISIONS.get(rhythmic_resolution, 2.0)
    beat_interval = 60.0 / bpm
    start_time = beat_times[0]
    end_time = beat_times[-1] + beat_interval
    
    if divisions_per_beat >= 4.0:
        # Fine resolution: short fixed windows for quasi real-time detection
        # eighth: ~100ms windows, sixteenth: ~50ms windows
        window_duration = 0.1 if divisions_per_beat < 6.0 else 0.05
        n_windows = int((end_time - start_time) / window_duration) + 1
        subdivisions = np.array([start_time + i * window_duration for i in range(n_windows)])
        
        if subdivisions[-1] < end_time:
            subdivisions = np.append(subdivisions, end_time)
        
        return subdivisions
    elif divisions_per_beat >= 1.0:
        # Medium resolution: subdivide each beat evenly
        subdivision_interval = beat_interval / divisions_per_beat
        n_subdivisions = int((end_time - start_time) / subdivision_interval) + 1
        subdivisions = np.array([start_time + i * subdivision_interval for i in range(n_subdivisions)])
        
        if subdivisions[-1] < end_time:
            subdivisions = np.append(subdivisions, end_time)
        
        return subdivisions
    else:
        # Coarse resolution: group beats together (1 point per N beats)
        beats_per_point = int(round(1.0 / divisions_per_beat))
        if beats_per_point < 2:
            beats_per_point = 2
        
        grouped_times = []
        for i in range(0, len(beat_times), beats_per_point):
            grouped_times.append(beat_times[i])
        
        # Add the final boundary
        last_beat_end = beat_times[-1] + beat_interval
        if grouped_times and grouped_times[-1] < last_beat_end - 0.01:
            grouped_times.append(last_beat_end)
        
        return np.array(grouped_times)


def get_chord_distribution(chord_labels: list) -> dict:
    """Get the distribution of chords in the prediction."""
    from collections import Counter
    counter = Counter(chord_labels)
    total = len(chord_labels)
    return {
        chord: round(count / total * 100, 1)
        for chord, count in counter.most_common()
        if chord != 'N'
    }
