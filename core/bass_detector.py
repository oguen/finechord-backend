"""
Bass note detection from raw CQT magnitude.

Analyzes the low-frequency bins of the CQT (first 3 octaves, ~30-250 Hz)
to detect the dominant bass note in each frame.
"""

import numpy as np


NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def detect_bass_note(cqt_raw: np.ndarray, frame_idx: int, threshold: float = 0.3) -> str:
    """
    Detect the dominant bass note in a single CQT frame.
    
    Analyzes the first 3 octaves of the CQT (bins 0-107, ~30-250 Hz).
    Groups energy by pitch class across octaves and returns the most energetic note.
    
    Args:
        cqt_raw: np.ndarray of shape (252, T) - raw CQT magnitude
        frame_idx: index of the frame to analyze
        threshold: minimum energy ratio to consider a bass note valid
    
    Returns:
        Note name (e.g., 'C', 'F#') or empty string if no clear bass.
    """
    if frame_idx < 0 or frame_idx >= cqt_raw.shape[1]:
        return ''
    
    # First 3 octaves = 108 bins (36 bins/octave * 3 octaves)
    # This covers ~30-250 Hz, where bass notes live
    n_octaves_for_bass = 3
    n_bins_per_octave = 36
    n_bass_bins = n_octaves_for_bass * n_bins_per_octave
    
    bass_frame = cqt_raw[:n_bass_bins, frame_idx]
    
    # Group energy by pitch class (every 36 bins = 1 octave)
    note_energies = np.zeros(12)
    for note_idx in range(12):
        # Sum energy across all octaves for this pitch class
        note_energies[note_idx] = np.sum(bass_frame[note_idx::12])
    
    # Normalize
    total_energy = np.sum(note_energies)
    if total_energy < 1e-10:
        return ''
    
    note_ratios = note_energies / total_energy
    max_ratio = np.max(note_ratios)
    
    if max_ratio < threshold:
        return ''
    
    return NOTE_NAMES[int(np.argmax(note_energies))]


def detect_bass_sequence(
    cqt_raw: np.ndarray,
    frame_indices: list,
    threshold: float = 0.3,
) -> list:
    """
    Detect bass notes for a sequence of frames.
    
    Args:
        cqt_raw: np.ndarray of shape (252, T)
        frame_indices: list of frame indices to analyze
        threshold: minimum energy ratio
    
    Returns:
        List of bass note names (empty string if no clear bass).
    """
    bass_notes = []
    for idx in frame_indices:
        bass = detect_bass_note(cqt_raw, idx, threshold)
        bass_notes.append(bass)
    return bass_notes


def detect_bass_for_intervals(
    cqt_raw: np.ndarray,
    start_frames: list,
    end_frames: list,
    threshold: float = 0.3,
) -> list:
    """
    Detect bass notes for time intervals by analyzing the first frame of each interval.
    
    Args:
        cqt_raw: np.ndarray of shape (252, T)
        start_frames: list of start frame indices for each interval
        end_frames: list of end frame indices
        threshold: minimum energy ratio
    
    Returns:
        List of bass note names.
    """
    bass_notes = []
    for start, end in zip(start_frames, end_frames):
        # Analyze the first 3 frames of the interval for stability
        frames_to_check = []
        for f in range(start, min(start + 3, end)):
            if f < cqt_raw.shape[1]:
                frames_to_check.append(f)
        
        if not frames_to_check:
            bass_notes.append('')
            continue
        
        # Get bass for each frame, take the most common
        frame_basses = [detect_bass_note(cqt_raw, f, threshold) for f in frames_to_check]
        valid_basses = [b for b in frame_basses if b]
        
        if not valid_basses:
            bass_notes.append('')
        else:
            # Take the most frequent bass note
            from collections import Counter
            most_common = Counter(valid_basses).most_common(1)[0][0]
            bass_notes.append(most_common)
    
    return bass_notes
