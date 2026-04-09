import uuid
import time
import os
import traceback
import numpy as np

from config import (
    UPLOAD_DIR,
    RESULTS_DIR,
    AUDIO_SAMPLE_RATE,
    HOP_LENGTH,
    CHORD_CLASSES,
    CHORD_TO_INDEX,
    INDEX_TO_CHORD,
    NUM_CHORD_CLASSES,
)
from core.audio_loader import extract_audio, get_duration
from core.source_separator import separate_sources
from core.feature_extractor import extract_cqt_chroma, extract_cqt_raw
from core.beat_tracker import detect_beats, sync_chords_to_beats
from core.chord_templates import predict_chords_beat_sync
from core.post_processor import (
    merge_short_segments,
    detect_key,
    detect_key_from_chroma,
    detect_key_from_audio,
    combine_key_detections,
    build_analysis_result,
)
from core.bass_detector import detect_bass_for_intervals


class AnalysisService:
    """Orchestrateur du pipeline d'analyse harmonique complet."""

    def __init__(self):
        pass

    def analyze(
        self,
        file_path: str,
        use_separation: bool = False,
        rhythmic_resolution: str = 'half',
        min_confidence: float = 0.4,
    ) -> dict:
        """
        Pipeline complet d'analyse harmonique.

        Args:
            file_path: chemin du fichier audio
            use_separation: activer la séparation de sources
            rhythmic_resolution: 'whole', 'half', 'quarter', 'eighth', 'sixteenth'
            min_confidence: seuil de confiance minimum
        """
        job_id = uuid.uuid4().hex[:12]
        start_time = time.time()

        print(f"\n[AnalysisService] Job {job_id} started")
        print(f"[AnalysisService] Input: {file_path}")
        print(f"[AnalysisService] Resolution: {rhythmic_resolution}, Confidence: {min_confidence}")

        try:
            print("[AnalysisService] Step 1/6: Extracting audio...")
            wav_path = extract_audio(file_path)
            duration = get_duration(wav_path)
            print(f"[AnalysisService] Audio extracted: {duration:.2f}s — {wav_path}")
        except Exception as e:
            print(f"[AnalysisService] ERROR at step 1 (extract audio): {e}")
            traceback.print_exc()
            raise

        backing_path = wav_path
        if use_separation:
            try:
                print("[AnalysisService] Step 2/6: Separating sources (Demucs)...")
                backing_path = separate_sources(wav_path)
                print(f"[AnalysisService] Backing track ready: {backing_path}")
            except Exception as e:
                print(f"[AnalysisService] Source separation failed, using full mix: {e}")
                traceback.print_exc()
                backing_path = wav_path
        else:
            print("[AnalysisService] Step 2/6: Skipping source separation")

        try:
            print("[AnalysisService] Step 3/6: Extracting CQT chroma features...")
            chroma = extract_cqt_chroma(backing_path)
            print(f"[AnalysisService] Chroma shape: {chroma.shape}")
        except Exception as e:
            print(f"[AnalysisService] ERROR at step 3 (extract chroma): {e}")
            traceback.print_exc()
            raise

        try:
            print("[AnalysisService] Step 3b/6: Extracting raw CQT for bass detection...")
            cqt_raw = extract_cqt_raw(backing_path)
            print(f"[AnalysisService] Raw CQT shape: {cqt_raw.shape}")
        except Exception as e:
            print(f"[AnalysisService] WARNING: Could not extract raw CQT for bass: {e}")
            cqt_raw = None

        try:
            print("[AnalysisService] Step 4/6: Detecting beats and BPM...")
            beat_times, bpm = detect_beats(backing_path)
            print(f"[AnalysisService] BPM: {bpm:.1f}, Beats: {len(beat_times)}")
        except Exception as e:
            print(f"[AnalysisService] ERROR at step 4 (beat detection): {e}")
            traceback.print_exc()
            raise

        try:
            print(f"[AnalysisService] Step 5/6: Predicting chords (beat-synced, {rhythmic_resolution})...")
            chord_times, chord_labels, chord_probs = predict_chords_beat_sync(
                chroma,
                beat_times,
                bpm,
                HOP_LENGTH,
                AUDIO_SAMPLE_RATE,
                rhythmic_resolution=rhythmic_resolution,
                min_confidence=min_confidence,
            )
            print(f"[AnalysisService] Predicted {len(chord_labels)} chord segments")

            print("[AnalysisService] Step 5b/6: Post-processing...")
            unique_before = len(set(chord_labels) - {"N"})
            print(f"[AnalysisService] Unique chords before merging: {unique_before}")

            chord_times, chord_labels = merge_short_segments(chord_times, chord_labels)
            unique_after = len(set(chord_labels) - {"N"})
            print(f"[AnalysisService] Segments after merging: {len(chord_labels)}, unique chords: {unique_after}")

            if cqt_raw is not None:
                print("[AnalysisService] Step 5c/6: Detecting bass notes...")
                frame_duration = HOP_LENGTH / AUDIO_SAMPLE_RATE
                start_frames = [int(t / frame_duration) for t in chord_times]
                end_frames = []
                for i in range(len(chord_times) - 1):
                    end_frames.append(int(chord_times[i + 1] / frame_duration))
                end_frames.append(int((chord_times[-1] + 1.0) / frame_duration))
                bass_notes = detect_bass_for_intervals(cqt_raw, start_frames, end_frames)
                print(f"[AnalysisService] Bass notes detected: {sum(1 for b in bass_notes if b)}/{len(bass_notes)} segments with bass")
            else:
                bass_notes = [''] * len(chord_labels)

            beat_times_synced, beat_chords = sync_chords_to_beats(chord_times, chord_labels, beat_times)

            # Improved key detection using both chord distribution and chroma features
            chord_based_key = detect_key(chord_labels)
            chroma_key, chroma_mode, chroma_confidence, chroma_corr = detect_key_from_chroma(chroma)
            
            # Combine both methods for better accuracy
            key = combine_key_detections(chord_based_key, chroma_key, chroma_confidence)
            
            confidence = float(np.mean(chord_probs)) if len(chord_probs) > 0 else 0.0

            print(f"[AnalysisService] Key detected (chord-based): {chord_based_key}")
            print(f"[AnalysisService] Key detected (chroma-based): {chroma_key} ({chroma_mode}), confidence: {chroma_confidence:.3f}")
            print(f"[AnalysisService] Final key: {key}")
            print(f"[AnalysisService] Chord confidence: {confidence:.3f}")
        except Exception as e:
            print(f"[AnalysisService] ERROR at step 5 (predict chords): {e}")
            traceback.print_exc()
            raise

        try:
            print("[AnalysisService] Step 6/6: Building results...")
            result = build_analysis_result(
                chord_times=chord_times,
                chord_labels=chord_labels,
                beat_times=beat_times_synced,
                beat_chords=beat_chords,
                bpm=round(bpm, -1),
                key=key,
                duration=duration,
                confidence=confidence,
                bass_notes=bass_notes,
            )

            from services.export_service import export_json, export_lrc, export_midi
            export_json(result, job_id)
            export_lrc(result, job_id)
            export_midi(result, job_id)

            elapsed = time.time() - start_time
            result["processing_time"] = round(elapsed, 2)
            result["job_id"] = job_id

            print(f"[AnalysisService] Job {job_id} completed in {elapsed:.2f}s")

            return result
        except Exception as e:
            print(f"[AnalysisService] ERROR at step 6 (build results): {e}")
            traceback.print_exc()
            raise
