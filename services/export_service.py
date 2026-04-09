import json
import os
from pathlib import Path
from config import RESULTS_DIR


def export_json(result: dict, job_id: str) -> str:
    """Exporte les résultats au format JSON."""
    output_path = RESULTS_DIR / f"{job_id}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return str(output_path)


def export_lrc(result: dict, job_id: str) -> str:
    """
    Exporte les accords au format LRC (synchronisés avec le temps).
    Format: [mm:ss.xx] Accord
    """
    output_path = RESULTS_DIR / f"{job_id}.lrc"
    lines = []

    lines.append(f"[ti:FineChord Analysis]")
    lines.append(f"[key:{result.get('key', 'C')}]")
    lines.append(f"[bpm:{result.get('bpm', 120)}]")
    lines.append("")

    for seg in result.get("segments", []):
        minutes = int(seg["start"] // 60)
        seconds = seg["start"] % 60
        timestamp = f"[{minutes:02d}:{seconds:05.2f}]"
        lines.append(f"{timestamp} {seg['chord']}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return str(output_path)


def export_midi(result: dict, job_id: str) -> str:
    """
    Exporte les accords au format MIDI basique.
    Chaque accord est représenté par ses notes sur un canal.
    """
    import mido
    from mido import MidiTrack, Message, MetaMessage

    CHORD_NOTES = {
        "C": [60, 64, 67], "Cm": [60, 63, 67], "C7M": [60, 64, 67, 71],
        "C#": [61, 65, 68], "C#m": [61, 64, 68],
        "D": [62, 66, 69], "Dm": [62, 65, 69], "D7": [62, 66, 69, 72], "D7M": [62, 66, 69, 73],
        "D#": [63, 67, 70], "D#m": [63, 66, 70],
        "E": [64, 68, 71], "Em": [64, 67, 71], "E7": [64, 68, 71, 74],
        "F": [65, 69, 72], "Fm": [65, 68, 72], "F7M": [65, 69, 72, 76],
        "F#": [66, 70, 73], "F#m": [66, 69, 73],
        "G": [67, 71, 74], "Gm": [67, 70, 74], "G7": [67, 71, 74, 77], "G7M": [67, 71, 74, 78],
        "G#": [68, 72, 75], "G#m": [68, 71, 75],
        "A": [69, 73, 76], "Am": [69, 72, 76], "A7": [69, 73, 76, 79], "A7M": [69, 73, 76, 80],
        "A#": [70, 74, 77], "A#m": [70, 73, 77],
        "B": [71, 75, 78], "Bm": [71, 74, 78], "B7": [71, 75, 78, 81],
    }

    BPM = result.get("bpm", 120)
    ticks_per_beat = 480
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)

    track = MidiTrack()
    mid.tracks.append(track)

    track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(BPM), time=0))
    track.append(MetaMessage('time_signature', numerator=4, denominator=4, time=0))
    track.append(MetaMessage('key_signature', key=result.get('key', 'C'), time=0))

    segments = result.get("segments", [])
    for i, seg in enumerate(segments):
        chord = seg["chord"]
        start_time = seg["start"]
        end_time = seg["end"]

        if chord == "N" or chord not in CHORD_NOTES:
            continue

        start_ticks = mido.second2tick(start_time, ticks_per_beat, BPM)
        end_ticks = mido.second2tick(end_time, ticks_per_beat, BPM)
        duration_ticks = end_ticks - start_ticks
        if duration_ticks <= 0:
            duration_ticks = ticks_per_beat

        for note in CHORD_NOTES.get(chord, []):
            track.append(Message('note_on', note=note, velocity=80, time=0))
        track.append(Message('note_off', note=0, time=duration_ticks))

    output_path = RESULTS_DIR / f"{job_id}.mid"
    mid.save(output_path)
    return str(output_path)
