import os
import sys
import subprocess
from pathlib import Path
from config import UPLOAD_DIR
from core.ffmpeg_utils import run_ffmpeg


def separate_sources(audio_path: str) -> str:
    """
    Sépare les sources audio avec Demucs v4 et retourne le chemin
    de la piste d'accompagnement (bass + drums + other, sans vocals).
    """
    audio_p = Path(audio_path)
    output_dir = UPLOAD_DIR / f"{audio_p.stem}_separated"

    print(f"[source_separator] Running Demucs with: {sys.executable}")
    subprocess.run(
        [sys.executable, "-m", "demucs", "--two-stems", "vocals", "-n", "htdemucs", "-o", str(output_dir), str(audio_p)],
        capture_output=True, text=True, timeout=1800,
    )

    demucs_out = output_dir / "htdemucs" / audio_p.stem
    backing_parts = []

    for stem in ["bass", "drums", "other"]:
        stem_file = demucs_out / f"{stem}.wav"
        if stem_file.exists():
            backing_parts.append(str(stem_file))

    if not backing_parts:
        no_vocals = demucs_out / "no_vocals.wav"
        if no_vocals.exists():
            return str(no_vocals)
        return str(audio_path)

    return _mix_stems(backing_parts, str(demucs_out / "backing.wav"))


def _mix_stems(stem_paths: list, output_path: str) -> str:
    """Mixe plusieurs stems audio avec FFmpeg."""
    inputs = []
    for s in stem_paths:
        inputs.extend(["-i", s])

    filter_complex = "".join(f"[{i}:a]" for i in range(len(stem_paths)))
    filter_complex += f"amix=inputs={len(stem_paths)}:duration=longest[out]"

    run_ffmpeg([
        "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-ar", "22050",
        "-ac", "1",
        output_path,
    ], timeout=300)

    if os.path.exists(output_path):
        return output_path

    return stem_paths[0]
