import os
import shutil
import subprocess
import sys


FFMPEG_CANDIDATES = [
    r"C:\FFmpeg\bin\ffmpeg.exe",
    r"C:\ffmpeg\bin\ffmpeg.exe",
    r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
    r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
    r"C:\Users\33651\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg\bin\ffmpeg.exe",
    r"C:\Users\33651\scoop\apps\ffmpeg\current\ffmpeg.exe",
]

FFPROBE_CANDIDATES = [
    r"C:\FFmpeg\bin\ffprobe.exe",
    r"C:\ffmpeg\bin\ffprobe.exe",
    r"C:\Program Files\ffmpeg\bin\ffprobe.exe",
    r"C:\Program Files (x86)\ffmpeg\bin\ffprobe.exe",
    r"C:\Users\33651\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg\bin\ffprobe.exe",
    r"C:\Users\33651\scoop\apps\ffmpeg\current\ffprobe.exe",
]


def _find_executable(candidates: list, name: str) -> str:
    for path in candidates:
        if os.path.isfile(path):
            print(f"[ffmpeg_utils] Found {name} at: {path}", file=sys.stderr)
            return path

    found = shutil.which(name)
    if found:
        print(f"[ffmpeg_utils] Found {name} via PATH: {found}", file=sys.stderr)
        return found

    env_paths = os.environ.get("PATH", "").split(os.pathsep)
    for p in env_paths:
        candidate = os.path.join(p, f"{name}.exe")
        if os.path.isfile(candidate):
            print(f"[ffmpeg_utils] Found {name} in PATH dir: {candidate}", file=sys.stderr)
            return candidate

    raise FileNotFoundError(
        f"{name} introuvable. Installez-le via :\n"
        f"  winget install Gyan.FFmpeg\n"
        f"ou téléchargez-le sur https://www.gyan.dev/ffmpeg/builds/\n"
        f"PATH actuel : {os.environ.get('PATH', '')}"
    )


def get_ffmpeg_path() -> str:
    return _find_executable(FFMPEG_CANDIDATES, "ffmpeg")


def get_ffprobe_path() -> str:
    return _find_executable(FFPROBE_CANDIDATES, "ffprobe")


def run_ffmpeg(args: list, timeout: int = 600) -> subprocess.CompletedProcess:
    ffmpeg = get_ffmpeg_path()
    print(f"[ffmpeg_utils] Running: {ffmpeg} {' '.join(args[:5])}...", file=sys.stderr)
    result = subprocess.run(
        [ffmpeg] + args,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {result.stderr}")
    return result


def run_ffprobe(args: list, timeout: int = 30) -> subprocess.CompletedProcess:
    ffprobe = get_ffprobe_path()
    result = subprocess.run(
        [ffprobe] + args,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"FFprobe error: {result.stderr}")
    return result
