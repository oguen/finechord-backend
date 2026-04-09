import numpy as np
import sys
sys.path.insert(0, '.')

# Use basic audio loading
try:
    import soundfile as sf
    audio_path = r'E:\FineChord_Final\audio test\piano test.mp3'
    print('Loading with soundfile...')
    y, sr = sf.read(audio_path)
    if len(y.shape) > 1:
        y = y.mean(axis=1)
    sr = 22050
except:
    print('soundfile failed, using scipy')
    from scipy.io import wavfile
    audio_path = r'E:\FineChord_Final\audio test\piano test.mp3'
    import subprocess
    subprocess.run(['ffmpeg', '-y', '-i', audio_path, '-ar', '22050', '-ac', '1', 'temp_audio.wav'])
    sr, y = wavfile.read('temp_audio.wav')

print(f'Audio: {len(y)/sr:.1f}s')

# Simple FFT-based chroma
from scipy.fft import fft
from scipy.signal import find_peaks

print('\n=== Simple Pitch Analysis ===')
# Take first 10 seconds for analysis
sample_len = min(10 * sr, len(y))
samples = y[:sample_len]

# Apply FFT
fft_result = np.abs(fft(samples))
freqs = np.fft.fftfreq(len(samples), 1/sr)

# Find dominant frequencies
peak_indices, _ = find_peaks(fft_result[1:], height=np.max(fft_result)*0.1)
if len(peak_indices) > 0:
    peak_freqs = freqs[peak_indices + 1]
    peak_freqs = peak_freqs[peak_freqs > 50]  # Filter low freqs
    
    print('Top frequencies:')
    sorted_peaks = sorted(zip(peak_freqs, fft_result[peak_indices + 1][:len(peak_freqs)]), key=lambda x: -x[1])[:10]
    for f, v in sorted_peaks:
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        midi_note = round(69 + 12 * np.log2(f/440))
        note_idx = midi_note % 12
        octave = midi_note // 12 - 1
        print(f'  {f:.1f} Hz -> {note_names[note_idx]}{octave}')
