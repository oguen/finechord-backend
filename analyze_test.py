import librosa
import numpy as np
import sys
sys.path.insert(0, '.')
from config import KRUMSHANSL_MAJOR, KRUMSHANSL_MINOR

audio_path = r'E:\FineChord_Final\audio test\piano test.mp3'

print('Loading audio...')
y, sr = librosa.load(audio_path, sr=22050, mono=True)
print(f'Audio loaded: {len(y)/sr:.1f}s at {sr}Hz')

print('\n=== Extracting chroma ===')
chroma = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=12)

print('\n=== Chroma Analysis ===')
chroma_sum = np.sum(chroma, axis=1)
notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

for note, val in sorted(zip(notes, chroma_sum), key=lambda x: -x[1]):
    print(f'{note}: {val:.1f}')

print('\n=== Top 5 Notes ===')
top5 = sorted(zip(notes, chroma_sum), key=lambda x: -x[1])[:5]
print('Top notes:', [n for n, v in top5])

print('\n=== Key Detection with Krumhansl ===')
major_profile = np.array(KRUMSHANSL_MAJOR) / np.sum(KRUMSHANSL_MAJOR)
minor_profile = np.array(KRUMSHANSL_MINOR) / np.sum(KRUMSHANSL_MINOR)
chroma_norm = chroma_sum / np.sum(chroma_sum)

best_key = 'C'
best_mode = 'major'
best_corr = -1

for shift in range(12):
    shifted = np.roll(chroma_norm, shift)
    major_corr = np.corrcoef(shifted, major_profile)[0, 1]
    minor_corr = np.corrcoef(shifted, minor_profile)[0, 1]
    
    if major_corr > best_corr:
        best_corr = major_corr
        best_key = notes[shift]
        best_mode = 'major'
    
    if minor_corr > best_corr:
        best_corr = minor_corr
        best_key = notes[shift]
        best_mode = 'minor'
    
    status = '***' if notes[shift] in ['D', 'A', 'B', 'F#'] else ''
    print(f'{notes[shift]:3s}: major={major_corr:+.3f}, minor={minor_corr:+.3f} {status}')

print(f'\nBest key: {best_key} {best_mode} (corr={best_corr:.3f})')
