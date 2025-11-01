# utils.py
import numpy as np
import simpleaudio as sa

def eye_aspect_ratio(points):
    """
    points: list/array of 6 (x,y) points for the eye in the order:
    [p1, p2, p3, p4, p5, p6] - any consistent ordering works as long as vertical/horizontal pairings are correct.
    We'll compute EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    p = np.array(points, dtype=np.float32)
    A = np.linalg.norm(p[1] - p[5])
    B = np.linalg.norm(p[2] - p[4])
    C = np.linalg.norm(p[0] - p[3])
    if C == 0:
        return 0.0
    ear = (A + B) / (2.0 * C)
    return float(ear)

def play_beep(frequency=880, duration_ms=300, volume=0.3):
    """
    Generate a sine beep and play it using simpleaudio.
    """
    fs = 44100  # sampling rate
    t = np.linspace(0, duration_ms/1000.0, int(fs * duration_ms/1000.0), False)
    tone = np.sin(frequency * 2 * np.pi * t)
    audio = tone * (32767 * volume)
    audio = audio.astype(np.int16)
    try:
        play_obj = sa.play_buffer(audio, 1, 2, fs)
        # Don't block long — we can allow it to play async
    except Exception as e:
        print("Beep playback failed:", e)
