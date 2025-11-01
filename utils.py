# utils.py
import numpy as np
import winsound  # Native Windows beep (no installation required)

def eye_aspect_ratio(points):
    """
    Computes the Eye Aspect Ratio (EAR) for an eye using 6 key points.
    """
    p = np.array(points, dtype=np.float32)
    A = np.linalg.norm(p[1] - p[5])
    B = np.linalg.norm(p[2] - p[4])
    C = np.linalg.norm(p[0] - p[3])
    if C == 0:
        return 0.0
    ear = (A + B) / (2.0 * C)
    return float(ear)

def play_beep():
    """
    Windows built-in beep: frequency 1000 Hz for 300 ms.
    """
    winsound.Beep(1000, 300)
