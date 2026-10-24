"""Small local imagehash compatibility shim for procvlm imports."""
from __future__ import annotations
import numpy as np
try:
    import cv2
except Exception:
    cv2 = None
class ImageHash:
    def __init__(self, bits): self.hash = np.asarray(bits, dtype=bool)
    def __sub__(self, other): return int(np.count_nonzero(self.hash != other.hash))
def phash(image, hash_size=8, highfreq_factor=4):
    size = hash_size * highfreq_factor
    arr = np.asarray(image.convert("L").resize((size, size)), dtype=np.float32)
    if cv2 is not None:
        dct = cv2.dct(arr)
    else:
        # DCT-II fallback, adequate for duplicate-frame detection.
        n = size; x = np.arange(n)
        basis = np.cos(np.pi / n * (x[:, None] + .5) * x[None, :])
        dct = basis.T @ arr @ basis
    low = dct[:hash_size, :hash_size]
    med = np.median(low[1:, :])
    return ImageHash(low > med)
