"""gestureEdge — simple Soli CNN+LSTM dual-radar gesture recognition.

Pattern follows:
  https://github.com/4uf04eG/FMCW-gesture-recognition
CNN frame features → LSTM temporal → FC. Dual BGT via light cross-attn fuse.
"""

from .model import GestureEdgeNet
from .preprocess import SOLI_ID_TO_NAME, SOLI_LABELS, soli_name

__all__ = ["GestureEdgeNet", "SOLI_LABELS", "SOLI_ID_TO_NAME", "soli_name"]
