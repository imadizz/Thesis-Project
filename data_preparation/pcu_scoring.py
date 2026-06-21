"""
pcu_scoring.py
PCU weighting and congestion class assignment.
Highway Capacity Manual (2010) PCU values.
"""

PCU_WEIGHTS = {
    'car':        1.0,
    'truck':      2.5,
    'bus':        3.0,
    'motorcycle': 0.5,
    'bicycle':    0.5,
    'person':     0.3,
}

CONGESTION_THRESHOLDS = {
    'FREE_FLOW': (0.00, 0.30),
    'MODERATE':  (0.30, 0.60),
    'HEAVY':     (0.60, 0.85),
    'GRIDLOCK':  (0.85, 1.01),
}


def compute_pcu_score(counts: dict) -> float:
    """Return raw PCU-weighted vehicle count from a category count dict."""
    return sum(counts.get(cat, 0) * w for cat, w in PCU_WEIGHTS.items())


def assign_congestion_class(normalised_score: float) -> str:
    """Map a normalised 0-1 PCU score to a congestion class string."""
    for label, (lo, hi) in CONGESTION_THRESHOLDS.items():
        if lo <= normalised_score < hi:
            return label
    return 'GRIDLOCK'


class RollingNormaliser:
    """
    Normalise PCU scores within a rolling window per scene category.
    Prevents dense urban scene outliers from compressing the range.
    """

    def __init__(self, window=100):
        self.window = window
        self._history: dict[str, list] = {}

    def normalise(self, scene: str, raw_score: float) -> float:
        if scene not in self._history:
            self._history[scene] = []
        hist = self._history[scene]
        hist.append(raw_score)
        if len(hist) > self.window:
            hist.pop(0)
        max_val = max(hist) if hist else 1.0
        return raw_score / max_val if max_val > 0 else 0.0
