# detector.py
import numpy as np
import hashlib
from scipy import stats


class WatermarkDetector:
    def __init__(self, vocab_size: int, key: str = "its-secret", fraction: float = 0.5):
        self.vocab_size = vocab_size
        self.key = key
        self.fraction = fraction

    def _normalize_tokens(self, tokens):
        """Normalize tokens to a plain Python list regardless of input type."""
        if hasattr(tokens, 'tolist'):
            return tokens.tolist()
        return list(tokens)

    def _hash(self, tokens):
        data = (str(tokens) + self.key).encode()
        return int(hashlib.sha256(data).hexdigest(), 16)

    def _allowed_tokens(self, tokens):
        h = self._hash(tokens)
        rng = np.random.RandomState(h % (2**32))
        k = int(self.fraction * self.vocab_size)
        return set(rng.choice(self.vocab_size, k, replace=False))

    def detect(self, tokens):
        
        tokens = self._normalize_tokens(tokens)

        matches = 0
        total = 0
        ratios = []

        for i in range(1, len(tokens)):
            prev = tokens[:i]           
            allowed = self._allowed_tokens(prev)

            if tokens[i] in allowed:
                matches += 1

            total += 1
            ratios.append(matches / total)

        final_ratio = matches / total if total else 0


        if total > 0:
            z_score = (matches - self.fraction * total) / np.sqrt(
                total * self.fraction * (1 - self.fraction)
            )
            # p < 0.01 one-sided  →  z > 2.326
            is_watermarked = bool(z_score > 2.326)
        else:
            z_score = 0.0
            is_watermarked = False

        return {
            "match_ratio": round(final_ratio, 4),
            "z_score": round(float(z_score), 4),
            "threshold_z": 2.326,
            "is_watermarked": is_watermarked,
            "trajectory": ratios,
        }