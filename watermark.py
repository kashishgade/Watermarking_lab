import torch
import numpy as np
import hashlib
from transformers import LogitsProcessor


class WatermarkProcessor(LogitsProcessor):
    def __init__(self, vocab_size, key="its-secret", fraction=0.5):
        self.vocab_size = vocab_size
        self.key = key
        self.fraction = fraction

    def _hash(self, tokens):
        token_list = tokens.tolist() if hasattr(tokens, 'tolist') else list(tokens)
        data = (str(token_list) + self.key).encode()
        return int(hashlib.sha256(data).hexdigest(), 16)

    def _allowed_tokens(self, tokens):
        h = self._hash(tokens)
        rng = np.random.RandomState(h % (2**32))
        k = int(self.fraction * self.vocab_size)
        return rng.choice(self.vocab_size, k, replace=False)

    def __call__(self, input_ids, scores):
        for i in range(input_ids.shape[0]):
            allowed = self._allowed_tokens(input_ids[i])
            scores[i, allowed] += 1.5

        return scores