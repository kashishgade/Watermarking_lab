# attacks.py

import random
def deletion(text: str, p: float = 0.2) -> str:
    words = text.split()
    return " ".join([w for w in words if random.random() > p])


def swap(text: str, n: int = 3) -> str:
    words = text.split()
    for _ in range(min(n, len(words) - 1)):
        i, j = random.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]
    return " ".join(words)


def insertion(text: str, n: int = 3) -> str:
    words = text.split()
    for _ in range(n):
        idx = random.randint(0, len(words))
        words.insert(idx, random.choice(words))
    return " ".join(words)