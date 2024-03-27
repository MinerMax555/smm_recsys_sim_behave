from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

def top_k_item_ids(scores: np.ndarray, k: int = 10) -> pd.DataFrame:
    n = len(scores)
    item_ids, score = [], []

    for s in tqdm(scores, desc=f'calculating top-{k} item_ids', smoothing=0):
        items = np.argsort(-s)[:k]
        # Martin: Only get country-specific items:

        for item in items:
            item_ids.append(item)
            score.append(s[item])

    user_ids = []
    for i in range(n):
        user_ids.extend([i] * k)

    return pd.DataFrame.from_dict({
        'user_id': user_ids,
        'item_id': item_ids,
        'rank': list(range(1, k + 1)) * n,
        'score': score
    })