"""Step 2a: the entropy ladder. Plug-in conditional entropy H(X_n | k previous chars), k = 0..5.

Usage: python ngram_entropy.py DATA_DIR OUT_JSON
  Uses the first N chars of DATA_DIR/{pt,en}/train.txt for N = 10M and 100M.

H(X_n | X_{n-k}..X_{n-1}) = H(k+1-gram) - H(k-gram), each H from empirical counts (plug-in).
The plug-in estimate is biased LOW when the number of distinct contexts is not small compared to N,
so comparing N = 10M and 100M shows where the high-k numbers stop being trustworthy.
"""
import json, sys
import numpy as np

ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789 \n.,;:!?'\"-()"
V = len(ALPHABET)


def to_ids(path, n):
    raw = np.fromfile(path, dtype=np.uint8, count=n)
    table = np.zeros(256, dtype=np.uint8)
    table[np.frombuffer(ALPHABET.encode(), dtype=np.uint8)] = np.arange(V, dtype=np.uint8)
    return table[raw]


def block_entropy(ids, m):
    """Entropy in bits of the empirical distribution of m-grams (m = 0 gives 0)."""
    if m == 0:
        return 0.0
    codes = np.zeros(len(ids) - m + 1, dtype=np.int64)
    for j in range(m):                          # m-gram as a base-V number; V^6 < 2^63
        codes = codes * V + ids[j:len(ids) - m + 1 + j]
    counts = np.unique(codes, return_counts=True)[1] if m > 3 else np.bincount(codes)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def main():
    data_dir, out_path = sys.argv[1], sys.argv[2]
    results = {}
    for lang in ("pt", "en"):
        for n in (10_000_000, 100_000_000):
            ids = to_ids(f"{data_dir}/{lang}/train.txt", n)
            h = [block_entropy(ids, m) for m in range(7)]
            results[f"{lang}_{n // 1_000_000}M"] = [h[k + 1] - h[k] for k in range(6)]
            print(lang, n, [round(x, 3) for x in results[f"{lang}_{n // 1_000_000}M"]], flush=True)
    json.dump(results, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
