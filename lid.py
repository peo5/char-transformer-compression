"""Step 6: language identification as a prediction we can test.

Usage: python lid.py RUNS_DIR OUT_JSON      (needs the .npy files written by evaluate.py)

For a snippet in language A, the per-character log-likelihood ratio is L_i = log2 q_A(x_i) - log2 q_B(x_i)
(B = the other language). Classify the snippet as A when sum_i L_i > 0.
From the first half of the test chunks we measure D = mean L_i and s^2 = var L_i and predict the snippet
length n needed for error eps:
  Stein / large deviations:  n ~ log2(1/eps) / D
  CLT (L_i treated as iid):   n ~ (z_eps * s / D)^2,  z_eps = normal quantile of 1 - eps
The second half of the chunks (disjoint text) gives the measured error of the rule at every n = 1..256.
"""
import json, sys
from pathlib import Path
from statistics import NormalDist
import numpy as np

SIZES = ("0.1M", "1M", "10M")
EPSILONS = (0.1, 0.01, 0.001)


def main():
    runs_dir = Path(sys.argv[1])
    results = {}
    for size in SIZES:
        for a, b in (("pt", "en"), ("en", "pt")):
            own, other = runs_dir / "eval" / f"{a}_{size}_on_{a}.npy", runs_dir / "eval" / f"{b}_{size}_on_{a}.npy"
            if not (own.exists() and other.exists()):
                continue
            llr = np.load(other) - np.load(own)              # bits: -log2 q_B - (-log2 q_A) = log2 q_A/q_B
            half = len(llr) // 2
            d, s = llr[:half].mean(), llr[:half].std()
            error_by_n = (np.cumsum(llr[half:], axis=1) <= 0).mean(0)       # ties count as errors
            r = {"D_bits": float(d), "std_bits": float(s), "test_snippets": int(len(llr) - half),
                 "error_by_n": error_by_n.tolist(), "eps": {}}
            for eps in EPSILONS:
                below = np.nonzero(error_by_n <= eps)[0]
                r["eps"][str(eps)] = {
                    "n_stein": float(np.log2(1 / eps) / d),
                    "n_clt": float((NormalDist().inv_cdf(1 - eps) * s / d) ** 2),
                    "n_measured": int(below[0] + 1) if len(below) else None}
            results[f"{size}_{a}_vs_{b}"] = r
            print(size, f"{a} text: D = {d:.3f} bits/char, std {s:.3f};",
                  {k: {kk: (round(vv, 1) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                   for k, v in r["eps"].items()}, flush=True)
    Path(sys.argv[2]).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
