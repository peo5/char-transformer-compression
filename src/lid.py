"""Language identification: a prediction we can test (RESULTS.md, section 7).

Usage: python src/lid.py RUNS_DIR OUT_JSON      (needs the .npy files written by evaluate.py)

Take a snippet written in language A, and let B be the other language. For each character, the log-likelihood
ratio is L_i = log2 q_A(x_i) - log2 q_B(x_i). The rule classifies the snippet as A when the sum of the L_i is
positive.

On the first half of the test chunks, the script measures the mean D and the variance s^2 of L_i. From them it
predicts the snippet length n needed to reach an error rate eps, in two ways:
  Stein's lemma (large deviations):              n ~ log2(1/eps) / D
  central limit theorem (L_i treated as iid):    n ~ (z_eps * s / D)^2, where z_eps is the normal quantile of 1 - eps
The second half of the chunks is different text. On it, the script measures the real error of the rule for every
n from 1 to 256.
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
