"""Step 7: description lengths. Model shared (like a dictionary) vs two-part (weights + code) vs prequential.

Usage: python mdl.py RUNS_DIR OUT_JSON

For each model, in bits per character of the text being described:
  shared        test cross-entropy (the receiver already has the model)
  two_part_test (32 * params + 10M * test bpc) / 10M     - ship fp32 weights with the 10 MB test file
  two_part_train (32 * params + N * val bpc) / N         - same for the N = 400M training chars; the final
                                                          model's train loss is taken as its val loss (one epoch,
                                                          every training char seen once, so no overfitting)
  prequential   sum of losses before each update / N     - no charge for weights at all
"""
import json, sys
from pathlib import Path

BITS_PER_WEIGHT = 32


def main():
    runs = Path(sys.argv[1])
    summary = json.loads((runs / "eval" / "summary.json").read_text())
    out = {}
    for log_path in sorted(runs.glob("*.log.json")):
        log = json.loads(log_path.read_text())
        lang, size, params = log["lang"], log["size"], log["params"]
        test = summary[f"{lang}_{size}_on_{lang}"]
        weights_bits = BITS_PER_WEIGHT * params
        n_train = log["prequential_chars"]
        out[f"{lang}_{size}"] = {
            "params": params, "weights_bits": weights_bits,
            "shared": test["bpc"],
            "two_part_test": (weights_bits + test["chars"] * test["bpc"]) / test["chars"],
            "two_part_train": (weights_bits + n_train * log["val_bpc"]) / n_train,
            "prequential": log["prequential_bpc"]}
        print(lang, size, {k: round(v, 4) if isinstance(v, float) else v for k, v in out[f"{lang}_{size}"].items()})
    Path(sys.argv[2]).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
