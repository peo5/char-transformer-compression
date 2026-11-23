"""Description lengths: four ways to count the bits of the same model (RESULTS.md, section 8).

Usage: python src/mdl.py RUNS_DIR OUT_JSON

For each model, all four numbers are in bits per character of the text being described:
  shared          the test cross-entropy; the receiver already has the model, like a dictionary
  two_part_test   (32 * params + 10M * test bpc) / 10M; the 32-bit weights are sent with the 10 MB test file
  two_part_train  (32 * params + N * val bpc) / N, the same for the N = 400M training characters. We use the
                  final validation loss as the training loss: with one epoch, the model saw each training
                  character only once, so it could not overfit them.
  prequential     the sum of the losses measured before each update, divided by N; the weights cost nothing
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
