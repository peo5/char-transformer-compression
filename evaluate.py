"""Steps 3 (calibration, loss by position) and 5 (cross-language matrix): score test sets with trained models.

Usage: python evaluate.py DATA_DIR RUNS_DIR
  For every RUNS_DIR/{lang}_{size}.pt and every test language, cuts DATA_DIR/{lang}/test.txt into
  non-overlapping 257-char windows (stride 256), each scored with no context from before it: position i
  predicts char i+1 from chars 0..i, so position i has i+1 chars of context. Float32, no AMP.
  Writes RUNS_DIR/eval/{model}_on_{test}.npy  (bits per char, float32, shape chunks x 256) and
           RUNS_DIR/eval/summary.json         (cross-entropy, mean predicted entropy, both per position).
"""
import json, sys
from pathlib import Path
import numpy as np
import torch
from model import CONTEXT, SIZES, CharTransformer, bits, chunks, load_ids

BATCH = 64


@torch.no_grad()
def score(model, raw):
    starts = np.arange((len(raw) - 1) // CONTEXT) * CONTEXT
    all_bits, entropy_sum = [], torch.zeros(CONTEXT, device="cuda", dtype=torch.float64)
    for i in range(0, len(starts), BATCH):
        x, y = chunks(raw, starts[i:i + BATCH])
        logits = model(x.cuda()).float()
        all_bits.append(bits(logits, y.cuda()).cpu().numpy())
        logp = torch.log_softmax(logits, -1)
        entropy_sum += (-(logp.exp() * logp).sum(-1) / np.log(2)).sum(0).double()
    return np.concatenate(all_bits), (entropy_sum / len(starts)).cpu().numpy()


def main():
    data_dir, runs_dir = Path(sys.argv[1]), Path(sys.argv[2])
    (runs_dir / "eval").mkdir(exist_ok=True)
    summary_path = runs_dir / "eval" / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for weights in sorted(runs_dir.glob("*_*.pt")):
        name = weights.stem
        size = name.split("_")[1]
        model = CharTransformer(**SIZES[size]).cuda().eval()
        model.load_state_dict(torch.load(weights, map_location="cuda"))
        for test in ("pt", "en"):
            key = f"{name}_on_{test}"
            if key in summary:
                continue
            per_char, entropy = score(model, load_ids(data_dir / test / "test.txt"))
            np.save(runs_dir / "eval" / f"{key}.npy", per_char)
            summary[key] = {"bpc": float(per_char.mean()), "mean_entropy": float(entropy.mean()),
                            "bpc_by_position": per_char.mean(0).tolist(), "entropy_by_position": entropy.tolist(),
                            "chars": int(per_char.size)}
            print(key, f"{summary[key]['bpc']:.4f} bpc, mean entropy {summary[key]['mean_entropy']:.4f}", flush=True)
            summary_path.write_text(json.dumps(summary))


if __name__ == "__main__":
    main()
