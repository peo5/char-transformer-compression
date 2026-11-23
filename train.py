"""Step 3: train one character transformer for one epoch over train.txt, logging bits/char.

Usage: python train.py LANG SIZE DATA_DIR OUT_DIR      e.g. python train.py pt 1M data runs

The epoch visits every non-overlapping 256-char chunk of train.txt once, in a random order (seed 0).
Each batch is scored BEFORE the update on it, so the sum of those losses is a prequential code length
of the training data: a decoder that knows the seed, the architecture and the training recipe can
replay the same updates, so no bits are charged for the weights.

Writes OUT_DIR/LANG_SIZE.pt (weights) and OUT_DIR/LANG_SIZE.log.json (curves and totals).
"""
import json, math, sys, time
from pathlib import Path
import numpy as np
import torch
from model import CONTEXT, SIZES, CharTransformer, bits, chunks, load_ids

BATCH = 64
LR = {"0.1M": 3e-3, "1M": 2e-3, "10M": 1e-3}
WARMUP = 500
LOG_EVERY = 200
VAL_EVERY = 2000
VAL_CHUNKS = 2048                               # 0.5M chars of val.txt for the curve; full val at the end


@torch.no_grad()
def evaluate(model, raw, n_chunks):
    model.eval()
    starts = np.arange(n_chunks) * CONTEXT
    total = 0.0
    for i in range(0, n_chunks, BATCH):
        x, y = chunks(raw, starts[i:i + BATCH])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x.cuda())
        total += bits(logits, y.cuda()).sum().item()
    model.train()
    return total / (n_chunks * CONTEXT)


def main():
    lang, size, data_dir, out_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    train, val = load_ids(data_dir / lang / "train.txt"), load_ids(data_dir / lang / "val.txt")
    model = CharTransformer(**SIZES[size]).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR[size], betas=(0.9, 0.95), weight_decay=0.1)

    order = np.random.default_rng(0).permutation((len(train) - 1) // CONTEXT) * CONTEXT
    steps = len(order) // BATCH                  # the last partial batch is dropped (and not counted)
    print(f"{lang} {size}: {n_params} parameters, {steps} steps, {steps * BATCH * CONTEXT} chars", flush=True)

    log = {"lang": lang, "size": size, "params": n_params, "batch": BATCH, "lr": LR[size], "steps": steps,
           "curve": [], "val_curve": []}
    prequential_bits, interval_bits, t0 = 0.0, 0.0, time.time()
    for step in range(steps):
        lr = LR[size] * min(1, (step + 1) / WARMUP) * 0.5 * (1 + math.cos(math.pi * step / steps))
        for group in opt.param_groups:
            group["lr"] = lr
        x, y = chunks(train, order[step * BATCH:(step + 1) * BATCH])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x.cuda(non_blocking=True))
        loss_bits = bits(logits, y.cuda(non_blocking=True)).sum()
        (loss_bits / (BATCH * CONTEXT)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        interval_bits += loss_bits.item()
        if (step + 1) % LOG_EVERY == 0:
            prequential_bits += interval_bits
            chars = (step + 1) * BATCH * CONTEXT
            log["curve"].append([step + 1, chars, interval_bits / (LOG_EVERY * BATCH * CONTEXT)])
            print(f"step {step + 1} chars {chars} train {log['curve'][-1][2]:.4f} bpc "
                  f"lr {lr:.2e} {time.time() - t0:.0f}s", flush=True)
            interval_bits = 0.0
        if (step + 1) % VAL_EVERY == 0:
            log["val_curve"].append([step + 1, (step + 1) * BATCH * CONTEXT, evaluate(model, val, VAL_CHUNKS)])
            print(f"step {step + 1} val {log['val_curve'][-1][2]:.4f} bpc", flush=True)

    prequential_bits += interval_bits
    log["prequential_bits"] = prequential_bits
    log["prequential_chars"] = steps * BATCH * CONTEXT
    log["prequential_bpc"] = prequential_bits / log["prequential_chars"]
    log["val_bpc"] = evaluate(model, val, (len(val) - 1) // CONTEXT)
    log["seconds"] = time.time() - t0
    log["max_vram_gb"] = torch.cuda.max_memory_allocated() / 1e9
    torch.save(model.state_dict(), out_dir / f"{lang}_{size}.pt")
    (out_dir / f"{lang}_{size}.log.json").write_text(json.dumps(log))
    print(f"done: prequential {log['prequential_bpc']:.4f} bpc, val {log['val_bpc']:.4f} bpc, "
          f"{log['seconds']:.0f}s, {log['max_vram_gb']:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
