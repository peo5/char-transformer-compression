"""Make every figure from the result files (labels in pt-BR). Each function writes one PNG into FIG_DIR.

Usage: python figures.py RESULTS_DIR FIG_DIR
  RESULTS_DIR holds ngram.json, compressors.tsv and runs/ (the logs, eval/summary.json, lid.json).
"""
import csv, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLOR = {"pt": "#2a78d6", "en": "#eb6834"}        # one hue per language, everywhere
STYLE = {"0.1M": ":", "1M": "--", "10M": "-"}     # line style per model size
MARKER = {"0.1M": "o", "1M": "s", "10M": "D"}
SIZES = ("0.1M", "1M", "10M")
plt.rcParams.update({"axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.8, "axes.spines.top": False,
                     "axes.spines.right": False, "lines.linewidth": 2, "figure.dpi": 150, "font.size": 10})


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def entropy_ladder(res, out):
    ngram = json.loads((res / "ngram.json").read_text())
    comp = list(csv.DictReader(open(res / "compressors.tsv"), delimiter="\t"))
    summary = json.loads((res / "runs/eval/summary.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, lang in zip(axes, ("pt", "en")):
        for n, style in (("10M", "--"), ("100M", "-")):
            ax.plot(range(6), ngram[f"{lang}_{n}"], style, marker="o", color=COLOR[lang],
                    label=f"H(X | k anteriores) plug-in, primeiros {n} caracteres do treino")
        for row in comp:
            if row["lang"] == lang:
                ax.axhline(float(row["bits_per_char"]), color="#8a8984", linewidth=1)
                ax.text(5, float(row["bits_per_char"]), row["compressor"].split()[0], ha="right",
                        va="top" if row["compressor"].startswith("xz") else "bottom",
                        fontsize=8, color="#52514e")
        for size in SIZES:
            key = f"{lang}_{size}_on_{lang}"
            if key in summary:
                ax.axhline(summary[key]["bpc"], color=COLOR[lang], linestyle=STYLE[size], linewidth=1)
                ax.text(0.05, summary[key]["bpc"], f"transformer {size}", va="bottom", fontsize=8, color="#52514e")
        ax.set_title(f"{lang}: escada de entropia, compressores e modelos")
        ax.set_xlabel("k (caracteres de contexto)")
        ax.legend(fontsize=8, loc="upper right")
    axes[0].set_ylabel("bits por caractere")
    save(fig, out / "entropy_ladder.png")


def training_curves(res, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, lang in zip(axes, ("pt", "en")):
        for size in SIZES:
            p = res / f"runs/{lang}_{size}.log.json"
            if not p.exists():
                continue
            log = json.loads(p.read_text())
            c = np.array(log["curve"])
            v = np.array(log["val_curve"])
            ax.plot(c[:, 1] / 1e6, c[:, 2], STYLE[size], color=COLOR[lang], alpha=0.5,
                    label=f"{size}: lotes de treino antes da atualização (prequencial)")
            ax.plot(v[:, 1] / 1e6, v[:, 2], STYLE[size], marker=MARKER[size], markersize=4, color=COLOR[lang],
                    label=f"{size}: validação, primeiros 0,5M caracteres")
        ax.set_xscale("log")
        ax.set_ylim(top=3.5)
        ax.set_title(f"{lang}: bits por caractere durante a única época de treino")
        ax.set_xlabel("caracteres de treino vistos (milhões)")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("bits por caractere")
    save(fig, out / "training_curves.png")


def by_position(res, out):
    summary = json.loads((res / "runs/eval/summary.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, lang in zip(axes, ("pt", "en")):
        for size in SIZES:
            key = f"{lang}_{size}_on_{lang}"
            if key not in summary:
                continue
            pos = np.arange(1, 257)
            ax.plot(pos, summary[key]["bpc_by_position"], STYLE[size], color=COLOR[lang], label=f"{size} entropia cruzada")
            ax.plot(pos, summary[key]["entropy_by_position"], STYLE[size], color="#8a8984", linewidth=1,
                    label=f"{size} entropia média prevista")
        ax.set_xscale("log")
        ax.set_title(f"teste {lang}: perda por tamanho do contexto")
        ax.set_xlabel("caracteres de contexto")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("bits por caractere")
    save(fig, out / "loss_by_position.png")


def lid(res, out):
    results = json.loads((res / "runs/lid.json").read_text())
    fig, axes = plt.subplots(1, len(results) // 2, figsize=(4 * (len(results) // 2), 3.6), sharey=True, squeeze=False)
    for ax, size in zip(axes[0], [s for s in SIZES if f"{s}_pt_vs_en" in results]):
        for a, b in (("pt", "en"), ("en", "pt")):
            r = results[f"{size}_{a}_vs_{b}"]
            err = np.array(r["error_by_n"])
            n = np.arange(1, len(err) + 1)
            ax.plot(n[err > 0], err[err > 0], color=COLOR[a], label=f"texto {a}, medido")
            for eps, p in r["eps"].items():
                ax.plot(p["n_stein"], float(eps), "o", color=COLOR[a], markersize=8, markerfacecolor="none")
                ax.plot(p["n_clt"], float(eps), "x", color=COLOR[a], markersize=8)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_title(f"modelos de {size}")
        ax.set_xlabel("tamanho do trecho n (caracteres)")
    axes[0][0].set_ylabel("taxa de erro")
    axes[0][0].plot([], [], "ko", markerfacecolor="none", label="previsto, log2(1/eps)/D (Stein)")
    axes[0][0].plot([], [], "kx", label="previsto, TCL")
    axes[0][0].legend(fontsize=7)
    save(fig, out / "language_id.png")


def main():
    res, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(exist_ok=True)
    for make in (entropy_ladder, training_curves, by_position, lid):
        try:
            make(res, out)
        except FileNotFoundError as e:
            print("skipped", make.__name__, "-", e)


if __name__ == "__main__":
    main()
