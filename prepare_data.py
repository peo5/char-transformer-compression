"""Step 1: Wikipedia parquet -> normalized ASCII text, split train/val/test by article.

Usage: python prepare_data.py RAW_DIR OUT_DIR
  RAW_DIR holds 20231101.pt/*.parquet and 20231101.en/*.parquet (from download.sh).
  Writes OUT_DIR/{pt,en}/{train,val,test}.txt and OUT_DIR/stats.json.

Recipe (lossy on purpose):
  - NFKD, drop combining marks (accents; c-cedilla -> c), lower case, a few quote/dash
    look-alikes mapped to ASCII, every character outside ALPHABET dropped.
  - Spaces collapsed, blank lines removed: one paragraph per line, articles separated by one blank line.
  - Articles shorter than MIN_CHARS after normalization are dropped (stubs, most bot articles).
  - Exact duplicates dropped, and near duplicates: two articles that are equal once every digit is
    replaced by 0 (templated bot text differing only in numbers) keep only the first.
  - Each article gets a pseudo-random key from its id; sorted by that key, articles go to test until
    TEST_CHARS, then val until VAL_CHARS, then train; every split has exactly the same size in both
    languages (the last article of a split is cut).
"""
import hashlib, json, re, sys, unicodedata
from pathlib import Path
import pyarrow.parquet as pq

ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789 \n.,;:!?'\"-()"
MIN_CHARS = 2000
TEST_CHARS = 10_000_000
VAL_CHARS = 5_000_000
MAX_TRAIN_CHARS = 400_000_000

LOOKALIKES = str.maketrans({"“": '"', "”": '"', "«": '"', "»": '"', "„": '"',
                            "‘": "'", "’": "'", "–": "-", "—": "-", "−": "-",
                            "\t": " "})
KEEP = set(ALPHABET)


def normalize(text):
    text = unicodedata.normalize("NFKD", text.translate(LOOKALIKES)).lower()
    text = "".join(c for c in text if c in KEEP)          # combining marks are not in KEEP
    lines = (re.sub(" +", " ", line).strip() for line in text.split("\n"))
    return "\n".join(line for line in lines if line)


def key(article_id, lang):
    return hashlib.md5(f"{lang}:{article_id}".encode()).hexdigest()


def collect(lang, raw_dir, out_dir):
    """Normalize and filter every article; append the kept ones to all.txt. Returns (key, offset, length) list."""
    seen, index, counts = set(), [], {"articles": 0, "short": 0, "duplicate": 0, "kept": 0}
    offset = 0
    with open(out_dir / "all.txt", "w", encoding="ascii") as out:
        for path in sorted((raw_dir / f"20231101.{lang}").glob("*.parquet")):
            for batch in pq.ParquetFile(path).iter_batches(columns=["id", "text"], batch_size=2000):
                for article_id, text in zip(*(c.to_pylist() for c in batch.columns)):
                    counts["articles"] += 1
                    text = normalize(text)
                    if len(text) < MIN_CHARS:
                        counts["short"] += 1
                        continue
                    fingerprint = hashlib.md5(re.sub("[0-9]", "0", text).encode()).digest()
                    if fingerprint in seen:
                        counts["duplicate"] += 1
                        continue
                    seen.add(fingerprint)
                    out.write(text + "\n\n")
                    index.append((key(article_id, lang), offset, len(text) + 2))
                    offset += len(text) + 2
                    counts["kept"] += 1
            print(lang, path.name, counts, f"{offset / 1e6:.0f}M chars", flush=True)
    counts["kept_chars"] = offset
    return sorted(index), counts


def write_split(all_path, index, out_path, budget):
    """Copy articles from the front of index into out_path until exactly budget chars (the last one is cut).
    Returns the articles not used and the chars written."""
    used = 0
    with open(all_path, "rb") as src, open(out_path, "wb") as out:
        for n, (_, offset, length) in enumerate(index):
            if used == budget:
                return index[n:], used
            src.seek(offset)
            out.write(src.read(min(length, budget - used)))
            used += min(length, budget - used)
    return [], used


def main():
    raw_dir, out_root = Path(sys.argv[1]), Path(sys.argv[2])
    stats, indexes = {}, {}
    for lang in ("pt", "en"):
        (out_root / lang).mkdir(parents=True, exist_ok=True)
        indexes[lang], stats[lang] = collect(lang, raw_dir, out_root / lang)
    test_val = TEST_CHARS + VAL_CHARS
    train_chars = min(MAX_TRAIN_CHARS, *(stats[l]["kept_chars"] - test_val - 100_000 for l in ("pt", "en")))
    for lang in ("pt", "en"):
        rest, all_path = indexes[lang], out_root / lang / "all.txt"
        for split, budget in (("test", TEST_CHARS), ("val", VAL_CHARS), ("train", train_chars)):
            n_before = len(rest)
            rest, used = write_split(all_path, rest, out_root / lang / f"{split}.txt", budget)
            stats[lang][split] = {"chars": used, "articles": n_before - len(rest)}
        all_path.unlink()
    stats["recipe"] = {"alphabet": ALPHABET, "min_chars": MIN_CHARS, "test_chars": TEST_CHARS,
                       "val_chars": VAL_CHARS, "max_train_chars": MAX_TRAIN_CHARS}
    (out_root / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
