"""Arithmetic coding driven by a trained model, with a bit-exact round trip (RESULTS.md, section 5).

Usage: python src/coder.py MODEL.pt SIZE TEXT_FILE N_CHARS OUT_PREFIX

The script encodes the first N_CHARS characters of TEXT_FILE into OUT_PREFIX.ac. It then decodes that file and
checks that the result equals the original text. It prints a JSON object with the real file size, the ideal size
(the sum of -log2 of the quantized probabilities) and the encoding and decoding times.

The encoder and decoder must see exactly the same probabilities, so they share one way of giving the model its
context. The text is cut into S contiguous streams that advance together, one character per step. Each stream
predicts its next character from a window with its last 256 characters, and a newline character stands before the
first one. While a stream has fewer than 256 characters, the window is padded on the right. The padding does not
change the prediction, because attention is causal: a position never looks at later positions.

Every step is one forward pass on a tensor of shape (S, 256), in float32 and with deterministic algorithms. The
encoder and the decoder run this same code, so they compute bit-identical integer frequencies.
"""
import json, math, os, struct, sys, time
import numpy as np
import torch
from model import ALPHABET, BYTE_TO_ID, CONTEXT, SIZES, V, CharTransformer

S = 512
TOTAL_BITS = 16
TOTAL = 1 << TOTAL_BITS
NEWLINE = ALPHABET.index("\n")

# ---- integer arithmetic coder (Witten, Neal & Cleary 1987), 32-bit registers -------------------------
HALF, QUARTER, MASK = 1 << 31, 1 << 30, (1 << 32) - 1


class Encoder:
    def __init__(self):
        self.low, self.high, self.pending, self.bits = 0, MASK, 0, []

    def emit(self, bit):
        self.bits.append(bit)
        self.bits.extend([1 - bit] * self.pending)
        self.pending = 0

    def encode(self, cum_low, freq):
        span = self.high - self.low + 1
        self.high = self.low + span * (cum_low + freq) // TOTAL - 1
        self.low = self.low + span * cum_low // TOTAL
        while True:
            if self.high < HALF:
                self.emit(0)
            elif self.low >= HALF:
                self.emit(1)
                self.low, self.high = self.low - HALF, self.high - HALF
            elif self.low >= QUARTER and self.high < HALF + QUARTER:
                self.pending += 1
                self.low, self.high = self.low - QUARTER, self.high - QUARTER
            else:
                break
            self.low, self.high = 2 * self.low, 2 * self.high + 1

    def finish(self):
        self.pending += 1
        self.emit(0 if self.low < QUARTER else 1)
        return self.bits


class Decoder:
    def __init__(self, bits):
        self.bits, self.pos = bits, 32
        self.low, self.high = 0, MASK
        self.value = int("".join(map(str, (bits[:32] + [0] * 32)[:32])), 2)

    def next_bit(self):
        self.pos += 1
        return self.bits[self.pos - 1] if self.pos <= len(self.bits) else 0

    def decode(self, cum):
        """cum: cumulative frequencies, length V+1, cum[0] = 0, cum[V] = TOTAL. Returns the symbol."""
        span = self.high - self.low + 1
        target = ((self.value - self.low + 1) * TOTAL - 1) // span
        symbol = int(np.searchsorted(cum, target, side="right")) - 1
        self.high = self.low + span * int(cum[symbol + 1]) // TOTAL - 1
        self.low = self.low + span * int(cum[symbol]) // TOTAL
        while True:
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.low, self.high, self.value = self.low - HALF, self.high - HALF, self.value - HALF
            elif self.low >= QUARTER and self.high < HALF + QUARTER:
                self.low, self.high, self.value = self.low - QUARTER, self.high - QUARTER, self.value - QUARTER
            else:
                break
            self.low, self.high = 2 * self.low, 2 * self.high + 1
            self.value = 2 * self.value + self.next_bit()
        return symbol


# ---- the model side: identical for encoder and decoder ------------------------------------------------
@torch.no_grad()
def frequencies(model, window, n_known):
    """window: (S, 256) ids on the GPU; position n_known-1 holds each stream's last known character.
    Returns integer cumulative frequencies, numpy (S, V+1), each row summing to TOTAL."""
    probs = torch.softmax(model(window)[:, n_known - 1].float(), dim=-1)
    freq = 1 + torch.floor(probs * (TOTAL - V)).long()
    freq[torch.arange(len(freq)), probs.argmax(-1)] += TOTAL - freq.sum(-1)
    cum = torch.zeros(len(freq), V + 1, dtype=torch.long, device=freq.device)
    cum[:, 1:] = freq.cumsum(-1)
    return cum.cpu().numpy()


def run(model, ids, decoders=None):
    """Encode ids (numpy, length N) or, with decoders, decode N characters. Same loop either way.
    Returns (ids, encoders or decoders, ideal bits)."""
    n = len(ids)
    length = math.ceil(n / S)
    history = torch.full((S, length + 1), NEWLINE, dtype=torch.long, device="cuda")
    if decoders is None:
        padded = np.full(S * length, NEWLINE, dtype=np.int64)
        padded[:n] = ids
        history[:, 1:] = torch.from_numpy(padded.reshape(S, length)).cuda()
        coders = [Encoder() for _ in range(S)]
    else:
        coders = decoders
    out, ideal = np.zeros((S, length), dtype=np.int64), 0.0
    live = [min(length, max(0, n - s * length)) for s in range(S)]       # characters in stream s
    for t in range(length):
        start = max(0, t + 1 - CONTEXT)
        window = history[:, start:t + 1]
        n_known = window.shape[1]
        if n_known < CONTEXT:
            window = torch.cat([window, window.new_full((S, CONTEXT - n_known), NEWLINE)], dim=1)
        cum = frequencies(model, window, n_known)
        for s in range(S):
            if t >= live[s]:
                continue
            if decoders is None:
                symbol = int(ids[s * length + t])
                coders[s].encode(int(cum[s, symbol]), int(cum[s, symbol + 1] - cum[s, symbol]))
            else:
                symbol = coders[s].decode(cum[s])
            out[s, t] = symbol
            ideal += TOTAL_BITS - math.log2(cum[s, symbol + 1] - cum[s, symbol])
        if decoders is not None:
            history[:, t + 1] = torch.from_numpy(out[:, t]).cuda()
    return out.reshape(-1)[:n], coders, ideal


def main():
    model_path, size, text_path, n_chars, prefix = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    model = CharTransformer(**SIZES[size]).cuda().eval()
    model.load_state_dict(torch.load(model_path, map_location="cuda"))
    ids = BYTE_TO_ID[np.fromfile(text_path, dtype=np.uint8, count=n_chars)].astype(np.int64)

    t0 = time.time()
    _, encoders, ideal = run(model, ids)
    streams = [np.packbits(np.array(e.finish(), dtype=np.uint8)).tobytes() for e in encoders]
    with open(prefix + ".ac", "wb") as f:            # header: N, then each stream's byte length
        f.write(struct.pack("<Q", len(ids)) + struct.pack(f"<{S}I", *map(len, streams)) + b"".join(streams))
    t_encode = time.time() - t0

    t0 = time.time()
    data = open(prefix + ".ac", "rb").read()
    n = struct.unpack("<Q", data[:8])[0]
    lengths, pos, decoders = struct.unpack(f"<{S}I", data[8:8 + 4 * S]), 8 + 4 * S, []
    for size_bytes in lengths:
        decoders.append(Decoder(np.unpackbits(np.frombuffer(data[pos:pos + size_bytes], np.uint8)).tolist()))
        pos += size_bytes
    decoded, _, _ = run(model, np.zeros(n, dtype=np.int64), decoders)
    t_decode = time.time() - t0

    result = {"model": model_path, "text": text_path, "chars": n, "streams": S,
              "round_trip_ok": bool(np.array_equal(decoded, ids)),
              "file_bytes": len(data), "header_bytes": 8 + 4 * S,
              "real_bpc": 8 * len(data) / n, "ideal_bpc": ideal / n,
              "encode_seconds": t_encode, "decode_seconds": t_decode}
    print(json.dumps(result, indent=1))
    with open(prefix + ".json", "w") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
