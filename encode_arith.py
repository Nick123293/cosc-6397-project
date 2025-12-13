#!/usr/bin/env python3
import argparse
import struct
from collections import Counter
import time

# ==============================
# Arithmetic coder parameters
# ==============================
CODE_BITS = 32
TOP_VALUE = (1 << CODE_BITS) - 1
HALF      = 1 << (CODE_BITS - 1)
FIRST_QTR = HALF >> 1
THIRD_QTR = FIRST_QTR * 3


def build_model(data: bytes):
    """
    Build symbol model (only symbols that appear).
    Returns:
      symbols: list of byte values (ints)
      freqs:   list of frequencies for those symbols
      cum:     cumulative frequencies (len = n+1)
      total:   total frequency
    """
    counts = Counter(data)
    # Use a stable, deterministic order: sorted by byte value
    symbols = sorted(counts.keys())
    freqs = [counts[s] for s in symbols]

    cum = [0]
    for f in freqs:
        cum.append(cum[-1] + f)
    total = cum[-1]

    return symbols, freqs, cum, total


class BitWriter:
    def __init__(self):
        self.bits = []

    def write_bit(self, bit: int):
        self.bits.append(bit & 1)

    def get_bytes(self) -> bytes:
        out = bytearray()
        cur = 0
        nbits = 0
        for b in self.bits:
            cur = (cur << 1) | b
            nbits += 1
            if nbits == 8:
                out.append(cur)
                cur = 0
                nbits = 0
        if nbits != 0:
            cur <<= (8 - nbits)  # pad the last byte with zeros
            out.append(cur)
        return bytes(out)


def arithmetic_encode(data: bytes):
    """
    Integer arithmetic encoding of 'data' (bytes).
    Returns:
      symbols, freqs, bitstream_bytes
    """
    symbols, freqs, cum, total = build_model(data)
    # Map byte value -> index in symbols
    sym_to_index = {s: i for i, s in enumerate(symbols)}

    low = 0
    high = TOP_VALUE
    bits_to_follow = 0
    bw = BitWriter()

    def output_bit(bit: int):
        nonlocal bits_to_follow
        bw.write_bit(bit)
        # emit bits_to_follow complement bits
        while bits_to_follow > 0:
            bw.write_bit(1 - bit)
            bits_to_follow -= 1

    for b in data:
        idx = sym_to_index[b]
        # symbol's cum range
        sym_low = cum[idx]
        sym_high = cum[idx + 1]

        range_ = high - low + 1
        high = low + (range_ * sym_high // total) - 1
        low  = low + (range_ * sym_low  // total)

        while True:
            if high < HALF:
                # Emit 0 + bits_to_follow 1s
                output_bit(0)
                low = low * 2
                high = high * 2 + 1
            elif low >= HALF:
                # Emit 1 + bits_to_follow 0s
                output_bit(1)
                low = (low - HALF) * 2
                high = (high - HALF) * 2 + 1
            elif low >= FIRST_QTR and high < THIRD_QTR:
                bits_to_follow += 1
                low = (low - FIRST_QTR) * 2
                high = (high - FIRST_QTR) * 2 + 1
            else:
                break

    # Termination
    bits_to_follow += 1
    if low < FIRST_QTR:
        output_bit(0)
    else:
        output_bit(1)

    bitstream = bw.get_bytes()
    return symbols, freqs, bitstream


def write_encoded_file(path: str, data: bytes, symbols, freqs, bitstream: bytes):
    """
    File format:
      - 8 bytes: original data length (unsigned long long, big endian)
      - 2 bytes: number of distinct symbols N (unsigned short)
      - For each symbol (N entries):
          1 byte: symbol value
          4 bytes: frequency (unsigned int, big endian)
      - Remaining bytes: arithmetic coded bitstream
    """
    with open(path, "wb") as f:
        # Original length
        f.write(struct.pack(">Q", len(data)))
        # Number of symbols
        f.write(struct.pack(">H", len(symbols)))
        # Symbol table
        for s, fr in zip(symbols, freqs):
            f.write(struct.pack("B", s))
            f.write(struct.pack(">I", fr))
        # Bitstream
        f.write(bitstream)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input text file (UTF-8)")
    ap.add_argument("--output", required=True, help="Output encoded binary file")
    args = ap.parse_args()
    start_time=time.perf_counter()

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()
    data = text.encode("utf-8")

    symbols, freqs, bitstream = arithmetic_encode(data)
    write_encoded_file(args.output, data, symbols, freqs, bitstream)
    end_time=time.perf_counter()
    print(f"Runtime: {end_time - start_time:.2f} sec")
    print(f"Encoded {len(data)} bytes into {args.output}")


if __name__ == "__main__":
    main()
