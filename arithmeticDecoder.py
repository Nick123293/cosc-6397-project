#!/usr/bin/env python3
import struct
from typing import List, Tuple


# ---- Same constants as encoder ----
CODE_BITS = 32
TOP = (1 << CODE_BITS) - 1
FIRST_QTR = TOP // 4 + 1
HALF = 2 * FIRST_QTR
THIRD_QTR = 3 * FIRST_QTR


def read_bits_from_bytes(data: bytes) -> str:
    return "".join(f"{b:08b}" for b in data)


class BitReader:
    def __init__(self, bits: List[int]):
        self.bits = bits
        self.idx = 0

    def read(self) -> int:
        if self.idx >= len(self.bits):
            # If we run out of bits, treat as 0
            return 0
        b = self.bits[self.idx]
        self.idx += 1
        return b


class ArithmeticDecoder:
    def __init__(self, bits: List[int]):
        self.reader = BitReader(bits)
        self.low = 0
        self.high = TOP
        self.value = 0

        # Initialize 'value' from first CODE_BITS bits
        for _ in range(CODE_BITS):
            self.value = (self.value << 1) | self.reader.read()

    def decode_symbol(self, cum_freq: List[int], total: int) -> int:
        low = self.low
        high = self.high
        range_ = high - low + 1

        # Map value into cumulative frequency space
        cum_value = ((self.value - low + 1) * total - 1) // range_

        # Binary search to find symbol index
        n = len(cum_freq) - 1  # number of symbols
        lo, hi = 0, n
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if cum_freq[mid] > cum_value:
                hi = mid
            else:
                lo = mid
        sym_idx = lo

        sym_low = cum_freq[sym_idx]
        sym_high = cum_freq[sym_idx + 1]

        # Update interval
        self.high = low + (range_ * sym_high) // total - 1
        self.low = low + (range_ * sym_low) // total

        # Renormalize
        while True:
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.low -= HALF
                self.high -= HALF
                self.value -= HALF
            elif self.low >= FIRST_QTR and self.high < THIRD_QTR:
                self.low -= FIRST_QTR
                self.high -= FIRST_QTR
                self.value -= FIRST_QTR
            else:
                break

            self.low <<= 1
            self.high = (self.high << 1) + 1
            self.value = (self.value << 1) | self.reader.read()

        return sym_idx


def decode_arithmetic_file(path: str) -> Tuple[str, List[int]]:
    """
    Decode a single combined arithmetic-encoded ranks file back to:
      - seed_text: str
      - ranks: List[int]
    """
    with open(path, "rb") as f:
        data = f.read()

    idx = 0

    # 1) Seed text
    seed_len = struct.unpack_from(">I", data, idx)[0]
    idx += 4
    seed_text = data[idx:idx + seed_len].decode("utf-8")
    idx += seed_len

    # 2) Frequency table
    num_symbols = struct.unpack_from(">I", data, idx)[0]
    idx += 4

    symbols: List[int] = []
    freqs: List[int] = []
    for _ in range(num_symbols):
        r, freq = struct.unpack_from(">II", data, idx)
        idx += 8
        symbols.append(r)
        freqs.append(freq)

    # Build cumulative frequencies
    cum_freq = [0]
    for fval in freqs:
        cum_freq.append(cum_freq[-1] + fval)
    total = cum_freq[-1]

    # 3) Encoded data
    padding = data[idx]
    idx += 1

    bitstream = read_bits_from_bytes(data[idx:])
    if padding > 0:
        bitstream = bitstream[:-padding]

    bits = [1 if b == "1" else 0 for b in bitstream]

    decoder = ArithmeticDecoder(bits)

    num_tokens = total  # sum of all freqs
    ranks: List[int] = []
    for _ in range(num_tokens):
        sym_idx = decoder.decode_symbol(cum_freq, total)
        ranks.append(symbols[sym_idx])

    return seed_text, ranks


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Decode combined arithmetic-encoded ranks file back to seed text + ranks."
    )
    parser.add_argument("encoded", help="Path to arithmetic-encoded binary file")
    parser.add_argument(
        "--out-ranks",
        required=True,
        help="Path to write decoded ranks (first line seed text, then one rank per line)",
    )
    args = parser.parse_args()

    seed_text, ranks = decode_arithmetic_file(args.encoded)

    with open(args.out_ranks, "w", encoding="utf-8") as f:
        f.write(seed_text + "\n")
        for r in ranks:
            f.write(f"{r}\n")

    print("Decoded seed text and ranks.")
    print(f"Seed text: {repr(seed_text)}")
    print(f"Number of ranks: {len(ranks)}")
    print(f"Decoded ranks file written to: {args.out_ranks}")
