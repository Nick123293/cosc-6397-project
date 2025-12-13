#!/usr/bin/env python3
import argparse
import struct
import time

# ==============================
# Arithmetic coder parameters
# ==============================
CODE_BITS = 32
TOP_VALUE = (1 << CODE_BITS) - 1
HALF      = 1 << (CODE_BITS - 1)
FIRST_QTR = HALF >> 1
THIRD_QTR = FIRST_QTR * 3


class BitReader:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0
        self.bit_buffer = 0
        self.bits_left = 0

    def read_bit(self) -> int:
        if self.bits_left == 0:
            if self.pos >= len(self.data):
                # If we run out of bits, pad with zeros
                return 0
            self.bit_buffer = self.data[self.pos]
            self.pos += 1
            self.bits_left = 8
        self.bits_left -= 1
        return (self.bit_buffer >> self.bits_left) & 1


def read_encoded_file(path: str):
    """
    Reads the file format written by encode_arith.py.
    Returns:
      data_len, symbols, freqs, bitstream_bytes
    """
    with open(path, "rb") as f:
        # Original length
        data_len = struct.unpack(">Q", f.read(8))[0]
        # Number of symbols
        n_syms = struct.unpack(">H", f.read(2))[0]

        symbols = []
        freqs = []
        for _ in range(n_syms):
            s = struct.unpack("B", f.read(1))[0]
            fr = struct.unpack(">I", f.read(4))[0]
            symbols.append(s)
            freqs.append(fr)

        bitstream = f.read()

    return data_len, symbols, freqs, bitstream


def build_model(symbols, freqs):
    """
    Build cumulative frequencies.
    Returns:
      cum: list, len = n+1
      total: int
    """
    cum = [0]
    for f in freqs:
        cum.append(cum[-1] + f)
    total = cum[-1]
    return cum, total


def arithmetic_decode(data_len: int, symbols, freqs, bitstream: bytes) -> bytes:
    cum, total = build_model(symbols, freqs)
    br = BitReader(bitstream)

    # Initial code value
    low = 0
    high = TOP_VALUE
    value = 0
    for _ in range(CODE_BITS):
        value = (value << 1) | br.read_bit()

    out = bytearray()
    n_syms = len(symbols)

    for _ in range(data_len):
        range_ = high - low + 1
        # Map code value into cumulative frequency
        cum_value = ((value - low + 1) * total - 1) // range_

        # Find symbol index
        # (linear search is fine for up to 256 symbols; can be binary if desired)
        lo = 0
        hi = n_syms
        # binary search for better performance
        while lo < hi:
            mid = (lo + hi) // 2
            if cum[mid + 1] <= cum_value:
                lo = mid + 1
            else:
                hi = mid
        idx = lo

        # Update interval
        sym_low = cum[idx]
        sym_high = cum[idx + 1]

        high = low + (range_ * sym_high // total) - 1
        low  = low + (range_ * sym_low  // total)

        # Renormalization
        while True:
            if high < HALF:
                # do nothing, bit is 0
                pass
            elif low >= HALF:
                low  -= HALF
                high -= HALF
                value -= HALF
            elif low >= FIRST_QTR and high < THIRD_QTR:
                low  -= FIRST_QTR
                high -= FIRST_QTR
                value -= FIRST_QTR
            else:
                break

            low = low * 2
            high = high * 2 + 1
            value = (value * 2) + br.read_bit()

        out.append(symbols[idx])

    return bytes(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Encoded binary file")
    ap.add_argument("--output", required=True, help="Output text file (UTF-8)")
    args = ap.parse_args()
    start_time=time.perf_counter()

    data_len, symbols, freqs, bitstream = read_encoded_file(args.input)
    decoded_bytes = arithmetic_decode(data_len, symbols, freqs, bitstream)
    text = decoded_bytes.decode("utf-8")

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(text)
    end_time=time.perf_counter()
    print(f"Runtime: {end_time - start_time:.2f} sec")
    print(f"Decoded {len(decoded_bytes)} bytes into {args.output}")


if __name__ == "__main__":
    main()
