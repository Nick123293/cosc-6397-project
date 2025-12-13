#!/usr/bin/env python3
import argparse
import struct


def read_header(f):
    magic = f.read(4)
    if magic != b"AC00":
        raise ValueError("Not a valid AC-compressed file")

    original_len = struct.unpack("<I", f.read(4))[0]
    freq = [struct.unpack("<I", f.read(4))[0] for _ in range(256)]

    total = sum(freq)
    if total == 0:
        total = 1
        freq[0] = 1

    cum = [0] * 257
    running = 0
    for i in range(256):
        cum[i] = running
        running += freq[i]
    cum[256] = running

    return original_len, freq, cum, total


def bit_stream(data: bytes):
    """Generator: yields bits (0/1) from a byte string."""
    for byte in data:
        for i in range(7, -1, -1):
            yield (byte >> i) & 1


def ac_decode(compressed: bytes, original_len: int, freq, cum, total):
    """
    Decode range-coded bitstream back to original bytes.
    This matches the encoder’s bit-level scheme.
    """
    # Use streaming bits instead of building a giant list
    it = bit_stream(compressed)

    low = 0.0
    high = 1.0
    code = 0.0

    # Initialize code with first 32 bits (or fewer)
    frac = 0.5
    for _ in range(32):
        try:
            b = next(it)
        except StopIteration:
            break
        code += b * frac
        frac *= 0.5

    out = bytearray()

    for _ in range(original_len):
        range_ = high - low
        target = (code - low) / range_ * total

        # Find smallest s with cum[s+1] > target
        s = 255
        for i in range(256):
            if cum[i + 1] > target:
                s = i
                break

        out.append(s)

        c_low = cum[s]
        c_high = cum[s + 1]

        high = low + range_ * (c_high / total)
        low = low + range_ * (c_low / total)

        # Renormalize using new bits
        while True:
            if high < 0.5:
                # MSB 0
                low *= 2.0
                high *= 2.0
                code *= 2.0
                try:
                    code += next(it)
                except StopIteration:
                    pass
            elif low >= 0.5:
                # MSB 1
                low = 2.0 * (low - 0.5)
                high = 2.0 * (high - 0.5)
                code = 2.0 * (code - 0.5)
                try:
                    code += next(it)
                except StopIteration:
                    pass
            elif 0.25 <= low and high < 0.75:
                # Underflow region
                low = 2.0 * (low - 0.25)
                high = 2.0 * (high - 0.25)
                code = 2.0 * (code - 0.25)
                try:
                    code += next(it)
                except StopIteration:
                    pass
            else:
                break

    return bytes(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="compressed file from encode_ac.py")
    parser.add_argument("output", help="decoded text file")
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        original_len, freq, cum, total = read_header(f)
        comp_len = struct.unpack("<I", f.read(4))[0]
        compressed = f.read(comp_len)

    decoded_bytes = ac_decode(compressed, original_len, freq, cum, total)

    with open(args.output, "wb") as f:
        f.write(decoded_bytes)

    print("Decoding finished.")


if __name__ == "__main__":
    main()
