#!/usr/bin/env python3
import argparse
import collections
import struct
import time
# We work on *bytes*, not Python str. Text is UTF-8 encoded.
# This is a simple range coder (arithmetic coding variant).

# Constants for 32-bit range coder
TOP = 1 << 24      # renormalization threshold
BOTTOM = 1 << 16   # not strictly needed but conventional


def build_model(data: bytes):
    """
    Build frequency table and CDF for a byte string.
    Returns:
      freq[256], cum[257], total
    freq[i]       = freq of byte value i
    cum[i]        = cumulative count up to i-1, cum[0] = 0, cum[256] = total
    total         = total symbol count
    """
    freq = [0] * 256
    for b in data:
        freq[b] += 1

    # Avoid zero total
    total = sum(freq)
    if total == 0:
        total = 1
        freq[0] = 1

    cum = [0] * 257
    running = 0
    for i in range(256):
        cum[i] = running
        running += freq[i]
    cum[256] = running  # = total

    return freq, cum, total


def ac_encode(data: bytes, freq, cum, total):
    """
    Standard range coding (32-bit). Encodes data into a bytearray.
    """
    low = 0
    high = 0xFFFFFFFF
    out = bytearray()
    pending = 0

    for b in data:
        # Current range
        range_ = high - low + 1

        # [cum[b], cum[b]+freq[b]) mapped into current range
        c_low = cum[b]
        c_high = cum[b] + freq[b]

        high = low + (range_ * c_high) // total - 1
        low = low + (range_ * c_low) // total

        # Renormalize
        while True:
            if (high & 0x80000000) == (low & 0x80000000):
                # MSB the same → output it
                out.append((high >> 31) & 1)  # temporarily just bits, pack later
                while pending > 0:
                    out.append(((~high) >> 31) & 1)
                    pending -= 1
                low = (low << 1) & 0xFFFFFFFF
                high = ((high << 1) | 1) & 0xFFFFFFFF
            elif (low & 0x40000000) and not (high & 0x40000000):
                # Underflow
                pending += 1
                low = (low << 1) & 0x7FFFFFFF
                high = ((high << 1) | 0x80000001) & 0xFFFFFFFF
            else:
                break

    # Final bits
    pending += 1
    out.append((low >> 30) & 1)
    while pending > 0:
        out.append(((~low) >> 30) & 1)
        pending -= 1

    # Pack bits into bytes
    packed = bytearray()
    byte = 0
    bit_count = 0
    for bit in out:
        byte = (byte << 1) | (bit & 1)
        bit_count += 1
        if bit_count == 8:
            packed.append(byte)
            byte = 0
            bit_count = 0
    if bit_count > 0:
        packed.append(byte << (8 - bit_count))

    return bytes(packed)


def write_header(f, original_len: int, freq):
    """
    Store:
      - magic
      - original length
      - 256 frequencies
    """
    f.write(b"AC00")  # magic
    f.write(struct.pack("<I", original_len))
    for v in freq:
        f.write(struct.pack("<I", v))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input text file")
    parser.add_argument("output", help="compressed binary output")
    args = parser.parse_args()
    start_time=time.perf_counter()

    # Read as bytes (UTF-8 text is fine)
    with open(args.input, "rb") as f:
        data = f.read()

    freq, cum, total = build_model(data)
    compressed = ac_encode(data, freq, cum, total)

    with open(args.output, "wb") as f:
        write_header(f, len(data), freq)
        f.write(struct.pack("<I", len(compressed)))
        f.write(compressed)
    end_time=time.perf_counter()
    print(f"Runtime: {end_time - start_time:.2f} sec")
    print(f"Encoded {len(data)} bytes → {len(compressed)} bytes")


if __name__ == "__main__":
    main()
