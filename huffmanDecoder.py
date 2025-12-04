#!/usr/bin/env python3
import struct


def read_bits_from_bytes(data: bytes) -> str:
    return "".join(f"{b:08b}" for b in data)


def decode_combined_file(path: str):
    """
    Decode a single combined binary file produced by eval_llama_streaming.py.

    Returns:
        seed_text: str
        ranks: List[int]
    """
    with open(path, "rb") as f:
        data = f.read()

    idx = 0

    # 1) Seed text
    seed_len = struct.unpack_from(">I", data, idx)[0]
    idx += 4
    seed_text = data[idx:idx + seed_len].decode("utf-8")
    idx += seed_len

    # 2) Codebook
    num_entries = struct.unpack_from(">I", data, idx)[0]
    idx += 4

    # Build reverse codebook: bitstring -> rank
    codebook_rev = {}
    for _ in range(num_entries):
        rank = struct.unpack_from(">I", data, idx)[0]
        idx += 4

        bitlen = data[idx]
        idx += 1

        nbytes = (bitlen + 7) // 8
        raw = data[idx:idx + nbytes]
        idx += nbytes

        bits = read_bits_from_bytes(raw)[:bitlen]
        codebook_rev[bits] = rank

    # 3) Encoded Huffman data
    padding = data[idx]
    idx += 1

    bitstream = read_bits_from_bytes(data[idx:])
    if padding > 0:
        bitstream = bitstream[:-padding]

    # Decode bitstream to ranks
    ranks = []
    cur = ""
    for b in bitstream:
        cur += b
        if cur in codebook_rev:
            ranks.append(codebook_rev[cur])
            cur = ""

    return seed_text, ranks


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Decode combined Huffman-encoded ranks file back to seed text + ranks."
    )
    parser.add_argument("encoded", help="Path to combined binary file (output of eval_llama_streaming.py)")
    parser.add_argument(
        "--out-ranks",
        required=True,
        help="Path to write decoded ranks file (same format as eval_llama_streaming --keep-intermediate)",
    )
    args = parser.parse_args()

    seed_text, ranks = decode_combined_file(args.encoded)

    # Write in the SAME format as eval_llama_streaming's intermediate file:
    # first line: seed text
    # subsequent lines: one rank per line
    with open(args.out_ranks, "w", encoding="utf-8") as f:
        f.write(seed_text + "\n")
        for r in ranks:
            f.write(f"{r}\n")

    print("Decoded seed text and ranks.")
    print(f"Seed text: {repr(seed_text)}")
    print(f"Number of ranks: {len(ranks)}")
    print(f"Decoded ranks file written to: {args.out_ranks}")
