#!/usr/bin/env python3
import argparse
import collections
import heapq
import json
import time
import struct
from typing import Dict, Optional


class HuffmanNode:
    def __init__(self, char: Optional[str], freq: int, left=None, right=None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right

    # For heapq to compare
    def __lt__(self, other):
        return self.freq < other.freq


def build_frequency_table(text: str) -> Dict[str, int]:
    return dict(collections.Counter(text))


def build_huffman_tree(freq_table: Dict[str, int]) -> Optional[HuffmanNode]:
    if not freq_table:
        return None

    heap = []
    # tie-breaker counter to avoid comparison issues if frequencies are equal
    counter = 0
    for ch, freq in freq_table.items():
        heapq.heappush(heap, (freq, counter, HuffmanNode(ch, freq)))
        counter += 1

    # Special case: only one unique character
    if len(heap) == 1:
        freq, _, node = heap[0]
        # Make a dummy parent to ensure at least one bit of code
        return HuffmanNode(None, freq, left=node, right=None)

    while len(heap) > 1:
        freq1, _, node1 = heapq.heappop(heap)
        freq2, _, node2 = heapq.heappop(heap)
        merged = HuffmanNode(None, freq1 + freq2, left=node1, right=node2)
        heapq.heappush(heap, (merged.freq, counter, merged))
        counter += 1

    return heap[0][2]


def build_codes(node: Optional[HuffmanNode]) -> Dict[str, str]:
    codes = {}

    if node is None:
        return codes

    def traverse(n: HuffmanNode, prefix: str):
        if n.char is not None:  # leaf
            # If only one unique character, ensure non-empty code
            codes[n.char] = prefix if prefix != "" else "0"
            return
        if n.left is not None:
            traverse(n.left, prefix + "0")
        if n.right is not None:
            traverse(n.right, prefix + "1")

    traverse(node, "")
    return codes


def encode_text(text: str, codes: Dict[str, str]):
    # Build the complete bitstring
    bitstring = "".join(codes[ch] for ch in text)

    # Pad to full bytes
    padding = (8 - (len(bitstring) % 8)) % 8
    bitstring_padded = bitstring + ("0" * padding)

    # Convert to bytes
    b = bytearray()
    for i in range(0, len(bitstring_padded), 8):
        byte = bitstring_padded[i:i + 8]
        b.append(int(byte, 2))

    return bytes(b), padding


def huffman_encode_file(input_path: str, output_path: str) -> None:
    # Read the entire input file as text (UTF-8)
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    if text == "":
        # Handle empty file: write a header with no codes and no data
        header = {"codes": {}, "padding": 0}
        header_bytes = json.dumps(header).encode("utf-8")
        with open(output_path, "wb") as out:
            out.write(struct.pack(">I", len(header_bytes)))
            out.write(header_bytes)
        return

    freq_table = build_frequency_table(text)
    tree = build_huffman_tree(freq_table)
    codes = build_codes(tree)

    encoded_bytes, padding = encode_text(text, codes)

    # Header will store the codes and padding
    header = {
        "codes": codes,   # char -> bitstring
        "padding": padding
    }
    header_bytes = json.dumps(header).encode("utf-8")
    header_len = len(header_bytes)

    with open(output_path, "wb") as out:
        # 4-byte big-endian header length
        out.write(struct.pack(">I", header_len))
        out.write(header_bytes)
        out.write(encoded_bytes)


def main():
    parser = argparse.ArgumentParser(description="Huffman encode a text file.")
    parser.add_argument("input", help="Path to input text file")
    parser.add_argument("output", help="Path to output encoded binary file")
    args = parser.parse_args()
    start_time=time.perf_counter()

    huffman_encode_file(args.input, args.output)
    end_time=time.perf_counter()
    print(f"Runtime: {end_time - start_time:.2f} sec")


if __name__ == "__main__":
    main()
