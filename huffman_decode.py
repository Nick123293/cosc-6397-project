#!/usr/bin/env python3
import argparse
import json
import struct
import time
from typing import Dict


def bytes_to_bitstring(data: bytes) -> str:
    bits = []
    for b in data:
        bits.append(f"{b:08b}")
    return "".join(bits)


def decode_bitstring(bitstring: str, code_to_char: Dict[str, str]) -> str:
    result_chars = []
    current = ""

    for bit in bitstring:
        current += bit
        if current in code_to_char:
            result_chars.append(code_to_char[current])
            current = ""

    # current should be empty at the end if everything was correct
    return "".join(result_chars)


def huffman_decode_file(input_path: str, output_path: str) -> None:
    with open(input_path, "rb") as f:
        # Read header length
        header_len_bytes = f.read(4)
        if len(header_len_bytes) != 4:
            raise ValueError("Invalid file: cannot read header length.")

        (header_len,) = struct.unpack(">I", header_len_bytes)

        # Read header JSON
        header_bytes = f.read(header_len)
        if len(header_bytes) != header_len:
            raise ValueError("Invalid file: incomplete header.")

        header = json.loads(header_bytes.decode("utf-8"))
        codes = header.get("codes", {})
        padding = header.get("padding", 0)

        # If no codes, this was an empty original file
        if not codes:
            with open(output_path, "w", encoding="utf-8") as out:
                out.write("")
            return

        # Remaining bytes are the encoded data
        data_bytes = f.read()

    # Invert mapping: code -> char
    code_to_char = {code: ch for ch, code in codes.items()}

    bitstring_full = bytes_to_bitstring(data_bytes)
    if padding > 0:
        bitstring = bitstring_full[:-padding]
    else:
        bitstring = bitstring_full

    decoded_text = decode_bitstring(bitstring, code_to_char)

    with open(output_path, "w", encoding="utf-8") as out:
        out.write(decoded_text)


def main():
    parser = argparse.ArgumentParser(description="Huffman decode an encoded file.")
    parser.add_argument("input", help="Path to encoded binary file")
    parser.add_argument("output", help="Path to output text file")
    args = parser.parse_args()
    start_time=time.perf_counter()

    huffman_decode_file(args.input, args.output)
    end_time=time.perf_counter()
    print(f"Runtime: {end_time - start_time:.2f} sec")


if __name__ == "__main__":
    main()
