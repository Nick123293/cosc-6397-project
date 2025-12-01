#!/usr/bin/env python3
import sys
import os

MAX_BYTES = 1024 * 512        # 1 MiB strict cutoff
EXTRA_BYTES = 200              # read ahead to find a clean word boundary

def truncate_full_word_after_limit(path):
    # Read 1 MiB + some extra to ensure we find the next space
    with open(path, "rb") as f:
        chunk = f.read(MAX_BYTES + EXTRA_BYTES)

    if not chunk:
        return ""

    whitespace = b" \t\r\n\f\v"

    # Find the FIRST whitespace *after* the 1 MiB mark
    cut_index = None
    for i in range(MAX_BYTES, len(chunk)):
        if chunk[i:i+1] in whitespace:
            cut_index = i
            break

    if cut_index is None:
        # No whitespace found in the extra read — extremely rare
        cut_index = MAX_BYTES

    trimmed = chunk[:cut_index]

    # Decode safely (ignore incomplete UTF-8 sequences)
    return trimmed.decode("utf-8", errors="ignore")


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {os.path.basename(sys.argv[0])} <input_file> <output_file>", file=sys.stderr)
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    text = truncate_full_word_after_limit(input_path)

    # Write output, overwriting if exists
    with open(output_path, "w", encoding="utf-8") as out:
        out.write(text)

    print(f"Truncated text written to {output_path}")

if __name__ == "__main__":
    main()
