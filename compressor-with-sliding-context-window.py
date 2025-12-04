#!/usr/bin/env python3
import re
import json
import argparse
from typing import Dict, Any, List
from collections import defaultdict
import heapq
import os
import struct
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------------- Config ----------------------
DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-1B"
DEFAULT_SEED_WORDS = 5
FLUSH_EVERY = 1048576   # ranks buffered before writing to tmp file


# ============================================================
# ---------------------- HUFFMAN LOGIC -----------------------
# ============================================================

class HuffmanNode:
    def __init__(self, freq, symbol=None, left=None, right=None):
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman_codebook(freqs: Dict[int, int]) -> Dict[int, str]:
    heap = []
    for rank, f in freqs.items():
        heapq.heappush(heap, HuffmanNode(f, symbol=rank))

    if len(heap) == 1:
        only = heapq.heappop(heap)
        return {only.symbol: "0"}

    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        merged = HuffmanNode(a.freq + b.freq, left=a, right=b)
        heapq.heappush(heap, merged)

    root = heap[0]
    codebook = {}

    def walk(node, path):
        if node.symbol is not None:
            codebook[node.symbol] = path
            return
        walk(node.left, path + "0")
        walk(node.right, path + "1")

    walk(root, "")
    return codebook


def write_combined_huffman_file(
    seed_text: str,
    codebook: Dict[int, str],
    ranks_file: str,
    output_bin: str,
    context_window: int,
):
    """
    Writes one binary file containing:
      seed text
      context_window (so decoder can match compressor)
      codebook
      encoded bitstream

    Layout (big-endian):
      [4 bytes]  seed_len
      [seed_len] seed_text (UTF-8)
      [4 bytes]  context_window (uint32, 0 = full history)
      [4 bytes]  num_entries
      per entry:
        [4 bytes] rank
        [1 byte]  bitlen
        [N]      code bits (padded to byte)
      [1 byte]   padding for bitstream
      [rest]     encoded bitstream
    """

    # ---------------------------------------------------------
    # Encode ranks into a single bitstring
    # ---------------------------------------------------------
    bitstring = ""
    with open(ranks_file, "r") as f:
        # First line is seed text → skip (we already have it)
        next(f)
        for line in f:
            rk = int(line.strip())
            bitstring += codebook[rk]

    # Pad out to full byte boundary
    padding = (8 - len(bitstring) % 8) % 8
    bitstring += "0" * padding

    # Convert full bitstring to bytes
    data_bytes = bytes(int(bitstring[i:i+8], 2) for i in range(0, len(bitstring), 8))

    # Sanitize context_window for storage
    cw_store = max(0, int(context_window))

    # ---------------------------------------------------------
    # Build binary file
    # ---------------------------------------------------------
    with open(output_bin, "wb") as bf:
        # 1) Seed text
        seed_bytes = seed_text.encode("utf-8")
        bf.write(struct.pack(">I", len(seed_bytes)))   # 4-byte length
        bf.write(seed_bytes)

        # 2) Context window (4 bytes, uint32)
        bf.write(struct.pack(">I", cw_store))

        # 3) Codebook
        bf.write(struct.pack(">I", len(codebook)))     # number of entries

        for rank, bits in codebook.items():
            bitlen = len(bits)
            bf.write(struct.pack(">I", rank))          # 4-byte rank
            bf.write(struct.pack("B", bitlen))         # 1-byte bit length

            # Write the bitstring compacted into bytes
            padded = bits + "0" * ((8 - bitlen % 8) % 8)
            code_bytes = bytes(int(padded[i:i+8], 2) for i in range(0, len(padded), 8))
            bf.write(code_bytes)

        # 4) Encoded data
        bf.write(struct.pack("B", padding))            # padding used
        bf.write(data_bytes)


# ============================================================
# ---------------------- MODEL UTILS -------------------------
# ============================================================

def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def choose_dtype(device):
    return torch.float16 if device.type in ("cuda", "mps") else torch.float32


def read_text_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def first_n_words_slice(text, n_words):
    matches = list(re.finditer(r"\S+", text))
    if not matches:
        return ""
    end_idx = matches[min(n_words, len(matches)) - 1].end()
    return text[:end_idx]


# ============================================================
# ------------------- STREAMING EVALUATION -------------------
# ============================================================

def run_sequence_eval_streaming(
    text,
    tok,
    model,
    tmp_ranks_path,
    seed_words,
    context_window: int = 0,
):
    """
    Compute ranks for the given text.

    If context_window <= 0:
        Use original streaming with KV cache (full history).

    If context_window > 0:
        For each position pos (>= seed token length),
        we feed at most the last `context_window` tokens into the model
        and compute the rank of the true token at pos.
    """
    device = model.device

    # Full tokenization of the text
    full = tok(text, return_tensors="pt")
    full_ids = full["input_ids"].to(device)
    T = full_ids.shape[1]

    # Seed text and its tokenization
    seed_text = first_n_words_slice(text, seed_words)
    seed_ids = tok(seed_text, return_tensors="pt")["input_ids"].to(device)
    L0 = seed_ids.shape[1]

    rank_freq = defaultdict(int)
    buffer = []

    with open(tmp_ranks_path, "w", encoding="utf-8") as f:
        f.write(seed_text + "\n")

        # ----------------------------------------------------
        # Case 1: full-history streaming with KV cache
        # ----------------------------------------------------
        if context_window is None or context_window <= 0:
            with torch.inference_mode():
                # Initial forward pass on the seed to get KV cache
                out = model(input_ids=seed_ids, use_cache=True)
                past = out.past_key_values

            for pos in range(L0, T):
                true_id = int(full_ids[0, pos])

                with torch.inference_mode():
                    # one-step incremental forward: previous token only
                    step_input = full_ids[:, pos - 1:pos]  # shape (1,1)
                    out = model(
                        input_ids=step_input,
                        past_key_values=past,
                        use_cache=True
                    )
                    past = out.past_key_values
                    logits = out.logits[0, -1, :]  # shape [vocab_size]

                    sorted_ids = torch.argsort(logits, descending=True)
                    true_pos = (sorted_ids == true_id).nonzero(as_tuple=False)
                    if true_pos.numel() == 0:
                        raise RuntimeError("True token id not found in sorted_ids.")
                    rk = int(true_pos.item()) + 1

                rank_freq[rk] += 1
                buffer.append(rk)

                if len(buffer) >= FLUSH_EVERY:
                    for r in buffer:
                        f.write(f"{r}\n")
                    buffer.clear()

        # ----------------------------------------------------
        # Case 2: sliding-window context (no KV cache)
        # ----------------------------------------------------
        else:
            cw = int(context_window)
            with torch.inference_mode():
                for pos in range(L0, T):
                    true_id = int(full_ids[0, pos])

                    # last cw tokens before position pos
                    ctx_start = max(0, pos - cw)
                    context_ids = full_ids[:, ctx_start:pos]  # (1, context_len)

                    out = model(input_ids=context_ids)
                    logits = out.logits[0, -1, :]

                    sorted_ids = torch.argsort(logits, descending=True)
                    true_pos = (sorted_ids == true_id).nonzero(as_tuple=False)
                    if true_pos.numel() == 0:
                        raise RuntimeError("True token id not found in sorted_ids.")
                    rk = int(true_pos.item()) + 1

                    rank_freq[rk] += 1
                    buffer.append(rk)

                    if len(buffer) >= FLUSH_EVERY:
                        for r in buffer:
                            f.write(f"{r}\n")
                        buffer.clear()

        # final flush
        for r in buffer:
            f.write(f"{r}\n")

    return seed_text, rank_freq


# ============================================================
# --------------------------- MAIN ---------------------------
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--seed-words", type=int, default=DEFAULT_SEED_WORDS)
    parser.add_argument("--huffman-encoding", action="store_true")
    parser.add_argument("--keep-intermediate", action="store_true")
    parser.add_argument(
        "--context-window",
        type=int,
        default=0,
        help="If > 0, only the last N tokens are used as context. "
             "If 0 or negative, use full history with KV cache.",
    )
    args = parser.parse_args()

    start_time = time.perf_counter()
    device = choose_device()
    dtype = choose_dtype(device)

    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, revision="main")
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=dtype).to(device)
    model.eval()

    text = read_text_file(args.input)
    tmp_ranks = args.output + ".ranks.txt"

    seed_text, rank_freq = run_sequence_eval_streaming(
        text=text,
        tok=tok,
        model=model,
        tmp_ranks_path=tmp_ranks,
        seed_words=args.seed_words,
        context_window=args.context_window,
    )

    if not args.huffman_encoding:
        print(f"Ranks written to {tmp_ranks}")
        end_time = time.perf_counter()
        print(f"Total runtime: {end_time - start_time:.2f} seconds")
        return

    # Build Huffman codebook
    codebook = build_huffman_codebook(rank_freq)

    # Build final output file (now includes context_window)
    write_combined_huffman_file(
        seed_text=seed_text,
        codebook=codebook,
        ranks_file=tmp_ranks,
        output_bin=args.output,
        context_window=args.context_window,
    )

    if not args.keep_intermediate:
        os.remove(tmp_ranks)

    end_time = time.perf_counter()
    print(f"Huffman-encoded combined file saved to {args.output}")
    print(f"Total runtime: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
