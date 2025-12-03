#!/usr/bin/env python3
import re
import json
import argparse
from typing import Dict, Any, List
from collections import defaultdict
import heapq
import os
import struct

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------------- Config ----------------------
DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-1B"
DEFAULT_SEED_WORDS = 5
FLUSH_EVERY = 10000   # ranks buffered before writing to tmp file


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


def build_huffman_codebook(freqs: Dict[int,int]) -> Dict[int,str]:
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
        walk(node.left,  path + "0")
        walk(node.right, path + "1")

    walk(root, "")
    return codebook


def write_combined_huffman_file(
    seed_text: str,
    codebook: Dict[int,str],
    ranks_file: str,
    output_bin: str
):
    """
    Writes one binary file containing:
      seed text
      codebook
      encoded bitstream
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

    # ---------------------------------------------------------
    # Build binary file
    # ---------------------------------------------------------
    with open(output_bin, "wb") as bf:

        # 1) Seed text
        seed_bytes = seed_text.encode("utf-8")
        bf.write(struct.pack(">I", len(seed_bytes)))   # 4-byte length
        bf.write(seed_bytes)

        # 2) Codebook
        bf.write(struct.pack(">I", len(codebook)))     # number of entries

        for rank, bits in codebook.items():
            bitlen = len(bits)
            bf.write(struct.pack(">I", rank))          # 4-byte rank
            bf.write(struct.pack("B", bitlen))         # 1-byte bit length

            # Write the bitstring compacted into bytes
            padded = bits + "0" * ((8 - bitlen % 8) % 8)
            code_bytes = bytes(int(padded[i:i+8],2) for i in range(0,len(padded),8))
            bf.write(code_bytes)

        # 3) Encoded data
        bf.write(struct.pack("B", padding))            # padding used
        bf.write(data_bytes)


# ============================================================
# ---------------------- MODEL UTILS -------------------------
# ============================================================

def choose_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def choose_dtype(device):
    return torch.float16 if device.type in ("cuda","mps") else torch.float32


def read_text_file(path):
    with open(path,"r",encoding="utf-8") as f:
        return f.read()


def first_n_words_slice(text, n_words):
    matches = list(re.finditer(r"\S+", text))
    if not matches: return ""
    end_idx = matches[min(n_words, len(matches))-1].end()
    return text[:end_idx]


def token_rank(prob_vector, true_token_id):
    true_p = prob_vector[true_token_id]
    higher = torch.sum(prob_vector > true_p).item()
    return int(higher) + 1


# ============================================================
# ------------------- STREAMING EVALUATION -------------------
# ============================================================

def run_sequence_eval_streaming(text, tok, model, tmp_ranks_path, seed_words):

    device = model.device

    full = tok(text, return_tensors="pt")
    full_ids = full["input_ids"].to(device)
    T = full_ids.shape[1]

    seed_text = first_n_words_slice(text, seed_words)
    seed_ids = tok(seed_text, return_tensors="pt")["input_ids"].to(device)
    L0 = seed_ids.shape[1]

    # Warm-up
    with torch.inference_mode():
        out = model(input_ids=seed_ids, use_cache=True)
        past = out.past_key_values

    rank_freq = defaultdict(int)
    buffer = []

    with open(tmp_ranks_path, "w") as f:
        f.write(seed_text + "\n")

        for pos in range(L0, T):

            true_id = int(full_ids[0,pos])

            with torch.inference_mode():
                out = model(
                    input_ids=full_ids[:, pos-1:pos],
                    past_key_values=past,
                    use_cache=True
                )
                past = out.past_key_values

            probs = torch.softmax(out.logits[:, -1, :], dim=-1)[0]
            rk = token_rank(probs, true_id)

            rank_freq[rk] += 1
            buffer.append(rk)

            if len(buffer) >= FLUSH_EVERY:
                for r in buffer: f.write(f"{r}\n")
                buffer.clear()

        # final flush
        for r in buffer: f.write(f"{r}\n")

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
    args = parser.parse_args()

    device = choose_device()
    dtype = choose_dtype(device)

    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype).to(device)
    model.eval()

    text = read_text_file(args.input)

    tmp_ranks = args.output + ".ranks.txt"

    seed_text, rank_freq = run_sequence_eval_streaming(
        text=text,
        tok=tok,
        model=model,
        tmp_ranks_path=tmp_ranks,
        seed_words=args.seed_words,
    )

    if not args.huffman_encoding:
        print(f"Ranks written to {tmp_ranks}")
        return

    # Build Huffman codebook
    codebook = build_huffman_codebook(rank_freq)

    # Build final output file
    write_combined_huffman_file(
        seed_text=seed_text,
        codebook=codebook,
        ranks_file=tmp_ranks,
        output_bin=args.output,
    )

    if not args.keep_intermediate:
        os.remove(tmp_ranks)

    print(f"Huffman-encoded combined file saved to {args.output}")


if __name__ == "__main__":
    main()
