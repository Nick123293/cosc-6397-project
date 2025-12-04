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
    """
    Compute ranks for the given text (streaming, using KV cache),
    write them to tmp_ranks_path, and ALSO do a short self-decode
    using the same model/tokenizer to verify correctness.

    If the self-decode diverges, you'll see a printed mismatch.
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

    # Initial forward pass on the seed to get KV cache
    with torch.inference_mode():
        out = model(input_ids=seed_ids, use_cache=True)
        past = out.past_key_values

    rank_freq = defaultdict(int)
    buffer = []

    # # ---- for self-decode debugging ----
    # SELF_DECODE_TOKENS = 1000  # how many tokens *after the seed* to test
    # debug_ranks = []          # first N ranks we'll self-decode
    # # -----------------------------------

    with open(tmp_ranks_path, "w", encoding="utf-8") as f:
        f.write(seed_text + "\n")

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

                # ---- rank definition consistent with decoder ----
                # Rank = 1-based index of true_id in argsort(logits, descending=True)
                sorted_ids = torch.argsort(logits, descending=True)
                true_pos = (sorted_ids == true_id).nonzero(as_tuple=False)
                if true_pos.numel() == 0:
                    raise RuntimeError("True token id not found in sorted_ids (should never happen)")
                rk = int(true_pos.item()) + 1
                # -----------------------------------------------

            rank_freq[rk] += 1
            buffer.append(rk)

            # # Keep the first SELF_DECODE_TOKENS ranks for self-check
            # if len(debug_ranks) < SELF_DECODE_TOKENS:
            #     debug_ranks.append(rk)

            if len(buffer) >= FLUSH_EVERY:
                for r in buffer:
                    f.write(f"{r}\n")
                buffer.clear()

        # final flush
        for r in buffer:
            f.write(f"{r}\n")

    # ============================================================
    # ---------------------- SELF-DECODE -------------------------
    # ============================================================
    # if debug_ranks:
    #     print(f"[SELF-DECODE] Checking first {len(debug_ranks)} tokens after seed...")

    #     with torch.inference_mode():
    #         # Re-run seed to get fresh KV cache
    #         seed_ids = tok(seed_text, return_tensors="pt")["input_ids"].to(device)
    #         out = model(seed_ids, use_cache=True)
    #         past = out.past_key_values

    #         decoded_ids = seed_ids[0].tolist()
    #         last_token_id = decoded_ids[-1]

    #         # Preallocate a 1x1 tensor for incremental input_ids
    #         step_ids = torch.empty(1, 1, dtype=torch.long, device=device)

    #         for rk in debug_ranks:
    #             step_ids[0, 0] = last_token_id
    #             out = model(
    #                 step_ids,
    #                 use_cache=True,
    #                 past_key_values=past,
    #             )
    #             past = out.past_key_values
    #             last_logits = out.logits[0, -1, :]

    #             # Interpret rank the same way as above:
    #             # position in logits-sorted-descending
    #             sorted_ids = torch.argsort(last_logits, descending=True)
    #             next_token_id = int(sorted_ids[rk - 1].item())

    #             decoded_ids.append(next_token_id)
    #             last_token_id = next_token_id

    #     # Compare decoded prefix to ground truth full_ids
    #     compare_len = len(decoded_ids)
    #     orig_slice = full_ids[0, :compare_len].tolist()

    #     if decoded_ids == orig_slice:
    #         print(f"[SELF-DECODE] OK: first {compare_len - L0} tokens after seed match ground truth.")
    #     else:
    #         # Find first mismatch
    #         mismatch_pos = None
    #         for i, (a, b) in enumerate(zip(decoded_ids, orig_slice)):
    #             if a != b:
    #                 mismatch_pos = i
    #                 break

    #         print("[SELF-DECODE] MISMATCH detected!")
    #         print(f"  First mismatch at token index {mismatch_pos} (0-based, in token space).")
    #         print(f"  Seed length (L0) = {L0}")
    #         if mismatch_pos is not None:
    #             print(f"  This is token {mismatch_pos - L0} after the seed.")
    #             print(f"  orig_id = {orig_slice[mismatch_pos]}, dec_id = {decoded_ids[mismatch_pos]}")

    #         # Optional: show a short text snippet around the mismatch
    #         try:
    #             orig_text_snip = tok.decode(orig_slice[max(0, mismatch_pos-10):mismatch_pos+10])
    #             dec_text_snip = tok.decode(decoded_ids[max(0, mismatch_pos-10):mismatch_pos+10])
    #             print("  Original context snippet:")
    #             print("  ", repr(orig_text_snip))
    #             print("  Decoded  context snippet:")
    #             print("  ", repr(dec_text_snip))
    #         except Exception as e:
    #             print(f"  (Could not decode snippet for inspection: {e})")

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
    end_time = time.perf_counter()
    print(f"Huffman-encoded combined file saved to {args.output}")
    print(f"Total runtime: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
