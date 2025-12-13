#!/usr/bin/env python3
import argparse
import json
import os
import struct
import time
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ---------------------- Config defaults ----------------------
DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-1B"
CODE_BITS = 32
TOP = (1 << CODE_BITS) - 1
FIRST_QTR = TOP // 4 + 1
HALF = 2 * FIRST_QTR
THIRD_QTR = 3 * FIRST_QTR


# ============================================================
# ---------------------- Device / Dtype ----------------------
# ============================================================
def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon
    else:
        return torch.device("cpu")


def choose_dtype(device: torch.device) -> torch.dtype:
    if device.type in ["cuda", "mps"]:
        return torch.float16
    else:
        return torch.float32


# ============================================================
# ---------------------- Text ranks reader -------------------
# ============================================================
def read_input_file(path: str):
    """
    Reads a single text file: first line is seed text, remaining lines are ranks.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if not lines:
        raise ValueError("Input file is empty.")

    seed_text = lines[0].rstrip("\n")
    ranks = [int(line.strip()) for line in lines[1:] if line.strip()]
    return seed_text, ranks


# ============================================================
# ---------------------- Shared Utils ------------------------
# ============================================================
def read_bits_from_bytes(data: bytes) -> str:
    return "".join(f"{b:08b}" for b in data)


# ============================================================
# ---------------------- Huffman Decoding --------------------
# ============================================================
def decode_huffman_file(path: str):
    """
    Decode a combined binary Huffman file produced by the compressor.
    """
    with open(path, "rb") as f:
        data = f.read()

    idx = 0

    # Seed text
    seed_len = struct.unpack_from(">I", data, idx)[0]
    idx += 4
    seed_text = data[idx:idx + seed_len].decode("utf-8")
    idx += seed_len

    # Codebook
    num_entries = struct.unpack_from(">I", data, idx)[0]
    idx += 4

    codebook_rev = {}
    for _ in range(num_entries):
        rank = struct.unpack_from(">I", data, idx)[0]
        idx += 4

        bitlen = data[idx]
        idx += 1

        nbytes = (bitlen + 7) // 8
        raw = data[idx:idx+nbytes]
        idx += nbytes

        bits = read_bits_from_bytes(raw)[:bitlen]
        codebook_rev[bits] = rank

    # Encoded bitstream
    padding = data[idx]
    idx += 1

    bitstream = read_bits_from_bytes(data[idx:])
    if padding > 0:
        bitstream = bitstream[:-padding]

    ranks = []
    cur = ""
    for b in bitstream:
        cur += b
        if cur in codebook_rev:
            ranks.append(codebook_rev[cur])
            cur = ""

    return seed_text, ranks


# ============================================================
# ---------------------- Arithmetic Decoding -----------------
# ============================================================
class BitReader:
    def __init__(self, bits: List[int]):
        self.bits = bits
        self.idx = 0

    def read(self) -> int:
        if self.idx >= len(self.bits):
            # If we run out of bits, treat as 0
            return 0
        b = self.bits[self.idx]
        self.idx += 1
        return b


class ArithmeticDecoder:
    def __init__(self, bits: List[int]):
        self.reader = BitReader(bits)
        self.low = 0
        self.high = TOP
        self.value = 0

        # Initialize 'value' from first CODE_BITS bits
        for _ in range(CODE_BITS):
            self.value = (self.value << 1) | self.reader.read()

    def decode_symbol(self, cum_freq: List[int], total: int) -> int:
        low = self.low
        high = self.high
        range_ = high - low + 1

        # Map value into cumulative frequency space
        cum_value = ((self.value - low + 1) * total - 1) // range_

        # Binary search to find symbol index
        # We want the largest index i such that cum_freq[i] <= cum_value
        # Actually standard arithmetic decoding logic:
        # Find symbol s such that cum_freq[s] <= cum_value < cum_freq[s+1]
        
        n = len(cum_freq) - 1  # number of symbols
        lo, hi = 0, n
        
        # We need index 'sym_idx' such that cum_freq[sym_idx] <= cum_value < cum_freq[sym_idx+1]
        # Using bisect_right equivalent logic manually to be safe
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if cum_freq[mid] > cum_value:
                hi = mid
            else:
                lo = mid
        sym_idx = lo

        sym_low = cum_freq[sym_idx]
        sym_high = cum_freq[sym_idx + 1]

        # Update interval
        self.high = low + (range_ * sym_high) // total - 1
        self.low = low + (range_ * sym_low) // total

        # Renormalize
        while True:
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.low -= HALF
                self.high -= HALF
                self.value -= HALF
            elif self.low >= FIRST_QTR and self.high < THIRD_QTR:
                self.low -= FIRST_QTR
                self.high -= FIRST_QTR
                self.value -= FIRST_QTR
            else:
                break

            self.low <<= 1
            self.high = (self.high << 1) + 1
            self.value = (self.value << 1) | self.reader.read()

        return sym_idx


def decode_arithmetic_file(path: str) -> Tuple[str, List[int]]:
    """
    Decode a single combined arithmetic-encoded ranks file back to:
      - seed_text: str
      - ranks: List[int]
    """
    with open(path, "rb") as f:
        data = f.read()

    idx = 0

    # 1) Seed text
    seed_len = struct.unpack_from(">I", data, idx)[0]
    idx += 4
    seed_text = data[idx:idx + seed_len].decode("utf-8")
    idx += seed_len

    # 2) Frequency table
    num_symbols = struct.unpack_from(">I", data, idx)[0]
    idx += 4

    symbols: List[int] = []
    freqs: List[int] = []
    for _ in range(num_symbols):
        r, freq = struct.unpack_from(">II", data, idx)
        idx += 8
        symbols.append(r)
        freqs.append(freq)

    # Build cumulative frequencies
    cum_freq = [0]
    for fval in freqs:
        cum_freq.append(cum_freq[-1] + fval)
    total = cum_freq[-1]

    # 3) Encoded data
    padding = data[idx]
    idx += 1

    bitstream = read_bits_from_bytes(data[idx:])
    if padding > 0:
        bitstream = bitstream[:-padding]

    # Convert bit string to list of ints
    bits = [1 if b == "1" else 0 for b in bitstream]

    decoder = ArithmeticDecoder(bits)

    num_tokens = total  # In this implementation, total freq = total tokens
    ranks: List[int] = []
    
    # Simple progress bar for arithmetic decoding since it can be slow
    for _ in tqdm(range(num_tokens), desc="Arith decoding"):
        sym_idx = decoder.decode_symbol(cum_freq, total)
        ranks.append(symbols[sym_idx])

    return seed_text, ranks


# ============================================================
# ---------------- Model loading & decoding ------------------
# ============================================================
def load_model_and_tokenizer(model_id: str, device: torch.device, dtype: torch.dtype):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, revision="main")
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
    model.eval()

    return tok, model


def decode_from_ranks_streaming(
    seed_text: str,
    ranks: list,
    output_file: str,
    model_id: str,
    device=None,
    dtype=None,
):
    device = device or choose_device()
    dtype = dtype or choose_dtype(device)

    tok, model = load_model_and_tokenizer(model_id, device, dtype)
    
    # Detect max context
    cfg = getattr(model, "config", None)
    max_ctx = getattr(cfg, "max_position_embeddings", 2048) if cfg else 2048

    seed_ids = tok(seed_text, return_tensors="pt").input_ids.to(device)
    decoded_ids = seed_ids[0].tolist()

    with torch.inference_mode():
        # Prefill with seed
        outputs = model(seed_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        last_token_id = decoded_ids[-1]
        
        step_ids = torch.empty(1, 1, dtype=torch.long, device=device)

        with open(output_file, "w", encoding="utf-8") as f_out:
            f_out.write(seed_text)
            text_buffer = []

            for i, rank in enumerate(tqdm(ranks, desc="Reconstructing text")):
                # Check context length strategy
                if len(decoded_ids) > max_ctx:
                    # Fallback to sliding window if context is full (slower but works)
                    context_tensor = torch.tensor([decoded_ids[-max_ctx:]], device=device)
                    outputs = model(context_tensor)
                    last_logits = outputs.logits[0, -1, :]
                    # Reset cache because we are sliding
                    past_key_values = None 
                else:
                    # Use KV Cache
                    step_ids[0, 0] = last_token_id
                    outputs = model(
                        step_ids,
                        use_cache=True,
                        past_key_values=past_key_values,
                    )
                    past_key_values = outputs.past_key_values
                    last_logits = outputs.logits[0, -1, :]

                sorted_ids = torch.argsort(last_logits, descending=True)
                next_token_id = int(sorted_ids[rank - 1].item())

                decoded_ids.append(next_token_id)
                last_token_id = next_token_id

                token_str = tok.decode([next_token_id], clean_up_tokenization_spaces=False)
                text_buffer.append(token_str)

                if len(text_buffer) >= 512:
                    f_out.write("".join(text_buffer))
                    text_buffer = []

            if text_buffer:
                f_out.write("".join(text_buffer))

    return len(decoded_ids)


# ============================================================
# ---------------------------- CLI ---------------------------
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Decode ranks using LLaMA. Supports Huffman, Arithmetic, or Plain Text ranks."
    )
    parser.add_argument("input_file", help="Input file path.")
    parser.add_argument("output_file", help="Write the reconstructed text here.")
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="HuggingFace model id",
    )
    parser.add_argument(
        "--huffman-binary",
        action="store_true",
        help="Treat input_file as a Huffman-compressed binary.",
    )
    parser.add_argument(
        "--arithmetic-binary",
        action="store_true",
        help="Treat input_file as an Arithmetic-compressed binary.",
    )
    
    args = parser.parse_args()

    start_time = time.perf_counter()

    device = choose_device()
    dtype = choose_dtype(device)
    print(f"Using device: {device}, dtype={dtype}")

    # 1. Decode the file to Ranks
    if args.huffman_binary:
        print(f"Decoding Huffman binary: {args.input_file}")
        seed_text, ranks = decode_huffman_file(args.input_file)
    elif args.arithmetic_binary:
        print(f"Decoding Arithmetic binary: {args.input_file}")
        seed_text, ranks = decode_arithmetic_file(args.input_file)
    else:
        # Default: Text file with ranks
        print(f"Reading ranks text file: {args.input_file}")
        seed_text, ranks = read_input_file(args.input_file)

    print(f"Seed text: {repr(seed_text)}")
    print(f"Ranks count: {len(ranks)}")

    # 2. Reconstruct Text from Ranks using LLM
    total_tokens = decode_from_ranks_streaming(
        seed_text,
        ranks,
        args.output_file,
        args.model_id,
        device=device,
        dtype=dtype,
    )

    end_time = time.perf_counter()
    print(f"Decoded {total_tokens} tokens.")
    print(f"Runtime: {end_time - start_time:.2f} sec")


if __name__ == "__main__":
    main()