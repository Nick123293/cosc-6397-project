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
    summary_json=None,
):
    """
    Decoder that mirrors the compressor logic in run_sequence_eval_streaming.

    Policy:
      - Let L0 = number of seed tokens.
      - Let T_total = L0 + len(ranks) be the final decoded length.
      - If T_total <= max_ctx:
          Use KV-cache incremental decoding over the full prefix.
      - If T_total > max_ctx:
          Use strict sliding-window decoding for *all* tokens,
          recomputing a forward pass at each step with the last
          <= max_ctx tokens as context (no KV cache).

    This matches the compressor, which does:
      - KV-cache walk if the *full* sequence fits in context.
      - Otherwise, sliding-window forward passes for every position.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import json

    # ---------------------------
    # Device / dtype defaults
    # ---------------------------
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    if dtype is None:
        if device.type in ["cuda", "mps"]:
            dtype = torch.float16
        else:
            dtype = torch.float32

    # ---------------------------
    # Load tokenizer / model
    # ---------------------------
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    model.to(device)
    model.eval()

    # ---------------------------
    # Determine max context length
    # ---------------------------
    cfg = getattr(model, "config", None)
    max_ctx = None
    if cfg is not None:
        max_ctx = getattr(cfg, "n_positions", None)
        if max_ctx is None:
            max_ctx = getattr(cfg, "max_position_embeddings", None)
    if max_ctx is None:
        max_ctx = 1024  # safe default (must match compressor's fallback)

    # ---------------------------
    # Tokenize seed and init state
    # ---------------------------
    seed_ids = tok(seed_text, return_tensors="pt")["input_ids"].to(device)
    decoded_ids = seed_ids[0].tolist()

    L0 = len(seed_ids[0])
    T_total = L0 + len(ranks)

    # Decide decoding mode to mirror compressor:
    # If the final sequence fits in context -> KV mode
    # Otherwise -> pure sliding-window mode (no KV at all)
    use_kv = T_total <= max_ctx

    # KV-cache state (used only if use_kv=True)
    kv_initialized = False
    past = None
    logits_for_next = None

    # ---------------------------
    # Decode and write output
    # ---------------------------
    with open(output_file, "w", encoding="utf-8") as f_out:
        f_out.write(seed_text)

        text_buffer = []

        for rank in ranks:
            prefix_len = len(decoded_ids)

            if use_kv:
                # --------------------------------------------------
                # KV-cache decoding: full-prefix causal context
                # (Only used when T_total <= max_ctx, so in practice
                #  prefix_len never exceeds max_ctx.)
                # --------------------------------------------------
                if not kv_initialized:
                    # First step: run model on the entire seed prefix
                    with torch.inference_mode():
                        out = model(input_ids=seed_ids, use_cache=True)
                    past = out.past_key_values
                    logits_for_next = out.logits[0, -1, :]
                    kv_initialized = True

                # Convert rank -> token using logits_for_next
                vocab_size = logits_for_next.shape[-1]
                if rank < 1 or rank > vocab_size:
                    raise ValueError(
                        f"Rank {rank} is out of bounds for vocab size {vocab_size} "
                        f"(must be between 1 and {vocab_size})"
                    )

                sorted_ids = torch.argsort(logits_for_next, descending=True)
                next_token_id = int(sorted_ids[rank - 1].item())

                decoded_ids.append(next_token_id)

                token_str = tok.decode(
                    [next_token_id],
                    clean_up_tokenization_spaces=False,
                )
                text_buffer.append(token_str)

                # Prepare logits for the *next* step
                next_input = torch.tensor([[next_token_id]], device=device)
                with torch.inference_mode():
                    out = model(
                        input_ids=next_input,
                        past_key_values=past,
                        use_cache=True,
                    )
                past = out.past_key_values
                logits_for_next = out.logits[0, -1, :]

            else:
                # --------------------------------------------------
                # Pure sliding-window decoding (Option A)
                # Mirrors compressor's sliding-window path:
                #   start_idx = max(0, pos - max_ctx)
                #   context = tokens[start_idx:pos]
                # --------------------------------------------------
                start_idx = max(0, prefix_len - max_ctx)
                ctx_slice = decoded_ids[start_idx:prefix_len]

                context_ids = torch.tensor(
                    ctx_slice,
                    dtype=torch.long,
                    device=device,
                ).unsqueeze(0)  # (1, <= max_ctx)

                with torch.inference_mode():
                    out = model(input_ids=context_ids)
                    last_logits = out.logits[0, -1, :]

                vocab_size = last_logits.shape[-1]
                if rank < 1 or rank > vocab_size:
                    raise ValueError(
                        f"Rank {rank} is out of bounds for vocab size {vocab_size} "
                        f"(must be between 1 and {vocab_size})"
                    )

                sorted_ids = torch.argsort(last_logits, descending=True)
                next_token_id = int(sorted_ids[rank - 1].item())

                decoded_ids.append(next_token_id)

                token_str = tok.decode(
                    [next_token_id],
                    clean_up_tokenization_spaces=False,
                )
                text_buffer.append(token_str)

            # Periodically flush text to disk
            if len(text_buffer) >= 1024:
                f_out.write("".join(text_buffer))
                text_buffer = []

        # Final flush
        if text_buffer:
            f_out.write("".join(text_buffer))

    total_tokens = len(decoded_ids)

    # Optional summary
    if summary_json is not None:
        summary = {
            "seed_text": seed_text,
            "num_ranks": len(ranks),
            "total_tokens": int(total_tokens),
            "max_ctx": int(max_ctx),
            "mode": "kv_only" if use_kv else "sliding_window_only",
            "model_id": model_id,
            "device": str(device),
            "dtype": str(dtype),
        }
        with open(summary_json, "w", encoding="utf-8") as f_js:
            json.dump(summary, f_js, indent=2)

    return total_tokens


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