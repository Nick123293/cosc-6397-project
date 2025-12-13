#!/usr/bin/env python3
import argparse
import json
import os
import struct
import time
import collections  # Added for ANS logic
from typing import List, Tuple

import torch
import zstandard as zstd 
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
# ---------------------- ZSTD ON RANKS -----------------------
# ============================================================
def decode_zstd_rank_file(path: str):
    """
    Reads [Seed Length][Seed][Zstd Compressed Ranks]
    Decompresses ranks and returns (seed_text, list_of_ranks).
    """
    with open(path, "rb") as f:
        data = f.read()

    idx = 0
    # 1. Read Seed
    seed_len = struct.unpack_from(">I", data, idx)[0]
    idx += 4
    seed_text = data[idx:idx + seed_len].decode("utf-8")
    idx += seed_len
    
    # 2. Decompress the rest
    compressed_ranks = data[idx:]
    dctx = zstd.ZstdDecompressor()
    ranks_text_bytes = dctx.decompress(compressed_ranks)
    
    # 3. Parse ranks
    ranks_text = ranks_text_bytes.decode('utf-8')
    ranks = []
    for line in ranks_text.splitlines():
        if line.strip():
            ranks.append(int(line.strip()))
            
    return seed_text, ranks


# ============================================================
# ---------------------- Huffman Decoding --------------------
# ============================================================
def decode_huffman_file(path: str):
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

        # Binary search
        n = len(cum_freq) - 1
        lo, hi = 0, n
        
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
    with open(path, "rb") as f:
        data = f.read()

    idx = 0
    # Seed text
    seed_len = struct.unpack_from(">I", data, idx)[0]
    idx += 4
    seed_text = data[idx:idx + seed_len].decode("utf-8")
    idx += seed_len

    # Frequency table
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

    # Encoded data
    padding = data[idx]
    idx += 1

    bitstream = read_bits_from_bytes(data[idx:])
    if padding > 0:
        bitstream = bitstream[:-padding]

    bits = [1 if b == "1" else 0 for b in bitstream]
    decoder = ArithmeticDecoder(bits)

    num_tokens = total
    ranks: List[int] = []
    
    for _ in tqdm(range(num_tokens), desc="Arith decoding"):
        sym_idx = decoder.decode_symbol(cum_freq, total)
        ranks.append(symbols[sym_idx])

    return seed_text, ranks


# ============================================================
# ---------------------- ANS Decoding ------------------------
# ============================================================
class StreamingrANS:
    """
    A Streaming rANS decoder (ported from ans.py).
    """
    def __init__(self, frequencies):
        self.L = 1 << 16  # Renormalization lower bound
        self.M_scale = sum(frequencies.values())
        self.freqs = frequencies
        
        # Build CDF
        self.cum_freq = {}
        self.symbol_map = {} 
        current = 0
        # Sort keys to ensure deterministic behavior across machines
        for sym in sorted(frequencies.keys()):
            f = frequencies[sym]
            self.cum_freq[sym] = current
            # Map range to symbol for decoding
            for i in range(current, current + f):
                self.symbol_map[i] = sym
            current += f

    def decode(self, initial_state, stream, num_symbols):
        state = initial_state
        stream_iter = iter(stream)
        decoded = []
        
        for _ in range(num_symbols):
            slot = state % self.M_scale
            sym = self.symbol_map[slot]
            
            freq = self.freqs[sym]
            start = self.cum_freq[sym]
            
            # Decode state
            state = freq * (state // self.M_scale) + slot - start
            decoded.append(sym)
            
            # Renormalize (pull bytes from stream)
            while state < self.L:
                try:
                    val = next(stream_iter) 
                    state = (state << 8) | val
                except StopIteration:
                    break
                    
        return decoded

def decode_ans_file(path: str):
    """
    Reads .ans binary file generated by ans.py
    Returns: (seed_text, ranks)
    """
    with open(path, "rb") as f:
        # 1. Read Header Length
        meta_len = struct.unpack("I", f.read(4))[0]
        # 2. Read Metadata JSON
        meta_json = f.read(meta_len)
        # 3. Read Bitstream (ranks compressed)
        bit_stream = f.read()
        
    metadata = json.loads(meta_json)
    
    # JSON keys are always strings, need to convert keys back to integers for ANS logic
    freqs = {int(k): v for k, v in metadata["frequencies"].items()}
    seed_text = metadata["seed_text"]
    final_state = metadata["final_state"]
    num_symbols = metadata["num_symbols"]
    
    # Instantiate Decoder
    rans = StreamingrANS(freqs)
    
    # Decode
    # Convert bytes to list of integers for the iterator
    ranks = rans.decode(final_state, list(bit_stream), num_symbols)
    
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
    """
    Inverse of run_sequence_eval_streaming.

    Given:
      - seed_text: the same seed prefix used at compression time
      - ranks: list of 1-based ranks for each subsequent token
      - model_id: HF model name (must match compressor)
    Reconstructs the token sequence and writes decoded text to output_file.
    """
    device = device or choose_device()
    dtype = dtype or choose_dtype(device)

    tok, model = load_model_and_tokenizer(model_id, device, dtype)

    # Config / context length
    cfg = getattr(model, "config", None)
    max_ctx = getattr(cfg, "max_position_embeddings", 1024) or 1024

    # Seed tokens
    seed_ids = tok(seed_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    L0 = seed_ids.shape[1]

    num_ranks = len(ranks)
    T = L0 + num_ranks

    # This will hold the full reconstructed token sequence (equivalent to full_ids[0] in compressor)
    decoded_ids = seed_ids[0].tolist()

    # -----------------------------
    # Case 1: Fits in context (KV cache)
    # Mirrors compressor's: Case 1: Fits in Context (KV Cache)
    # -----------------------------
    if T <= max_ctx:
        with torch.inference_mode():
            # Match compressor:
            #   prefix = full_ids[:, :L0]
            #   out = model(input_ids=prefix, use_cache=True)
            #   past = out.past_key_values
            out = model(input_ids=seed_ids, use_cache=True)
            past = out.past_key_values

            for i, rank in enumerate(
                tqdm(ranks, desc="Reconstructing (KV cache)", unit="token")
            ):
                # Compressor uses out.logits[0, -1, :] for token at position pos = L0 + i
                logits = out.logits[0, -1, :]
                sorted_ids = torch.argsort(logits, descending=True)

                if rank < 1 or rank > sorted_ids.numel():
                    raise ValueError(
                        f"Invalid rank {rank} at step {i} (vocab size={sorted_ids.numel()})"
                    )

                next_token_id = int(sorted_ids[rank - 1].item())
                decoded_ids.append(next_token_id)

                # Mirror compressor:
                #   if pos + 1 < T:
                #       next_input = full_ids[:, pos:pos+1]
                #       out = model(input_ids=next_input, past_key_values=past, use_cache=True)
                #       past = out.past_key_values
                if i + 1 < num_ranks:
                    next_input = torch.tensor(
                        [[next_token_id]], dtype=torch.long, device=device
                    )
                    out = model(
                        input_ids=next_input,
                        past_key_values=past,
                        use_cache=True,
                    )
                    past = out.past_key_values

    # -----------------------------
    # Case 2: Exceeds context (Sliding window)
    # Mirrors compressor's: Case 2: Exceeds Context (Sliding Window)
    # -----------------------------
    else:
        with torch.inference_mode():
            for i, rank in enumerate(
                tqdm(ranks, desc="Reconstructing (sliding window)", unit="token")
            ):
                # Position in the full sequence (same pos as in compressor loop)
                pos = L0 + i

                # Compressor:
                #   start_idx = max(0, pos - max_ctx)
                #   context_ids = full_ids[:, start_idx:pos]
                #   if context_ids.shape[1] == 0: context_ids = full_ids[:, 0:1]
                start_idx = max(0, pos - max_ctx)
                context_slice = decoded_ids[start_idx:pos]

                if len(context_slice) == 0:
                    # Fallback exactly like compressor: first token only
                    context_slice = decoded_ids[0:1]

                context_ids = torch.tensor(
                    [context_slice], dtype=torch.long, device=device
                )

                out = model(input_ids=context_ids)
                logits = out.logits[0, -1, :]
                sorted_ids = torch.argsort(logits, descending=True)

                if rank < 1 or rank > sorted_ids.numel():
                    raise ValueError(
                        f"Invalid rank {rank} at step {i} (vocab size={sorted_ids.numel()})"
                    )

                next_token_id = int(sorted_ids[rank - 1].item())
                decoded_ids.append(next_token_id)

    # -----------------------------
    # Decode full token sequence to text
    # -----------------------------
    full_text = tok.decode(
        decoded_ids,
        clean_up_tokenization_spaces=False,
    )

    with open(output_file, "w", encoding="utf-8") as f_out:
        f_out.write(full_text)

    return len(decoded_ids)



# ============================================================
# ---------------------------- CLI ---------------------------
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Decode ranks using LLaMA or run Zstandard/ANS/Huffman/Arithmetic Baselines."
    )
    parser.add_argument("input_file", help="Path to input file.")
    parser.add_argument("output_file", help="Path to write the reconstructed text.")
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
    parser.add_argument(
        "--ans-binary",
        action="store_true",
        help="Treat input_file as an rANS-compressed binary.",
    )
    parser.add_argument(
        "--zstd",
        action="store_true",
        help="Treat input_file as a Zstd-compressed RANKS file.",
    )
    
    args = parser.parse_args()

    start_time = time.perf_counter()

    device = choose_device()
    dtype = choose_dtype(device)
    print(f"Using device: {device}, dtype={dtype}")

    if args.huffman_binary:
        print(f"Decoding Huffman binary: {args.input_file}")
        seed_text, ranks = decode_huffman_file(args.input_file)
    elif args.arithmetic_binary:
        print(f"Decoding Arithmetic binary: {args.input_file}")
        seed_text, ranks = decode_arithmetic_file(args.input_file)
    elif args.ans_binary:
        print(f"Decoding ANS binary: {args.input_file}")
        seed_text, ranks = decode_ans_file(args.input_file)
    elif args.zstd:
        print(f"Decoding Zstandard ranks: {args.input_file}")
        seed_text, ranks = decode_zstd_rank_file(args.input_file)
    else:
        print(f"Reading ranks text file: {args.input_file}")
        seed_text, ranks = read_input_file(args.input_file)

    print(f"Seed text: {repr(seed_text)}")
    print(f"Ranks count: {len(ranks)}")

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