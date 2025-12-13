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
import collections # Re-added for explicit usage if needed, though defaultdict covers most

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------------- Config ----------------------
DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-1B"
DEFAULT_SEED_WORDS = 5
# FLUSH_EVERY = 1048576   # ranks buffered before writing to tmp file
CODE_BITS = 32
TOP = (1 << CODE_BITS) - 1
FIRST_QTR = TOP // 4 + 1
HALF = 2 * FIRST_QTR
THIRD_QTR = 3 * FIRST_QTR


# ============================================================
# ---------------------- RANS LOGIC --------------------------
# ============================================================

class StreamingrANS:
    """
    A Streaming rANS encoder/decoder.
    Integrated from ans.py
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

    def encode(self, symbols):
        state = self.L 
        bit_stream = [] 

        # Encode in reverse (LIFO)
        for sym in reversed(symbols):
            freq = self.freqs[sym]
            start = self.cum_freq[sym]

            # Renormalize to stay within bounds
            while state >= (self.L * 4): 
                bit_stream.append(state & 0xFF) 
                state >>= 8                     

            # Update state
            state = (state // freq) * self.M_scale + start + (state % freq)

        return state, bytearray(reversed(bit_stream))

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

def write_combined_ans_file(
    seed_text: str,
    rank_freq: Dict[int, int],
    ranks_file: str,
    output_bin: str
):
    """
    Writes the binary file using the rANS method.
    Matches the file format expected by the original ans.py loader.
    """
    # 1. Read ranks from the temp file
    # We need the full list in memory to encode in reverse.
    ranks = []
    with open(ranks_file, 'r', encoding='utf-8') as f:
        # First line is seed text, skip it
        lines = f.readlines()
        # The first line is the seed text
        # The rest are ranks
        for line in lines[1:]:
            if line.strip():
                try:
                    ranks.append(int(line.strip()))
                except ValueError:
                    pass

    num_symbols = len(ranks)
    
    # 2. Initialize rANS
    rans = StreamingrANS(rank_freq)
    
    # 3. Encode
    final_state, stream = rans.encode(ranks)
    
    # 4. Prepare Metadata (matching ans.py format)
    metadata = {
        "seed_text": seed_text,
        "final_state": final_state,
        "num_symbols": num_symbols,
        "frequencies": rank_freq
    }
    
    # 5. Write to File
    meta_json = json.dumps(metadata).encode('utf-8')
    
    with open(output_bin, "wb") as f:
        f.write(struct.pack("I", len(meta_json))) # 4 bytes for header length
        f.write(meta_json)                        # Metadata
        f.write(stream)                           # Compressed Ranks


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


class ArithmeticEncoder:
    def __init__(self):
        self.low = 0
        self.high = TOP
        self.bits_to_follow = 0
        self.bits = []  # type: List[int]

    def _output_bit_plus_follow(self, bit: int):
        self.bits.append(bit)
        inv = 1 - bit
        for _ in range(self.bits_to_follow):
            self.bits.append(inv)
        self.bits_to_follow = 0

    def encode_symbol(self, symbol_idx: int, cum_freq: List[int], total: int):
        low = self.low
        high = self.high
        range_ = high - low + 1

        sym_low = cum_freq[symbol_idx]
        sym_high = cum_freq[symbol_idx + 1]

        self.high = low + (range_ * sym_high) // total - 1
        self.low = low + (range_ * sym_low) // total

        # Renormalization (E1, E2, E3 cases)
        while True:
            if self.high < HALF:
                self._output_bit_plus_follow(0)
            elif self.low >= HALF:
                self._output_bit_plus_follow(1)
                self.low -= HALF
                self.high -= HALF
            elif self.low >= FIRST_QTR and self.high < THIRD_QTR:
                self.bits_to_follow += 1
                self.low -= FIRST_QTR
                self.high -= FIRST_QTR
            else:
                break

            self.low <<= 1
            self.high = (self.high << 1) + 1

    def finish(self) -> List[int]:
        # Emit final bits
        self.bits_to_follow += 1
        if self.low < FIRST_QTR:
            self._output_bit_plus_follow(0)
        else:
            self._output_bit_plus_follow(1)
        return self.bits

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

def build_cumulative_freq(rank_freq: Dict[int, int]):
    """
    From {rank: freq} build:
      - symbols: sorted list of ranks
      - cum_freq: list such that cum_freq[i] is sum of freqs for symbols[:i]
      - total: total number of ranks
    """
    symbols = sorted(rank_freq.keys())
    cum_freq = [0]
    for r in symbols:
        cum_freq.append(cum_freq[-1] + rank_freq[r])
    total = cum_freq[-1]
    return symbols, cum_freq, total

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


def write_combined_arithmetic_file(
    seed_text: str,
    rank_freq: Dict[int, int],
    ranks_file: str,
    output_bin: str,
):
    """
    Writes one binary file containing:
      seed text
      frequency table
      arithmetic-coded bitstream of ranks
    """
    # Build cumulative distribution over ranks
    symbols, cum_freq, total = build_cumulative_freq(rank_freq)
    symbol_to_idx = {r: i for i, r in enumerate(symbols)}

    # Encode ranks (skip first line = seed)
    encoder = ArithmeticEncoder()
    with open(ranks_file, "r", encoding="utf-8") as f:
        next(f)  # skip seed line
        for line in f:
            line = line.strip()
            if not line:
                continue
            rk = int(line)
            idx = symbol_to_idx[rk]
            encoder.encode_symbol(idx, cum_freq, total)

    bits = encoder.finish()

    # Convert bits -> bytes with padding
    bitstring = "".join("1" if b else "0" for b in bits)
    padding = (8 - (len(bitstring) % 8)) % 8
    bitstring += "0" * padding

    data_bytes = bytes(
        int(bitstring[i:i + 8], 2) for i in range(0, len(bitstring), 8)
    )

    import struct

    with open(output_bin, "wb") as bf:
        # 1) Seed text
        seed_bytes = seed_text.encode("utf-8")
        bf.write(struct.pack(">I", len(seed_bytes)))
        bf.write(seed_bytes)

        # 2) Frequency table
        bf.write(struct.pack(">I", len(symbols)))  # number of distinct ranks
        for r in symbols:
            freq = rank_freq[r]
            bf.write(struct.pack(">II", r, freq))   # rank, freq

        # 3) Encoded data
        bf.write(struct.pack("B", padding))        # padding bits at end
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
    Compute ranks for the given text.

    If the total length T of the tokenized text satisfies T <= max_ctx
    (i.e., the model can attend to the entire prefix without truncation),
    we use a KV-cache-based left-to-right evaluation, which is O(T)
    forward calls and reuses the cached keys/values.

    Otherwise, we fall back to a sliding-window evaluation that respects
    the model's maximum context length by taking the last `max_ctx` tokens
    as context for each position and running a fresh forward pass.
    This matches the original semantics.
    """

    device = model.device

    # Full tokenization of the whole text
    full = tok(text, return_tensors="pt")
    full_ids = full["input_ids"].to(device)
    T = full_ids.shape[1]

    # Seed text and its tokenization
    seed_text = first_n_words_slice(text, seed_words)
    seed_ids = tok(seed_text, return_tensors="pt")["input_ids"].to(device)
    L0 = seed_ids.shape[1]

    # Determine maximum context length from model config
    cfg = getattr(model, "config", None)
    max_ctx = None
    if cfg is not None:
        max_ctx = getattr(cfg, "n_positions", None)
        if max_ctx is None:
            max_ctx = getattr(cfg, "max_position_embeddings", None)
    if max_ctx is None:
        max_ctx = 1024  # safe default

    rank_freq = defaultdict(int)
    buffer = []

    with open(tmp_ranks_path, "w", encoding="utf-8") as f:
        # First line is the seed text (metadata used by your encoders)
        f.write(seed_text + "\n")

        # ---------------------------------------------------------
        # Case 1: Entire sequence fits into model context
        #         → use KV cache, left-to-right.
        #
        # For each pos in [L0, T-1], we want logits that predict token
        # at index `pos` given tokens [0..pos-1]. With a causal model:
        #
        #   - Run model on prefix [0..L0-1] with use_cache=True.
        #   - The logits for predicting token L0 are out.logits[:, -1, :].
        #   - Then, for pos = L0..T-1:
        #       - compute rank from current logits (predicting pos),
        #       - feed the true token at `pos` back in with past_key_values
        #         to get logits for `pos+1`, etc.
        # ---------------------------------------------------------
        if T <= max_ctx:
            with torch.inference_mode():
                # Initial prefix: tokens before the first position we score (pos = L0)
                prefix = full_ids[:, :L0]  # shape (1, L0)
                out = model(input_ids=prefix, use_cache=True)
                past = out.past_key_values

                # We will use out.logits from each step as the prediction
                # for the current `pos`, then advance using the true token.
                for pos in range(L0, T):
                    true_id = int(full_ids[0, pos])

                    logits = out.logits[0, -1, :]  # prediction for token at `pos`
                    sorted_ids = torch.argsort(logits, descending=True)
                    true_pos = (sorted_ids == true_id).nonzero(as_tuple=False)
                    if true_pos.numel() == 0:
                        raise RuntimeError(
                            "True token id not found in sorted_ids (should never happen)"
                        )
                    rk = int(true_pos.item()) + 1

                    rank_freq[rk] += 1
                    buffer.append(rk)

                    # Prepare logits for the next position (pos+1) by
                    # feeding in the *true* token at `pos`.
                    if pos + 1 < T:
                        next_input = full_ids[:, pos:pos+1]  # shape (1, 1)
                        out = model(
                            input_ids=next_input,
                            past_key_values=past,
                            use_cache=True,
                        )
                        past = out.past_key_values

        # ---------------------------------------------------------
        # Case 2: Sequence exceeds context length
        #         → fall back to original sliding-window logic.
        # ---------------------------------------------------------
        else:
            for pos in range(L0, T):
                true_id = int(full_ids[0, pos])

                # Take up to max_ctx previous tokens as context
                # (pos is the index of the token to predict)
                start_idx = max(0, pos - max_ctx)
                context_ids = full_ids[:, start_idx:pos]  # shape (1, <=max_ctx)

                if context_ids.shape[1] == 0:
                    # Degenerate case: extremely short text
                    # Use the first token as context to avoid empty input.
                    context_ids = full_ids[:, 0:1]

                with torch.inference_mode():
                    # No persistent KV cache: one forward per window,
                    # exactly as in the original implementation.
                    out = model(input_ids=context_ids)
                    logits = out.logits[0, -1, :]  # prediction for the "next token"

                    # Rank = 1-based index of true_id in argsort(logits, descending=True)
                    sorted_ids = torch.argsort(logits, descending=True)
                    true_pos = (sorted_ids == true_id).nonzero(as_tuple=False)
                    if true_pos.numel() == 0:
                        raise RuntimeError(
                            "True token id not found in sorted_ids (should never happen)"
                        )
                    rk = int(true_pos.item()) + 1

                rank_freq[rk] += 1
                buffer.append(rk)

        # Final flush
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
    parser.add_argument("--huffman-encoding", action="store_true", help="Use Huffman encoding")
    parser.add_argument("--arith-encoding", action="store_true", help="Use Arithmetic encoding")
    parser.add_argument("--ans-encoding", action="store_true", help="Use rANS encoding (integrated from ans.py)")
    parser.add_argument("--keep-intermediate", action="store_true", help="Keep the intermediate .ranks.txt file")
    
    args = parser.parse_args()
    start_time = time.perf_counter()
    device = choose_device()
    dtype = choose_dtype(device)

    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, revision="main")
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=dtype).to(device)
    model.eval()

    text = read_text_file(args.input)

    tmp_ranks = args.output + ".ranks.txt"

    print(f"Generating ranks using {args.model_id}...")
    seed_text, rank_freq = run_sequence_eval_streaming(
        text=text,
        tok=tok,
        model=model,
        tmp_ranks_path=tmp_ranks,
        seed_words=args.seed_words,
    )
    print(f"Ranks generation complete. Found {len(rank_freq)} unique ranks.")

    # 1. No encoding selected
    if not args.huffman_encoding and not args.arith_encoding and not args.ans_encoding:
        print(f"No encoding flag set. Raw ranks written to {tmp_ranks}")
        return

    # 2. Prevent multiple encodings
    active_encodings = [args.huffman_encoding, args.arith_encoding, args.ans_encoding]
    if sum(active_encodings) > 1:
        raise ValueError("Choose only one of --huffman-encoding, --arith-encoding, or --ans-encoding")

    # 3. Apply selected encoding
    scheme = "Unknown"
    
    if args.huffman_encoding:
        scheme = "Huffman"
        codebook = build_huffman_codebook(rank_freq)
        write_combined_huffman_file(
            seed_text=seed_text,
            codebook=codebook,
            ranks_file=tmp_ranks,
            output_bin=args.output,
        )
        
    elif args.arith_encoding:
        scheme = "Arithmetic"
        write_combined_arithmetic_file(
            seed_text=seed_text,
            rank_freq=rank_freq,
            ranks_file=tmp_ranks,
            output_bin=args.output,
        )
        
    elif args.ans_encoding:
        scheme = "rANS"
        write_combined_ans_file(
            seed_text=seed_text,
            rank_freq=rank_freq,
            ranks_file=tmp_ranks,
            output_bin=args.output,
        )

    # 4. Cleanup
    if not args.keep_intermediate:
        os.remove(tmp_ranks)
    
    end_time = time.perf_counter()
    print(f"{scheme} encoded combined file saved to {args.output}")
    print(f"Total runtime: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()