#!/usr/bin/env python3
import argparse
import json
import os
import struct
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ---------------------- Config defaults ----------------------
DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-1B"


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
# ---------------------- Huffman decoding --------------------
# ============================================================
def read_bits_from_bytes(data: bytes) -> str:
    return "".join(f"{b:08b}" for b in data)


def decode_combined_file(path: str):
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

    seed_ids = tok(seed_text, return_tensors="pt").input_ids.to(device)
    decoded_ids = seed_ids[0].tolist()

    with torch.inference_mode():
        outputs = model(seed_ids, use_cache=True)
        past_key_values = outputs.past_key_values

        last_token_id = decoded_ids[-1]
        step_ids = torch.empty(1, 1, dtype=torch.long, device=device)

        with open(output_file, "w", encoding="utf-8") as f_out:
            f_out.write(seed_text)
            text_buffer = []

            for rank in tqdm(ranks, desc="Decoding tokens"):
                step_ids[0, 0] = last_token_id

                outputs = model(
                    step_ids,
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                past_key_values = outputs.past_key_values

                last_logits = outputs.logits[0, -1, :]
                vocab_size = last_logits.shape[-1]
                if not (1 <= rank <= vocab_size):
                    raise ValueError(f"Rank {rank} is out of bounds for vocab {vocab_size}")

                sorted_ids = torch.argsort(last_logits, descending=True)
                next_token_id = int(sorted_ids[rank - 1].item())

                decoded_ids.append(next_token_id)
                last_token_id = next_token_id

                token_str = tok.decode([next_token_id], clean_up_tokenization_spaces=False)
                text_buffer.append(token_str)

                if len(text_buffer) >= 1024:
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
        description="Decode ranks using LLaMA; optionally Huffman-decode a binary file first."
    )
    parser.add_argument("input_file", help="Text ranks file OR binary combined file.")
    parser.add_argument("output_file", help="Write the reconstructed text here.")
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="HuggingFace model id",
    )
    parser.add_argument(
        "--huffman-binary",
        action="store_true",
        help="Treat input_file as a Huffman-compressed binary file.",
    )
    args = parser.parse_args()

    start_time = time.perf_counter()

    device = choose_device()
    dtype = choose_dtype(device)
    print(f"Using device: {device}, dtype={dtype}")

    if args.huffman_binary:
        print(f"Decoding Huffman binary: {args.input_file}")
        seed_text, ranks = decode_combined_file(args.input_file)
    else:
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
