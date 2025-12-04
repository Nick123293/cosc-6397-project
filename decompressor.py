#!/usr/bin/env python3
import argparse
import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ---------------------- Config defaults ----------------------
DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-1B"

# ---------------------- Helpers ----------------------
def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon
    else:
        return torch.device("cpu")

def choose_dtype(device: torch.device) -> torch.dtype:
    # float16 on CPU is usually a bad idea; keep fp32 there
    if device.type in ["cuda", "mps"]:
        return torch.float16
    else:
        return torch.float32

def read_input_file(path: str):
    """Reads a single file: first line is seed, rest are ranks."""
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if not lines:
        raise ValueError("Input file is empty.")

    # Preserve all spaces in the seed text, only remove the newline
    seed_text = lines[0].rstrip("\n")

    # For ranks, stripping is OK
    ranks = [int(line.strip()) for line in lines[1:] if line.strip()]

    return seed_text, ranks

def load_model_and_tokenizer(model_id: str, device: torch.device, dtype: torch.dtype):
    """
    Load tokenizer and model in a more memory-friendly way.
    """
    # Suppress tokenizer parallelism warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, revision="main")

    # low_cpu_mem_usage avoids some extra copies when loading
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
    ).to(device)
    model.eval()

    return tok, model

# ---------------------- Rank decoder ----------------------
def decode_from_ranks_streaming(
    seed_text: str,
    ranks: list,
    output_file: str,
    model_id: str,
    device=None,
    dtype=None,
    summary_json=None,
):
    device = device or choose_device()
    dtype = dtype or choose_dtype(device)

    tok, model = load_model_and_tokenizer(model_id, device, dtype)

    # Encode the seed and run a single forward pass to get initial past_key_values
    seed_ids = tok(seed_text, return_tensors="pt").input_ids.to(device)
    decoded_ids = seed_ids[0].tolist()
    decoded_ids_sample = decoded_ids[:20]  # for summary

    with torch.inference_mode():
        # Initial forward pass on the full seed to get cache
        outputs = model(seed_ids, use_cache=True)
        past_key_values = outputs.past_key_values

        # Last token from the seed (for next step)
        last_token_id = decoded_ids[-1]

        # Preallocate a 1x1 tensor for incremental input_ids (matches self-decode)
        step_ids = torch.empty(1, 1, dtype=torch.long, device=device)

        # Open output file and write the seed text up front
        with open(output_file, "w", encoding="utf-8") as f_out:
            f_out.write(seed_text)

            text_buffer = []

            for rank in tqdm(ranks, desc="Decoding tokens"):
                # Prepare input with only the last token, in-place
                step_ids[0, 0] = last_token_id

                # Incremental forward using past_key_values (KV cache)
                outputs = model(
                    step_ids,
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                past_key_values = outputs.past_key_values
                last_logits = outputs.logits[0, -1, :]

                # Ensure rank is valid
                vocab_size = last_logits.shape[-1]
                if rank < 1 or rank > vocab_size:
                    raise ValueError(
                        f"Rank {rank} is out of bounds for vocab size {vocab_size} "
                        f"(must be between 1 and {vocab_size})"
                    )

                # **Use argsort, just like the compressor + self-decode**
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

    total_tokens = len(decoded_ids)

    if summary_json is not None:
        summary = {
            "seed_text": seed_text,
            "num_ranks": len(ranks),
            "total_tokens": total_tokens,
            "decoded_ids_sample": decoded_ids_sample,
            "decoded_text_start": seed_text + "...",
        }
        with open(summary_json, "w", encoding="utf-8") as f_json:
            json.dump(summary, f_json, indent=2, ensure_ascii=False)

    return total_tokens

# ---------------------- CLI ----------------------
def main():
    parser = argparse.ArgumentParser(
        description="Decode text from token ranks using LLaMA (streaming, single file input)"
    )
    parser.add_argument(
        "input_file",
        help="Path to input file (first line seed, rest ranks)",
    )
    parser.add_argument(
        "output_file",
        help="Path to write the decoded text",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="HuggingFace model id",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional path to write JSON summary",
    )
    args = parser.parse_args()

    device = choose_device()
    dtype = choose_dtype(device)
    print(f"Using device: {device}, dtype: {dtype}")

    seed_text, ranks = read_input_file(args.input_file)
    print(f"Seed text: '{seed_text}'")
    print(f"Number of ranks to decode: {len(ranks)}")

    total_tokens = decode_from_ranks_streaming(
        seed_text,
        ranks,
        args.output_file,
        args.model_id,
        device=device,
        dtype=dtype,
        summary_json=args.summary_json,
    )

    print(f"Decoding complete. Total tokens written: {total_tokens}")
    if args.summary_json:
        print(f"Summary JSON saved to: {args.summary_json}")

if __name__ == "__main__":
    main()
