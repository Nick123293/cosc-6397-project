#!/usr/bin/env python3
import argparse
import json
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
    if device.type in ["cuda", "mps"]:
        return torch.float16
    else:
        return torch.float32

def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def read_ranks_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [int(line.strip()) for line in f if line.strip()]

# ---------------------- Rank decoder ----------------------
def decode_from_ranks_streaming(seed_text: str, ranks: list, output_file: str, model_id: str, device=None, dtype=None, summary_json=None):
    device = device or choose_device()
    dtype = dtype or choose_dtype(device)

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device)
    model.eval()

    # Encode seed
    seed_ids = tok(seed_text, return_tensors="pt").input_ids.to(device)
    decoded_ids_sample = seed_ids[0, :20].tolist()  # store first few tokens for summary

    # Write seed text first
    with open(output_file, "w", encoding="utf-8") as f_out:
        f_out.write(seed_text)
        f_out.write(" ")  # optional space after seed

        # Initialize decoded_ids for autoregressive input
        decoded_ids = seed_ids[0].tolist()

        with torch.inference_mode():
            for rank in tqdm(ranks, desc="Decoding tokens"):
                input_ids = torch.tensor([decoded_ids], device=device)
                logits = model(input_ids).logits
                last_logits = logits[0, -1]
                probs = torch.softmax(last_logits, dim=-1)
                sorted_ids = torch.argsort(probs, descending=True)
                next_token_id = int(sorted_ids[rank - 1].item())
                decoded_ids.append(next_token_id)

                # Decode only the last token to save memory
                token_str = tok.decode([next_token_id])
                f_out.write(token_str)

    total_tokens = len(decoded_ids)

    # Write summary JSON if requested
    if summary_json is not None:
        summary = {
            "seed_text": seed_text,
            "num_ranks": len(ranks),
            "total_tokens": total_tokens,
            "decoded_ids_sample": decoded_ids_sample,
            "decoded_text_start": seed_text + "..."  # could add first 100 chars if desired
        }
        with open(summary_json, "w", encoding="utf-8") as f_json:
            json.dump(summary, f_json, indent=2, ensure_ascii=False)

    return total_tokens

# ---------------------- CLI ----------------------
def main():
    parser = argparse.ArgumentParser(
        description="Decode text from token ranks using LLaMA (streaming)"
    )
    parser.add_argument("seed_file", help="Path to file containing the first few words (seed)")
    parser.add_argument("ranks_file", help="Path to ranks file (one rank per line)")
    parser.add_argument("output_file", help="Path to write the decoded text")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="HuggingFace model id")
    parser.add_argument("--summary-json", default=None, help="Optional path to write JSON summary")
    args = parser.parse_args()

    device = choose_device()
    dtype = choose_dtype(device)
    print(f"Using device: {device}, dtype: {dtype}")

    seed_text = read_text_file(args.seed_file)
    ranks = read_ranks_file(args.ranks_file)
    print(f"Seed text: '{seed_text}'")
    print(f"Number of ranks to decode: {len(ranks)}")

    total_tokens = decode_from_ranks_streaming(
        seed_text, ranks, args.output_file, args.model_id, device=device, dtype=dtype, summary_json=args.summary_json
    )

    print(f"Decoding complete. Total tokens written: {total_tokens}")
    if args.summary_json:
        print(f"Summary JSON saved to: {args.summary_json}")

if __name__ == "__main__":
    main()
