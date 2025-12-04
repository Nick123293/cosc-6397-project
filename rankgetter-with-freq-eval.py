#!/usr/bin/env python3
import re
import json
import argparse
from typing import Dict, Any, List
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------- Config defaults ----------------------
DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-1B"
DEFAULT_SEED_WORDS = 5
# FLUSH_EVERY is the number of ranks to buffer before writing to disk.
# You set it to 1,048,576 which is fine; if you want more frequent flushing, use e.g. 1000.
FLUSH_EVERY = 1048576


# ---------------------- Helpers ----------------------
def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon
    else:
        return torch.device("cpu")


def choose_dtype(device: torch.device) -> torch.dtype:
    if device.type in ("cuda", "mps"):
        return torch.float16
    else:
        return torch.float32


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def first_n_words_slice(text: str, n_words: int) -> str:
    """
    Return substring of 'text' containing exactly the first n words (regex \S+),
    preserving spacing/punctuation up to that point.
    """
    if n_words <= 0:
        return ""
    matches = list(re.finditer(r"\S+", text))
    if not matches:
        return ""
    end_idx = matches[min(n_words, len(matches)) - 1].end()
    return text[:end_idx]


def token_rank(prob_vector: torch.Tensor, true_token_id: int) -> int:
    """
    prob_vector: shape [vocab], already softmaxed.
    Return 1-based rank of true_token_id among all tokens (descending prob).
    """
    true_p = prob_vector[true_token_id]
    higher = torch.sum(prob_vector > true_p).item()
    return int(higher) + 1


# ---------------------- Streaming eval ----------------------
def run_sequence_eval_streaming(
    text: str,
    tok: AutoTokenizer,
    model: AutoModelForCausalLM,
    ranks_out_path: str,
    seed_words: int = 5,
    topk_show: int = 0,        # currently unused, but kept for interface compatibility
    sample_steps_max: int = 10,
) -> Dict[str, Any]:
    """
    Sequential, cache-based evaluation that streams ranks to disk and tracks rank frequencies.

    Output file layout:
        # SEED_TEXT_BEGIN
        <seed text line>
        # SEED_TEXT_END

        <rank_0>
        <rank_1>
        ...
    """
    device = model.device

    # Tokenize full text once
    full = tok(text, return_tensors="pt")
    full_ids = full["input_ids"].to(device)  # [1, T]
    T = full_ids.shape[1]

    # Seed text by first N words
    seed_text = first_n_words_slice(text, seed_words)
    if not seed_text.strip():
        raise ValueError("Seed text is empty; increase SEED_WORDS or provide non-empty text.")

    seed = tok(seed_text, return_tensors="pt").to(device)
    seed_ids = seed["input_ids"]  # [1, L0]
    L0 = seed_ids.shape[1]

    if L0 >= T:
        raise ValueError("Seed covers the entire text; decrease SEED_WORDS.")

    summary: Dict[str, Any] = {
        "seed_text": seed_text,
        "seed_len_tokens": int(L0),
        "total_len_tokens": int(T),
        "num_evaluated_tokens": 0,
        "mean_rank": None,
        "top1_accuracy": None,
        "mrr": None,
        "sample_steps": [],
    }

    # Stats accumulators
    total_tokens_eval = 0
    sum_ranks = 0.0
    sum_recip = 0.0
    top1_hits = 0

    sample_steps: List[Dict[str, Any]] = []

    # Frequency table for Huffman coding
    rank_freq = defaultdict(int)

    # Buffer for streaming to disk
    ranks_buffer: List[int] = []

    # Warm up model with seed to build KV cache
    with torch.inference_mode():
        out = model(input_ids=seed_ids, use_cache=True)
        past = out.past_key_values

    # Open output file and write seed header + ranks
    with open(ranks_out_path, "w", encoding="utf-8") as f_out:
        # Seed header
        f_out.write("# SEED_TEXT_BEGIN\n")
        f_out.write(seed_text + "\n")
        f_out.write("# SEED_TEXT_END\n\n")

        # Iterate over positions from L0..T-1
        for pos in range(L0, T):
            true_id = int(full_ids[0, pos].item())

            with torch.inference_mode():
                step_input = full_ids[:, pos - 1 : pos]  # [1, 1]
                out = model(
                    input_ids=step_input,
                    past_key_values=past,
                    use_cache=True,
                )
                past = out.past_key_values
                logits_last = out.logits[:, -1, :]  # [1, V]
                probs = torch.softmax(logits_last, dim=-1)[0]  # [V]

            rk = token_rank(probs, true_id)
            pred_id = int(torch.argmax(probs).item())
            pred_tok = tok.decode([pred_id])
            pred_p = float(probs[pred_id].item())

            # Update streaming stats
            total_tokens_eval += 1
            sum_ranks += rk
            sum_recip += 1.0 / rk
            if rk == 1:
                top1_hits += 1

            # Update frequency table
            rank_freq[rk] += 1

            # Stream rank to disk in buffered chunks
            ranks_buffer.append(rk)
            if len(ranks_buffer) >= FLUSH_EVERY:
                for r in ranks_buffer:
                    f_out.write(f"{r}\n")
                ranks_buffer.clear()

            # Save a few sample steps for inspection
            if len(sample_steps) < sample_steps_max:
                sample_steps.append(
                    {
                        "pos": pos,
                        "true_id": true_id,
                        "true_tok": tok.decode([true_id]),
                        "rank_of_true": rk,
                        "pred_id": pred_id,
                        "pred_tok": pred_tok,
                        "pred_p": pred_p,
                    }
                )

        # Flush any remaining ranks
        if ranks_buffer:
            for r in ranks_buffer:
                f_out.write(f"{r}\n")
            ranks_buffer.clear()

    # Fill metrics
    if total_tokens_eval > 0:
        summary["num_evaluated_tokens"] = int(total_tokens_eval)
        summary["mean_rank"] = float(sum_ranks / total_tokens_eval)
        summary["top1_accuracy"] = float(top1_hits / total_tokens_eval)
        summary["mrr"] = float(sum_recip / total_tokens_eval)

    summary["sample_steps"] = sample_steps
    summary["rank_freq"] = {str(k): int(v) for k, v in rank_freq.items()}

    return summary


# ---------------------- CLI script ----------------------
def main():
    parser = argparse.ArgumentParser(
        description="Streaming next-word prediction evaluation with Llama 3.2, GPU-aware."
    )
    parser.add_argument("input", help="Path to input text file")
    parser.add_argument("output", help="Path to output ranks file (text)")
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"HF model id (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--seed-words",
        type=int,
        default=DEFAULT_SEED_WORDS,
        help=f"Number of seed words (default: {DEFAULT_SEED_WORDS})",
    )
    parser.add_argument(
        "--topk-show",
        type=int,
        default=0,
        help="(Currently unused in streaming mode; reserved for future debug features.)",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional: path to write a JSON summary (stats, freq table, sample steps).",
    )

    args = parser.parse_args()

    # Device + dtype
    device = choose_device()
    dtype = choose_dtype(device)
    print(f"Using device: {device}, dtype: {dtype}")

    # Load model & tokenizer
    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    # Read text
    text = read_text_file(args.input)
    print(f"Loaded text with {len(text)} characters.")

    # Run streaming evaluation (low memory)
    summary = run_sequence_eval_streaming(
        text=text,
        tok=tok,
        model=model,
        ranks_out_path=args.output,
        seed_words=args.seed_words,
        topk_show=args.topk_show,
    )

    print("\n--- SUMMARY ---")
    print(f"Seed text: {repr(summary['seed_text'])}")
    print(f"Evaluated tokens: {summary['num_evaluated_tokens']}")
    print(
        f"Top-1 accuracy: {summary['top1_accuracy']:.4f} | "
        f"MRR: {summary['mrr']:.4f} | "
        f"Mean rank: {summary['mean_rank']:.2f}"
    )
    print(f"Ranks (and seed header) written to: {args.output}")

    # Write JSON summary if requested
    if args.summary_json is not None:
        with open(args.summary_json, "w", encoding="utf-8") as f_js:
            json.dump(summary, f_js, indent=2, ensure_ascii=False)
        print(f"Summary JSON written to: {args.summary_json}")


if __name__ == "__main__":
    main()
