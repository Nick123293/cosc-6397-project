#!/usr/bin/env python3
import re
import json
import argparse
from typing import Dict, Any, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------- Config defaults ----------------------
DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-1B"
DEFAULT_SEED_WORDS = 5


# ---------------------- Helpers ----------------------
def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon
    else:
        return torch.device("cpu")


def choose_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda" or device.type == "mps":
        # Mixed/low precision on GPU is fine
        return torch.float16
    else:
        # safer on CPU
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


# ---------------------- Core eval (vectorized) ----------------------
def run_sequence_eval_batched(
    text: str,
    tok: AutoTokenizer,
    model: AutoModelForCausalLM,
    seed_words: int = 5,
    topk_show: int = 0,
) -> Dict[str, Any]:
    """
    Vectorized version:
      - Tokenize entire text once.
      - Single forward pass through the model.
      - For positions pos = L0..T-1:
          - use logits[pos-1] to score true token at full_ids[pos]
    """
    device = model.device

    # Full tokenization once
    full = tok(text, return_tensors="pt").to(device)
    full_ids = full["input_ids"]  # [1, T]
    T = full_ids.shape[1]

    # Seed by words (on raw text) to align with tokenization
    seed_text = first_n_words_slice(text, seed_words)
    if not seed_text.strip():
        raise ValueError("Seed text is empty; increase SEED_WORDS or provide non-empty text.")

    seed_ids = tok(seed_text, return_tensors="pt")["input_ids"].to(device)
    L0 = seed_ids.shape[1]

    if L0 >= T:
        raise ValueError("Seed covers the entire text; decrease SEED_WORDS.")

    results: Dict[str, Any] = {
        "seed_text": seed_text,
        "seed_len_tokens": int(L0),
        "total_len_tokens": int(T),
        "steps": [],  # each: {pos, true_id, true_tok, rank_of_true, pred_id, pred_tok, pred_p, [topk]}
    }

    # One big forward pass
    with torch.inference_mode():
        outputs = model(input_ids=full_ids)
        logits = outputs.logits[0]  # [T, V]

    # We predict token at position pos using logits at pos-1
    # Evaluate from pos = L0..T-1
    start = L0
    end = T

    eval_logits = logits[start - 1 : end - 1, :]  # [S, V], where S = T - L0
    true_ids = full_ids[0, start:end]             # [S]

    # Softmax over vocab for all positions at once
    probs_all = torch.softmax(eval_logits, dim=-1)  # [S, V]

    ranks: List[int] = []
    for i in range(eval_logits.shape[0]):
        pos = start + i
        probs = probs_all[i]              # [V]
        true_id = int(true_ids[i].item())
        rk = token_rank(probs, true_id)
        ranks.append(rk)

        # Greedy prediction
        pred_id = int(torch.argmax(probs).item())
        pred_tok = tok.decode([pred_id])
        pred_p = float(probs[pred_id].item())

        step_info = {
            "pos": pos,
            "true_id": true_id,
            "true_tok": tok.decode([true_id]),
            "rank_of_true": rk,
            "pred_id": pred_id,
            "pred_tok": pred_tok,
            "pred_p": pred_p,
        }

        if topk_show and topk_show > 0:
            top_p, top_i = torch.topk(probs, k=min(topk_show, probs.numel()))
            step_info["topk"] = [
                (tok.decode([int(i.item())]), float(p.item()))
                for p, i in zip(top_p, top_i)
            ]

        results["steps"].append(step_info)

    # Summary stats
    results["num_evaluated_tokens"] = len(ranks)
    results["mean_rank"] = float(sum(ranks) / len(ranks))
    results["top1_accuracy"] = float(sum(1 for r in ranks if r == 1) / len(ranks))
    results["mrr"] = float(sum(1.0 / r for r in ranks) / len(ranks))
    results["ranks_array"] = ranks

    return results


# ---------------------- CLI script ----------------------
def main():
    parser = argparse.ArgumentParser(
        description="Next-word prediction evaluation with Llama 3.2, vectorized & GPU-enabled."
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
        help="If > 0, store top-k predictions per step in results (debug/inspection).",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional: path to write a JSON summary (stats + maybe first few steps).",
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

    # Run evaluation (vectorized)
    out = run_sequence_eval_batched(
        text,
        tok=tok,
        model=model,
        seed_words=args.seed_words,
        topk_show=args.topk_show,
    )

    ranks_array = out["ranks_array"]

    # --------- Write ranks array to disk ---------
    # One rank per line, simple & easy to parse.
    with open(args.output, "w", encoding="utf-8") as f:
        for r in ranks_array:
            f.write(f"{r}\n")

    print(f"Saved ranks array ({len(ranks_array)} values) to: {args.output}")

    # Optional JSON summary with metrics + first few steps
    if args.summary_json is not None:
        summary = {
            "seed_text": out["seed_text"],
            "seed_len_tokens": out["seed_len_tokens"],
            "total_len_tokens": out["total_len_tokens"],
            "num_evaluated_tokens": out["num_evaluated_tokens"],
            "mean_rank": out["mean_rank"],
            "top1_accuracy": out["top1_accuracy"],
            "mrr": out["mrr"],
            # include a small sample of steps for inspection
            "sample_steps": out["steps"][:10],
        }
        with open(args.summary_json, "w", encoding="utf-8") as f_json:
            json.dump(summary, f_json, indent=2, ensure_ascii=False)
        print(f"Wrote summary JSON to: {args.summary_json}")


if __name__ == "__main__":
    main()

