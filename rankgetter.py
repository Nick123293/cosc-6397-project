import re
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

# ---------------------- Config ----------------------
MODEL_ID = "meta-llama/Llama-3.2-1B"   # or "meta-llama/Llama-3.2-3B"
SEED_WORDS = 5                          # "first few words" to prime the model
DEVICE_MAP = "auto"                     # "cuda", "cpu", or "auto"
DTYPE = torch.bfloat16                  # float16 on GPU is fine too

# ---------------------- Load ------------------------
print(f"Loading model: {MODEL_ID}...")
try:
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map=DEVICE_MAP, torch_dtype=DTYPE
    ).eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"\nCRITICAL ERROR: Could not load model. Error details: {e}")
    print("If you see 'No module named transformers', run: pip install transformers torch")
    sys.exit(1)

def read_text_file(path: str) -> str:
    """Read the full content of a text file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def first_n_words_slice(text: str, n_words: int) -> str:
    """
    Return the substring of 'text' containing exactly the first n words (regex \S+).
    Preserves the original spacing/punctuation up to that point.
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
    # Rank = 1 + number of tokens with strictly higher prob
    higher = torch.sum(prob_vector > true_p).item()
    return int(higher) + 1

def run_sequence_eval(text: str, seed_words: int = 5, topk_show: int = 0) -> Dict[str, Any]:
    """
    Slides through the sequence:
      - starts with first 'seed_words' words,
      - at each step, gets next-token distribution,
      - records rank of the true next token,
      - appends the true token (teacher forcing),
      - continues to the end.
    Returns a dict with ranks, predictions, etc.
    """
    # Prepare full tokenization once
    full = tok(text, return_tensors="pt")
    full_ids = full["input_ids"].to(model.device)  # [1, T]
    T = full_ids.shape[1]

    # Build seed prefix by words (preserving original spacing)
    seed_text = first_n_words_slice(text, seed_words)
    if not seed_text.strip():
        raise ValueError("Seed text is empty; increase SEED_WORDS or provide non-empty text.")

    seed = tok(seed_text, return_tensors="pt").to(model.device)
    seed_ids = seed["input_ids"]                # [1, L0]
    L0 = seed_ids.shape[1]

    if L0 >= T:
        raise ValueError("Seed covers the entire text; decrease SEED_WORDS.")

    results = {
        "seed_text": seed_text,
        "seed_len_tokens": int(L0),
        "total_len_tokens": int(T),
        "steps": []  # each step: {pos, true_id, true_tok, rank, topk: [(tok, p), ...], pred_id, pred_tok, pred_p}
    }

    # Warm-up with the seed (use cache)
    with torch.inference_mode():
        out = model(input_ids=seed_ids, use_cache=True)
        past = out.past_key_values

    # Iterate over the remaining positions: target positions are L0..T-1
    # At each step, the model should predict token at 'pos' given all previous (teacher forcing).
    prev_id = seed_ids[0, -1]
    for pos in range(L0, T):
        true_id = full_ids[0, pos].item()

        with torch.inference_mode():
            # Feed only the last true token, using cache for efficiency
            out = model(
                input_ids=full_ids[:, pos-1:pos],  # the previous true token
                past_key_values=past,
                use_cache=True
            )
            past = out.past_key_values
            logits_last = out.logits[:, -1, :]       # [1, V]
            probs = torch.softmax(logits_last, dim=-1)[0]  # [V]

        # Rank of the true token
        rk = token_rank(probs, true_id)

        # Greedy predicted token (argmax) + info
        pred_id = int(torch.argmax(probs).item())
        pred_tok = tok.decode([pred_id])
        pred_p = float(probs[pred_id].item())

        step_info = {
            "pos": pos,                              # index in token space
            "true_id": true_id,
            "true_tok": tok.decode([true_id]),
            "rank_of_true": rk,
            "pred_id": pred_id,
            "pred_tok": pred_tok,
            "pred_p": pred_p
        }

        # Optional: show top-k list for inspection
        if topk_show and topk_show > 0:
            top_p, top_i = torch.topk(probs, k=min(topk_show, probs.numel()))
            step_info["topk"] = [
                (tok.decode([int(i.item())]), float(p.item()))
                for p, i in zip(top_p, top_i)
            ]

        results["steps"].append(step_info)
        prev_id = true_id  # teacher forcing (we already used true token via full_ids and cache)

    # Summary stats
    ranks = [s["rank_of_true"] for s in results["steps"]]
    results["num_evaluated_tokens"] = len(ranks)
    results["mean_rank"] = float(sum(ranks) / len(ranks))
    results["top1_accuracy"] = float(sum(1 for r in ranks if r == 1) / len(ranks))
    # Mean Reciprocal Rank (MRR)
    results["mrr"] = float(sum(1.0 / r for r in ranks) / len(ranks))

    return results

# ---------------------- Example ----------------------
if __name__ == "__main__":
    file_path = input("Enter the path to your text file: ").strip()
    text = read_text_file(file_path)

    print(f"\nLoaded text ({len(text)} characters):\n{text[:200]}...\n")

    out = run_sequence_eval(text, seed_words=SEED_WORDS, topk_show=5)

    print(f"Seed text (first {SEED_WORDS} words): {out['seed_text']!r}")
    print(f"Evaluated tokens: {out['num_evaluated_tokens']}")
    print(f"Top-1 accuracy: {out['top1_accuracy']:.3f}  |  MRR: {out['mrr']:.3f}  |  Mean rank: {out['mean_rank']:.2f}\n")

    # Show first few steps for sanity
    # for s in out["steps"]:
    #     # Clean display of tokens (SentencePiece often uses leading spaces)
    #     t_true = s["true_tok"].replace("▁", " ")
    #     t_pred = s["pred_tok"].replace("▁", " ")
    #     print(f"[pos {s['pos']:>3}] true={t_true!r:>10}  rank={s['rank_of_true']:>4}  "
    #           f"pred={t_pred!r:<10}  p={s['pred_p']:.4f}")
    #     if "topk" in s:
    #         top_show = ", ".join([f"{tok.replace('▁',' ').strip()!r}:{p:.3f}" for tok, p in s["topk"]])
    #         print("        topk:", top_show)
    ranks_array = [s["rank_of_true"] for s in out["steps"]]

    print("\n--- SUMMARY OUTPUT ---")
    print("Starting input text:")
    print(repr(out["seed_text"]))
    print("\nRanks array (in order):")
    print(len(ranks_array))
