#!/usr/bin/env python3

"""
Download the GPT-2 model and tokenizer from Hugging Face
and exit without running any inference.

Requirements:
    pip install transformers torch
"""

from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    model_name = "gpt2"

    print(f"Downloading tokenizer for {model_name}...")
    _ = AutoTokenizer.from_pretrained(model_name)

    print(f"Downloading model weights for {model_name}...")
    _ = AutoModelForCausalLM.from_pretrained(model_name)

    print("Download complete. Exiting without using the model.")


if __name__ == "__main__":
    main()
