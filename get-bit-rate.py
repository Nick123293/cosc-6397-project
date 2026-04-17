import os

def compute_bitrate_ranks(ranks_file, original_file):
    compressed_bits = os.path.getsize(ranks_file) * 8
    original_bytes = os.path.getsize(original_file)
    return compressed_bits / original_bytes

print(compute_bitrate_ranks("data/gpt2-ans-128.bin", "text8-128kB.txt"))
