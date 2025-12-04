import os

def compute_bitrate_ranks(ranks_file, original_file):
    compressed_bits = os.path.getsize(ranks_file) * 8
    original_bytes = os.path.getsize(original_file)
    return compressed_bits / original_bytes

print(compute_bitrate_ranks("data/huffmanEnc-512kB", "text8-512kB.txt"))
