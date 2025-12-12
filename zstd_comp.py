import os
import struct
import heapq
import collections
import subprocess
import re
import sys

# --- 1. SETTINGS ---
TEXT_FILE = "text8-128kB.txt"
RANKS_FILE = "ranks.txt"
ANS_SCRIPT = "ans.py"

# --- 2. HELPER FUNCTIONS ---
def get_file_size(filename):
    if os.path.exists(filename):
        return os.path.getsize(filename)
    return 0

def first_n_words_slice(text, n_words=5):
    """Extracts the first 5 words to use as a seed."""
    matches = list(re.finditer(r"\S+", text))
    if not matches: return ""
    end_idx = matches[min(n_words, len(matches))-1].end()
    return text[:end_idx]

# --- 3. HUFFMAN LOGIC ---
class HuffmanNode:
    def __init__(self, freq, symbol=None, left=None, right=None):
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_codebook(freqs):
    heap = []
    for rank, f in freqs.items():
        heapq.heappush(heap, HuffmanNode(f, symbol=rank))
    if len(heap) == 1:
        return {heap[0].symbol: "0"}
    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        heapq.heappush(heap, HuffmanNode(a.freq + b.freq, left=a, right=b))
    
    codebook = {}
    def walk(node, path):
        if node.symbol is not None:
            codebook[node.symbol] = path
            return
        walk(node.left, path + "0")
        walk(node.right, path + "1")
    walk(heap[0], "")
    return codebook

def save_huffman(seed_text, ranks, output_file):
    freqs = collections.Counter(ranks)
    codebook = build_huffman_codebook(freqs)
    
    bitstring = "".join(codebook[r] for r in ranks)
    padding = (8 - len(bitstring) % 8) % 8
    bitstring += "0" * padding
    data_bytes = bytes(int(bitstring[i:i+8], 2) for i in range(0, len(bitstring), 8))

    with open(output_file, "wb") as f:
        f.write(struct.pack(">I", len(seed_text)))
        f.write(seed_text.encode("utf-8"))
        f.write(struct.pack(">I", len(codebook)))
        for rank, bits in codebook.items():
            b_len = len(bits)
            f.write(struct.pack(">I", rank))
            f.write(struct.pack("B", b_len))
            # Pad bits to byte
            padded = bits + "0" * ((8 - b_len % 8) % 8)
            f.write(bytes(int(padded[i:i+8], 2) for i in range(0, len(padded), 8)))
        f.write(struct.pack("B", padding))
        f.write(data_bytes)

# --- 4. ARITHMETIC LOGIC ---
class ArithmeticEncoder:
    def __init__(self):
        self.low = 0
        self.high = 0xFFFFFFFF # 32-bit
        self.follow = 0
        self.bits = []
        self.HALF = 0x80000000
        self.QTR  = 0x40000000
        self.TQTR = 0xC0000000

    def write_bit(self, bit):
        self.bits.append(bit)
        while self.follow > 0:
            self.bits.append(1 - bit)
            self.follow -= 1

    def encode(self, idx, cum_freq, total):
        r = self.high - self.low + 1
        self.high = self.low + (r * cum_freq[idx+1]) // total - 1
        self.low  = self.low + (r * cum_freq[idx])   // total
        
        while True:
            if self.high < self.HALF:
                self.write_bit(0)
            elif self.low >= self.HALF:
                self.write_bit(1)
                self.low -= self.HALF
                self.high -= self.HALF
            elif self.low >= self.QTR and self.high < self.TQTR:
                self.follow += 1
                self.low -= self.QTR
                self.high -= self.QTR
            else:
                break
            self.low = (self.low << 1) & 0xFFFFFFFF
            self.high = ((self.high << 1) + 1) & 0xFFFFFFFF

    def finish(self):
        self.follow += 1
        if self.low < self.QTR: self.write_bit(0)
        else: self.write_bit(1)
        return self.bits

def save_arithmetic(seed_text, ranks, output_file):
    freqs = collections.Counter(ranks)
    symbols = sorted(freqs.keys())
    cum_freq = [0]
    for s in symbols: cum_freq.append(cum_freq[-1] + freqs[s])
    total = cum_freq[-1]
    sym_map = {s: i for i, s in enumerate(symbols)}

    enc = ArithmeticEncoder()
    for r in ranks:
        enc.encode(sym_map[r], cum_freq, total)
    bits = enc.finish()

    bitstring = "".join(str(b) for b in bits)
    padding = (8 - len(bitstring) % 8) % 8
    bitstring += "0" * padding
    data_bytes = bytes(int(bitstring[i:i+8], 2) for i in range(0, len(bitstring), 8))

    with open(output_file, "wb") as f:
        f.write(struct.pack(">I", len(seed_text)))
        f.write(seed_text.encode("utf-8"))
        f.write(struct.pack(">I", len(symbols)))
        for s in symbols:
            f.write(struct.pack(">II", s, freqs[s]))
        f.write(struct.pack("B", padding))
        f.write(data_bytes)

# --- 5. MAIN EXECUTION ---
if __name__ == "__main__":
    print(f"Reading {TEXT_FILE} and {RANKS_FILE}...")
    
    # Read Seed
    with open(TEXT_FILE, "r", encoding="utf-8") as f:
        text = f.read().strip()
    seed = first_n_words_slice(text, 5)

    # Read Ranks
    with open(RANKS_FILE, "r") as f:
        ranks = [int(line.strip()) for line in f if line.strip()]

    print(f"Loaded {len(ranks)} ranks.")

    # 1. Run Huffman
    print("Running Huffman...")
    save_huffman(seed, ranks, "output.huff")

    # 2. Run Arithmetic
    print("Running Arithmetic...")
    save_arithmetic(seed, ranks, "output.arith")

    # 3. Run ANS (Using your ans.py script)
    print("Running ANS...")
    # ans.py needs a file with Seed on line 1, ranks on line 2+
    with open("temp_ranks_for_ans.txt", "w", encoding="utf-8") as f:
        f.write(seed + "\n")
        for r in ranks: f.write(str(r) + "\n")
    
    # Call ans.py
    subprocess.run([sys.executable, ANS_SCRIPT, "compress", "temp_ranks_for_ans.txt", "output.ans"])

    # 4. Run Zstandard
    try:
        import zstandard as zstd
        cctx = zstd.ZstdCompressor(level=3)
        with open(TEXT_FILE, "rb") as f:
            zdata = cctx.compress(f.read())
        with open("output.zst", "wb") as f:
            f.write(zdata)
        zstd_size = get_file_size("output.zst")
    except ImportError:
        zstd_size = "Not Installed"

    # 5. Print Table
    orig_size = get_file_size(TEXT_FILE)
    print("\n" + "="*40)
    print(f"{'METHOD':<15} | {'SIZE (BYTES)':<15}")
    print("="*40)
    print(f"{'Original':<15} | {orig_size:<15}")
    print(f"{'Zstandard':<15} | {zstd_size:<15}")
    print(f"{'Huffman':<15} | {get_file_size('output.huff'):<15}")
    print(f"{'Arithmetic':<15} | {get_file_size('output.arith'):<15}")
    print(f"{'ANS':<15} | {get_file_size('output.ans'):<15}")
    print("="*40)
    
    # Cleanup
    if os.path.exists("temp_ranks_for_ans.txt"): os.remove("temp_ranks_for_ans.txt")