import sys
import struct
import json
import collections
import argparse

class StreamingrANS:
    """
    A Streaming rANS encoder/decoder.
    """
    def __init__(self, frequencies):
        self.L = 1 << 16  # Renormalization lower bound
        self.M_scale = sum(frequencies.values())
        self.freqs = frequencies
        
        # Build CDF
        self.cum_freq = {}
        self.symbol_map = {} 
        current = 0
        # Sort keys to ensure deterministic behavior across machines
        for sym in sorted(frequencies.keys()):
            f = frequencies[sym]
            self.cum_freq[sym] = current
            # Map range to symbol for decoding
            for i in range(current, current + f):
                self.symbol_map[i] = sym
            current += f

    def encode(self, symbols):
        state = self.L 
        bit_stream = [] 

        # Encode in reverse (LIFO)
        for sym in reversed(symbols):
            freq = self.freqs[sym]
            start = self.cum_freq[sym]

            # Renormalize to stay within bounds
            while state >= (self.L * 4): 
                bit_stream.append(state & 0xFF) 
                state >>= 8                     

            # Update state
            state = (state // freq) * self.M_scale + start + (state % freq)

        return state, bytearray(reversed(bit_stream))

    def decode(self, initial_state, stream, num_symbols):
        state = initial_state
        stream_iter = iter(stream)
        decoded = []
        
        for _ in range(num_symbols):
            slot = state % self.M_scale
            sym = self.symbol_map[slot]
            
            freq = self.freqs[sym]
            start = self.cum_freq[sym]
            
            # Decode state
            state = freq * (state // self.M_scale) + slot - start
            decoded.append(sym)
            
            # Renormalize (pull bytes from stream)
            while state < self.L:
                try:
                    val = next(stream_iter) 
                    state = (state << 8) | val
                except StopIteration:
                    break
                    
        return decoded

# --- File I/O Helpers ---

def save_compressed(filepath, seed_text, state, bit_stream, frequencies, num_symbols):
    """Saves compressed binary with metadata header."""
    metadata = {
        "seed_text": seed_text,
        "final_state": state,
        "num_symbols": num_symbols,
        "frequencies": frequencies
    }
    # Serialize metadata
    meta_json = json.dumps(metadata).encode('utf-8')
    
    with open(filepath, "wb") as f:
        f.write(struct.pack("I", len(meta_json))) # 4 bytes for header length
        f.write(meta_json)                        # Metadata
        f.write(bit_stream)                       # Compressed Ranks

def load_compressed(filepath):
    """Loads compressed binary and metadata."""
    with open(filepath, "rb") as f:
        meta_len = struct.unpack("I", f.read(4))[0]
        meta_json = f.read(meta_len)
        bit_stream = f.read()
        
    metadata = json.loads(meta_json)
    # JSON keys are strings, convert back to int for the rANS logic
    freqs = {int(k): v for k, v in metadata["frequencies"].items()}
    
    return metadata["seed_text"], metadata["final_state"], bit_stream, freqs, metadata["num_symbols"]

# --- Main Logic ---

def main():
    parser = argparse.ArgumentParser(description="rANS Compressor for Integer Ranks")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Compress Command
    # Input: Ranks Text File -> Output: .ans binary file
    comp = subparsers.add_parser("compress", help="Compress a file containing ranks.")
    comp.add_argument("input_ranks_file", help="File with Seed (line 1) and Ranks (lines 2+)")
    comp.add_argument("output_ans_file", help="The binary output file.")

    # Decompress Command
    # Input: .ans binary file -> Output: Ranks Text File
    decomp = subparsers.add_parser("decompress", help="Decompress .ans file back to text format.")
    decomp.add_argument("input_ans_file", help="The .ans binary file.")
    decomp.add_argument("output_ranks_file", help="The output text file (compatible with decompressor.py)")

    args = parser.parse_args()

    if args.command == "compress":
        print(f"Reading ranks from {args.input_ranks_file}...")
        
        # 1. READ RANKS DIRECTLY FROM FILE
        # Assumes format: 
        # Line 1: Seed text
        # Lines 2+: Integer ranks
        with open(args.input_ranks_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if not lines:
                print("Error: Input file is empty.")
                sys.exit(1)
            
            seed_text = lines[0].strip()
            # Convert remaining lines to integers, skipping empty lines
            ranks = []
            for line in lines[1:]:
                if line.strip():
                    try:
                        ranks.append(int(line.strip()))
                    except ValueError:
                        print(f"Warning: Skipping non-integer line: '{line.strip()}'")

        print(f"Loaded {len(ranks)} ranks. Compressing...")

        # 2. COMPRESS
        freqs = collections.Counter(ranks)
        rans = StreamingrANS(freqs)
        final_state, stream = rans.encode(ranks)
        
        # 3. SAVE
        save_compressed(args.output_ans_file, seed_text, final_state, stream, freqs, len(ranks))
        print(f"Success! Compressed to {args.output_ans_file} ({len(stream)} bytes stream).")

    elif args.command == "decompress":
        print(f"Decompressing {args.input_ans_file}...")
        
        # 1. LOAD
        seed_text, state, stream, freqs, num_syms = load_compressed(args.input_ans_file)
        
        # 2. DECODE
        rans = StreamingrANS(freqs)
        decoded_ranks = rans.decode(state, list(stream), num_syms)
        
        # 3. WRITE formatted output
        with open(args.output_ranks_file, 'w', encoding='utf-8') as f:
            f.write(seed_text + "\n")
            for r in decoded_ranks:
                f.write(f"{r}\n")
                
        print(f"Success! Ranks restored to {args.output_ranks_file}.")

if __name__ == "__main__":
    main()