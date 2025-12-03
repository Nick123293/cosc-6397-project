class rANS:
    def __init__(self, frequencies):
        self.freqs = frequencies
        self.M = sum(frequencies.values()) # Normalization factor
        
        # Build Cumulative Distribution Function (CDF)
        self.cum_freq = {}
        current = 0
        for char, freq in frequencies.items():
            self.cum_freq[char] = current
            current += freq
            
        # Reverse lookup for decoding (slot -> char)
        self.symbol_map = []
        for char, freq in frequencies.items():
            self.symbol_map.extend([char] * freq)

    def encode(self, message):
        state = 0 # Start with state 0 usually implies an empty buffer, 
                  # but usually initialized to a lower bound in streaming.
                  # For this math demo, we start at a base offset to avoid x=0 issues.
        state = 1 
        
        for char in message:
            fs = self.freqs[char]
            cs = self.cum_freq[char]
            
            # The rANS Encoding Formula
            state = (state // fs) * self.M + cs + (state % fs)
            
        return state

    def decode(self, final_state, length):
        state = final_state
        decoded_message = []
        
        for _ in range(length):
            # 1. Find the symbol based on modulo M
            slot = state % self.M
            char = self.symbol_map[slot]
            
            # 2. Retrieve Frequency and Cumulative Frequency
            fs = self.freqs[char]
            cs = self.cum_freq[char]
            
            # 3. Update State (Inverse of Encoding)
            state = fs * (state // self.M) + slot - cs
            
            decoded_message.append(char)
            
        return "".join(reversed(decoded_message)) # LIFO structure

# --- Usage ---
# Frequencies must be known by both encoder and decoder
freqs = {'A': 3, 'B': 1, 'C': 1} # A is very common
rans = rANS(freqs)

original = "ABACABA"
encoded_state = rans.encode(original)
decoded = rans.decode(encoded_state, len(original))

print(f"Original: {original}")
print(f"State Integer: {encoded_state}")
print(f"Decoded:  {decoded}")