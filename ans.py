from collections import Counter
import json
# Import the class from your file
from ans import rANS 

# 1. Simulate output from rankgetter.py
# In reality, you would get this list from rankgetter's "ranks_array"
ranks_from_llm = [1, 1, 2, 1, 5, 1, 1, 3, 10, 1] 

# 2. Build the Frequency Table (The Missing Link)
# We must count how often each rank appears to build the probability model
freqs = Counter(ranks_from_llm)

# 3. Initialize rANS with integer keys
rans = rANS(freqs)

# 4. Encode
# Note: We pass the list of integers, not a string
encoded_state = rans.encode(ranks_from_llm)

print(f"Compressed State (Integer): {encoded_state}")

# ---------------------------------------------------------
# To send this to decompressor.py, you need to save:
# A. The Encoded State
# B. The Frequency Table (Decoder needs this to reconstruct probabilities)
# C. The Length of the message
# ---------------------------------------------------------

# 5. Decode (What needs to happen before decompressor.py runs)
decoded_ranks = rans.decode(encoded_state, len(ranks_from_llm))

print(f"Decoded Ranks: {decoded_ranks}")

# Verify exact match
assert ranks_from_llm == list(decoded_ranks)
print("Success: rANS encoded and decoded the LLM ranks.")