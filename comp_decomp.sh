#!/bin/bash
python3 rankgetter-with-huffman-enc-option.py text8-128kB.txt data/test1
python3 decompressor.py data/test1.ranks.txt decompressedtext8-128kB.txt
