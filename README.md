# cosc-6397-project
information theory and data compression project

To-Do:  
1) Test on smaller models
2) Get baseline compression ratios and throughputs of huffman, arithmetic, and ANS (will there be a baseline for arithmetic and ANS since it is text data)?
3) Test with differing number of seed words, and a sliding context window while still keeping KV cache
4) Implement non entropy based encoders to see if we can increase compression ratio (maybe use multiple encoders, since we do not need to worry about runtime)
5) Since throughput is very poor, we need very low bitrate to make this work
6) Address the issue of having to encode and decode on the same architecture
7) Explore using the float8 datatype to increase throughput
Compression ratio with huffman encoding on a 128kB file is 5.913
