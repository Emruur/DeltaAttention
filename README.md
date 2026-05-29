# DeltaAttention

**[Read the draft paper](https://emruur.github.io/DeltaAttention/)**

A prefill acceleration method for LLMs that compresses the key matrix $K$ along the sequence dimension before the $QK$ multiplication, reducing FLOPs proportionally to the compression ratio. Custom Triton kernels realize the speedup in practice, demonstrating wall-clock acceleration on long contexts.
