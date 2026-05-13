### 1. Use Case: Sole Prefill Optimization
**Goal:** Evaluate the method's ability to accelerate the prompt-processing (pre-filling) stage by compressing KV matrices while maintaining long-context understanding.

*   **Selected Benchmark:** **RULER** (Specifically focusing on accuracy across variable sequence lengths from 4K to 128K).
*   **Selected Model:** **Llama-3.1-8B-Instruct (128K Variant)**.
*   **Selected Baselines:** 
    *   **Full Attention:** The dense, uncompressed baseline.
    *   **MInference 1.0:** Dynamic sparse attention utilizing Vertical-Slash, A-shape, and Block-Sparse patterns.
    *   **FlexPrefill:** Adaptive sparse pattern determination (Query-Aware vs. Vertical-Slash).
    *   **XAttention:** Antidiagonal scoring block-sparse attention (Reporting Stride=8 and Stride=16 configurations).

#### RULER Benchmark Accuracy (Llama-3.1-8B-Instruct, 128K Variant)
| Method | 4K | 8K | 16K | 32K | 64K | 128K | **Average** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Full Attention** | 96.74 | 94.03 | 92.02 | 84.17 | 81.32 | 76.89 | **87.52** |
| **FlexPrefill** | 95.99 | 93.67 | 92.73 | 88.14 | 81.14 | 74.67 | **87.72** |
| **MInference 1.0** | 96.54 | 94.06 | 91.37 | 85.79 | 83.03 | 54.12 | **84.15** |
| **XAttention** (S=8) | 96.83 | 94.07 | 93.17 | 90.75 | 84.08 | 72.31 | **88.47** |
| **XAttention** (S=16)| 96.11 | 93.95 | 93.56 | 90.64 | 83.12 | 71.11 | **88.08** |

*(Source: XAttention Reproduction Table)*

***

### 2. Use Case: Sole Decode Optimization (KV Cache Compression)
**Goal:** Evaluate how well the compressed KV matrices reduce memory footprints and maintain text generation quality during the token-by-token decoding phase.

*   **Selected Benchmark:** **LongBench** (A shared subset of 9 summarization and QA tasks to ensure direct comparability).
*   **Selected Model:** **Llama-2-7B-chat (Standard 4K Variant)**.
    *   *Crucial Specification:* Because the base Llama-2-7B-chat has a native 4K context window, the standard THUDM LongBench evaluation script automatically truncates input texts from the middle to fit this 4K limit. Your method must be evaluated on these *truncated 4K snippets*—compressing them down to 2K tokens—to match the baseline numbers exactly.
*   **Selected Baselines:**
    *   **H2O (Heavy-Hitter Oracle):** Evicts tokens dynamically by keeping a balance of recent tokens and high-attention "heavy hitters".
    *   **KVMerger:** Adaptively fuses key-value states together into single token states using a Gaussian kernel weighted merging algorithm.
*   **Cache Budget:** 50%.

#### LongBench Subset Accuracy (Llama-2-7B-chat, 50% KV Cache Budget)
| LongBench Task | KVMerger | H2O |
| :--- | :--- | :--- |
| **GovReport** | 25.31 | 24.48 |
| **MultiNews** | 26.29 | 25.37 |
| **NarrativeQA** | 18.50 | 17.85 |
| **Qasper** | 20.04 | 20.04 |
| **MultifieldQA-en** | 36.89 | 32.17 |
| **TREC** | 64.00 | 63.00 |
| **2WikiMQA** | 32.99 | 30.57 |
| **TriviaQA** | 83.62 | 80.89 |
| **PassageRetrieval-en** | 7.33 | 7.00 |
| **Average** | **35.02** | **33.49** |

*(Source: KVMerger Reproduction Table)*