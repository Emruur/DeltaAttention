# Transformer Prefill vs. Decoding Phases

The lifecycle of a Transformer inference request is split into two distinct mechanical phases: **Prefill** (often called the profile phase) and **Decoding**.

---

## 1. The Prefill Phase (The "Understanding" Step)
* **Goal:** Process the entire user prompt simultaneously to compute the initial mathematical representations.
* **Parallelism:** This phase is highly parallel. Because all prompt tokens are available at once, the GPU can utilize its thousands of cores to compute them in one "giant" matrix operation.
* **The KV Cache:** During this phase, the model generates the **Key (K)** and **Value (V)** tensors for every token in the prompt. These are stored in the **KV Cache** so they never have to be re-calculated.
* **Constraint:** This phase is **Compute-bound** (limited by the GPU's raw TFLOPS).

---

## 2. The Decoding Phase (The "Generation" Step)
* **Goal:** Predict the next token, one at a time, based on the context.
* **Recurrence:** This is an **autoregressive** process. To generate token $N$, you must have finished token $N-1$. This creates a sequential bottleneck.
* **Causal Logic:** Because Transformers use **Causal Masking**, tokens only look at the past. Since the past doesn't change, we can "freeze" the KV Cache and only calculate the math for the single newest token.
* **Constraint:** This phase is **Memory-bandwidth bound**. The GPU spends most of its time moving massive weight matrices from VRAM to the processor just to perform a small calculation on a single token.

---

## 3. The MLP (Multi-Layer Perceptron) Component
* **Independence:** The MLP operates on each token vector in isolation. It does not "look" at other tokens.
* **No Caching Needed:** Unlike the Attention mechanism, the MLP has no temporal dependency. It doesn't need a cache because it doesn't need to know the MLP outputs of previous tokens to compute the current one.
* **The Heavy Lifter:** The MLP typically contains the majority of the model's parameters. During decoding, loading these parameters for just one token is the primary reason generation can feel slow compared to the initial "snap" of the prefill.
