# Proof of Concept: Logical Reasoning with LLMs

This repository explores **what happens inside LLMs when inferring the truth value of a logical formula**.  
Inspired by [this paper](https://arxiv.org/pdf/2411.04105v4), we created a synthetic dataset for testing logical reasoning and probing LLMs.

---

## Dataset: Implications

A synthetic dataset of **logical formulas with Boolean variables** and operators (`and`, `or`, `implies`).  
Each example contains:

- `facts`: truth values assigned to variables (e.g., `a is true, b is false`)  
- `formula`: linear logical formula without parentheses (e.g., `a and b implies c or d`)  
- `num_operators`: number of logical operators  
- `gold_formula`: overall truth value (`True` or `False`)  

Duplicates `(facts, formula)` are removed to ensure uniqueness.  

---

## Models: currently testing


| Model                     | Parameters | Layers | Hidden Units | Attention Heads | Embedding Size | Feedforward Dim | Activation | LayerNorm Type | Attention Type | Dropout | Positional Encoding |
|----------------------------|------------|--------|--------------|----------------|----------------|----------------|------------|----------------|----------------|---------|-------------------|
| **gpt2**                   | 117M       | 12     | 768          | 12             | 768            | 3072           | GELU       | PostNorm       | Scaled Dot-Product | 0.1     | Learned            |
| **gpt2-medium**            | 345M       | 24     | 1024         | 16             | 1024           | 4096           | GELU       | PostNorm       | Scaled Dot-Product | 0.1     | Learned            |
| **EleutherAI/gpt-neo-2.7B**| 2.7B       | 32     | 2560         | 32             | 2560           | 10240          | GELU       | PreNorm        | Scaled Dot-Product | 0.1     | Sinusoidal         |
| **EleutherAI/pythia-70m**  | 70M        | 12     | 512          | 8              | 512            | 2048           | GELU       | PreNorm        | Scaled Dot-Product | 0.1     | Sinusoidal         |
| **EleutherAI/pythia-2.8b** | 2.8B       | 32     | 2560         | 32             | 2560           | 10240          | GELU       | PreNorm        | Scaled Dot-Product | 0.1     | Sinusoidal         |
| **Qwen/Qwen3-4B**          | 4B         | 32     | 2560         | 32             | 2560           | 10240          | GELU       | PreNorm        | Scaled Dot-Product | 0.1     | Sinusoidal         |
| **Qwen/Qwen2.5-Math-1.5B** | 1.5B       | 24     | 2048         | 32             | 2048           | 8192           | GELU       | PreNorm        | Scaled Dot-Product | 0.1     | Sinusoidal         |
| **microsoft/Phi-4-mini-instruct** | 1.3B? | 24     | 2048         | 16-32          | 2048           | 8192           | GELU       | PreNorm        | Scaled Dot-Product | 0.1     | Sinusoidal         |
| **google/gemma-3-4b**      | 4B         | 32     | 2560         | 32             | 2560           | 10240          | GELU       | PreNorm        | Scaled Dot-Product | 0.1     | Sinusoidal         |
