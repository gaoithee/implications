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

### Examples

```csv
facts,formula,num_operators,gold_formula
"a is true, b is false, c is true","a and b or c implies d",5,True
"p is false, q is true","p or q implies r",2,False

