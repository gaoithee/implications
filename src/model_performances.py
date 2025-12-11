import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

# ----------------------------
# 1. Carica dataset
# ----------------------------
dataset = load_dataset("saracandu/implications")
train_df = dataset['train'].to_pandas()
test_df = dataset['test'].to_pandas()
df = pd.concat([train_df, test_df], ignore_index=True)

# ----------------------------
# 2. Lista modelli da testare
# ----------------------------
models_to_test = [
    "gpt2",                 # GPT-2 small
    "gpt2-medium",          # GPT-2 medium
    "EleutherAI/gpt-neo-2.7B",  # GPT-Neo 2.7B
    "EleutherAI/pythia-70m",    # Pythia (esempio piccolo, puoi cambiare)
    "EleutherAI/pythia-2.8b",
    "Qwen/Qwen3-4B",        # Qwen3-4B
    "Qwen/Qwen2.5-Math-1.5B"
    "microsoft/Phi-4-mini-instruct",  # nonostante sia fine-tuned, la versione base è 7B
    "google/gemma-3-4b-pt"
]

device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# 3. Loop su modelli
# ----------------------------
for model_name in models_to_test:
    print(f"\n=== Processing model: {model_name} ===")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device=="cuda" else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    # Prendi gli ID dei token True/False
    true_id = tokenizer.convert_tokens_to_ids("True")
    false_id = tokenizer.convert_tokens_to_ids("False")
    
    prediction_records = []

    # Loop sul dataset
    for idx, row in enumerate(tqdm(df.itertuples(), total=len(df), desc=f"{model_name}")):
        text = f"{row.facts}; {row.formula}; answer:"
        gold = row.gold_formula

        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            logits = model(**inputs).logits

        next_token_logits = logits[0, -1]
        probs = torch.softmax(next_token_logits, dim=-1)

        prob_true = float(probs[true_id]) if true_id is not None else 0.0
        prob_false = float(probs[false_id]) if false_id is not None else 0.0

        predicted = "True" if prob_true >= prob_false else "False"

        prediction_records.append({
            "text": text,
            "gold": gold,
            "predicted": predicted,
            "prob_true": prob_true,
            "prob_false": prob_false
        })

    # Salva CSV
    preds_df = pd.DataFrame(prediction_records)
    file_name = f"predictions_{model_name.replace('/', '_')}.csv"
    preds_df.to_csv(file_name, index=False)
    print(f"Saved predictions to {file_name}")
