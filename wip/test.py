import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import numpy as np

# ----------------------------
# 1. Carica dataset
# ----------------------------
dataset = load_dataset("saracandu/implications")
train_df = dataset['train'].to_pandas()
test_df = dataset['test'].to_pandas()
df = pd.concat([train_df, test_df], ignore_index=True)

# ----------------------------
# 2. Tokenizer e modello Qwen
# ----------------------------
MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

# Per attivazioni dei layer
model = AutoModel.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

# Per generazione
lm_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto",
    trust_remote_code=True
)
lm_model.eval()

# ----------------------------
# 3. Tabelle per output
# ----------------------------
activation_records = []
prediction_records = []

true_id = tokenizer.convert_tokens_to_ids("True")
false_id = tokenizer.convert_tokens_to_ids("False")

# ----------------------------
# 4. Loop principale
# ----------------------------
for idx, row in enumerate(tqdm(df.itertuples(), total=len(df), desc="Processing examples")):

    text = f"{row.facts}; {row.formula}; answer:"
    gold = row.gold_formula

    inputs = tokenizer(text, return_tensors="pt").to(device)
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    # ==========================
    # A) Estrarre hidden states
    # ==========================
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states

    # Debug ogni 1000
    if idx % 1000 == 0:
        print(f"[DEBUG] Example {idx}")
        print(f"Text: {text[:80]}...")
        print(f"Tokens: {tokens[:10]} ... ({len(tokens)} total)")
        print(f"Layers: {len(hidden_states)}, dim = {hidden_states[0].shape[-1]}")

    # Salva attivazioni
    for layer_idx, layer_h in enumerate(hidden_states):
        h = layer_h[0]  # [seq_len, hidden_dim]
        for token_idx, token in enumerate(tokens):
            activation_vector = h[token_idx].float().cpu().numpy()
            activation_records.append({
                'token': token,
                'layer': layer_idx,
                'activation': activation_vector,
                'gold_formula': gold
            })

    # ==========================
    # B) Predizione del modello
    # ==========================
    with torch.no_grad():
        logits = lm_model(**inputs).logits  # [1, seq_len, vocab]
    next_token_logits = logits[0, -1]  # logits dell'ultimo token

    probs = torch.softmax(next_token_logits, dim=-1)

    prob_true = float(probs[true_id])
    prob_false = float(probs[false_id])

    predicted = "True" if prob_true >= prob_false else "False"

    # Salva risultato
    prediction_records.append({
        "text": text,
        "gold": gold,
        "predicted": predicted,
        "prob_true": prob_true,
        "prob_false": prob_false
    })

# ----------------------------
# 5. Salva risultati
# ----------------------------
activations_df = pd.DataFrame(activation_records)
activations_df.to_pickle("implications_activations_qwen.pkl")
print(f"Saved activations: {len(activations_df)} vectors.")

preds_df = pd.DataFrame(prediction_records)
preds_df.to_csv("implications_qwen_predictions.csv", index=False)
print("Saved predictions to implications_qwen_predictions.csv")
