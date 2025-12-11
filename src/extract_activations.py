import torch
from transformers import GPT2Tokenizer, GPT2Model
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm  # <--- import tqdm

# ----------------------------
# 1. Carica dataset
# ----------------------------
dataset = load_dataset("saracandu/implications")
train_df = dataset['train'].to_pandas()
test_df = dataset['test'].to_pandas()
df = pd.concat([train_df, test_df], ignore_index=True)

# ----------------------------
# 2. Tokenizer e modello
# ----------------------------
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")
model.eval()

# ----------------------------
# 3. DataFrame per attivazioni
# ----------------------------
activation_records = []

# ----------------------------
# 4. Estrazione token-level hidden states con tqdm
# ----------------------------
for idx, row in enumerate(tqdm(df.itertuples(), total=len(df), desc="Processing examples")):
    # Combina facts + formula + solo 'answer:' come testo target
    text = f"{row.facts}; {row.formula}; answer:"
    gold = row.gold_formula

    inputs = tokenizer(text, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states  # tuple di (layer+1) tensor [1, seq_len, hidden_dim]

    # Debug: stampa alcune informazioni ogni 1000 esempi
    if idx % 1000 == 0:
        print(f"[DEBUG] Processing example {idx}")
        print(f"Text: {text[:80]}...")  # primi 80 caratteri
        print(f"Number of tokens: {len(tokens)}, first 5 tokens: {tokens[:5]}")
        print(f"Number of layers: {len(hidden_states)}, hidden_dim: {hidden_states[0].shape[-1]}")

    for layer_idx, layer_h in enumerate(hidden_states):
        for token_idx, token in enumerate(tokens):
            activation_vector = layer_h[0, token_idx, :].numpy()
            activation_records.append({
                'token': token,
                'layer': layer_idx,
                'activation': activation_vector,
                'gold_formula': gold
            })

# ----------------------------
# 5. Salva DataFrame
# ----------------------------
activations_df = pd.DataFrame(activation_records)
activations_df.to_pickle("implications_activations_token_level.pkl")
print(f"Saved {len(activations_df)} token-level activations to 'implications_activations_token_level.pkl'")
