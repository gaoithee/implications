import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# ----------------------------
# 1. Modello e tokenizer
# ----------------------------
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
lm_model = GPT2LMHeadModel.from_pretrained("gpt2")
lm_model.eval()

# ----------------------------
# 2. Esempi dummy con tutti i facts necessari
# ----------------------------
examples = [
    ("a is true, b is false, c is true, d is false; a and b implies c or d; answer:", True),
    ("x is false, y is true, z is true; x or y implies z; answer:", True),
    ("p is true, q is true, r is true; p and q implies r; answer:", True),

    ("m is true, n is false, o is false; m and n implies o; answer:", False),
    ("s is true, t is true, u is false; s or t implies u; answer:", False),
    ("v is false, w is false, x is true; v and w implies x; answer:", False)
]

true_id = tokenizer.convert_tokens_to_ids("True")
false_id = tokenizer.convert_tokens_to_ids("False")

# ----------------------------
# 3. Funzione logit lens normalizzata True/False
# ----------------------------
def compute_logit_lens_normalized(hidden_states, token_position):
    probs_per_layer = []
    for layer_idx, h in enumerate(hidden_states):
        logits = lm_model.lm_head(h)[0, token_position, :]
        probs = torch.softmax(logits, dim=-1)

        true_prob = probs[true_id].detach()
        false_prob = probs[false_id].detach()
        denom = true_prob + false_prob
        if denom > 0:
            true_norm = true_prob / denom
            false_norm = false_prob / denom
        else:
            true_norm = false_norm = 0.5
        probs_per_layer.append((float(true_norm), float(false_norm)))
    return probs_per_layer

# ----------------------------
# 4. Loop sugli esempi
# ----------------------------
for idx, (text, gold) in enumerate(examples):
    inputs = tokenizer(text, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    token_pos = len(tokens) - 1  # ultimo token 'answer:'

    with torch.no_grad():
        outputs = lm_model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        # Predizione GPT-2 sull'ultimo token
        logits_last = lm_model.lm_head(hidden_states[-1])[0, token_pos, :]
        predicted_id = torch.argmax(logits_last).item()
        model_answer = tokenizer.convert_ids_to_tokens(predicted_id)

    probs_per_layer = compute_logit_lens_normalized(hidden_states, token_pos)

    # ----------------------------
    # Stampa risultati
    # ----------------------------
    print(f"\nExample {idx} - Gold answer: {gold}")
    print(f"Text: {text}")
    print(f"GPT-2 predicted answer: {model_answer}\n")
    for layer_idx, (p_true, p_false) in enumerate(probs_per_layer):
        print(f"Layer {layer_idx}: Prob True={p_true:.4f}, Prob False={p_false:.4f}")
