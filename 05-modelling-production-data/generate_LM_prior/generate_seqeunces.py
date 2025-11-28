import csv
import itertools
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Adjektivklassen
D_adj = {'großer'}
F_adj = {'dreieckiger', 'kreisförmiger', 'quadratischer', 'sternförmiger', 'herzförmiger', 'diamantförmiger'}
C_adj = {'schwarzer', 'grüner', 'orangener', 'grauer', 'blauer', 'brauner'}

# Nomen
noun = "Aufkleber"

# Klassen in einem Dictionary sammeln (falls du später mehr ergänzen willst)
classes = {
    "D": list(D_adj),
    "F": list(F_adj),
    "C": list(C_adj),
}

# for each form and farben pair, together with groesse, generate 15 combinations, and then head noun at the end

# Generate the 15 permissible adjective orderings (no repetition, STOP = end)
orderings = [
    ("D",),
    ("F",),
    ("C",),
    ("D","F"),
    ("D","C"),
    ("F","D"),
    ("F","C"),
    ("C","D"),
    ("C","F"),
    ("D","F","C"),
    ("D","C","F"),
    ("F","D","C"),
    ("F","C","D"),
    ("C","D","F"),
    ("C","F","D"),
]

# Collect all sequences in memory first
rows = []
for D in classes["D"]:
    for F in classes["F"]:
        for C in classes["C"]:
            for ordering in orderings:
                # map class keys to adjectives
                realised = [{"D": D, "F": F, "C": C}[slot] for slot in ordering]
                surface = " ".join(realised + [noun])
                order_key = "".join([slot[0].upper() for slot in ordering])
                rows.append({
                    "D_adj": D,
                    "F_adj": F,
                    "C_adj": C,
                    "order_key": order_key,
                    "ordered_string": surface,
                })

# ------------------------------------------------------------------
# GPT-2 surprisal computation using a German GPT-2 model
# ------------------------------------------------------------------

MODEL_NAME = "dbmdz/german-gpt2"  # or "gpt2-german" depending on your setup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

# ensure tokenizer has a padding token (for batch processing if needed)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

import math


def compute_surprisal_bits(text: str) -> float:
    """Return average surprisal in bits per token for a given string."""
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        # outputs.loss is mean cross-entropy (nats) over tokens
        loss_nats = outputs.loss.item()
    # convert nats to bits
    loss_bits = loss_nats / math.log(2.0)
    return loss_bits

# Compute surprisal for each row
for row in rows:
    text = row["ordered_string"]
    bits_per_tok = compute_surprisal_bits(text)
    row["surprisal_bits_per_token"] = bits_per_tok

# Output CSV with surprisal
out_file = "LM_adjective_sequences_with_surprisal.csv"
with open(out_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "D_adj",
        "F_adj",
        "C_adj",
        "order_key",
        "ordered_string",
        "surprisal_bits_per_token",
    ])
    for row in rows:
        writer.writerow([
            row["D_adj"],
            row["F_adj"],
            row["C_adj"],
            row["order_key"],
            row["ordered_string"],
            row["surprisal_bits_per_token"],
        ])

print(f"Wrote {len(rows)} rows with surprisal to {out_file}.")

# Calculate surprisal for all adjective-noun combinations using gpt-2-german

# Aggregate mean surprisal per adjective type pair, in the end, surpress to a distribution over 15 categories, e.g. D, DCF, DFC, FDC, FCD, CDF, CFD... 
