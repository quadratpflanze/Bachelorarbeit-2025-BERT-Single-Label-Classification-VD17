from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from collections import defaultdict

# 1. Gerät
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Test-Set laden (CSV mit Spalten "text","label")
test_ds = load_dataset(
    "csv",
    data_files={"test": "D:\\Wolf\\Testdaten1651-1700-gekuerzt1306.csv"},
    column_names=["text", "label"],
    header=None
)["test"]

# 3. Modell & Tokenizer laden
model_path = "2BAbert5e-5-2ep"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
model.eval()

# 4. Tokenisierung und DataLoader
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_test = test_ds.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer)

#Spalten umbenennen, damit DataCollator das Label als "labels" behält
tokenized_test = tokenized_test.rename_column("label", "labels")
tokenized_test.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
loader = DataLoader(tokenized_test, batch_size=16, collate_fn=data_collator)

# 5. Wahrscheinlichkeiten sammeln
probs_per_label = defaultdict(list)

with torch.no_grad():
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        labels = labels.cpu().numpy()

        # pro Beispiel in diesem Batch
        for p, lbl in zip(probs, labels):
            probs_per_label[lbl].append(p)

# 6. Durchschnitt pro Label berechnen
avg_probs = {}
for lbl, plist in probs_per_label.items():
    avg_probs[lbl] = np.mean(np.stack(plist, axis=0), axis=0)

# 7. Ausgabe
id2label = {0: "Lyrik", 1: "Leichenpredigt", 2: "Flugschrift",
            3: "Streitschrift:theol.", 4: "Verordnung", 5: "Meßkatalog"}

for lbl, probs in avg_probs.items():
    print(f"Label {lbl} ({id2label[lbl]}):")
    for class_id, p in enumerate(probs):
        print(f"  P({id2label[class_id]}) = {p:.4f}")
    print()
