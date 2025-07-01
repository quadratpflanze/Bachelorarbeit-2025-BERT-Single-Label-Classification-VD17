from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoModelForMaskedLM
from transformers import TrainerCallback #neu1306
import evaluate
import numpy as np
from transformers import DataCollatorWithPadding
from collections import defaultdict
from sklearn.utils.class_weight import compute_class_weight #neu2505
import torch #neu2505


# 1. Dataset laden (absolute Pfade durch eigene ersetzen)
dataset_dict = load_dataset("csv", data_files={
    "train": "D:\\Wolf\\Trainingsdaten1600-1650-gekuerzt1306.csv",
    "test": "D:\\Wolf\\Testdaten1651-1700-gekuerzt1306.csv"},
column_names=["text", "label"],
    header=None
)

# 2. 10% des Traingssatzes als Validationssplit festlegen
split = dataset_dict["train"].train_test_split(test_size=0.1, seed=42)
train_ds = split["train"]
valid_ds = split["test"]
# Testset für finale Auswertung
test_ds = dataset_dict["test"]

# 3. Device definieren (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 4. Modell & Tokenizer
model_path = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 5. Label-Mappings
id2label = {0: "Lyrik", 1: "Leichenpredigt", 2: "Flugschrift", 3: "Streitschrift:theol.", 4: "Verordnung", 5: "Meßkatalog"}
label2id = {v: k for k, v in id2label.items()}

#(optional1) Class Weights berechnen (NACH dem Dataset-Laden!)
#labels = dataset_dict["train"]["label"]
#class_weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
#class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# 6. Modell & Tokenizer
model_path = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    num_labels=6,
    id2label=id2label,
    label2id=label2id,
    problem_type="single_label_classification"
).to(device)  # Modell auf Device verschieben

#(optional2) Freezing: Nur letzter Layer + Classifier
#for name, param in model.named_parameters():
#    if not any(n in name for n in ["classifier", "encoder.layer.10", "encoder.layer.11"]):
#        param.requires_grad = False

# 7. Preprocessing
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

# 8. Tokenisierung des Splits
tokenized_train = train_ds.map(preprocess_function, batched=True)
tokenized_valid = valid_ds.map(preprocess_function, batched=True)
tokenized_test = test_ds.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 9. Metriken laden
accuracy = evaluate.load("accuracy")

# 10. compute_metrics-Funktion definieren
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predicted_classes = np.argmax(predictions, axis=-1)
    acc = accuracy.compute(predictions=predicted_classes, references=labels)["accuracy"]
    return {"Accuracy": acc}

# 11. Training (Parameter je nach Testung anpassen)
training_args = TrainingArguments(
    output_dir="bert-genre-classifier",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    #weight_decay=0.01,            #optional udn funktioniert nur mit optional1 zusammen
    eval_strategy="epoch",
    logging_strategy="epoch",      #fügt Training-Loss zum Log hinzu
    save_strategy="epoch",
    load_best_model_at_end=True
)

# 12. Callback definieren
class TrainMetricsCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        # train_dataset ist hier tokenized_train aus 8.
        train_metrics = trainer.evaluate(eval_dataset=tokenized_train)
        valid_metrics = trainer.evaluate(eval_dataset=tokenized_valid)
        print(
            f">>> Epoch {state.epoch:.0f} – "
            f"Train Loss: {train_metrics['eval_loss']:.4f}, "
            f"Train Acc:  {train_metrics['eval_Accuracy']:.4f}"
        )

#13. Trainer definieren
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics  # Siehe Schritt 4
)

# 14. Callback registrieren neu1306
trainer.add_callback(TrainMetricsCallback())

trainer.train()

#15. Modell speichern (kann auskommentiert werden, wenn Modell nicht gespeichert werden soll)
trainer.save_model("4BAbert2e-5-3ep-weightd-5")

#16. Test Metriken ausgeben in der letzten epoche
test_metrics = trainer.evaluate(tokenized_test)
print("Finale Test-Metriken:", test_metrics)

