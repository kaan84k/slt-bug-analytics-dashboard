# Fine-tune a transformer for bug/non-bug review classification using HuggingFace Trainer
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

# Load labeled data (use your labeled_examples or expand as needed)
labeled_examples = [
    ("App crashes every time I open it", 1),
    ("Keeps freezing after login", 1),
    ("Very helpful app!", 0),
    ("Great design, but needs dark mode", 0),
    ("Bug: Can't upload images", 1),
    ("Everything works smoothly", 0),
    ("Crash on opening camera feature", 1),
    ("I love the app", 0),
    ("The app won't start after the latest update", 1),
    ("Best app I've ever used", 0),
    ("Excellent features and very intuitive", 0),
    ("Error message appears when trying to save", 1),
    ("App is unstable and keeps closing", 1),
    ("Worth every penny, highly recommend", 0),
    ("Stuck on loading screen", 1),
    ("Could use more customization options", 0),
    ("Perfect for my needs", 0),
    ("The UI is beautiful and responsive", 0),
    ("Can't log in, keeps saying invalid credentials", 1),
    ("Videos won't play, just shows black screen", 1)
]

train_texts, train_labels = zip(*labeled_examples)
train_df = pd.DataFrame({"text": train_texts, "label": train_labels})

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=128)

dataset = Dataset.from_pandas(train_df)
tokenized_dataset = dataset.map(preprocess_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)

training_args = TrainingArguments(
    output_dir="./bug_classifier_model",
    num_train_epochs=10,
    per_device_train_batch_size=4,
    learning_rate=2e-5,
    evaluation_strategy="no",
    save_strategy="epoch",
    logging_steps=10,
    report_to=[],
    seed=42
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=None,
)

trainer.train()

# Save the model
trainer.save_model("./bug_classifier_model")
print("✅ Fine-tuned model saved to ./bug_classifier_model")
