import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, load_metric
import joblib
import mlflow

# 1. Load data
df = pd.read_csv('data/final_dataset.csv')
df = df.dropna(subset=['Document', 'Topic_group'])

# 2. Encode labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['Topic_group'])
num_labels = len(label_encoder.classes_)

# 3. Train/test split
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

# 4. Tokenization
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-multilingual-cased')

def preprocess(batch):
    return tokenizer(batch['Document'], truncation=True, padding='max_length', max_length=128)

train_dataset = Dataset.from_pandas(train_df[['Document', 'label']])
test_dataset = Dataset.from_pandas(test_df[['Document', 'label']])
train_dataset = train_dataset.map(preprocess, batched=True)
test_dataset = test_dataset.map(preprocess, batched=True)

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# 5. Model
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-multilingual-cased', num_labels=num_labels
)

# 6. Metrics
metric = load_metric("f1")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": (preds == labels).mean(),
        "f1": metric.compute(predictions=preds, references=labels, average='weighted')["f1"],
    }

# 7. Training arguments
training_args = TrainingArguments(
    output_dir="src/transformer/checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="src/transformer/logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

# 8. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# 9. Train
mlflow.set_experiment("distilbert_multilingual")
with mlflow.start_run():
    trainer.train()
    eval_results = trainer.evaluate()
    mlflow.log_metric("accuracy", eval_results["eval_accuracy"])
    mlflow.log_metric("f1_score", eval_results["eval_f1"])
    model.save_pretrained("src/transformer/model")
    tokenizer.save_pretrained("src/transformer/model")
    joblib.dump(label_encoder, "src/transformer/model/label_encoder.joblib")
    mlflow.log_artifact("src/transformer/model")

print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
print(f"F1 Score: {eval_results['eval_f1']:.4f}")