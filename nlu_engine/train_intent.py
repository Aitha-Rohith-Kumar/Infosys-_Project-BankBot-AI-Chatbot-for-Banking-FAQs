# nlu_engine/train_intent.py
"""
Train intent classifier (HuggingFace Trainer).
This version is robust to different transformers versions.
"""
import os
import json
import argparse
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import transformers

HERE = os.path.dirname(__file__)
INTENTS_PATH = os.path.join(HERE, "intents.json")

def load_intents(path=INTENTS_PATH):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = []
    labels = []
    if isinstance(data, dict) and "intents" in data and isinstance(data["intents"], list):
        for item in data["intents"]:
            name = item.get("name")
            examples = item.get("examples", [])
            for ex in examples:
                texts.append(ex)
                labels.append(name)
    else:
        for name, examples in data.items():
            for ex in examples:
                texts.append(ex)
                labels.append(name)
    return texts, labels

def build_dataset(texts, labels, label_list):
    label2id = {l: i for i, l in enumerate(label_list)}
    data = {"text": texts, "label": [label2id[l] for l in labels]}
    ds = Dataset.from_dict(data)
    return ds, label2id

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

def main(args):
    print("Transformers version:", transformers.__version__)
    texts, labels = load_intents()
    if len(texts) == 0:
        raise ValueError("No training examples found in intents.json")
    label_list = sorted(list(set(labels)))
    print(f"Found {len(label_list)} intents with {len(texts)} examples total.")
    out_dir = args.out_dir

    ds, label2id = build_dataset(texts, labels, label_list)

    if args.test_frac and len(texts) > 20:
        ds = ds.train_test_split(test_size=args.test_frac, seed=42)
        train_ds = ds["train"]
        test_ds = ds["test"]
    else:
        print("Tiny dataset or test_frac==0 -> using full dataset as train (no separate test split).")
        train_ds = ds
        test_ds = ds

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize_fn(example):
        text = example.get("text", "") if isinstance(example, dict) else example
        if not isinstance(text, str):
            text = str(text)
        return tokenizer(text, truncation=True, padding="max_length", max_length=args.max_len)

    train_ds = train_ds.map(lambda ex: tokenize_fn(ex), batched=False)
    test_ds  = test_ds.map(lambda ex: tokenize_fn(ex), batched=False)

    train_ds = train_ds.rename_column("label", "labels")
    test_ds  = test_ds.rename_column("label", "labels")

    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=len(label_list))

    # Build TrainingArguments with graceful fallback for older/newer transformers
    try:
        training_args = TrainingArguments(
            output_dir=out_dir,
            evaluation_strategy="epoch" if (len(texts) > 20 and args.test_frac>0) else "no",
            save_strategy="epoch",
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.lr,
            logging_steps=10,
            weight_decay=0.01,
            save_total_limit=2,
            fp16=False,
            remove_unused_columns=False
        )
    except TypeError:
        print("Warning: TrainingArguments did not accept some parameters. Falling back to a simpler signature.")
        training_args = TrainingArguments(
            output_dir=out_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.lr,
            logging_steps=10,
            weight_decay=0.01,
            do_train=True,
            do_eval=False if args.test_frac==0 else True
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds if (len(texts) > 20 and args.test_frac>0) else None,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics if (len(texts) > 20 and args.test_frac>0) else None
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to {out_dir} ...")
    # Save model + tokenizer (may write safetensors)
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    labels_json = os.path.join(out_dir, "labels.json")
    with open(labels_json, "w", encoding="utf-8") as f:
        json.dump(label_list, f, indent=2)

    # Safe check for evaluation_strategy using getattr
    eval_strategy = getattr(trainer.args, "evaluation_strategy", None)
    if eval_strategy is not None and trainer.eval_dataset is not None:
        try:
            eval_res = trainer.evaluate()
            print("Evaluation:", eval_res)
        except Exception as e:
            print("Warning: evaluation failed:", e)
    else:
        print("No evaluation performed (tiny dataset or no test split).")

    print(f"Model & tokenizer saved to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="models/intent_model", help="output folder")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased", help="pretrained HF model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--test_frac", type=float, default=0.15)
    parser.add_argument("--max_len", type=int, default=128)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    main(args)
