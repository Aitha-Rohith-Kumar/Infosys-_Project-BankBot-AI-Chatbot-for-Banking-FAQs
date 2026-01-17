# nlu_engine/infer_intent.py
import os, json
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "intent_model")

class IntentClassifier:
    def __init__(self, model_dir: str = MODEL_DIR):
        self.model_dir = model_dir
        if not os.path.isdir(self.model_dir):
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}. Train first.")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        # load labels if present
        labels_path = os.path.join(self.model_dir, "labels.json")
        if os.path.exists(labels_path):
            with open(labels_path, "r", encoding="utf-8") as f:
                self.labels = json.load(f)
        else:
            # fallback to model config labels
            num_labels = self.model.config.num_labels
            self.labels = [f"label_{i}" for i in range(num_labels)]
        self.pipe = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer, return_all_scores=True)

    def predict(self, text: str, top_k: int = 3) -> Dict[str, Any]:
        out = self.pipe(text)
        if isinstance(out, list) and len(out) > 0:
            scores = out[0]
            sorted_scores = sorted(scores, key=lambda x: x["score"], reverse=True)[:top_k]
            preds = []
            for s in sorted_scores:
                lab = s["label"]
                # HF often returns 'LABEL_0' — map to our labels if possible
                if lab.startswith("LABEL_") and lab[6:].isdigit():
                    idx = int(lab[6:])
                    mapped = self.labels[idx] if idx < len(self.labels) else lab
                else:
                    mapped = lab
                preds.append({"intent": mapped, "confidence": float(s["score"])})
            return {"text": text, "predictions": preds}
        return {"text": text, "predictions": []}

if __name__ == "__main__":
    ic = IntentClassifier()
    print(ic.predict("Please transfer ₹2,500 to account 9988776655", top_k=5))

# -------------------- SAFE WRAPPER FOR UI --------------------
_classifier = None


def predict_intent(text: str, top_k: int = 3):
    """
    Safe intent prediction:
    - Tries ML model
    - Falls back to rule-based demo if model/tokenizer fails
    """
    global _classifier

    # ---------- TRY ML ----------
    try:
        if _classifier is None:
            _classifier = IntentClassifier()

        result = _classifier.predict(text, top_k=top_k)

        if result["predictions"]:
            top = result["predictions"][0]

            if top["confidence"] >= 0.6:
                return "out_of_scope", top["confidence"], {}
            
            return top["intent"], top["confidence"], {}

    except Exception:
        pass  # silently fall back

    # ---------- FALLBACK DEMO LOGIC ----------
    t = text.lower()

    if any(w in t for w in ["hi", "hello", "hey","good morning","good afternoon","good evening","thanks","thank you","thanks alot","nice work"]):
        return "greet", 0.95, {}

    if "balance" in t:
        return "check_balance", 0.92, {}

    if "transfer" in t and any(char.isdigit() for char in t):
        # simple entity extraction
        import re
        amount = re.findall(r"\b\d+\b", t)
        acc = re.findall(r"account\s+(\d+)", t)

        entities = {}
        if amount:
            entities["amount"] = amount[0]
        if acc:
            entities["account_number"] = acc[0]

        return "transfer_money", 0.90, entities
    
    if any(w in t for w in ["unblock card", "activate card", "enable my card","reactivate card","unblock my card","activate atm card","enable card","reactivate my card","unblock atm card","activate debit card","enable debit card","reactivate debit card","activate credit card","enable credit card","reactivate credit card","unblock credit card","unblock my credit card","unblock debit card"]):
        return "unblock_card", 0.95, {}

    if "atm" in t:
        return "atm_info", 0.90, {}
    
    if "loan" in t:
        return "loan_info", 0.88, {}
    
    if "interest rate" in t:
        return "interest_rate", 0.85, {}

    if any(w in t for w in ["account details", "account info", "account information","my account details","show my account details","view account details"]):
        return "account_details", 0.90, {}

    if any(k in t for k in ["block my card", "lost my card", "stolen my card","block atm card","block card","lost card","stolen card"]):
        return "block_card", 0.95, {}

    if any(w in t for w in ["bye", "exit", "quit","see you","goodbye","have a nice day"]):
        return "goodbye", 0.93, {}

    return "out_of_scope", 0.50, {}
