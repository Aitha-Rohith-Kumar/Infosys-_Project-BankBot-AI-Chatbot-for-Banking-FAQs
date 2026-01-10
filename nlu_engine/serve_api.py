# nlu_engine/serve_api.py
from flask import Flask, request, jsonify
from infer_intent import IntentClassifier
from entity_extractor import EntityExtractor

app = Flask(__name__)

# load once
ic = None
ee = None

def ensure_models_loaded():
    global ic, ee
    if ic is None:
        ic = IntentClassifier()
    if ee is None:
        ee = EntityExtractor()

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status":"ok"})

@app.route("/parse", methods=["POST"])
def parse():
    ensure_models_loaded()
    data = request.json or {}
    text = data.get("text") or data.get("query") or ""
    if not text:
        return jsonify({"error":"provide 'text' field"}), 400
    intents = ic.predict(text, top_k=5)
    entities = ee.extract(text)
    # simple slots
    slots = {}
    for e in entities:
        if e["entity"] == "amount":
            slots["amount"] = e.get("normalized", e["value"])
        elif e["entity"] == "account_number":
            slots["account_number"] = e["value"]
        elif e["entity"] == "transaction_id":
            slots["transaction_id"] = e["value"]
    return jsonify({"text": text, "intents": intents, "entities": entities, "slots": slots})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5006, debug=True)
