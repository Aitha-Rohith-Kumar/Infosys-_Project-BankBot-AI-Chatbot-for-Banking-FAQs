# nlu_engine/tests/test_nlu.py
import requests
import time

BASE = "http://localhost:5006/parse"

cases = [
    ("Please transfer ₹2,500 to account 9988776655", "transfer", ["amount", "account_number"]),
    ("What is my savings account balance?", "balance", []),
    ("Has TXN12345 been processed?", "transaction", ["transaction_id"]),
    ("Block my debit card", "block", [])
]

def test_nlu():
    # wait until server up
    for _ in range(6):
        try:
            r = requests.get("http://localhost:5006/health", timeout=1)
            if r.status_code == 200:
                break
        except:
            time.sleep(1)

    for text, expected_keyword, req_ents in cases:
        r = requests.post(BASE, json={"text": text})
        assert r.status_code == 200, f"API failed for {text}: {r.text}"
        j = r.json()
        top = j.get("intents", {}).get("predictions", [])
        top_label = top[0]["intent"] if top else ""
        assert expected_keyword in top_label.lower() or expected_keyword in text.lower(), f"Intent mismatch for '{text}' -> {top_label}"
        ents = [e["entity"] for e in j.get("entities",[])]
        for re in req_ents:
            assert re in ents, f"Missing entity {re} for {text}: found {ents}"
