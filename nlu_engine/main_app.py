# main_app.py
import streamlit as st
import os
import json
import re
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

st.set_page_config(page_title="BankBot NLU — Intent & Entity Engine", layout="wide")

# ---------------- Paths ----------------
INTENTS_PATH = os.path.join("nlu_engine", "intents.json")
MODEL_DIR = os.path.join("models", "intent_model")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
CLF_PATH = os.path.join(MODEL_DIR, "clf.pkl")

# ---------------- Default intents (fallback) ----------------
DEFAULT_INTENTS = {
    "check_balance": [
        "What's my account balance?",
        "Show balance for savings account",
        "How much money do I have?"
    ],
    "transfer_money": [
        "Transfer ₹5000 from savings to checking",
        "Move ₹1500 to account 12345678",
        "Please transfer ₹250 to my friend",
        "I want to send ₹1000 to account 9876543210"
    ],
    "card_block": [
        "Block my debit card",
        "My credit card is lost, block it",
        "Disable my card immediately"
    ],
    "find_atm": [
        "Find nearest ATM",
        "Where is the nearest branch or ATM?",
        "ATM near me in Hyderabad"
    ],
    "apply_loan": [
        "How can I apply for a home loan?",
        "Loan eligibility for 24 months income",
        "I want to apply for a personal loan"
    ],
    "open_account": [
        "How do I open a savings account?",
        "Documents required for account opening",
        "Open a new salary account"
    ]
}

# ---------------- Helpers: load/save intents ----------------
def ensure_dirs():
    os.makedirs(os.path.dirname(INTENTS_PATH), exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

def load_intents_from_disk():
    if os.path.exists(INTENTS_PATH):
        try:
            with open(INTENTS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                # ensure lists
                return {k: (v if isinstance(v, list) else [v]) for k, v in data.items()}
        except Exception as e:
            st.warning(f"Failed to load {INTENTS_PATH}: {e}")
            return None
    return None

def save_intents_to_disk():
    try:
        ensure_dirs()
        with open(INTENTS_PATH, "w", encoding="utf-8") as f:
            json.dump(st.session_state.intents, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.error(f"Failed to save intents file: {e}")

# ---------------- NLU training / persistence ----------------
def train_nlu_and_persist(intents_map):
    X = []
    y = []
    for intent, examples in intents_map.items():
        for ex in examples:
            X.append(ex)
            y.append(intent)
    if not X:
        return None, None
    vec = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
    Xv = vec.fit_transform(X)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xv, y)
    # persist
    ensure_dirs()
    joblib.dump(vec, VECTORIZER_PATH)
    joblib.dump(clf, CLF_PATH)
    return vec, clf

def load_model_if_exists():
    try:
        if os.path.exists(VECTORIZER_PATH) and os.path.exists(CLF_PATH):
            vec = joblib.load(VECTORIZER_PATH)
            clf = joblib.load(CLF_PATH)
            return vec, clf
    except Exception as e:
        st.warning(f"Failed to load saved model: {e}")
    return None, None

# ---------------- Entity extraction (currency-only amounts) ----------------
ACCOUNT_RE = re.compile(
    r"(?:account|acct|a/c|a c|a\.c\.|acc)\s*(?:no[:#]?\s*)?[:#]?\s*([0-9]{5,16})",
    flags=re.I,
)
FALLBACK_DIGIT_RE = re.compile(r"\b([0-9]{6,16})\b")

# two robust currency-number regexes (handles prefix & suffix, commas, decimals)
CURRENCY_NUM_RE = re.compile(
    r"(?:₹|\$|rs\.?|rupees|rupee|inr)\s*([0-9][0-9,\.]*)", flags=re.I
)
NUM_CURRENCY_WORD_RE = re.compile(
    r"([0-9][0-9,\.]*)\s*(?:rupees|rupee|rs\.?|inr|\$)", flags=re.I
)

def _clean_number_str(s: str) -> str:
    return re.sub(r"[,\s]", "", s)

def extract_entities(text: str):
    """
    Returns dict with optional keys:
      - 'Account_Number' (string)
      - 'Amount' (float)  -> only if currency marker present
    """
    ents = {}
    # Account extraction
    acc = ACCOUNT_RE.search(text)
    if acc:
        ents["Account_Number"] = acc.group(1)
    else:
        fb = FALLBACK_DIGIT_RE.findall(text)
        if fb:
            ents["Account_Number"] = sorted(fb, key=lambda x: -len(x))[0]

    # Amount extraction: ONLY if currency marker present
    low = text.lower()
    currency_markers = ["₹", "rs", "rs.", "rupee", "rupees", "inr", "$"]
    if any(mark in low for mark in currency_markers):
        # Try currency-number (marker before/near num)
        m = CURRENCY_NUM_RE.search(text)
        if m:
            num_str = m.group(1)
            cleaned = _clean_number_str(num_str)
            try:
                ents["Amount"] = float(cleaned)
                return ents
            except:
                pass
        # Try number followed by currency word
        m2 = NUM_CURRENCY_WORD_RE.search(text)
        if m2:
            num_str = m2.group(1)
            cleaned = _clean_number_str(num_str)
            try:
                ents["Amount"] = float(cleaned)
                return ents
            except:
                pass
        # If currency marker present but above fails, try first numeric token
        first_num = re.search(r"([0-9][0-9,\.]*)", text)
        if first_num:
            cleaned = _clean_number_str(first_num.group(1))
            try:
                ents["Amount"] = float(cleaned)
            except:
                pass

    return ents

# ---------------- Utility: clamp probs ----------------
def clamp_prob(p):
    p_clamped = max(0.01, float(p))
    return min(1.0, p_clamped)

# ---------------- Session state init ----------------
if "intents" not in st.session_state:
    on_disk = load_intents_from_disk()
    if on_disk:
        st.session_state.intents = on_disk
    else:
        st.session_state.intents = {k: v[:] for k, v in DEFAULT_INTENTS.items()}

# Try load saved model
if "vectorizer" not in st.session_state or "clf" not in st.session_state:
    vec, clf = load_model_if_exists()
    if vec is not None and clf is not None:
        st.session_state.vectorizer = vec
        st.session_state.clf = clf
        st.session_state.trained_model = True
    else:
        st.session_state.vectorizer = None
        st.session_state.clf = None
        st.session_state.trained_model = False

if "train_params" not in st.session_state:
    st.session_state.train_params = {"epochs": 2, "batch_size": 8, "lr": 0.00002}

# ---------------- UI CSS ----------------
st.markdown(
    """
    <style>
    :root { color-scheme: dark; }
    .stApp { background: #0b0f12; color: #e6eef3; }
    .panel { background: #0f1113; border-radius: 6px; padding: 12px; }
    .right-card { background:#0b0d0f; padding:18px; border-radius:8px; margin-bottom:12px; }
    .badge-true { display:inline-block; padding:6px 10px; background:#d4f5d6; color:#063d10; border-radius:6px; font-weight:600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Layout ----------------
st.markdown('<div style="font-size:28px;font-weight:700;margin-bottom:6px;">BankBot NLU – Intent & Entity Engine</div>', unsafe_allow_html=True)
st.markdown('<div style="color:#98a0a6;margin-bottom:10px;">1. Edit & Train Intents</div>', unsafe_allow_html=True)

left_col, right_col = st.columns([2, 3])

# ---------------- Left: intent editor ----------------
with left_col:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)

    # Render expanders from session_state (so counts are always up-to-date)
    for intent_name in list(st.session_state.intents.keys()):
        examples = st.session_state.intents[intent_name]
        with st.expander(f"{intent_name} ({len(examples)} examples)", expanded=False):
            new_name = st.text_input("Intent name", value=intent_name, key=f"name_{intent_name}")
            examples_text = st.text_area("Examples (one per line)", value="\n".join(examples), key=f"ex_{intent_name}", height=140)
            c1, c2 = st.columns([1,1])
            with c1:
                if st.button("Save changes", key=f"save_{intent_name}"):
                    ex_list = [e.strip() for e in examples_text.splitlines() if e.strip()]
                    # rename/merge logic
                    if new_name != intent_name:
                        if new_name in st.session_state.intents:
                            st.session_state.intents[new_name].extend(ex_list)
                            del st.session_state.intents[intent_name]
                            st.success(f"Merged `{intent_name}` into existing `{new_name}`.")
                        else:
                            st.session_state.intents[new_name] = ex_list
                            del st.session_state.intents[intent_name]
                            st.success(f"Renamed `{intent_name}` → `{new_name}`.")
                    else:
                        st.session_state.intents[intent_name] = ex_list
                        st.success(f"Updated `{intent_name}` examples ({len(ex_list)}).")
                    save_intents_to_disk()
                    st.experimental_rerun()
            with c2:
                if st.button("Delete intent", key=f"del_{intent_name}"):
                    del st.session_state.intents[intent_name]
                    save_intents_to_disk()
                    st.experimental_rerun()

    st.markdown("---")
    st.subheader("Add new intent")
    with st.form("add_intent_form", clear_on_submit=True):
        ai_name = st.text_input("Intent name (snake_case)", placeholder="transfer_money")
        ai_examples = st.text_area("Examples (one per line)", placeholder="Transfer ₹500 to account 12345678")
        submitted = st.form_submit_button("Add Intent")
        if submitted:
            if not ai_name.strip():
                st.error("Provide an intent name.")
            else:
                exs = [e.strip() for e in ai_examples.splitlines() if e.strip()]
                if ai_name in st.session_state.intents:
                    st.session_state.intents[ai_name].extend(exs)
                    st.success(f"Appended {len(exs)} examples to `{ai_name}`.")
                else:
                    st.session_state.intents[ai_name] = exs if exs else ["example sentence"]
                    st.success(f"Added new intent `{ai_name}`.")
                save_intents_to_disk()
                st.experimental_rerun()

    st.markdown("---")
    st.download_button(
        label="Save intents.json (download)",
        data=json.dumps(st.session_state.intents, indent=2, ensure_ascii=False),
        file_name="intents.json",
        mime="application/json",
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Right: NLU Visualizer ----------------
with right_col:
    st.markdown("<div class='right-card'>", unsafe_allow_html=True)
    st.subheader("NLU Visualizer")
    user_query = st.text_area("User Query", value="Move ₹1500 to account 12345678", height=100)
    top_k = st.number_input("Top intents to show", min_value=1, max_value=10, value=4, step=1)
    if st.button("Analyze"):
        # ensure a model exists (train on-the-fly if nothing persisted and no trained model)
        if st.session_state.trained_model and st.session_state.clf and st.session_state.vectorizer:
            vec = st.session_state.vectorizer
            clf = st.session_state.clf
        else:
            vec, clf = train_nlu_and_persist(st.session_state.intents)
            if vec is None:
                st.warning("No training data available. Add intents/examples first.")
                vec = None
                clf = None
            else:
                # update session state to reflect persisted model
                st.session_state.vectorizer = vec
                st.session_state.clf = clf
                st.session_state.trained_model = True

        if vec is not None and clf is not None:
            Xq = vec.transform([user_query])
            probs = clf.predict_proba(Xq)[0]
            classes = clf.classes_
            idxs = np.argsort(probs)[::-1][:top_k]

            st.markdown("### Intent Recognition")
            for i in idxs:
                prob = clamp_prob(probs[i])
                st.write(f"- **{classes[i]}:** {prob:.2f}")

            st.markdown("### Entity Extraction")
            ents = extract_entities(user_query)
            if ents:
                if "Account_Number" in ents:
                    st.write(f"**Account_Number:** {ents['Account_Number']}")
                if "Amount" in ents:
                    st.write(f"**Amount:** {ents['Amount']:.2f}")
            else:
                st.write("_No entities found._")
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- Train model (below visualizer) ----------------
    st.markdown("<div class='right-card'>", unsafe_allow_html=True)
    st.subheader("Train model")
    if st.session_state.trained_model:
        st.markdown('<span class="badge-true">Trained model found</span>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        epochs = st.number_input("Epochs", min_value=1, max_value=100, value=st.session_state.train_params.get("epochs", 2), step=1, key="epochs2")
    with c2:
        batch_size = st.number_input("Batch size", min_value=1, max_value=1024, value=st.session_state.train_params.get("batch_size", 8), step=1, key="batch2")
    with c3:
        lr = st.number_input("Learning rate", min_value=0.0, format="%.8f", value=float(st.session_state.train_params.get("lr", 0.00002)), step=1e-6, key="lr2")

    if st.button("Start training", key="start_training"):
        st.session_state.train_params = {"epochs": int(epochs), "batch_size": int(batch_size), "lr": float(lr)}
        progress = st.progress(0)
        total_steps = max(10, int(epochs) * 5)
        for i in range(total_steps):
            time.sleep(0.03)
            progress.progress(int((i+1)/total_steps * 100))
        # train & persist
        vec, clf = train_nlu_and_persist(st.session_state.intents)
        if vec is not None and clf is not None:
            st.success("Training completed and model saved.")
            st.session_state.vectorizer = vec
            st.session_state.clf = clf
            st.session_state.trained_model = True
        else:
            st.warning("No training data available; add intents/examples first.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Footer ----------------
st.markdown("---")
