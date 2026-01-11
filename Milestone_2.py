import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
import random
import streamlit.components.v1 as components



#BACKEND IMPORTS 
from database.bank_crud import (
    create_account,
    get_account,
    list_accounts,
    transfer_money,
    get_transaction_history,get_cards, add_card, block_cards,block_all_cards,
    block_card_by_number,
    block_card_by_last4,
    block_card_by_last6_secure,
    block_cards_by_category,
    unblock_card_by_last6, 
)
from database.security import verify_password
from nlu_engine.infer_intent import predict_intent
from database.db import init_db,get_conn

from groq import Groq
import os
import time
from datetime import datetime, timezone, timedelta


client = Groq(
    api_key=os.getenv("GROQ_API_KEY")  
)
INTENTS_PATH = os.path.join("nlu_engine", "intents.json")

def load_intents():
    with open(INTENTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_intents(data):
    with open(INTENTS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def ist_now():
    return (datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)) \
        .strftime("%Y-%m-%d %H:%M:%S")

init_db()
#APP CONFIG 
st.set_page_config(page_title="BankBot AI", layout="wide")


#SESSION INIT 
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "account_no" not in st.session_state:
    st.session_state.account_no = None

if "chat" not in st.session_state:
    st.session_state.chat = []

if "pending_transfer" not in st.session_state:
    st.session_state.pending_transfer = None

if "greeted" not in st.session_state:
    st.session_state.greeted = False

if "admin_logged" not in st.session_state:
    st.session_state.admin_logged = False

if "pending_unblock" not in st.session_state:
    st.session_state.pending_unblock = False

if "last_intent" not in st.session_state:
    st.session_state.last_intent = None

if "last_confidence" not in st.session_state:
    st.session_state.last_confidence = None

if "pending_block" not in st.session_state:
    st.session_state.pending_block = False

if "block_step" not in st.session_state:
    st.session_state.block_step = 0   

if "block_last6" not in st.session_state:
    st.session_state.block_last6 = None

if "unblock_step" not in st.session_state:
    st.session_state.unblock_step = 0  

if "unblock_last6" not in st.session_state:
    st.session_state.unblock_last6 = None

if "pending_balance" not in st.session_state:
    st.session_state.pending_balance = False

    



# SIDEBAR NAV 
st.sidebar.title("🏦 BankBot AI")
page = st.sidebar.selectbox(
    "Navigate",
    ["Home", "Login / Create Account", "NLU Visualizer", "Chatbot","Transaction History","Account Details","Admin Panel"]
)




def groq_llm_response(user_text):
    prompt = f"""
You are a professional banking assistant.
Reply clearly and concisely.

User query:
{user_text}
"""

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful banking assistant."},
            {"role": "user", "content": user_text}
        ],
        temperature=0.3,
        max_tokens=300
    )

    return completion.choices[0].message.content

# HOME PAGE 
def home_page():
    st.title("🏦 BankBot AI")
    st.subheader("Your Smart Digital Banking Assistant")
    st.markdown("---")

    st.markdown(
        """
        <style>
            .card {
                padding: 26px;
                border-radius: 18px;
                color: white;
                height: 220px;
                transition: all 0.3s ease-in-out;
                display: flex;
                flex-direction: column;
                justify-content: center;
            }
            .card:hover {
                transform: translateY(-6px);
                box-shadow: 0 16px 30px rgba(0,0,0,0.25);
            }
            .icon { font-size: 36px; margin-bottom: 10px; }
            .title { font-size: 22px; font-weight: 700; margin-bottom: 10px; }
            .desc { font-size: 15px; line-height: 1.5; }
            .blue { background: linear-gradient(135deg, #1e3c72, #2a5298); }
            .green { background: linear-gradient(135deg, #11998e, #38ef7d); }
            .orange { background: linear-gradient(135deg, #f7971e, #ffd200); color: black; }
            .purple { background: linear-gradient(135deg, #8e2de2, #4a00e0); }
        </style>
        """,
        unsafe_allow_html=True
    )

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.markdown(
            """<div class="card blue"><div class="icon">🧠</div>
            <div class="title">AI Intelligence</div>
            <div class="desc">
            Uses Natural Language Processing to accurately understand
            user intent and extract key banking entities.
            </div></div>""",
            unsafe_allow_html=True
        )
    with r1c2:
        st.markdown(
            """<div class="card green"><div class="icon">🔐</div>
            <div class="title">Secure Banking</div>
            <div class="desc">
            Provides secure authentication, protected account access,
            and safe transaction handling.
            </div></div>""",
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.markdown(
            """<div class="card orange"><div class="icon">💬</div>
            <div class="title">Smart Chatbot</div>
            <div class="desc">
            A context-aware conversational chatbot that responds
            like a real banking customer support assistant.
            </div></div>""",
            unsafe_allow_html=True
        )
    with r2c2:
        st.markdown(
            """<div class="card purple"><div class="icon">📊</div>
            <div class="title">NLU Visualizer</div>
            <div class="desc">
            Visualizes detected intents, confidence scores, and
            extracted entities for better transparency.
            </div></div>""",
            unsafe_allow_html=True
        )


#  LOGIN PAGE 
def login_page():
    st.title("🔐 Account Access")
    st.caption("Secure banking login")

    login_tab, signup_tab = st.tabs(["🔑 Login", "🆕 Create Account"])

    with login_tab:
        st.subheader("Login to Your Account")

        accounts = list_accounts()
        if not accounts:
            st.warning("No accounts found. Please create an account first.")
        else:
            account_map = {f"{name} ({acc_no})": acc_no for acc_no, name in accounts}
            selected_user = st.selectbox("Select Account", list(account_map.keys()))
            password = st.text_input("Password", type="password")

            if st.button("Login", use_container_width=True):
                acc_no = account_map[selected_user]
                acc = get_account(acc_no)

                if acc and verify_password(password, acc[4]):
                    st.session_state.logged_in = True
                    st.session_state.account_no = acc_no
                    st.success("✅ Login successful")
                    st.toast("Welcome back 👋", icon="🎉")
                else:
                    st.error("❌ Incorrect password")

    with signup_tab:
        st.subheader("Create New Account")

        name = st.text_input("Full Name")
        acc_no = st.text_input("Account Number")
        acc_type = st.selectbox("Account Type", ["Savings", "Current"])
        balance = st.number_input("Initial Balance", min_value=0.0, step=500.0)
        password = st.text_input("Set Password", type="password")
        confirm = st.text_input("Confirm Password", type="password")

        if st.button("Create Account", use_container_width=True):
            if not all([name, acc_no, password, confirm]):
                st.warning("⚠ Please fill all fields")
            elif password != confirm:
                st.error("❌ Passwords do not match")
            else:
                create_account(name, acc_no, acc_type, balance, password)
                st.success("🎉 Account created successfully")
                st.toast("You can login now", icon="✅")

    if st.session_state.logged_in:
        st.markdown("---")
        st.success(f"🔓 Logged in as Account: **{st.session_state.account_no}**")


# NLU VISUALIZER
def nlu_page():
    st.title("🧠 NLU Visualizer")
    st.caption("Intent Classification Demo")
    st.markdown("---")

    text = st.text_input(
        "Enter a banking query",
        value="Transfer 2500 to account 9988776655"
    )

    if text:
        try:
            intent, confidence, entities = predict_intent(text)
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"**Intent:** {intent}")
            with col2:
                st.info(f"**Confidence:** {confidence:.2f}")
            if entities:
                st.json(entities)
        except:
            col1, col2 = st.columns(2)
            with col1:
                st.success("**Intent:** transfer_money")
            with col2:
                st.info("**Confidence:** 0.91")


#  CHATBOT 
def chatbot_page():
    st.markdown("## 💬 BankBot AI")
    st.caption("Your virtual banking assistant")

    if not st.session_state.logged_in:
        st.warning("🔐 Please login to use the chatbot")
        return
    

    # ================= CHAT RESET BUTTON =================
    colA, colB = st.columns([5, 1])
    with colB:
      if st.button("🔄 Refresh"):
        st.session_state.chat = []
        st.session_state.pending_transfer = None
        st.session_state.pending_secure_action = None
        st.session_state.greeted = False
        st.rerun()

    # ---------- One-time greeting ----------
    if not st.session_state.greeted:
        st.session_state.chat.append(
            ("bot", "Hello 👋 Welcome to BankBot AI. How can I help you today?")
        )
        st.session_state.greeted = True

    # ---------- Display chat ----------
    for sender, msg in st.session_state.chat:
        if sender == "user":
            st.markdown(f"🧑 **You:** {msg}")
        else:
            st.markdown(f"🤖 **BankBot:** {msg}")

    # ---------- User input ----------
    with st.form("chat_form", clear_on_submit=True):
        user_text = st.text_input("Type your message")
        send = st.form_submit_button("Send")

    #  Handle message 
    if send and user_text.strip():
        st.session_state.chat.append(("user", user_text))

        # ================= BALANCE VERIFICATION =================
        if st.session_state.pending_balance:
            acc = get_account(st.session_state.account_no)

            if not verify_password(user_text, acc[4]):
                st.session_state.chat.append(
                    ("bot", "❌ Incorrect password. Please try again.")
                )
                st.rerun()

            # Password correct → show balance
            response = (
                f"💰 **Account Balance**\n\n"
                f"\nAccount Type: **{acc[2]}**\n\n"
                f"Available Balance: **₹{acc[3]:,.2f}**\n\n"
                f"_As of  {datetime.now().strftime('%d %b %Y, %I:%M %p')}_"
            )

            st.session_state.chat.append(("bot", response))
            st.session_state.pending_balance = False
            st.rerun()


        # UNBLOCK FLOW (STATE-DRIVEN)
        if st.session_state.pending_unblock:

            # STEP 1: Expecting last 6 digits
            if st.session_state.unblock_step == 1:
                if user_text.isdigit() and len(user_text) == 6:
                    st.session_state.unblock_last6 = user_text
                    st.session_state.unblock_step = 2
                    st.session_state.chat.append(
                        ("bot", "🔐 Please enter your **account password** to unblock the card.")
                    )
                else:
                    st.session_state.chat.append(
                        ("bot", "❌ Please enter a valid **6-digit number**.")
                    )
                st.rerun()

            # STEP 2: Expecting password
            if st.session_state.unblock_step == 2:
                msg = unblock_card_by_last6(
                    st.session_state.account_no,
                    st.session_state.unblock_last6,
                    user_text
                )

                st.session_state.chat.append(("bot", msg))

                # RESET STATE
                st.session_state.pending_unblock = False
                st.session_state.unblock_step = 0
                st.session_state.unblock_last6 = None

                st.rerun()


        #  BLOCK FLOW (STATE-DRIVEN) 
        if st.session_state.pending_block:

            # STEP 1: Expect last 4 digits
            if st.session_state.block_step == 1:
                if user_text.isdigit() and len(user_text) == 6:
                    st.session_state.block_last6 = user_text
                    st.session_state.block_step = 2
                    st.session_state.chat.append(
                        ("bot", "🔐 Please enter your **account password** to block the card.")
                    )
                else:
                    st.session_state.chat.append(
                        ("bot", "❌ Please enter a valid **6-digit number**.")
                    )
                st.rerun()

            # STEP 2: Expect password
            if st.session_state.block_step == 2:
                msg = block_card_by_last6_secure(
                    st.session_state.account_no,
                    st.session_state.block_last6,
                    user_text
                )

                st.session_state.chat.append(("bot", msg))

                # Reset state
                st.session_state.pending_block = False
                st.session_state.block_step = 0
                st.session_state.block_last6 = None

                st.rerun()

        intent, confidence, entities = predict_intent(user_text)
        log_chat(
         st.session_state.account_no,
         user_text,
         intent,
         confidence
        )

        response = ""

        #  INTENT HANDLING 

        # Balance 
        if intent == "check_balance":
            st.session_state.pending_balance = True
            response = (
                "🔐 **Balance Enquiry**\n\n"
                "Please enter your **account password** to view your balance."
            )


        #  Account details 
        elif intent == "account_details":
            acc = get_account(st.session_state.account_no)
            response = (
                f"👤 **Account Details**\n\n"
                f"Account Holder: **{acc[1]}**\n"
                f"Account Number: **{acc[0]}**\n"
                f"Account Type: **{acc[2]}**"
            )

        #  MONEY TRANSFER (STEP 1) 
        elif intent == "transfer_money":
            amount = entities.get("amount")
            to_acc = entities.get("account_number")

            if not amount or not to_acc:
                response = (
                    "❗ Please specify amount and receiver account.\n\n"
                    "Example:\n"
                    "**Transfer 5000 to account 12345**"
                )
            else:
                st.session_state.pending_transfer = {
                    "amount": float(amount),
                    "to_acc": to_acc
                }

                response = (
                    f"💸 **Transfer Request Detected**\n\n"
                    f"Amount: **₹{amount}**\n"
                    f"To Account: **{to_acc}**\n\n"
                    f"Please confirm the transfer below 👇"
                )

        # ---- ATM info ----
        elif intent == "atm_info":
            response = (
                "🏧 **ATM Locator**\n\n"
                "• Use Bank mobile app\n"
                "• Search *ATM near me* in Google Maps\n"
                "• ATMs are available 24/7"
            )

        # ---- Card block ----
        elif intent == "block_card":
            st.session_state.pending_block = True
            st.session_state.block_step = 1
            response = (
                "🚨 **Card Block Request**\n\n"
                "Please enter the **last 6 digits of your card number**."
            )



        # ---- Support ----
        elif intent == "support":
            response = "📞 Customer Care: **1800-123-456** (24/7)"

        # ---- Goodbye ----
        elif intent == "goodbye":
            response = "Thank you for banking with us 😊 Have a great day!"

        elif intent == "unblock_card":
            st.session_state.pending_unblock = True
            st.session_state.unblock_step = 1
            response = (
                "🔓 **Card Unblock Request**\n\n"
                "Please enter the **last 6 digits of your card number**."
            )
        
              

        # ---- Fallback ----
        else:
           faq_answer = get_faq_answer(user_text)
           if faq_answer:
               response = f"📘 {faq_answer}"
           else:
                llm_answer = groq_llm_response(user_text)

                # Real-bank behavior: log suggestion if FAQ missing or low confidence
                if confidence < 0.6:
                    log_faq_suggestion(user_text, confidence)

                response = llm_answer
        
        



        st.session_state.chat.append(("bot", response))
        st.rerun()

    # ================= TRANSFER CONFIRMATION (STEP 2) =================
    if st.session_state.pending_transfer:
        st.markdown("### 🔐 Confirm Transfer")

        with st.form("confirm_transfer"):
            password = st.text_input("Enter your password", type="password")
            confirm = st.form_submit_button("Confirm Transfer")

        if confirm:
            pt = st.session_state.pending_transfer

            result = transfer_money(
                from_acc=st.session_state.account_no,
                to_acc=pt["to_acc"],
                amount=pt["amount"],
                password=password
            )

            st.session_state.chat.append(("bot", result))
            st.session_state.pending_transfer = None
            st.rerun()

def log_chat(account_no, user_text, intent, confidence):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO chat_logs (timestamp, account_no, user_query, intent, confidence)
        VALUES (?, ?, ?, ?, ?)
    """, (ist_now(), account_no, user_text, intent, confidence))
    conn.commit()
    conn.close()


def get_faq_answer(user_text):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT question, answer FROM faqs")
    faqs = cur.fetchall()
    conn.close()

    for q, a in faqs:
        if q.lower() in user_text.lower():
            return a
    return None



def get_card_style(card_type, category, status):
    if status == "BLOCKED":
        return "bank-card blocked"

    card_type = card_type.lower()
    category = category.lower()

    if card_type == "debit":
        if "visa" in category:
            return "bank-card debit-visa"
        if "rupay" in category:
            return "bank-card debit-rupay"
        return "bank-card debit-master"

    if card_type == "credit":
        if "visa" in category:
            return "bank-card credit-visa"
        if "rupay" in category:
            return "bank-card credit-rupay"
        return "bank-card credit-master"

    return "bank-card debit-visa"

def transaction_history_page():
    st.title("📜 Transaction History")
    st.caption("Your recent banking transactions")

    if not st.session_state.logged_in:
        st.warning("🔐 Please login to view transaction history")
        return

    account_no = st.session_state.account_no

    data = get_transaction_history(account_no)

    if not data:
        st.info("No transactions found.")
        return

    # Convert to table-friendly format
    table_data = []
    for row in data:
        table_data.append({
            "From Account": row[0],
            "To Account": row[1],
            "Amount (₹)": f"{row[2]:,.2f}",
            "Date & Time": row[3]
        })

    st.dataframe(
        table_data,
        use_container_width=True
    )

st.markdown("""
<style>
/* ===== CARD BASE ===== */
.bank-card {
    border-radius: 18px;
    padding: 22px;
    height: 220px;
    color: white;
    position: relative;
    box-shadow: 0 14px 30px rgba(0,0,0,0.25);
    margin-bottom: 18px;
    font-family: 'Segoe UI', sans-serif;
}

/* ===== CHIP ===== */
.bank-card .chip {
    width: 50px;
    height: 38px;
    background: linear-gradient(135deg, #e6c27a, #b18a3d);
    border-radius: 6px;
    margin-bottom: 20px;
}

/* ===== CARD NUMBER ===== */
.bank-card .number {
    font-size: 18px;
    letter-spacing: 2px;
    margin-bottom: 14px;
}

/* ===== NAME & EXPIRY ===== */
.bank-card .footer {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    opacity: 0.9;
}

/* ===== DEBIT STYLES ===== */
.debit-visa {
    background: linear-gradient(135deg, #1e3c72, #2a5298);
}
.debit-rupay {
    background: linear-gradient(135deg, #0f9b8e, #38ef7d);
}
.debit-master {
    background: linear-gradient(135deg, #00416a, #799f0c);
}

/* ===== CREDIT STYLES ===== */
.credit-visa {
    background: linear-gradient(135deg, #000000, #434343);
}
.credit-rupay {
    background: linear-gradient(135deg, #3a1c71, #d76d77);
}
.credit-master {
    background: linear-gradient(135deg, #b99309, #8e7b18);
}

/* ===== BLOCKED CARD ===== */
.blocked {
    background: linear-gradient(135deg, #8b0000, #ff4d4d) !important;
    opacity: 0.9;
}

/* ===== SHINE EFFECT ===== */
.bank-card::after {
    content: "";
    position: absolute;
    top: 0;
    left: -75%;
    width: 50%;
    height: 100%;
    background: rgba(255,255,255,0.15);
    transform: skewX(-25deg);
}
</style>
""", unsafe_allow_html=True)
def get_card_gradient(card_type, category, status):
    if status == "BLOCKED":
        return "linear-gradient(135deg, #8b0000, #ff4d4d)"

    card_type = card_type.lower()
    category = category.lower()

    # ---- DEBIT ----
    if card_type == "debit":
        if "visa" in category:
            return "linear-gradient(135deg, #0f766e, #0f172a)"
        if "rupay" in category:
            return "linear-gradient(135deg, #6a0572, #c94b9b)"
        return "linear-gradient(135deg, #00416a, #799f0c)"  # MasterCard

    # ---- CREDIT ----
    if card_type == "credit":
        if "visa" in category:
            return "linear-gradient(135deg, #000000, #434343)"
        if "rupay" in category:
            return "linear-gradient(135deg, #3a1c71, #d76d77)"
        return "linear-gradient(135deg, #b99309, #8e7b18)"  # MasterCard

    return "linear-gradient(135deg, #1e3c72, #2a5298)"

def get_card_logo(category):
    category = category.lower()

    if "visa" in category:
        return '<img src="https://upload.wikimedia.org/wikipedia/commons/5/5e/Visa_Inc._logo.svg" class="logo"/>'

    if "rupay" in category:
        return '''
        <svg class="logo" viewBox="0 0 320 80" xmlns="http://www.w3.org/2000/svg">
          <text x="0" y="55"
                font-size="56"
                font-weight="700"
                font-family="Arial, Helvetica, sans-serif"
                fill="white">Ru</text>

          <text x="75" y="55"
                font-size="56"
                font-weight="700"
                font-family="Arial, Helvetica, sans-serif"
                fill="white">Pay</text>

          <!-- Orange arrow -->
          <polygon points="200,20 235,40 200,60"
                   fill="#f58220"/>

          <!-- Green arrow -->
          <polygon points="215,20 250,40 215,60"
                   fill="#0f9d58"/>
        </svg>
        '''
    if "master" in category:
        return '<img src="https://upload.wikimedia.org/wikipedia/commons/2/2a/Mastercard-logo.svg" class="logo"/>'
    return ""

def account_details_page():
    st.title("👤 Account Details")

    if not st.session_state.logged_in:
        st.warning("🔐 Please login to view account details")
        return

    # ================= ACCOUNT INFO =================
    acc = get_account(st.session_state.account_no)

    st.subheader("📄 Account Information")
    st.markdown(f"""
    **Account Holder:** {acc[1]}  
    **Account Number:** {acc[0]}  
    **Account Type:** {acc[2]}  
    **Account Balance:** ₹{acc[3]:,.2f}
    """)

    st.markdown("---")

    # ================= CARDS SECTION =================
    st.subheader("💳 Your Cards")

    cards = get_cards(acc[0])

    if not cards:
        st.info("No cards added yet.")
    else:
        cols = st.columns(2)
        for i, card in enumerate(cards):
            with cols[i % 2]:
                gradient = get_card_gradient(card[2], card[3], card[7])
                logo_html = get_card_logo(card[3])

                components.html(
                    f"""
                    <style>
                    .bank-card {{
                        background: {gradient};
                        border-radius: 20px;
                        padding: 22px;
                        height: 220px;
                        color: white;
                        font-family: 'Segoe UI', sans-serif;
                        position: relative;
                        box-shadow: 0 16px 30px rgba(0,0,0,0.3);
                    }}

                    .chip {{
                        width: 50px;
                        height: 38px;
                        background: linear-gradient(135deg, #f7d774, #b38b2f);
                        border-radius: 6px;
                        margin-bottom: 20px;
                    }}

                    .logo {{
                        position: absolute;
                        top: 18px;
                        right: 22px;
                        height: 32px;
                    }}

                    .card-title {{
                        font-size: 18px;
                        font-weight: 600;
                        opacity: 0.9;
                        margin-bottom: 16px;
                    }}

                    .card-number {{
                        font-size: 18px;
                        letter-spacing: 2px;
                        margin-bottom: 14px;
                    }}

                    .card-footer {{
                        display: flex;
                        justify-content: space-between;
                        font-size: 18px;
                        opacity: 0.95;
                    }}

                    .status {{
                        position: absolute;
                        bottom: 18px;
                        left: 22px;
                        font-size: 18px;
                        font-weight: 600;
                    }}

                    .cvv {{
                        position: absolute;
                        bottom: 18px;
                        right: 22px;
                        font-size: 18px;
                    }}
                    </style>

                    <div class="bank-card">
                        {logo_html}
                        <div class="chip"></div>

                        <div class="card-title">
                            {card[2]} {card[3]} Card
                        </div>

                        <div class="card-number">
                            {card[0][0:]}
                        </div>

                        <div class="card-footer">
                            <div>
                                <b>{card[1]}</b>
                            </div>
                            <div>
                                EXP<br>{card[4]}/{card[5]}
                            </div>
                        </div>

                        <div class="status">
                            Status: {card[7]}
                        </div>

                        <div class="cvv">
                            CVV: ***
                        </div>
                    </div>
                    """,
                    height=280
                )





    st.markdown("---")

    # ================= ADD CARD =================
    st.subheader("➕ Add New Card")

    with st.form("add_card_form"):
        card_no = st.text_input("Card Number")
        holder = st.text_input("Card Holder Name")
        card_type = st.selectbox("Card Type", ["Debit", "Credit"])
        category = st.selectbox(
            "Card Category", ["RuPay", "VISA", "MasterCard", "Other"]
        )

        col1, col2 = st.columns(2)
        with col1:
            exp_month = st.selectbox(
                "Expiry Month", [f"{i:02d}" for i in range(1, 13)]
            )
        with col2:
            exp_year = st.selectbox(
                "Expiry Year", ["2025", "2026", "2027", "2028", "2029", "2030", "2031", "2032", "2033", "2034", "2035", "2036", "2037", "2038", "2039", "2040", "2041", "2042", "2043", "2044", "2045", "2046", "2047", "2048", "2049", "2050", "2051", "2052", "2053", "2054", "2055", "2056", "2057", "2058", "2059", "2060", "2061", "2062", "2063", "2064", "2065", "2066", "2067", "2068", "2069", "2070", "2071", "2072", "2073", "2074", "2075", "2076", "2077", "2078", "2079", "2080", "2081", "2082", "2083", "2084", "2085", "2086", "2087", "2088", "2089", "2090", "2091", "2092", "2093", "2094", "2095", "2096", "2097", "2098", "2099", "2100"]
            )

        cvv = st.text_input("CVV", type="password", max_chars=3)

        submit = st.form_submit_button("Add Card")
        

    if submit:
        if not all([card_no, holder, cvv]):
            st.warning("⚠ Please fill all required fields")
        elif not cvv.isdigit() or len(cvv) != 3:
            st.error("❌ CVV must be a 3-digit number")
        else:
            add_card(
                acc[0],
                card_no,
                holder,
                card_type,
                category,
                exp_month,
                exp_year
            )
            st.success("✅ Card added successfully")
            st.rerun()

st.markdown("""
<style>
/* ---- CONFETTI BASE ---- */
.confetti {
  position: fixed;
  bottom: 0;
  width: 10px;
  height: 20px;
  background-color: red;
  animation: confetti-fall 3s linear forwards;
  opacity: 0.9;
  z-index: 9999;
}

/* Left & Right origins */
.confetti.left {
  left: 5%;
  animation-name: confetti-left;
}
.confetti.right {
  right: 5%;
  animation-name: confetti-right;
}

/* Colors */
.confetti.c1 { background: #ff4d4d; }
.confetti.c2 { background: #4da6ff; }
.confetti.c3 { background: #4dff88; }
.confetti.c4 { background: #ffd24d; }
.confetti.c5 { background: #c77dff; }

/* Left animation */
@keyframes confetti-left {
  0% { transform: translate(0, 0) rotate(0deg); }
  100% { transform: translate(300px, -600px) rotate(720deg); opacity: 0; }
}

/* Right animation */
@keyframes confetti-right {
  0% { transform: translate(0, 0) rotate(0deg); }
  100% { transform: translate(-300px, -600px) rotate(-720deg); opacity: 0; }
}
</style>
""", unsafe_allow_html=True)

def confetti_blast():
    confetti_html = ""
    colors = ["c1", "c2", "c3", "c4", "c5"]

    for i in range(15):
        confetti_html += f"""
        <div class="confetti left {colors[i % 5]}" style="animation-delay:{i*0.05}s"></div>
        <div class="confetti right {colors[i % 5]}" style="animation-delay:{i*0.05}s"></div>
        """

    st.markdown(confetti_html, unsafe_allow_html=True)

st.markdown("""
<style>
/* ================= CONFETTI PARTICLES ================= */
.confetti-particle {
  position: fixed;
  width: 8px;
  height: 8px;
  border-radius: 2px;
  opacity: 0.9;
  z-index: 9999;
  animation-duration: 2.5s;
  animation-timing-function: ease-out;
  animation-fill-mode: forwards;
}

/* Some circles */
.confetti-circle {
  border-radius: 50%;
}

/* Colors */
.c-red { background: #ff4d4d; }
.c-blue { background: #4da6ff; }
.c-green { background: #4dff88; }
.c-yellow { background: #ffd24d; }
.c-purple { background: #c77dff; }
.c-cyan { background: #4dffff; }

/* LEFT explosion */
@keyframes explode-left {
  0%   { transform: translate(0, 0) rotate(0deg); opacity: 1; }
  100% { transform: translate(300px, -600px) rotate(720deg); opacity: 0; }
}

/* RIGHT explosion */
@keyframes explode-right {
  0%   { transform: translate(0, 0) rotate(0deg); opacity: 1; }
  100% { transform: translate(-300px, -600px) rotate(-720deg); opacity: 0; }
}
</style>
""", unsafe_allow_html=True)

def confetti_explosion():
    colors = ["c-red", "c-blue", "c-green", "c-yellow", "c-purple", "c-cyan"]
    html = ""

    for i in range(30):
        color = colors[i % len(colors)]
        shape = "confetti-circle" if i % 3 == 0 else ""
        delay = i * 0.02

        # LEFT SIDE
        html += f"""
        <div class="confetti-particle {color} {shape}"
             style="left:5%; bottom:0;
             animation-name: explode-left;
             animation-delay:{delay}s;">
        </div>
        """

        # RIGHT SIDE
        html += f"""
        <div class="confetti-particle {color} {shape}"
             style="right:5%; bottom:0;
             animation-name: explode-right;
             animation-delay:{delay}s;">
        </div>
        """

    st.markdown(html, unsafe_allow_html=True)

def admin_panel_page():
    st.title("🛠️ Admin Panel")

    # -------- LOGIN --------
    if not st.session_state.admin_logged:
        st.subheader("🔐 Admin Login")
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")

        if st.button("Login"):
            if u == "admin" and p == "admin123":
                st.session_state.admin_logged = True
                st.success("Admin logged in")
                st.rerun()
            else:
                st.error("Invalid credentials")
        return

    tab_dashboard, tab_logs, tab_faq, tab_suggestions, tab_analytics, tab_retrain = st.tabs([
    "📊 Dashboard",
    "🧾 User Logs",
    "📚 FAQ Manager",
    "🧠 FAQ Suggestions",
    "📈 Intent Analytics",
    "🔁 Retrain Model"
    ])

    # -------- DASHBOARD --------
    with tab_dashboard:
        st.markdown("""
        <div class="section-box">
            <div class="section-title">📊 Dashboard Overview</div>
            <div class="section-sub">Real-time chatbot usage metrics</div>
        </div>
        """, unsafe_allow_html=True)

        conn = get_conn()
        df = pd.read_sql("SELECT * FROM chat_logs", conn)
        conn.close()
        df["timestamp"] = pd.to_datetime(df["timestamp"])


        st.markdown("""
        <style>
        .section-box {
            background: #f8f9fa;
            padding: 22px;
            border-radius: 18px;
            margin-bottom: 28px;
            box-shadow: 0 8px 18px rgba(0,0,0,0.08);
        }
        .section-title {
            font-size: 20px;
            font-weight: 700;
            margin-bottom: 14px;
        }
        .section-sub {
            font-size: 14px;
            color: #555;
            margin-bottom: 10px;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown("""
        <style>
        .kpi-card {
            padding: 22px;
            border-radius: 18px;
            color: white;
            height: 150px;
            transition: all 0.3s ease-in-out;
        }
        .kpi-card:hover {
            transform: translateY(-6px);
            box-shadow: 0 18px 35px rgba(0,0,0,0.25);
        }
        .kpi-title {
            font-size: 16px;
            opacity: 1.5;
        }
        .kpi-value {
            font-size: 28px;
            font-weight: 700;
            margin-top: 10px;
        }
        .kpi-icon {
            font-size: 30px;
        }
        .blue { background: linear-gradient(135deg, #1e3c72, #2a5298); }
        .green { background: linear-gradient(135deg, #11998e, #38ef7d); }
        .purple { background: linear-gradient(135deg, #8e2de2, #4a00e0); }
        .orange { background: linear-gradient(135deg, #f7971e, #ffd200); color: white; }
        </style>
        """, unsafe_allow_html=True)

        total_queries = len(df)
        active_users = df["account_no"].nunique() if not df.empty else 0
        unique_intents = df["intent"].nunique() if not df.empty else 0
        last_activity = df["timestamp"].iloc[-1] if not df.empty else "N/A"
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="kpi-card blue">
                <div class="kpi-icon">📊</div>
                <div class="kpi-title">Total Queries</div>
                <div class="kpi-value">{total_queries}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="kpi-card green">
                <div class="kpi-icon">👥</div>
                <div class="kpi-title">Active Users</div>
                <div class="kpi-value">{active_users}</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="kpi-card purple">
                <div class="kpi-icon">🧠</div>
                <div class="kpi-title">Unique Intents</div>
                <div class="kpi-value">{unique_intents}</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="kpi-card orange">
                <div class="kpi-icon">🕒</div>
                <div class="kpi-title">Last Activity</div>
                <div class="kpi-value">{last_activity}</div>
            </div>
            """, unsafe_allow_html=True)
        
        
        st.markdown("""
        <div class="section-box">
            <div class="section-title">🔍 Filters</div>
            <div class="section-sub">Refine analytics by date or intent</div>
        </div>
        """, unsafe_allow_html=True)


        f1, f2, f3 = st.columns(3)

        with f1:
            start_date = st.date_input(
                "From Date",
                value=df["timestamp"].min().date() if not df.empty else None
            )
        with f2:
            end_date = st.date_input(
                "To Date",
                value=df["timestamp"].max().date() if not df.empty else None
            )
        with f3:
            intent_filter = st.selectbox(
                "Intent",
                options=["All"] + sorted(df["intent"].unique().tolist())
            if not df.empty else ["All"]
            )
        
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            if start_date:
                df = df[df["timestamp"].dt.date >= start_date]
            if end_date:
                df = df[df["timestamp"].dt.date <= end_date]
            if intent_filter != "All":
                df = df[df["intent"] == intent_filter]

        st.markdown("""
        <div class="section-box">
            <div class="section-title">📊 Intent Distribution</div>
            <div class="section-sub">Most frequently used chatbot intents</div>
        </div>
        """, unsafe_allow_html=True)

        


        if not df.empty:
            intent_counts = df["intent"].value_counts()

            st.bar_chart(intent_counts)
        else:
            st.info("No data available yet.")

        st.markdown("""
        <div class="section-box">
            <div class="section-title">📈 Queries Over Time</div>
            <div class="section-sub">Daily chatbot usage trend</div>
        </div>
        """, unsafe_allow_html=True)

        


        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            trend = df.groupby(df["timestamp"].dt.date).size()

            st.line_chart(trend)
        else:
            st.info("No trend data available yet.")



    
        

    # -------- USER LOGS --------
    with tab_logs:
        st.subheader("🧾 User Logs")
        conn = get_conn()
        df = pd.read_sql("SELECT * FROM chat_logs", conn)
        conn.close()

        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Download CSV",
            df.to_csv(index=False),
            "chat_logs.csv"
        )

    


    # -------- FAQ MANAGER --------
    with tab_faq:
        st.subheader("📚 FAQ Manager")

        # ---------- ADD FAQ SECTION ----------
        st.markdown("""
        <div class="section-box">
            <div class="section-title">➕ Add New FAQ</div>
            <div class="section-sub">Create knowledge base answers for the chatbot</div>
        </div>
        """, unsafe_allow_html=True)

        with st.form("add_faq_form"):
            q = st.text_input("❓ Question")
            a = st.text_area("✍️ Answer")
            submit = st.form_submit_button("Add FAQ")

        if submit:
            if not q or not a:
                st.warning("Please fill both Question and Answer")
            else:
                conn = get_conn()
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO faqs (question, answer) VALUES (?, ?)",
                    (q, a)
                )
                conn.commit()
                conn.close()
                st.success("✅ FAQ added successfully")
                st.rerun()

        # ---------- EXISTING FAQS ----------
        st.markdown("""
        <div class="section-box">
            <div class="section-title">📖 Existing FAQs</div>
            <div class="section-sub">Edit or remove existing knowledge base entries</div>
        </div>
        """, unsafe_allow_html=True)

        conn = get_conn()
        faqs = pd.read_sql("SELECT * FROM faqs", conn)
        conn.close()

        if faqs.empty:
            st.info("No FAQs added yet.")
        else:
            for _, row in faqs.iterrows():
                with st.expander(f"❓ {row['question']}"):
                    new_answer = st.text_area(
                        "Update Answer",
                        value=row["answer"],
                        key=f"faq_{row['id']}"
                    )

                    col1, col2 = st.columns([1, 1])

                    with col1:
                        if st.button("💾 Update", key=f"upd_{row['id']}"):
                            conn = get_conn()
                            cur = conn.cursor()
                            cur.execute(
                                "UPDATE faqs SET answer=? WHERE id=?",
                                (new_answer, row["id"])
                            )
                            conn.commit()
                            conn.close()
                            st.success("Updated successfully")
                            st.rerun()

                    with col2:
                        if st.button("🗑 Delete", key=f"del_{row['id']}"):
                            conn = get_conn()
                            cur = conn.cursor()
                            cur.execute(
                                "DELETE FROM faqs WHERE id=?",
                                (row["id"],)
                            )
                            conn.commit()
                            conn.close()
                            st.warning("FAQ deleted")
                            st.rerun()
    
    with tab_suggestions:
        st.markdown("""
            <div class="section-box">
                <div class="section-title">🧠 FAQ Suggestions (Human Review)</div>
                <div class="section-sub">
                     Approval or rejection of user-suggested FAQs for chatbot knowledge base
                </div>
            </div>
            """, unsafe_allow_html=True)

        conn = get_conn()
        df = pd.read_sql("""
            SELECT * FROM faq_suggestions
            WHERE status='PENDING'
            ORDER BY frequency DESC
        """, conn)
        conn.close()

        if df.empty:
            st.success("No pending FAQ suggestions 🎉")
        else:
            for _, row in df.iterrows():
                with st.expander(f"❓ {row['question']} (Asked {row['frequency']} times)"):
                    st.write(f"Average Confidence: {round(row['avg_confidence'], 2)}")
                    answer = st.text_area(
                        "Approved Answer",
                        key=f"ans_{row['id']}"
                    )

                    col1, col2 = st.columns(2)

                    with col1:
                        if st.button("✅ Approve", key=f"app_{row['id']}"):
                            conn = get_conn()
                            cur = conn.cursor()

                            cur.execute(
                                "INSERT INTO faqs (question, answer) VALUES (?, ?)",
                                (row["question"], answer)
                            )
                            cur.execute(
                                "UPDATE faq_suggestions SET status='APPROVED' WHERE id=?",
                                (row["id"],)
                            )

                            conn.commit()
                            conn.close()
                            st.success("FAQ approved & published")
                            st.rerun()

                    with col2:
                        if st.button("❌ Reject", key=f"rej_{row['id']}"):
                            conn = get_conn()
                            cur = conn.cursor()
                            cur.execute(
                                "UPDATE faq_suggestions SET status='REJECTED' WHERE id=?",
                                (row["id"],)
                            )
                            conn.commit()
                            conn.close()
                            st.warning("FAQ rejected")
                            st.rerun()


    # -------- INTENT ANALYTICS --------
    with tab_analytics:
        st.subheader("📈 Intent Analytics")

        conn = get_conn()
        df = pd.read_sql("SELECT * FROM chat_logs", conn)
        conn.close()
        if df.empty:
            st.info("No intent data available yet.")
            st.stop()

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        intent_counts = df["intent"].value_counts()

    # ---------- ROW 1 ----------
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="section-box">
                <div class="section-title">📊 Intent Usage Frequency</div>
                <div class="section-sub">How often each intent is triggered</div>
            </div>
             """, unsafe_allow_html=True)

            st.bar_chart(intent_counts)

        with col2:
            st.markdown("""
            <div class="section-box">
                <div class="section-title">🥧 Intent Distribution (%)</div>
                <div class="section-sub">Relative share of intents</div>
            </div>
            """, unsafe_allow_html=True)

            fig1, ax1 = plt.subplots()
            ax1.pie(
                intent_counts,
                labels=intent_counts.index,
                autopct="%1.1f%%",
                startangle=90
            )
            ax1.axis("equal")
            st.pyplot(fig1)

    # ---------- ROW 2 ----------
        col3, col4 = st.columns(2)

        with col3:
            st.markdown("""
            <div class="section-box">
                <div class="section-title">📈 Intent Trend Over Time</div>
                <div class="section-sub">Daily usage trend of intents</div>
            </div>
            """, unsafe_allow_html=True)

            trend = (
                df.groupby([df["timestamp"].dt.date, "intent"])
                 .size()
                 .unstack(fill_value=0)
            )
            st.line_chart(trend)

        with col4:
            st.markdown("""
            <div class="section-box">
                <div class="section-title">🎯 Confidence Distribution</div>
                <div class="section-sub">Model confidence across predictions</div>
            </div>
            """, unsafe_allow_html=True)

            fig2, ax2 = plt.subplots()
            ax2.hist(df["confidence"], bins=10)
            ax2.set_xlabel("Confidence Score")
            ax2.set_ylabel("Query Count")
            st.pyplot(fig2)
    # ---------- INSIGHTS ----------
        st.markdown("---")
        st.markdown("### 🧠 Key Insights")

        st.success(f"""
        ✔ Most frequent intent: **{intent_counts.idxmax()}**  
        ✔ Average confidence: **{round(df['confidence'].mean(), 2)}**  
        ✔ Total intents detected: **{df['intent'].nunique()}**
        """)

    if "intent_added" not in st.session_state:
        st.session_state.intent_added = False

    if "example_added" not in st.session_state:
        st.session_state.example_added = False

    
    
    # -------- RETRAIN --------
    with tab_retrain:

        # ================= HEADER =================
        

        st.markdown("""
        <div class="section-box">
            <div class="section-title">🔁 Retrain Model</div>
            <div class="section-sub">
                Edit intents, analyze predictions, train and retrain the NLU model
            </div>
        </div>
        """, unsafe_allow_html=True)


        intents_data = load_intents()

        # ================= TOP ROW =================
        left_col, right_col = st.columns([3, 2])

        # =====================================================
        # LEFT SIDE — EDIT & TRAIN INTENTS
        # =====================================================
        with left_col:
            st.markdown("""
            <div class="section-box">
                <div class="section-title">✏️ Edit & Train Intents</div>
                <div class="section-sub">
                     Modify existing intents and their training examples
                </div>
            </div>
            """, unsafe_allow_html=True)


            for idx, intent in enumerate(intents_data["intents"]):
                intent_name = intent["name"]
                examples = intent["examples"]

                with st.expander(f"{intent_name} ({len(examples)} examples)"):
                    # -------- EDIT EXAMPLES --------
                    updated_examples = st.text_area(
                        "Examples (one per line)",
                        value="\n".join(examples),
                        key=f"edit_{idx}_{intent_name}"
                    )

                    intent["examples"] = [
                        e.strip() for e in updated_examples.split("\n") if e.strip()
                    ]

                    # -------- DELETE INTENT --------
                    st.markdown("---")
                    col_del1, col_del2 = st.columns([3, 1])

                    with col_del2:
                        if st.button(
                            "🗑️ Delete Intent",
                            key=f"delete_intent_{idx}"
                        ):
                            st.session_state.intent_to_delete = idx

            # -------- DELETE CONFIRMATION --------
            if "intent_to_delete" not in st.session_state:
                st.session_state.intent_to_delete = None

            if st.session_state.intent_to_delete is not None:
                del_idx = st.session_state.intent_to_delete
                del_name = intents_data["intents"][del_idx]["name"]

                st.warning(f"Are you sure you want to delete intent **{del_name}**?")

                col1, col2 = st.columns(2)

                with col1:
                    if st.button("❌ Cancel", key="cancel_delete_intent"):
                        st.session_state.intent_to_delete = None

                with col2:
                    if st.button("✅ Confirm Delete", key="confirm_delete_intent"):
                        intents_data["intents"].pop(del_idx)
                        save_intents(intents_data)
                        st.session_state.intent_to_delete = None
                        st.success(f"Intent **{del_name}** deleted successfully")
                        st.rerun()

            intent_names = [i["name"] for i in intents_data["intents"]]
            if len(intent_names) != len(set(intent_names)):
                st.error("Duplicate intent names found. Intent names must be unique.")
                

            if st.button("💾 Save"):
                save_intents(intents_data)
                st.success("✅ Intents saved successfully")


        # =====================================================
        # RIGHT SIDE — NLU VISUALIZER
        # =====================================================
        with right_col:
            st.markdown("""
            <div class="section-box">
                <div class="section-title">🔍 NLU Visualizer</div>
                <div class="section-sub">
                    Analyze intent prediction, confidence, and entities
                </div>
            </div>
            """, unsafe_allow_html=True)

            query = st.text_area(
                "User Query",
                placeholder="Move ₹1500 to account 12345678"
            )

            top_k = st.number_input(
            "Top intents to show",
            min_value=1,
            max_value=5,
            value=4
            )

            if st.button("Analyze"):
                intent, confidence, entities = predict_intent(query, top_k=top_k)

                st.markdown("#### 🎯 Intent Recognition")
                st.write(f"**{intent}**  —  confidence: `{round(confidence, 3)}`")
                st.markdown("#### 🧩 Entity Extraction")
                if entities:
                    st.json(entities)
                else:
                    st.info("No entities detected")

        # ================= SECOND ROW =================
        left_col2, right_col2 = st.columns([3, 2])

        # =====================================================
        # LEFT BOTTOM — ADD NEW INTENT + QUICK ADD EXAMPLE
        # =====================================================
        with left_col2:
            st.markdown("""
            <div class="section-box">
                <div class="section-title">➕ Add New Intent</div>
                <div class="section-sub">
                    Create a new intent and add training examples
                </div>
            </div>
            """, unsafe_allow_html=True)


            new_intent_name = st.text_input("Intent name ")
            new_intent_examples = st.text_area("Examples (one per line)")

            if st.button("Add Intent"):
                if not new_intent_name:
                    st.error("Intent name is required")
                else:
                    intents_data["intents"].append({
                        "name": new_intent_name,
                        "examples": [
                            e.strip()
                            for e in new_intent_examples.split("\n")
                            if e.strip()
                        ]
                    })
                    save_intents(intents_data)
                    st.session_state.intent_added = True
                    st.rerun()
            if st.session_state.intent_added:
                st.success("✅ New intent added successfully")
                st.session_state.intent_added = False

                    
            st.markdown("""
            <div class="section-box">
                <div class="section-title">⚡ Quick Add Example</div>
                <div class="section-sub">
                    Add a new example to an existing intent
                </div>
            </div>
            """, unsafe_allow_html=True)


            intent_names = [i["name"] for i in intents_data["intents"]]
            selected_intent = st.selectbox("Select intent", intent_names)
            quick_example = st.text_input("Example sentence")

            if st.button("Add Example"):
                for i in intents_data["intents"]:
                    if i["name"] == selected_intent:
                        i["examples"].append(quick_example)

                save_intents(intents_data)
                st.session_state.example_added = True
                st.rerun()
            
            if st.session_state.example_added:
                st.success("✅ Example added to intent successfully")

                st.session_state.example_added = False


        # =====================================================
        # RIGHT BOTTOM — TRAIN MODEL
        # =====================================================
        with right_col2:
            st.markdown("""
            <div class="section-box">
                <div class="section-title">⚙️ Train Model</div>
                <div class="section-sub">
                    Train the intent classifier using current configuration
                </div>
            </div>
            """, unsafe_allow_html=True)


            epochs = st.number_input("Epochs", min_value=1, value=2)
            batch_size = st.number_input("Batch size", min_value=1, value=8)
            lr = st.number_input("Learning rate", value=0.001, format="%.4f")

            if st.button("🚀 Start Training"):
                progress = st.progress(0)
                status = st.empty()

                losses = []
                total_epochs = epochs

                conn = get_conn()
                cur = conn.cursor()

                for ep in range(1, total_epochs + 1):
                    status.info(f"Training epoch {ep}/{total_epochs}")

                    # ---- Simulated decreasing loss ----
                    loss = round(random.uniform(0.8, 1.2) / ep, 4)
                    losses.append(loss)

                    cur.execute(
                        "INSERT INTO training_logs (stage, epoch, loss, timestamp) VALUES (?, ?, ?, ?)",
                        ("TRAIN", ep, loss, ist_now())
                    )

                    progress.progress(int((ep / total_epochs) * 100))
                    time.sleep(1)

                conn.commit()
                conn.close()

                status.success("Training completed")
                st.success("✅ Model training completed")
                st.balloons()

                

                # -------- LOSS CHART --------
                st.markdown("### 📊 Epoch-wise Training Loss")
                loss_df = pd.DataFrame({
                    "Epoch": list(range(1, total_epochs + 1)),
                    "Loss": losses
                }).set_index("Epoch")

                st.line_chart(loss_df)



        # =====================================================
        # FULL WIDTH — MODEL RETRAINING
        # =====================================================
        st.markdown("---")
        st.markdown("""
        <div class="section-box">
            <div class="section-title">🔁 Model Retraining</div>
            <div class="section-sub">
                Retrain the deployed NLU model using approved intents and logs
            </div>
        </div>
        """, unsafe_allow_html=True)


        st.warning(
            "Retraining updates the deployed NLU model using latest intents "
            "and approved chatbot data."
        )

        confirm = st.checkbox("I understand retraining impact")

        # ----- RETRAIN BUTTON -----
        retrain_clicked = st.button("🔁 Retrain Model", key="retrain_model_btn")

        if retrain_clicked:
            if not confirm:
                st.warning("Please confirm before retraining the model.")
            else:
                progress = st.progress(0)
                status = st.empty()

                steps = [
                    "Loading latest intents",
                    "Preparing training dataset",
                    "Fine-tuning model",
                    "Validating model",
                    "Deploying updated model"
                ]

                losses = []

                conn = get_conn()
                cur = conn.cursor()

                total = len(steps)

                for idx, step in enumerate(steps, start=1):
                    status.info(step)

                    # ---- Simulated loss ----
                    loss = round(random.uniform(0.5, 1.0) / idx, 4)
                    losses.append(loss)

                    # 🔥 LOGGING (THIS WAS GETTING SKIPPED BEFORE)
                    cur.execute(
                        """
                        INSERT INTO training_logs (stage, epoch, loss, timestamp)
                        VALUES (?, ?, ?, ?)
                        """,
                        ("RETRAIN", idx, loss, ist_now())
                    )

                    progress.progress(int((idx / total) * 100))
                    time.sleep(1)

                conn.commit()
                conn.close()

                status.success("Retraining completed successfully")
                st.success("✅ Model retrained successfully")
                st.balloons()

        

            

            


        #Training Logs
        st.markdown("---")
        st.subheader("🧾 Training Logs")

        conn = get_conn()
        logs_df = pd.read_sql(
            "SELECT stage, epoch, loss, timestamp FROM training_logs ORDER BY timestamp DESC",
            conn
        )
        conn.close()

        if logs_df.empty:
            st.info("No training logs available yet.")
        else:
            st.dataframe(logs_df, use_container_width=True)



def log_faq_suggestion(user_text, confidence):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        "SELECT id, frequency FROM faq_suggestions WHERE question=?",
        (user_text,)
    )
    row = cur.fetchone()

    if row:
        cur.execute("""
            UPDATE faq_suggestions
            SET frequency = frequency + 1,
                last_asked = ist_now(),
            WHERE id=?
        """, (row[0],)) 
    else:
        cur.execute("""
            INSERT INTO faq_suggestions (question, avg_confidence, last_asked)
            VALUES (?, ?, ?)
        """, (user_text, confidence, ist_now()))
    conn.commit()
    conn.close()

# ================= ROUTER =================
if page == "Home":
    home_page()
elif page == "Login / Create Account":
    login_page()
elif page == "NLU Visualizer":
    nlu_page()
elif page == "Chatbot":
    chatbot_page()
elif page == "Transaction History":
    transaction_history_page()
elif page == "Account Details":
    account_details_page()
elif page == "Admin Panel":
    admin_panel_page()


