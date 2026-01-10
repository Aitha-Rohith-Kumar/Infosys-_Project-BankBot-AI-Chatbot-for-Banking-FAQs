from database.db import get_conn
from database.security import hash_password, verify_password
from datetime import datetime

def create_account(name, acc_no, acc_type, balance, password):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("INSERT OR IGNORE INTO users(name) VALUES (?)", (name,))
    pwd_hash = hash_password(password)

    cur.execute("""
    INSERT INTO accounts(account_number, user_name, account_type, balance, password_hash)
    VALUES (?, ?, ?, ?, ?)
    """, (acc_no, name, acc_type, balance, pwd_hash))

    conn.commit()
    conn.close()

def get_account(acc_no):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    SELECT account_number, user_name, account_type, balance, password_hash
    FROM accounts WHERE account_number=?
    """, (acc_no,))
    row = cur.fetchone()
    conn.close()
    return row

def list_accounts():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT account_number, user_name FROM accounts")
    rows = cur.fetchall()
    conn.close()
    return rows

def transfer_money(from_acc, to_acc, amount, password):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT balance, password_hash FROM accounts WHERE account_number=?", (from_acc,))
    row = cur.fetchone()
    if not row:
        return "❌ Invalid sender account"

    balance, pwd_hash = row
    if not verify_password(password, pwd_hash):
        return "❌ Incorrect password"

    if balance < amount:
        return "❌ Insufficient balance"

    # Transaction (ACID)
    cur.execute("UPDATE accounts SET balance = balance - ? WHERE account_number=?", (amount, from_acc))
    cur.execute("UPDATE accounts SET balance = balance + ? WHERE account_number=?", (amount, to_acc))

    cur.execute("""
    INSERT INTO transactions(from_account, to_account, amount, timestamp)
    VALUES (?, ?, ?, ?)
    """, (from_acc, to_acc, amount, datetime.now().isoformat()))

    conn.commit()
    conn.close()
    return "✅ Transfer Successful"

def get_transaction_history(account_no):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT from_account, to_account, amount, timestamp
        FROM transactions
        WHERE from_account = ? OR to_account = ?
        ORDER BY timestamp DESC
    """, (account_no, account_no))

    rows = cur.fetchall()
    conn.close()
    return rows


from datetime import datetime


def add_card(account_no, card_no, holder, card_type, category, exp_month, exp_year):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO cards (
            account_number, card_number, holder_name,
            card_type, card_category,
            expiry_month, expiry_year,
            cvv_masked, status, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, '***', 'ACTIVE', ?)
    """, (
        account_no, card_no, holder,
        card_type, category,
        exp_month, exp_year,
        datetime.now().isoformat()
    ))

    conn.commit()
    conn.close()


def get_cards(account_no):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT
            card_number,
            holder_name,
            card_type,
            card_category,
            expiry_month,
            expiry_year,
            cvv_masked,
            status
        FROM cards
        WHERE account_number = ?
    """, (account_no,))

    rows = cur.fetchall()
    conn.close()
    return rows




def block_cards(account_no):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        UPDATE cards
        SET status = 'BLOCKED'
        WHERE account_number = ?
    """, (account_no,))

    conn.commit()
    conn.close()

def block_all_cards(account_no):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        UPDATE cards
        SET status = 'BLOCKED'
        WHERE account_number = ?
    """, (account_no,))

    conn.commit()
    conn.close()


def block_card_by_number(account_no, last6):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        UPDATE cards
        SET status = 'BLOCKED'
        WHERE account_number = ?
        AND substr(card_number, -6) = ?
    """, (account_no, last6))

    conn.commit()
    conn.close()

def block_card_by_last4(account_no, last4):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT card_number FROM cards
        WHERE account_number = ?
          AND status = 'ACTIVE'
          AND substr(card_number, -4) = ?
        """,
        (account_no, last4)
    )
    card = cur.fetchone()

    if not card:
        conn.close()
        return "❌ No active card found with those last 4 digits."

    cur.execute(
        """
        UPDATE cards
        SET status = 'BLOCKED'
        WHERE account_number = ? AND card_number = ?
        """,
        (account_no, card[0])
    )

    conn.commit()
    conn.close()

    return f"🚨 Card ending with **{last4}** has been blocked."

def block_cards_by_category(account_no, category):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        UPDATE cards
        SET status = 'BLOCKED'
        WHERE account_number = ?
        AND card_category = ?
    """, (account_no, category))

    conn.commit()
    conn.close()

def unblock_card_by_last6(account_no, last6_digits, password):
    conn = get_conn()
    cur = conn.cursor()

    # ---- Verify account password ----
    cur.execute(
        "SELECT password_hash FROM accounts WHERE account_number = ?",
        (account_no,)
    )
    row = cur.fetchone()
    if not row or not verify_password(password, row[0]):
        conn.close()
        return "❌ Incorrect password. Unblock failed."

    # ---- Find BLOCKED card matching last 6 digits ----
    cur.execute(
        """
        SELECT card_number FROM cards
        WHERE account_number = ?
          AND status = 'BLOCKED'
          AND substr(card_number, -6) = ?
        """,
        (account_no, last6_digits)
    )
    card = cur.fetchone()

    if not card:
        conn.close()
        return "❌ No blocked card found with those last 6 digits."

    # ---- Unblock card ----
    cur.execute(
        """
        UPDATE cards
        SET status = 'ACTIVE'
        WHERE account_number = ? AND card_number = ?
        """,
        (account_no, card[0])
    )

    conn.commit()
    conn.close()

    return f"✅ Card ending with **{last6_digits}** has been successfully unblocked."

def block_card_by_last6_secure(account_no, last6, password):
    conn = get_conn()
    cur = conn.cursor()

    # Verify password
    cur.execute(
        "SELECT password_hash FROM accounts WHERE account_number = ?",
        (account_no,)
    )
    row = cur.fetchone()
    if not row or not verify_password(password, row[0]):
        conn.close()
        return "❌ Incorrect password. Card block failed."

    # Find active card by last 6 digits
    cur.execute(
        """
        SELECT card_number FROM cards
        WHERE account_number = ?
          AND status = 'ACTIVE'
          AND substr(card_number, -6) = ?
        """,
        (account_no, last6)
    )
    card = cur.fetchone()

    if not card:
        conn.close()
        return "❌ No active card found with those last 6 digits."

    # Block card
    cur.execute(
        """
        UPDATE cards
        SET status = 'BLOCKED'
        WHERE account_number = ? AND card_number = ?
        """,
        (account_no, card[0])
    )

    conn.commit()
    conn.close()

    return f"🚨 Card ending with **{last6}** has been blocked successfully."
