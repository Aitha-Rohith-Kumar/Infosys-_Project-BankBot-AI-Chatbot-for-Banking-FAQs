import sqlite3
import bcrypt
from datetime import datetime

DB_NAME = "bankbot.db"

def get_conn():
    return sqlite3.connect(DB_NAME, check_same_thread=False)

def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS accounts (
        account_number TEXT PRIMARY KEY,
        user_name TEXT,
        account_type TEXT,
        balance INTEGER,
        password_hash BLOB,
        FOREIGN KEY(user_name) REFERENCES users(name)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        from_account TEXT,
        to_account TEXT,
        amount INTEGER,
        timestamp TEXT
    )
    """)

    

    cur.execute("""
    CREATE TABLE IF NOT EXISTS cards (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account_number TEXT,
    card_number TEXT,
    holder_name TEXT,
    card_type TEXT,        
    card_category TEXT,
    expiry_month TEXT,
    expiry_year TEXT,
    cvv_masked TEXT,    
    status TEXT DEFAULT 'ACTIVE',
    created_at TEXT,
    FOREIGN KEY(account_number) REFERENCES accounts(account_number)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS faq_suggestions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT UNIQUE,
        frequency INTEGER DEFAULT 1,
        avg_confidence REAL,
        last_asked TEXT,
        status TEXT DEFAULT 'PENDING'
    )
    """)



    cur.execute("""
        CREATE TABLE IF NOT EXISTS training_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stage TEXT,
            epoch INTEGER,
            loss REAL,
            timestamp TEXT
        )
    """)

    # Chat Logs
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            account_no TEXT,
            user_query TEXT,
            intent TEXT,
            confidence REAL
        )
    """)

    # FAQs
    cur.execute("""
        CREATE TABLE IF NOT EXISTS faqs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            answer TEXT
        )
    """)


    conn.commit()
    conn.close()


