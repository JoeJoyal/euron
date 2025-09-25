# db.py
import sqlite3
from src.config import DB_PATH

def ensure_schema():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS invoices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT,
            factory TEXT,
            order_type TEXT,
            proforma_no TEXT,
            date TEXT,
            bill_to TEXT,
            delivery_to TEXT,
            place_of_supply TEXT,
            state_code TEXT,
            gstin_no TEXT,
            sales_reference TEXT,
            terms_conditions TEXT,
            raw_json TEXT
        )
    """)
    con.commit()
    con.close()

def fetch_all():
    import pandas as pd
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM invoices", con)
    con.close()
    return df

def fetch_one_by_id(rowid):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT * FROM invoices WHERE id=?", (rowid,))
    row = cur.fetchone()
    con.close()
    return row

ensure_schema()