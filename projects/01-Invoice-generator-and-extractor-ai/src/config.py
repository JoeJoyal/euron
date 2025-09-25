# config.py
import os

EURI_API_KEY = "euri-7ef3fcb8f73531cef0506158e5d52c7a857ae04f6f3b8c96333539fdd0715904"
MODEL = "gpt-4.1-nano"
INPUT_DIR = "invoice"
DB_PATH = "invoices.sqlite"
PROCESSED_LOG = "processed.json"
POLL_SEC = 5
OCR_LANGS = ['en']

os.makedirs(INPUT_DIR, exist_ok=True)