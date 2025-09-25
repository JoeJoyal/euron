import easyocr
import os, re, json, io
from typing import Dict, Any, List
from euriai.langgraph import EuriaiLangGraph
from src.config import (EURI_API_KEY, MODEL, INPUT_DIR, DB_PATH, PROCESSED_LOG, OCR_LANGS)
from src.db import ensure_schema
from wand.image import Image as WandImage
from PIL import Image as PILImage

ocr = easyocr.Reader(OCR_LANGS, gpu=False)

# --- OCR Node handling PDF ---
def pdf_to_images(pdf_path_or_file, dpi=300):
    """
    Convert the PDF to a list of PIL images using Wand/ImageMagick.
    Accepts path or file-like.
    """
    images = []
    if hasattr(pdf_path_or_file, 'read'):
        f = pdf_path_or_file
        f.seek(0)
        with WandImage(file=f, resolution=dpi) as w_pdf:
            for page in w_pdf.sequence:
                with WandImage(page) as img:
                    img.format = 'png'
                    with io.BytesIO() as output:
                        img.save(file=output)
                        output.seek(0)
                        pil_img = PILImage.open(output).convert("RGB")
                        images.append(pil_img)
    else:
        with WandImage(filename=pdf_path_or_file, resolution=dpi) as w_pdf:
            for page in w_pdf.sequence:
                with WandImage(page) as img:
                    img.format = 'png'
                    with io.BytesIO() as output:
                        img.save(file=output)
                        output.seek(0)
                        pil_img = PILImage.open(output).convert("RGB")
                        images.append(pil_img)
    return images


def load_seen() -> set:
    if not os.path.exists(PROCESSED_LOG): return set()
    try:
        with open(PROCESSED_LOG, "r", encoding='utf-8') as f: return set(json.load(f))
    except Exception: return set()

def save_seen(seen: set) -> None:
    with open(PROCESSED_LOG, "w", encoding="utf-8") as f:
        json.dump(sorted(list(seen)), f, ensure_ascii=False, indent=2)

clean_graph = EuriaiLangGraph(api_key=EURI_API_KEY, default_model=MODEL)
clean_graph.add_ai_node(
    "CLEAN",
    """You clean noisy OCR text from invoicesDB into structured JSON data with the following fields:
    - factory
    - order_type
    - proforma_no
    - date
    - bill_to
    - delivery_to
    - place_of_supply
    - state_code
    - gstin_no
    - sales_reference
    - terms_conditions
    - Keep facts only, do not add any explanations.
    - No guessing, if something is not found, use empty string.
    - Keep table rows readable.
    OCR:
    {ocr_text}"""
)
clean_graph.set_entry_point("CLEAN")
clean_graph.set_finish_point("CLEAN")

extract_graph = EuriaiLangGraph(api_key=EURI_API_KEY, default_model=MODEL)
extract_graph.add_ai_node(
    "EXTRACT",
    """From CLEAN_TEXT, return STRICT JSON with the following fields:
    - factory
    - order_type
    - proforma_no
    - date
    - bill_to
    - delivery_to
    - place_of_supply
    - state_code
    - gstin_no
    - sales_reference
    - terms_conditions
    Unknown -> null. Numbers numeric. Dates YYYY-MM-DD if possible.
    - keep facts only, do not add any explanations.
    CLEAN_TEXT:
    {clean_text}"""
)
extract_graph.set_entry_point("EXTRACT")
extract_graph.set_finish_point("EXTRACT")

def pick_text(x, *, prefer_key=None):
    """Return a plain string from various possible structures.
       If x is a dict, try prefer_key or common keys; else stringify.
    """
    if isinstance(x, str): return x
    if isinstance(x, dict):
        if prefer_key and prefer_key in x and isinstance(x[prefer_key], str):
            return x[prefer_key]
        for k in ("output", "text", "CLEAN_output", "EXTRACT_output"):
            if k in x and isinstance(x[k], str): return x[k]
        return json.dumps(x, ensure_ascii=False)
    return str(x)

def parse_json_safe(raw):
    """Parse JSON robustly. Accepts dict or str; falls back to substring."""
    if isinstance(raw, dict): return raw
    if not isinstance(raw, str): return {"__raw__": raw}
    try: return json.loads(raw)
    except Exception: pass
    try:
        s, e = raw.index("{"), raw.rfind("}")
        if s != -1 and e !=-1 and e > s:
            return json.loads(raw[s:e+1])
    except Exception: pass
    return {"__raw__": raw}

def _heuristic_extract(clean_text: str) -> dict:
    """Very simple regex-based extractor to keep DB flowing when AI is down."""
    def find(pat, s):
        m = re.search(pat, s, re.IGNORECASE | re.DOTALL)
        return m.group(1).strip() if m else None
    
    factory = find(r"Factory\s*:\s*(.+?)\n", clean_text)
    order_type = find(r"Order Type\s*:\s*([^\n]+)", clean_text)
    proforma_no = find(r"Proforma No\s*:\s*([^\n]+)", clean_text)
    date = find(r"Date\s*:\s*([0-9\-]+)", clean_text)
    bill_to = find(r"BILL TO\s*\n([\s\S]+?)\n(?:Ph|DELIVERY TO)", clean_text)
    delivery_to = find(r"DELIVERY TO\s*\n([\s\S]+?)(?=\nState|$)", clean_text)
    place_of_supply = find(r"Place of Supply\s*([a-zA-Z\s]+)", clean_text)
    state_code = find(r"State Code\s*:\s*([0-9]+)", clean_text)
    gstin_no = find(r"GSTIN No\s*:\s*([A-Za-z0-9]+)", clean_text)
    sales_reference = find(r"Sales Reference\s*:\s*([^\n]+)", clean_text)
    terms_conditions = find(r"NOTE\s*:\s*\n([\s\S]+)", clean_text)

    return {
        "factory": factory,
        "order_type": order_type,
        "proforma_no": proforma_no,
        "date": date,
        "bill_to": bill_to,
        "delivery_to": delivery_to,
        "place_of_supply": place_of_supply,
        "state_code": state_code,
        "gstin_no": gstin_no,
        "sales_reference": sales_reference,
        "terms_conditions": terms_conditions
    }

def NODE_OCR(file_path: str) -> dict:
    """Read an image file and return OCR text."""
    if file_path.lower().endswith('.pdf'):
        images = pdf_to_images(file_path, dpi=300)
        texts = []
        for page_num, img in enumerate(images, 1):
            temp_img_path = f"{file_path}_page_{page_num}.png"
            img.save(temp_img_path)
            page_text = "\n".join(ocr.readtext(temp_img_path, detail=0, paragraph=True))
            texts.append(page_text)
            os.remove(temp_img_path)
        text = "\n\n".join(texts)
    else:
        text = "\n".join(ocr.readtext(file_path, detail=0, paragraph=True))
    return {"ocr_text": text}

def NODE_CLEAN(ocr_text: str) -> Dict[str, Any]:
    """Normalize noisy OCR text using the CLEAN AI node; fallback to pass-through on error."""
    try:
        clean_raw = clean_graph.run({"ocr_text": ocr_text})
        clean_text = pick_text(clean_raw, prefer_key="CLEAN_output")
        if not isinstance(clean_text, str) or not clean_text.strip():
            raise RuntimeError("Empty CLEAN output")
        return {"clean_text": clean_text, "CLEAN_raw": clean_raw}
    except Exception as e:
        print(f"[CLEAN: FALLBACK] {e}")
        return {"clean_text": ocr_text, "CLEAN_raw": {"fallback": True}}

def NODE_EXTRACT(clean_text_any) -> dict:
    """Extract structured JSON using the EXTRACT AI node; fallback to heuristic on error."""
    clean_text = pick_text(clean_text_any, prefer_key="CLEAN_output")
    try:
        result = extract_graph.run({"clean_text": clean_text})
        raw_json = pick_text(result, prefer_key="EXTRACT_output")
        return {"raw_json": raw_json, "EXTRACT_raw": result}
    except Exception as e:
        print(f"[EXTRACT: FALLBACK] {e}")
        heuristic = _heuristic_extract(clean_text)
        return {"raw_json": json.dumps(heuristic, ensure_ascii=False), "EXTRACT_raw": {"fallback": True}}

def NODE_VALIDATE(data: Dict[str, Any]) -> Dict[str, Any]:
    """Minimal schema checks and numeric sanity for demo purposes."""
    issues: List[str] = []
    for k in ("proforma_no", "date", "state_code", "gstin_no"):
        if k not in data or not data[k]:
            issues.append(f"Missing Key: {k}")
    try:
        if "state_code" in data and data["state_code"]:
            sc = int(data["state_code"])
            if not (1 <= sc <= 37):
                issues.append(f"Invalid State Code: {data['state_code']}")
        if "gstin_no" in data and data["gstin_no"]:
            if not re.match(r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$", data["gstin_no"], re.IGNORECASE):
                issues.append(f"Invalid GSTIN No: {data['gstin_no']}")
        if "date" in data and data["date"]:
            if not re.match(r"^\d{4}-\d{2}-\d{2}$", data["date"]):
                issues.append(f"Invalid Date format: {data['date']}")
    except Exception:
        issues.append(f"Non-numeric State Code: {data.get('state_code','')}")
    if not isinstance(data.get("raw_json"), str): issues.append("raw_json not string")
    return {"valid": len(issues) == 0, "issues": issues}

def NODE_PERSIST(file_name: str, raw_json: str, CLEAN_raw: dict, EXTRACT_raw: dict, valid: bool, issues: List[str]) -> Dict[str, Any]:
    data = parse_json_safe(raw_json)
    import sqlite3
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO invoices (
            file_name, factory, order_type, proforma_no, date, bill_to, delivery_to, place_of_supply,
            state_code, gstin_no, sales_reference, terms_conditions, raw_json
        )
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            file_name,
            data.get("factory"),
            data.get("order_type"),
            data.get("proforma_no"),
            data.get("date"),
            data.get("bill_to"),
            data.get("delivery_to"),
            data.get("place_of_supply"),
            data.get("state_code"),
            data.get("gstin_no"),
            data.get("sales_reference"),
            data.get("terms_conditions"),
            raw_json
        ),
    )
    rowid = cur.lastrowid
    con.commit()
    con.close()
    print(f"[DB] Inserted row id={rowid} at {os.path.abspath(DB_PATH)}, valid={valid}, issues={issues}")
    return {"db": "sqlite", "rowid": rowid, "valid": valid, "issues": issues}

def NODE_NOTIFY(file_name: str, valid: bool, issues: List[str], rowid: int) -> Dict[str, Any]:
    status = "VALID" if valid else "INVALID"
    print(f"[NOTIFY] Processed {file_name}: Status={status}, Issues={issues}, DB RowID={rowid}")
    return {"notified": True}


# NODES = ["WATCH", "OCR", "CLEAN", "EXTRACT", "VALIDATE", "PERSIST", "NOTIFY"]
# EDGES = [
#     ("WATCH", "OCR"),
#     ("OCR", "CLEAN"),
#     ("CLEAN", "EXTRACT"),
#     ("EXTRACT", "VALIDATE"),
#     ("VALIDATE", "PERSIST"),
#     ("PERSIST", "NOTIFY"),
# ]

def run_pipeline_for_file(file_path: str) -> dict:
    file_name = os.path.basename(file_path)
    ocr_text = NODE_OCR(file_path)
    clean_result = NODE_CLEAN(ocr_text)
    extract_result = NODE_EXTRACT(clean_result)
    raw_json = pick_text(extract_result, prefer_key="raw_json")
    validate_result = NODE_VALIDATE({**parse_json_safe(raw_json), "raw_json": raw_json})
    persist_result = NODE_PERSIST(
        file_name,
        raw_json,
        clean_result.get("CLEAN_raw", {}),
        extract_result.get("EXTRACT_raw", {}),
        validate_result["valid"],
        validate_result["issues"]
    )
    NODE_NOTIFY(
        file_name,
        validate_result["valid"],
        validate_result["issues"],
        persist_result["rowid"]
    )
    # Return all for UI use
    return {
        "file_name": file_name,
        "ocr_text": ocr_text,
        "cleaned": clean_result,
        "extracted": extract_result,
        "raw_json": raw_json,
        "validation": validate_result,
        "persistence": persist_result,
    }