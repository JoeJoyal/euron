import os
import streamlit as st 
from src.config import INPUT_DIR, PROCESSED_LOG
from src.db import fetch_all, fetch_one_by_id, ensure_schema
from src.pipeline import run_pipeline_for_file, load_seen, save_seen, pdf_to_images


st.set_page_config(page_title="Intelligent Invoice Extractor", layout="wide")
st.title("Invoice Extraction Pipeline")
st.sidebar.header("Options")

uploaded = st.sidebar.file_uploader("Upload new Invoice (.jpg/.png/.pdf)", type=('png', 'jpg', 'jpeg', 'pdf'))

if uploaded:
    fpath = os.path.join(INPUT_DIR, uploaded.name)
    with open(fpath, "wb") as f: f.write(uploaded.read())
    # PDF preview with Wand
    if uploaded.name.lower().endswith(".pdf"):
        try:
            # Reopen as file-like for preview
            with open(fpath, "rb") as f:
                pages = pdf_to_images(f, dpi=120)
                if pages:
                    st.image(pages[0], caption=f"First page: {uploaded.name}")
        except Exception as e:
            st.warning(f"PDF preview not available: {e}")
    st.success(f"Saved {uploaded.name}. Processing...")
    result = run_pipeline_for_file(fpath)
    seen = load_seen()
    seen.add(uploaded.name)
    save_seen(seen)
    st.info(f"Finished {uploaded.name}!")
    st.write(result)

st.sidebar.markdown("---")
df = fetch_all()
st.sidebar.subheader(f"View Invoice ({len(df)})")
choose_id = st.sidebar.selectbox(
    "Select by Row ID", df["id"].tolist() if not df.empty else [], format_func=lambda r: f'Invoice #{r}'
)
if choose_id:
    row = fetch_one_by_id(choose_id)
    if row:
        st.write("### Selected Invoice Details")
        st.json(dict(zip(df.columns, row)))

st.header("All Extracted Invoice Data")
if df.shape[0]:
    st.dataframe(df)
    with st.expander("📈 Analytics (Sample Insights)"):
        st.write("**Invoice count by Factory:**")
        st.bar_chart(df["factory"].value_counts())
        st.write("**Order Types Distribution:**")
        st.bar_chart(df["order_type"].value_counts())
        st.write("**Date Range:**")
        st.write(dict(Min=df["date"].min(), Max=df["date"].max()))
else:
    st.info("No invoice data found. Upload or drop images or PDFs into the input folder.")

st.sidebar.markdown("---")
if st.sidebar.button("Process all images in input folder"):
    seen = load_seen()
    count = 0
    for fname in os.listdir(INPUT_DIR):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".pdf")): continue
        if fname in seen: continue
        fpath = os.path.join(INPUT_DIR, fname)
        try:
            run_pipeline_for_file(fpath)
            seen.add(fname)
            count += 1
        except Exception as e:
            st.warning(f"[{fname}] Failed: {e}")
    save_seen(seen)
    st.success(f"Processed {count} new invoices.")

st.sidebar.info(f"Upload OR drop .pdf/.jpg/.png images in '{INPUT_DIR}'. Already processed files will be skipped.")