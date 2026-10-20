import json
import re
from typing import Optional

import fitz
import pandas as pd
import spacy
import streamlit as st
from spacy.language import Language
from spacy.tokens import Doc

st.set_page_config(page_title="NLP Invoice Extractor", layout="wide")
st.title("NLP Invoice Information Extractor")
st.caption("Extract structured invoice fields using spaCy NLP, Named Entity Recognition (NER).")

MODEL_NAME = "en_core_web_sm"


@st.cache_resource
def load_nlp():
    """Load spaCy's locally installed English NLP pipeline."""
    return spacy.load(MODEL_NAME)


def normalize_text(text: str) -> str:
    text = text.replace("\xa0", " ").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def first_match(patterns, text, flags=re.IGNORECASE | re.MULTILINE) -> Optional[str]:
    for pattern in patterns:
        match = re.search(pattern, text, flags)
        if match:
            return match.group(1).strip(" \t:,-")
    return None


def to_number(value):
    if value is None:
        return None
    value = re.sub(r"(?i)\bINR\b|\bRs\.?\b", "", str(value))
    value = re.sub(r"[₹$€£,\s]", "", value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_ner_entities(doc: Doc):
    """Return named entities detected by spaCy's pretrained NER model."""
    return [
        {"text": ent.text, "label": ent.label_, "description": spacy.explain(ent.label_) or ent.label_}
        for ent in doc.ents
    ]


def extract_line_items(text: str):
    """Extract common narrative lines such as '2 keyboards at ₹800 each'."""
    items = []
    pattern = re.compile(
        r"(?P<qty>\d+(?:\.\d+)?)\s+"
        r"(?P<description>[A-Za-z][A-Za-z0-9 &()/_\-]*?)\s+"
        r"(?:at|@)\s*(?:₹|INR|Rs\.?|USD|\$|€|£)?\s*"
        r"(?P<unit_price>[\d,]+(?:\.\d{1,2})?)\s*(?:each|per\s+unit)?",
        re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        qty = to_number(match.group("qty"))
        price = to_number(match.group("unit_price"))
        description = match.group("description").strip(" .,:;-")
        if qty is not None and price is not None and description:
            items.append({
                "description": description,
                "quantity": qty,
                "unit_price": price,
                "line_total": round(qty * price, 2),
            })

    # Simple table format: Description    Qty    Unit price    Line total
    if not items:
        table_pattern = re.compile(
            r"^\s*(?P<description>[A-Za-z][A-Za-z0-9 &()/_\-]{1,45}?)\s{2,}"
            r"(?P<qty>\d+(?:\.\d+)?)\s{1,}"
            r"(?:₹|INR|Rs\.?)?\s*(?P<unit_price>[\d,]+(?:\.\d{1,2})?)\s{1,}"
            r"(?:₹|INR|Rs\.?)?\s*(?P<line_total>[\d,]+(?:\.\d{1,2})?)\s*$",
            re.IGNORECASE | re.MULTILINE,
        )
        for match in table_pattern.finditer(text):
            qty = to_number(match.group("qty"))
            price = to_number(match.group("unit_price"))
            line_total = to_number(match.group("line_total"))
            items.append({
                "description": match.group("description").strip(),
                "quantity": qty,
                "unit_price": price,
                "line_total": line_total,
            })
    return items


def extract_invoice(text: str, nlp: Language):
    cleaned = normalize_text(text)
    doc = nlp(cleaned)

    invoice_number = first_match([
        r"\b(?:invoice\s*(?:number|no\.?|#|id))\s*[:\-]?\s*([A-Z0-9][A-Z0-9/_\-]+)",
        r"\b(INV[-/][A-Z0-9/_\-]+)\b",
    ], cleaned)

    invoice_date = first_match([
        r"(?:invoice\s*date|date\s*of\s*issue|issued\s*on|date)\s*[:\-]?\s*"
        r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}|"
        r"\d{1,2}\s+[A-Za-z]+\s+\d{4}|[A-Za-z]+\s+\d{1,2},?\s+\d{4})"
    ], cleaned)

    due_date = first_match([
        r"(?:due\s*date|payment\s*due)\s*[:\-]?\s*"
        r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}|"
        r"\d{1,2}\s+[A-Za-z]+\s+\d{4}|[A-Za-z]+\s+\d{1,2},?\s+\d{4})"
    ], cleaned)

    supplier = first_match([
        r"(?:supplier|seller|vendor|billed\s*by|company\s*name)\s*[:\-]\s*([^\n]+)",
        r"(?:invoice\s*(?:from|issued\s*by))\s*[:\-]?\s*([^\n]+)",
    ], cleaned)

    customer = first_match([
        r"(?:customer|buyer|bill\s*to|billed\s*to|client)\s*[:\-]\s*([^\n]+)"
    ], cleaned)

    def amount_for(labels):
        label_pattern = "|".join(labels)
        raw = first_match([
            rf"(?:{label_pattern})\s*[:\-]?\s*(?:₹|INR|Rs\.?|USD|\$|€|£)?\s*([\d,]+(?:\.\d{{1,2}})?)"
        ], cleaned)
        return to_number(raw)

    subtotal = amount_for([r"sub\s*total", r"subtotal", r"total\s*before\s*tax"])
    tax_amount = amount_for([r"total\s*tax", r"tax\s*amount", r"GST(?:\s*\(\s*\d+(?:\.\d+)?\s*%\s*\))?", r"VAT", r"sales\s*tax", r"tax"])
    total = amount_for([r"grand\s*total", r"total\s*amount", r"amount\s*due", r"total\s*payable", r"balance\s*due", r"invoice\s*total", r"total"])

    tax_rate_raw = first_match([
        r"(?:GST|VAT|tax)\s*\(\s*(\d+(?:\.\d+)?)\s*%\s*\)",
        r"\b(\d+(?:\.\d+)?)\s*%\s*(?:GST|VAT|tax)\b",
    ], cleaned)
    tax_rate = to_number(tax_rate_raw)

    currency = (
        "INR" if re.search(r"₹|\bINR\b|\bRs\.?", cleaned, re.IGNORECASE)
        else "USD" if re.search(r"\$|\bUSD\b", cleaned, re.IGNORECASE)
        else "EUR" if "€" in cleaned
        else "GBP" if "£" in cleaned
        else "Not detected"
    )

    items = extract_line_items(cleaned)
    if subtotal is None and items:
        subtotal = round(sum(item["line_total"] or 0 for item in items), 2)
    if tax_amount is None and subtotal is not None and total is not None and total >= subtotal:
        tax_amount = round(total - subtotal, 2)

    # If labels were not present, use spaCy NER as a suggestion for organization/date/money.
    ner_entities = extract_ner_entities(doc)
    organizations = [e["text"] for e in ner_entities if e["label"] == "ORG"]
    ner_dates = [e["text"] for e in ner_entities if e["label"] == "DATE"]
    ner_money = [e["text"] for e in ner_entities if e["label"] == "MONEY"]

    result = {
        "invoice_number": invoice_number,
        "invoice_date": invoice_date or (ner_dates[0] if ner_dates else None),
        "due_date": due_date,
        "supplier": supplier or (organizations[0] if organizations else None),
        "customer": customer or (organizations[1] if len(organizations) > 1 else None),
        "currency": currency,
        "items": items,
        "subtotal": subtotal,
        "tax_rate_percent": tax_rate,
        "tax_amount": tax_amount,
        "total": total,
        "arithmetic_check": None,
        "ner_entities": ner_entities,
        "ner_money_candidates": ner_money,
    }

    if subtotal is not None and tax_amount is not None and total is not None:
        result["arithmetic_check"] = {
            "subtotal_plus_tax": round(subtotal + tax_amount, 2),
            "matches_total": abs(subtotal + tax_amount - total) < 0.02,
        }

    return result, doc


def pdf_to_text(file):
    pdf = fitz.open(stream=file.getvalue(), filetype="pdf")
    text = "\n".join(page.get_text("text") for page in pdf)
    pdf.close()
    return text.strip()


SAMPLE = """Invoice No: INV-2026-105
Invoice Date: 15 October 2026
Supplier: ABC Technologies
Customer: Rahul Enterprises, Nagpur

Items:
2 keyboards at ₹800 each
1 mouse at ₹500 each

Subtotal: ₹2,100
GST (18%): ₹378
Total Amount: ₹2,478
"""

with st.sidebar:
    st.header("Pipeline")
    st.markdown("1. Text normalization\n2. spaCy NLP pipeline\n3. Named Entity Recognition\n4. Invoice-specific rules\n5. Structured JSON/CSV")
    st.info("NER uses spaCy's local pretrained English model. No external AI API is called.")
    st.caption("General-purpose NER may miss invoice-specific entities; rule-based extraction handles labeled invoice fields.")


mode = st.radio(
    "Invoice input",
    ["Paste text", "Upload PDF"],
    horizontal=True,
    key="invoice_input_mode"
)

text = ""
uploaded_files = []

if mode == "Paste text":
    text = st.text_area(
        "Paste invoice text",
        value=SAMPLE,
        height=250,
        key="invoice_text_input"
    )

else:
    uploaded_files = st.file_uploader(
        "Upload invoice PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key="invoice_pdf_uploader"
    )

    if uploaded_files:
        st.success(f"{len(uploaded_files)} PDF(s) uploaded.")

        selected_index = st.selectbox(
            "Preview invoice",
            options=range(len(uploaded_files)),
            format_func=lambda i: uploaded_files[i].name,
            key="invoice_preview_selector"
        )

        try:
            preview_text = pdf_to_text(uploaded_files[selected_index])
            with st.expander("Preview extracted PDF text"):
                st.text(preview_text[:15000])

            if not preview_text:
                st.warning(
                    "No selectable text found. "
                    "Scanned PDFs require OCR."
                )
        except Exception as exc:
            st.error(f"Could not preview PDF: {exc}")


if st.button(
    "Extract invoice information",
    type="primary",
    use_container_width=True
):
    if mode == "Paste text" and not text.strip():
        st.warning("Please enter invoice text.")

    elif mode == "Upload PDF" and not uploaded_files:
        st.warning("Please upload at least one PDF.")

    else:
        try:
            nlp = load_nlp()
            batch_results = []

            if mode == "Paste text":
                files_to_process = [
                    ("Pasted invoice text", text)
                ]
            else:
                files_to_process = []

                for uploaded_file in uploaded_files:
                    try:
                        pdf_text = pdf_to_text(uploaded_file)
                        files_to_process.append(
                            (uploaded_file.name, pdf_text)
                        )
                    except Exception as exc:
                        st.error(
                            f"Could not read {uploaded_file.name}: {exc}"
                        )

            progress = st.progress(0)

            for index, (filename, invoice_text) in enumerate(
                files_to_process
            ):
                if not invoice_text.strip():
                    st.warning(
                        f"No readable text found in {filename}."
                    )
                    progress.progress(
                        (index + 1) / len(files_to_process)
                    )
                    continue

                try:
                    result, doc = extract_invoice(invoice_text, nlp)

                    batch_results.append({
                        "filename": filename,
                        "result": result
                    })

                except Exception as exc:
                    st.error(f"Error processing {filename}: {exc}")

                progress.progress(
                    (index + 1) / len(files_to_process)
                )

            st.session_state["batch_results"] = batch_results

        except OSError:
            st.error(
                "spaCy model missing. Run:\n\n"
                "`uv run python -m spacy download en_core_web_sm`"
            )
        except Exception as exc:
            st.error(f"Extraction failed: {exc}")


if "batch_results" in st.session_state:
    batch_results = st.session_state["batch_results"]

    if batch_results:
        st.divider()
        st.subheader("Extracted invoice results")

        st.success(
            f"Successfully processed {len(batch_results)} invoice(s)."
        )

        # Create a summary table with one row per invoice.
        summary_rows = []

        for entry in batch_results:
            result = entry["result"]

            summary_rows.append({
                "Source File": entry["filename"],
                "Invoice Number": result.get("invoice_number"),
                "Invoice Date": result.get("invoice_date"),
                "Due Date": result.get("due_date"),
                "Supplier": result.get("supplier"),
                "Customer": result.get("customer"),
                "Currency": result.get("currency"),
                "Subtotal": result.get("subtotal"),
                "Tax Rate (%)": result.get("tax_rate_percent"),
                "Tax Amount": result.get("tax_amount"),
                "Total": result.get("total"),
                "Line Items": json.dumps(
                    result.get("items", []),
                    ensure_ascii=False
                ),
                "NER Entities": json.dumps(
                    result.get("ner_entities", []),
                    ensure_ascii=False
                ),
                "Arithmetic Check": json.dumps(
                    result.get("arithmetic_check"),
                    ensure_ascii=False
                )
            })

        summary_df = pd.DataFrame(summary_rows)

        st.markdown("### All invoices in one table")
        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True
        )

        # Download all extracted invoice information in one CSV.
        csv_data = summary_df.to_csv(
            index=False
        ).encode("utf-8-sig")

        st.download_button(
            label="Download ALL invoices as one CSV",
            data=csv_data,
            file_name="all_invoices_extracted.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True
        )

        # Download all results together as JSON.
        combined_json = json.dumps(
            batch_results,
            indent=2,
            ensure_ascii=False
        )

        st.download_button(
            label="Download ALL invoices as one JSON",
            data=combined_json,
            file_name="all_invoices_extracted.json",
            mime="application/json",
            use_container_width=True
        )

        # Show detailed extraction results for each file.
        st.markdown("### Individual invoice details")

        for entry in batch_results:
            result = entry["result"]

            with st.expander(entry["filename"]):
                st.json({
                    key: value
                    for key, value in result.items()
                    if key not in {"items", "ner_entities"}
                })

                st.markdown("**Line items**")

                if result.get("items"):
                    st.dataframe(
                        pd.DataFrame(result["items"]),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("No line items detected.")

                st.markdown("**NER entities**")

                if result.get("ner_entities"):
                    st.dataframe(
                        pd.DataFrame(result["ner_entities"]),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("No named entities detected.")
