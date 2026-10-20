# Intelligent Invoice Data Extractor

A Streamlit mini-project that converts pasted invoice text or text-based PDF invoices into structured fields and line items.

## Requirements

- Python 3.10 or newer
- `uv`

## Run on Windows

Open a terminal in this project folder and run:

```powershell
uv sync
uv run streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`.

## Features

- Paste invoice text
- Upload text-based PDF invoices
- Extract invoice number, invoice/due dates, supplier, customer, currency, line items, subtotal, tax, and total
- Check whether subtotal plus tax matches the total when all values are available
- Download JSON and CSV

## Limitations

This is a starter implementation based on regular expressions. Invoice formats vary, so fields may not be found if labels or layouts are unusual. Scanned PDFs need OCR, which is not included in this first version. Always review the extracted values before relying on them.
