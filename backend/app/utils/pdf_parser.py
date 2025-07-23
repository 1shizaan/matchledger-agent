import pdfplumber
import pytesseract
from pdf2image import convert_from_path
import pandas as pd
import tempfile
import os

def extract_text_from_pdf(path):
    # Try extracting text directly
    with pdfplumber.open(path) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    return text.strip()

def extract_text_via_ocr(path):
    # Convert PDF to images
    images = convert_from_path(path)
    text = ""
    for image in images:
        text += pytesseract.image_to_string(image)
    return text.strip()

def parse_bank_pdf_to_df(pdf_path):
    # First try direct text extraction
    text = extract_text_from_pdf(pdf_path)
    if not text or len(text.splitlines()) < 5:
        text = extract_text_via_ocr(pdf_path)

    # Very naive line parsing — we’ll improve it later
    rows = []
    for line in text.splitlines():
        if any(char.isdigit() for char in line):
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    date = parts[0]
                    amount = float(parts[-2].replace(",", ""))
                    narration = " ".join(parts[1:-2])
                    ref_no = parts[-1]
                    rows.append({
                        "date": date,
                        "amount": amount,
                        "narration": narration,
                        "ref_no": ref_no
                    })
                except:
                    continue

    df = pd.DataFrame(rows)
    return df