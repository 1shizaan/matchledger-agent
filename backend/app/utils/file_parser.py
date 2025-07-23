from io import StringIO, BytesIO # <-- Add BytesIO for Excel
import pandas as pd
from app.utils.smart_parser import auto_map_columns

def parse_csv(file_content_bytes: bytes):
    """Parses CSV content from bytes."""
    # Add a check to ensure we truly get bytes, though FastAPI's .read() should provide them.
    if not isinstance(file_content_bytes, bytes):
        raise TypeError("parse_csv expects bytes content, but received a non-bytes object.")

    content_string = file_content_bytes.decode("utf-8")
    df = pd.read_csv(StringIO(content_string))

    # Normalize columns
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    if 'amount' in df.columns:
        df['amount'] = df['amount'].astype(str).str.replace(",", "").astype(float)
    if 'narration' in df.columns:
        df['narration'] = df['narration'].str.strip().str.lower()

    return df

def parse_excel(file_content_bytes: bytes): # <-- NEW FUNCTION FOR EXCEL
    """Parses Excel content from bytes."""
    # Add a check to ensure we truly get bytes.
    if not isinstance(file_content_bytes, bytes):
        raise TypeError("parse_excel expects bytes content, but received a non-bytes object.")

    # Use BytesIO for binary Excel file content
    df = pd.read_excel(BytesIO(file_content_bytes))

    # Normalize columns (same as CSV)
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    if 'amount' in df.columns:
        df['amount'] = df['amount'].astype(str).str.replace(",", "").astype(float)
    if 'narration' in df.columns:
        df['narration'] = df['narration'].str.strip().str.lower()

    return df