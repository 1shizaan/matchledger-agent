# app/utils/column_utils.py - NEW FILE

import pandas as pd
import io
from typing import Dict, List, Optional, Any
from app.utils.aliases import COLUMN_ALIASES
# ===== COLUMN DETECTION HELPER =====
def detect_csv_headers(file_content: bytes, filename: str) -> List[str]:
    """
    Reads the first row of a CSV/Excel file to extract column headers
    """
    try:
        if filename.lower().endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_content), nrows=0)
            return df.columns.tolist()
        elif filename.lower().endswith(('.xls', '.xlsx')):
            df = pd.read_excel(io.BytesIO(file_content), nrows=0)
            return df.columns.tolist()
        else:
            return []
    except Exception as e:
        print(f"Error detecting headers for {filename}: {e}")
        return []

def suggest_column_mapping(headers: List[str]) -> Dict[str, Optional[str]]:
    """
    Intelligently suggests column mappings based on common aliases
    """
    suggestions = {}
    headers_lower = [h.lower() for h in headers]

    for standard_col, aliases in COLUMN_ALIASES.items():
        suggestion = None
        for alias in aliases:
            if alias.lower() in headers_lower:
                # Find the original header with correct case
                for original_header in headers:
                    if original_header.lower() == alias.lower():
                        suggestion = original_header
                        break
                break
        suggestions[standard_col] = suggestion

    return suggestions

# ===== COLUMN RENAMING AND COMBINING HELPER =====
def rename_and_combine_columns(df: pd.DataFrame, column_map: Dict[str, Any], file_type: str) -> pd.DataFrame:
    """
    Renames DataFrame columns based on mapping and combines debit/credit into amount
    """
    print(f"[{file_type.upper()}] Original columns: {df.columns.tolist()}")
    print(f"[{file_type.upper()}] Column mapping: {column_map}")

    rename_dict = {}

    # Build rename dictionary
    for standard_col, user_col in column_map.items():
        if user_col is None or user_col == "":
            continue

        if user_col not in df.columns:
            raise ValueError(
                f"Selected {file_type} column '{user_col}' not found in file. "
                f"Available columns: {df.columns.tolist()}"
            )

        rename_dict[user_col] = standard_col

    # Perform renaming
    df.rename(columns=rename_dict, inplace=True)
    print(f"[{file_type.upper()}] After renaming: {df.columns.tolist()}")

    # Handle debit/credit combination with IMPROVED data cleaning
    if 'debit' in df.columns or 'credit' in df.columns:
        # Convert to numeric, handle empty strings and NaN properly
        if 'debit' in df.columns:
            # Clean empty strings first, then convert to numeric
            df['debit'] = df['debit'].replace('', '0').fillna('0')
            df['debit'] = pd.to_numeric(df['debit'], errors='coerce').fillna(0)
        else:
            df['debit'] = 0

        if 'credit' in df.columns:
            # Clean empty strings first, then convert to numeric
            df['credit'] = df['credit'].replace('', '0').fillna('0')
            df['credit'] = pd.to_numeric(df['credit'], errors='coerce').fillna(0)
        else:
            df['credit'] = 0

        # Create unified amount column (Credit - Debit for positive values)
        df['amount'] = df['credit'] - df['debit']

        print(f"[{file_type.upper()}] Created amount column from debit/credit")
        print(f"[{file_type.upper()}] Sample amounts: {df['amount'].head().tolist()}")

    # Validate required columns exist
    required_cols = ['date', 'narration', 'amount']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in {file_type} file after mapping: {missing_cols}. "
            f"Available columns: {df.columns.tolist()}"
        )

    print(f"[{file_type.upper()}] Final columns: {df.columns.tolist()}")
    return df