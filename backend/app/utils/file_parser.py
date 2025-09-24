# from io import StringIO, BytesIO # <-- Add BytesIO for Excel
# import pandas as pd
# from app.utils.smart_parser import auto_map_columns

# def parse_csv(file_content_bytes: bytes):
#     """Parses CSV content from bytes."""
#     # Add a check to ensure we truly get bytes, though FastAPI's .read() should provide them.
#     if not isinstance(file_content_bytes, bytes):
#         raise TypeError("parse_csv expects bytes content, but received a non-bytes object.")

#     content_string = file_content_bytes.decode("utf-8")
#     df = pd.read_csv(StringIO(content_string))

#     # Normalize columns
#     df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

#     if 'date' in df.columns:
#         df['date'] = pd.to_datetime(df['date'], errors='coerce')
#     if 'amount' in df.columns:
#         df['amount'] = df['amount'].astype(str).str.replace(",", "").astype(float)
#     if 'narration' in df.columns:
#         df['narration'] = df['narration'].str.strip().str.lower()

#     return df

# def parse_excel(file_content_bytes: bytes): # <-- NEW FUNCTION FOR EXCEL
#     """Parses Excel content from bytes."""
#     # Add a check to ensure we truly get bytes.
#     if not isinstance(file_content_bytes, bytes):
#         raise TypeError("parse_excel expects bytes content, but received a non-bytes object.")

#     # Use BytesIO for binary Excel file content
#     df = pd.read_excel(BytesIO(file_content_bytes))

#     # Normalize columns (same as CSV)
#     df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

#     if 'date' in df.columns:
#         df['date'] = pd.to_datetime(df['date'], errors='coerce')
#     if 'amount' in df.columns:
#         df['amount'] = df['amount'].astype(str).str.replace(",", "").astype(float)
#     if 'narration' in df.columns:
#         df['narration'] = df['narration'].str.strip().str.lower()

#     return df

# app/utils/file_parser.py - ENHANCED WITH SMART COLUMN MAPPING
from io import StringIO, BytesIO
import pandas as pd
import numpy as np
import logging

# Import the smart column mapper
from app.utils.column_mapping import smart_map_columns, analyze_column_mapping_quality

logger = logging.getLogger(__name__)

def preserve_ref_no_column(df: pd.DataFrame, column_name: str = 'ref_no') -> pd.DataFrame:
    """
    🔧 CRITICAL: Preserve reference number columns during processing
    """
    if column_name not in df.columns:
        logger.warning(f"⚠️ Column '{column_name}' not found for preservation")
        return df

    logger.info(f"🔧 Preserving {column_name} column...")

    def preserve_ref_value(value):
        """Preserve individual reference values exactly as they should be"""
        if value is None or pd.isna(value) or value == '':
            return None  # Return None for truly missing values

        # Convert to string but preserve original format
        str_value = str(value).strip()

        # Handle common pandas/numpy artifacts
        if str_value.lower() in ['nan', 'null', 'undefined', '<na>', 'none']:
            return None

        # If it's a float like 123.0, convert to integer string
        if '.' in str_value and str_value.replace('.', '').replace('-', '').isdigit():
            try:
                float_val = float(str_value)
                if float_val.is_integer():
                    return str(int(float_val))
            except ValueError:
                pass

        return str_value if str_value else None

    # Apply preservation logic
    original_non_null = df[column_name].notna().sum()
    df[column_name] = df[column_name].apply(preserve_ref_value)
    preserved_non_null = df[column_name].notna().sum()

    logger.info(f"🔧 Preserved {column_name}: {original_non_null} -> {preserved_non_null} non-null values")
    logger.info(f"📋 Sample {column_name} values: {df[column_name].head().tolist()}")

    return df

def process_amount_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process amount columns (Debit/Credit or single Amount column)
    """
    logger.info("💰 Processing amount columns...")

    # Check if we have Debit/Credit columns or single Amount column
    has_debit_credit = 'debit' in df.columns or 'credit' in df.columns
    has_amount = 'amount' in df.columns

    if has_debit_credit:
        logger.info("💰 Found Debit/Credit columns, consolidating...")

        # Ensure both columns exist
        if 'debit' not in df.columns:
            df['debit'] = 0
        if 'credit' not in df.columns:
            df['credit'] = 0

        # Clean the values
        df['debit'] = pd.to_numeric(df['debit'].astype(str).str.replace(",", "").str.replace("$", ""), errors='coerce').fillna(0.0)
        df['credit'] = pd.to_numeric(df['credit'].astype(str).str.replace(",", "").str.replace("$", ""), errors='coerce').fillna(0.0)

        # Create consolidated amount column (Credit positive, Debit negative)
        df['amount'] = df['credit'] - df['debit']

        logger.info(f"💰 Consolidated amounts: Credit({df['credit'].notna().sum()}) - Debit({df['debit'].notna().sum()}) = Amount")

    elif has_amount:
        logger.info("💰 Found single Amount column, cleaning...")
        df['amount'] = pd.to_numeric(df['amount'].astype(str).str.replace(",", "").str.replace("$", ""), errors='coerce').fillna(0.0)
    else:
        logger.warning("⚠️ No amount columns found, creating empty amount column")
        df['amount'] = 0.0

    return df

def parse_csv(file_content_bytes: bytes, filename: str = "unknown.csv"):
    """Enhanced CSV parser with smart column mapping"""
    if not isinstance(file_content_bytes, bytes):
        raise TypeError("parse_csv expects bytes content")

    logger.info(f"📝 Parsing CSV file: {filename}")

    try:
        # Step 1: Basic parsing
        content_string = file_content_bytes.decode("utf-8")
        df = pd.read_csv(StringIO(content_string))

        logger.info(f"📊 Loaded {len(df)} rows, {len(df.columns)} columns")
        logger.info(f"📋 Original columns: {list(df.columns)}")

        # Step 2: Apply smart column mapping
        df = smart_map_columns(df, filename)
        logger.info(f"📋 Mapped columns: {list(df.columns)}")

        # Step 3: Process specific columns

        # Date processing
        if 'date' in df.columns:
            logger.info("📅 Processing date column...")
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Amount processing (handles Debit/Credit or Amount)
        df = process_amount_columns(df)

        # Narration processing
        if 'narration' in df.columns:
            logger.info("📝 Processing narration column...")
            df['narration'] = df['narration'].astype(str).str.strip()

        # Reference number preservation (CRITICAL)
        if 'ref_no' in df.columns:
            df = preserve_ref_no_column(df, 'ref_no')

        # Step 4: Final validation
        quality_report = analyze_column_mapping_quality(pd.read_csv(StringIO(content_string)), filename)
        logger.info(f"✅ CSV parsing complete - Quality: {quality_report.get('overall_quality', 'unknown')}")

        if 'ref_no_analysis' in quality_report:
            ref_stats = quality_report['ref_no_analysis']
            logger.info(f"📌 ref_no final stats: {ref_stats['coverage_percentage']:.1f}% coverage ({ref_stats['non_null_refs']}/{ref_stats['total_rows']})")

        return df

    except Exception as e:
        logger.error(f"❌ Error parsing CSV {filename}: {e}")
        raise

def parse_excel(file_content_bytes: bytes, filename: str = "unknown.xlsx"):
    """Enhanced Excel parser with smart column mapping"""
    if not isinstance(file_content_bytes, bytes):
        raise TypeError("parse_excel expects bytes content")

    logger.info(f"📊 Parsing Excel file: {filename}")

    try:
        # Step 1: Basic parsing
        df = pd.read_excel(BytesIO(file_content_bytes))

        logger.info(f"📊 Loaded {len(df)} rows, {len(df.columns)} columns")
        logger.info(f"📋 Original columns: {list(df.columns)}")

        # Step 2: Apply smart column mapping
        df = smart_map_columns(df, filename)
        logger.info(f"📋 Mapped columns: {list(df.columns)}")

        # Step 3: Process specific columns

        # Date processing
        if 'date' in df.columns:
            logger.info("📅 Processing date column...")
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Amount processing (handles Debit/Credit or Amount)
        df = process_amount_columns(df)

        # Narration processing
        if 'narration' in df.columns:
            logger.info("📝 Processing narration column...")
            df['narration'] = df['narration'].astype(str).str.strip()

        # Reference number preservation (CRITICAL)
        if 'ref_no' in df.columns:
            df = preserve_ref_no_column(df, 'ref_no')

        # Step 4: Final validation
        original_df = pd.read_excel(BytesIO(file_content_bytes))  # Re-read for analysis
        quality_report = analyze_column_mapping_quality(original_df, filename)
        logger.info(f"✅ Excel parsing complete - Quality: {quality_report.get('overall_quality', 'unknown')}")

        if 'ref_no_analysis' in quality_report:
            ref_stats = quality_report['ref_no_analysis']
            logger.info(f"📌 ref_no final stats: {ref_stats['coverage_percentage']:.1f}% coverage ({ref_stats['non_null_refs']}/{ref_stats['total_rows']})")

        return df

    except Exception as e:
        logger.error(f"❌ Error parsing Excel {filename}: {e}")
        raise

# Utility functions for parsing ledger and bank files
def parse_ledger_file(file_content: bytes, filename: str) -> pd.DataFrame:
    """
    Parse ledger file with smart column mapping - Use this in tasks.py
    """
    try:
        if filename.lower().endswith('.xlsx') or filename.lower().endswith('.xls'):
            return parse_excel(file_content, filename)
        else:
            return parse_csv(file_content, filename)
    except Exception as e:
        logger.error(f"❌ Failed to parse ledger file {filename}: {e}")
        raise

def parse_bank_file(file_content: bytes, filename: str) -> pd.DataFrame:
    """
    Parse bank file with smart column mapping - Use this in tasks.py
    """
    try:
        if filename.lower().endswith('.xlsx') or filename.lower().endswith('.xls'):
            return parse_excel(file_content, filename)
        else:
            return parse_csv(file_content, filename)
    except Exception as e:
        logger.error(f"❌ Failed to parse bank file {filename}: {e}")
        raise