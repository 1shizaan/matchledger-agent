# backend/app/services/recon_service.py - FIXED VERSION

import pandas as pd
from app.utils.file_parser import parse_csv, parse_excel
from app.core.recon_engine import match_transactions
import asyncio
from typing import Union, Dict, Any
import traceback

async def reconcile_files(
    ledger_file: Union[bytes, str],
    bank_file: Union[bytes, str],
    ledger_filename: str = None,
    bank_filename: str = None,
    bank_is_pdf: bool = False,
    ledger_column_map: Dict[str, str] = None,
    bank_column_map: Dict[str, str] = None
) -> Dict[str, Any]:
    """
    Enhanced reconciliation service that handles various file formats and column mappings.

    Args:
        ledger_file: File content as bytes or file path as string
        bank_file: File content as bytes or file path as string
        ledger_filename: Original filename for format detection
        bank_filename: Original filename for format detection
        bank_is_pdf: Whether the bank file is a PDF
        ledger_column_map: Column mapping for ledger file
        bank_column_map: Column mapping for bank file

    Returns:
        Dict containing reconciliation results with matched and unmatched transactions
    """
    try:
        print(f"🔧 Starting reconciliation service...")
        print(f"Ledger filename: {ledger_filename}")
        print(f"Bank filename: {bank_filename}")
        print(f"Bank is PDF: {bank_is_pdf}")
        print(f"Ledger column map: {ledger_column_map}")
        print(f"Bank column map: {bank_column_map}")

        # Parse ledger file
        try:
            if ledger_filename and ledger_filename.lower().endswith('.xlsx'):
                # For Excel files, use parse_excel if available
                try:
                    from app.utils.file_parser import parse_excel
                    ledger_df = await parse_excel(ledger_file)
                except ImportError:
                    # Fallback to CSV parser
                    ledger_df = await parse_csv(ledger_file)
            else:
                # Default to CSV parsing
                ledger_df = await parse_csv(ledger_file)

            print(f"✅ Ledger parsed successfully. Shape: {ledger_df.shape}")
            print(f"Ledger columns: {list(ledger_df.columns)}")

        except Exception as e:
            print(f"❌ Error parsing ledger file: {str(e)}")
            raise ValueError(f"Failed to parse ledger file: {str(e)}")

        # Parse bank file
        try:
            if bank_is_pdf:
                # Handle PDF bank statements
                try:
                    from app.utils.pdf_parser import parse_bank_pdf_to_df
                    bank_df = parse_bank_pdf_to_df(bank_file)
                    # Convert to async if needed
                    if asyncio.iscoroutine(bank_df):
                        bank_df = await bank_df
                except ImportError:
                    raise ValueError("PDF parsing not available. Please use CSV format.")
            elif bank_filename and bank_filename.lower().endswith('.xlsx'):
                # For Excel files
                try:
                    from app.utils.file_parser import parse_excel
                    bank_df = await parse_excel(bank_file)
                except ImportError:
                    bank_df = await parse_csv(bank_file)
            else:
                # Default to CSV parsing
                bank_df = await parse_csv(bank_file)

            print(f"✅ Bank file parsed successfully. Shape: {bank_df.shape}")
            print(f"Bank columns: {list(bank_df.columns)}")

        except Exception as e:
            print(f"❌ Error parsing bank file: {str(e)}")
            raise ValueError(f"Failed to parse bank file: {str(e)}")

        # Apply column mappings if provided
        if ledger_column_map:
            try:
                print(f"🔄 Applying ledger column mapping: {ledger_column_map}")
                ledger_df = ledger_df.rename(columns=ledger_column_map)
                print(f"✅ Ledger columns after mapping: {list(ledger_df.columns)}")
            except Exception as e:
                print(f"⚠️ Warning: Could not apply ledger column mapping: {str(e)}")

        if bank_column_map:
            try:
                print(f"🔄 Applying bank column mapping: {bank_column_map}")
                bank_df = bank_df.rename(columns=bank_column_map)
                print(f"✅ Bank columns after mapping: {list(bank_df.columns)}")
            except Exception as e:
                print(f"⚠️ Warning: Could not apply bank column mapping: {str(e)}")

        # Validate that we have the minimum required columns
        required_columns = ['date', 'amount', 'description']  # Adjust based on your needs

        for col in required_columns:
            if col not in ledger_df.columns:
                print(f"⚠️ Warning: Missing column '{col}' in ledger data")
            if col not in bank_df.columns:
                print(f"⚠️ Warning: Missing column '{col}' in bank data")

        # Run the reconciliation matching
        print(f"🔄 Starting transaction matching...")

        # Check if match_transactions is async or sync
        try:
            result = match_transactions(ledger_df, bank_df)

            # If it returns a coroutine, await it
            if asyncio.iscoroutine(result):
                result = await result

        except Exception as e:
            print(f"❌ Error in match_transactions: {str(e)}")
            print(f"Full traceback: {traceback.format_exc()}")
            raise ValueError(f"Reconciliation matching failed: {str(e)}")

        print(f"✅ Reconciliation completed successfully")

        # Validate result structure
        if not isinstance(result, dict):
            raise ValueError("Reconciliation engine returned invalid result format")

        required_keys = ['matched', 'unmatched_ledger', 'unmatched_bank']
        for key in required_keys:
            if key not in result:
                print(f"⚠️ Warning: Missing key '{key}' in reconciliation result")
                result[key] = []

        # Log summary
        matched_count = len(result.get('matched', []))
        unmatched_ledger_count = len(result.get('unmatched_ledger', []))
        unmatched_bank_count = len(result.get('unmatched_bank', []))

        print(f"📊 Reconciliation Summary:")
        print(f"   ✅ Matched: {matched_count}")
        print(f"   🔴 Unmatched Ledger: {unmatched_ledger_count}")
        print(f"   🔵 Unmatched Bank: {unmatched_bank_count}")

        return result

    except Exception as e:
        print(f"🔥 Critical error in reconcile_files: {str(e)}")
        print(f"Full traceback: {traceback.format_exc()}")
        raise


# ✅ NEW: Helper function for quick reconciliation stats
async def get_reconciliation_stats(ledger_file, bank_file) -> Dict[str, int]:
    """
    Quick function to get basic stats without full reconciliation.
    Useful for previews or validation.
    """
    try:
        ledger_df = await parse_csv(ledger_file)
        bank_df = await parse_csv(bank_file)

        return {
            "ledger_transactions": len(ledger_df),
            "bank_transactions": len(bank_df),
            "ledger_date_range": {
                "start": str(ledger_df['date'].min()) if 'date' in ledger_df.columns else None,
                "end": str(ledger_df['date'].max()) if 'date' in ledger_df.columns else None
            },
            "bank_date_range": {
                "start": str(bank_df['date'].min()) if 'date' in bank_df.columns else None,
                "end": str(bank_df['date'].max()) if 'date' in bank_df.columns else None
            }
        }
    except Exception as e:
        print(f"Error getting reconciliation stats: {str(e)}")
        return {
            "ledger_transactions": 0,
            "bank_transactions": 0,
            "ledger_date_range": None,
            "bank_date_range": None,
            "error": str(e)
        }