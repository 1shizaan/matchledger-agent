import pandas as pd
from app.utils.file_parser import parse_csv
from app.core.recon_engine import match_transactions  # Update import

async def reconcile_files(ledger_file, bank_file):
    ledger_df = await parse_csv(ledger_file)
    bank_df = await parse_csv(bank_file)
    result = match_transactions(ledger_df, bank_df)  # Update function call
    return result