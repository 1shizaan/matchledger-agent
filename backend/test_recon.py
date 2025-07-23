from app.core.recon_engine import parse_and_clean, match_transactions

ledger_path = "test_data/sample_ledger.csv"
bank_path = "test_data/sample_bank.csv"

# Parse & clean
ledger_df = parse_and_clean(ledger_path, "ledger")
bank_df = parse_and_clean(bank_path, "bank")

# Match logic
result = match_transactions(ledger_df, bank_df)

# Show results
import json
print(json.dumps(result, indent=2, default=str))
