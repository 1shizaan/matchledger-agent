import pandas as pd

# Define known column aliases
COLUMN_ALIASES = {
    'date': ['txn_date', 'transaction_date', 'dt', 'date'],
    'amount': ['amount', 'amt', 'value', 'debit/credit', 'transaction_amount'],
    'narration': ['narration', 'description', 'desc', 'remarks', 'details'],
    'ref_no': ['ref_no', 'reference', 'ref', 'id', 'txn_id']
}

def auto_map_columns(df: pd.DataFrame):
    column_map = {}
    df_cols = [col.strip().lower().replace(" ", "_") for col in df.columns]

    for std_col, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df_cols:
                column_map[std_col] = alias
                break

    # Rename matched columns
    df = df.rename(columns={v: k for k, v in column_map.items()})
    return df