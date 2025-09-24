# app/utils/aliases.py

COLUMN_ALIASES = {
    "date": [
        "date", "txn date", "transaction date", "trans date", "value date",
        "posting date", "entry date", "dated", "dt", "transaction_date",
        "txn_date", "value_date", "posting_date", "Date"
    ],
    "narration": [
        "narration", "description", "particulars", "details", "remarks",
        "transaction details", "txn details", "memo", "reference", "note",
        "Narration", "Description", "Particulars", "Details", "Remarks"
    ],
    "debit": [
        "debit", "dr", "debit amount", "withdrawal", "outgoing", "paid",
        "Debit", "DR", "Debit Amount", "Withdrawal", "Outgoing", "Paid"
    ],
    "credit": [
        "credit", "cr", "credit amount", "deposit", "incoming", "received",
        "Credit", "CR", "Credit Amount", "Deposit", "Incoming", "Received"
    ],
    "ref_no": [
        "reference no", "ref no", "reference", "ref", "transaction id", "txn id",
        "Reference No", "Ref No", "Reference", "Ref", "Transaction ID", "Txn ID",
        "check no", "cheque no", "voucher no", "Check No", "Cheque No", "Voucher No"
    ]
}