import pandas as pd
from datetime import timedelta
import os
import numpy as np
from openai import OpenAI
from app.utils.memory_db import is_match_from_memory, add_to_memory

# ✅ Initialize OpenAI client (reads key from env var)
try:
    client = OpenAI()
    print("OpenAI client initialized for embeddings.")
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    client = None

# ✅ Global embedding cache to avoid redundant OpenAI calls
embedding_cache = {}

# ✅ 1. Introduce semantic_alias_map (optional but powerful)
NARRATION_ALIASES = {
    "uber ride": "travel expense",
    "office depot": "supplies purchase",
    # Add more aliases as needed for common variations
    # Example: "amazon prime" : "online subscription",
    # Example: "starbucks" : "food and beverage"
}

def classify_match(score: float) -> str:
    if score >= 0.95:
        return "exact"
    elif score >= 0.60:
        return "fuzzy"
    elif score >= 0.40:
        return "partial"
    else:
        return "no_match"

def clean_text(text: str) -> str:
    if isinstance(text, str):
        text = text.lower()
        text = ' '.join(text.split())
        text = text.replace("-", "")
        return text.strip()
    return ""

def get_embedding(text: str):
    if not client:
        print("OpenAI client not initialized.")
        return None

    cleaned = text.strip().lower()
    if cleaned in embedding_cache:
        return embedding_cache[cleaned]

    try:
        response = client.embeddings.create(
            input=cleaned,
            model="text-embedding-3-small"
        )
        embedding = response.data[0].embedding
        embedding_cache[cleaned] = embedding
        return embedding
    except Exception as e:
        print(f"Embedding error for '{text}': {e}")
        return None

def cosine_similarity(embedding1, embedding2) -> float:
    if embedding1 is None or embedding2 is None:
        return 0.0
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return dot_product / (norm_vec1 * norm_vec2)

def ai_narration_match(n1: str, n2: str, threshold: float = 0.60):
    cleaned_n1 = clean_text(n1)
    cleaned_n2 = clean_text(n2)

    if not cleaned_n1 or not cleaned_n2:
        return False, 0.0

    cleaned_n1 = NARRATION_ALIASES.get(cleaned_n1, cleaned_n1)
    cleaned_n2 = NARRATION_ALIASES.get(cleaned_n2, cleaned_n2)

    if cleaned_n1 == cleaned_n2:
        print(f"🧠 Comparing '{n1}' vs '{n2}' — Cleaned (Aliased) '{cleaned_n1}' vs '{cleaned_n2}' — Similarity: 1.00 (Exact Match)")       
        return True, 1.0

    embedding1 = get_embedding(cleaned_n1)
    embedding2 = get_embedding(cleaned_n2)

    if embedding1 is None or embedding2 is None:
        return False, 0.0

    similarity = cosine_similarity(embedding1, embedding2)
    print(f"🧠 Comparing '{n1}' vs '{n2}' — Cleaned (Aliased) '{cleaned_n1}' vs '{cleaned_n2}' — Similarity: {similarity:.2f}")

    if similarity >= threshold or (cleaned_n1 in cleaned_n2 or cleaned_n2 in cleaned_n1):
        return True, similarity
    else:
        if 0.55 < similarity < threshold:
            print(f"🤔 Potential match (needs review): '{n1}' vs '{n2}' (sim: {similarity:.2f})")
        return False, similarity

def match_transactions(ledger_df, bank_df):
    matched = []
    unmatched_ledger = []
    unmatched_bank = bank_df.copy()
    soft_matches_for_review = []

    for _, ledger_row in ledger_df.iterrows():
        found_match = False
        print(f"\n📘 Processing Ledger: {ledger_row['narration']} (${ledger_row['amount']}) on {ledger_row['date'].date()}")

        best_non_match_bank_row = None
        highest_non_match_similarity = 0.0

        for bank_index, bank_row in unmatched_bank.iterrows():
            amount_match = abs(ledger_row["amount"] - bank_row["amount"]) <= 0.01
            date_diff = abs((ledger_row["date"] - bank_row["date"]).days)
            date_match = date_diff <= 1
            narration_match, similarity = ai_narration_match(
                ledger_row["narration"], bank_row["narration"]
            )

            # FIX: Corrected the print statement
            print(f"➡️ Bank: {bank_row['narration']} | Amount: {bank_row['amount']} | Date: {bank_row['date'].date()}")
            print(f"   Match: Amount ✅ {amount_match} | Date ✅ {date_match} | Narration 🧠 {similarity:.2f} -> {narration_match}")        

            if amount_match and date_match and narration_match:
                match_tag = classify_match(similarity)
                matched.append({
                    "ledger": {
                        "date": str(ledger_row["date"].date()),
                        "amount": float(ledger_row["amount"]),  # Ensure it's a standard float
                        "narration": ledger_row["narration"],
                        "ref_no": ledger_row["ref_no"]
                    },
                    "bank": {
                        "date": str(bank_row["date"].date()),
                        "amount": float(bank_row["amount"]),  # Ensure it's a standard float
                        "narration": bank_row["narration"],
                        "ref_no": bank_row["ref_no"]
                    },
                    "similarity_score": round(float(similarity), 2),
                    "match_type": match_tag,
                    "match_on": {
                        "amount": bool(amount_match),
                        "date": bool(date_match),
                        "narration": bool(similarity >= 0.6)
                    }
                })
                unmatched_bank.drop(index=bank_index, inplace=True)
                found_match = True
                break
            else:
                if amount_match and date_match and (0.55 < similarity < 0.60):
                    if similarity > highest_non_match_similarity:
                        highest_non_match_similarity = similarity
                        best_non_match_bank_row = bank_row

        if not found_match:
            print(f"❌ No match found for: {ledger_row['narration']}")
            unmatched_ledger.append({
                "date": str(ledger_row["date"].date()),
                "amount": float(ledger_row["amount"]),  # Ensure it's a standard float
                "narration": ledger_row["narration"],
                "ref_no": ledger_row["ref_no"],
                "reason": "No good match found"
            })

            if best_non_match_bank_row is not None:
                soft_matches_for_review.append({
                    "ledger_item": {
                        "date": str(ledger_row["date"].date()),
                        "amount": float(ledger_row["amount"]),
                        "narration": ledger_row["narration"],
                        "ref_no": ledger_row["ref_no"]
                    },
                    "bank_item": {
                        "date": str(best_non_match_bank_row["date"].date()),
                        "amount": float(best_non_match_bank_row["amount"]),
                        "narration": best_non_match_bank_row["narration"],
                        "ref_no": best_non_match_bank_row["ref_no"]
                    },
                    "similarity_score": round(float(highest_non_match_similarity), 2),
                    "status": "Potential Soft Match - Needs Review"
                })

    # Convert the final unmatched_bank DataFrame to records with proper types
    unmatched_bank_records = []
    for _, row in unmatched_bank.iterrows():
        unmatched_bank_records.append({
            "date": str(row["date"].date()),
            "amount": float(row["amount"]),
            "narration": row["narration"],
            "ref_no": row["ref_no"]
        })

    return {
        "matched": matched,
        "unmatched_ledger": unmatched_ledger,
        "unmatched_bank": unmatched_bank_records,
        "soft_matches_for_review": soft_matches_for_review
    }