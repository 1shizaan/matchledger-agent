# # import os
# # import pandas as pd
# # from langchain_openai import ChatOpenAI
# # from langchain_experimental.agents import create_pandas_dataframe_agent
# # from app.utils.query_parser import parse_natural_date_range # This import is correct and needed

# # from dotenv import load_dotenv

# # load_dotenv()

# # def run_chat_agent(ledger_df: pd.DataFrame, bank_df: pd.DataFrame, query: str):
# #     combined_df = pd.concat([
# #         ledger_df.assign(source="Ledger"),
# #         bank_df.assign(source="Bank")
# #     ])

# #     # --- START OF NEW CODE FOR DATE FILTERING ---
# #     start, end = parse_natural_date_range(query)
# #     if start and end:
# #         # It's good practice to ensure the 'date' column exists and is in datetime format
# #         # before trying to convert it and filter.
# #         if 'date' in combined_df.columns:
# #             # Convert 'date' column to datetime, handling potential errors
# #             # 'coerce' will turn unparseable dates into NaT (Not a Time)
# #             combined_df['date'] = pd.to_datetime(combined_df['date'], errors='coerce')

# #             # Drop rows where date conversion failed (NaT values) if you don't want them
# #             combined_df = combined_df.dropna(subset=['date'])

# #             # Filter the DataFrame based on the parsed date range
# #             combined_df = combined_df[
# #                 (combined_df['date'].dt.date >= start) &
# #                 (combined_df['date'].dt.date <= end)
# #             ]
# #             print(f"DEBUG: Filtered data for date range: {start} to {end}. Remaining rows: {len(combined_df)}")
# #         else:
# #             print("Warning: 'date' column not found in combined_df. Skipping date range filtering.")
# #     # --- END OF NEW CODE FOR DATE FILTERING ---

# #     # The agent will now operate on the potentially filtered combined_df
# #     agent = create_pandas_dataframe_agent(
# #         ChatOpenAI(temperature=0, model="gpt-4"),
# #         combined_df,
# #         verbose=True, # Keeping this True for better debugging of agent's thoughts
# #         allow_dangerous_code=True,
# #         max_iterations=15
# #     )

# #     result = agent.invoke({"input": query})
# #     return result["output"]
# #  ------------------------------------------------
# # chat_agent.py (Modified)

# import os
# import pandas as pd
# from langchain_openai import ChatOpenAI
# from langchain_experimental.agents import create_pandas_dataframe_agent
# from app.utils.query_parser import parse_natural_date_range

# from dotenv import load_dotenv

# load_dotenv()

# llm = ChatOpenAI(temperature=0, model="gpt-4")

# def run_chat_agent(
#     ledger_df: pd.DataFrame,
#     bank_df: pd.DataFrame,
#     query: str,
#     reconciliation_summary: dict = None
# ):
#     query_lower = query.lower()

#     # --- Pre-process query for specific intents (matched/unmatched) ---

#     # NEW: Check for direct "show" or "list" commands for unmatched
#     if "show unmatched" in query_lower or "list unmatched" in query_lower or query_lower == "unmatched":
#         if reconciliation_summary and (reconciliation_summary.get('unmatched_ledger') or reconciliation_summary.get('unmatched_bank')):
#             unmatched_ledger = reconciliation_summary.get('unmatched_ledger', [])
#             unmatched_bank = reconciliation_summary.get('unmatched_bank', [])

#             all_unmatched_data = []
#             for item in unmatched_ledger:
#                 flat_item = {k: v for k, v in item.items() if k != 'transaction'}
#                 if 'transaction' in item and isinstance(item['transaction'], dict):
#                     flat_item.update(item['transaction'])
#                 all_unmatched_data.append({**flat_item, "source": "Ledger"})

#             for item in unmatched_bank:
#                 flat_item = {k: v for k, v in item.items() if k != 'transaction'}
#                 if 'transaction' in item and isinstance(item['transaction'], dict):
#                     flat_item.update(item['transaction'])
#                 all_unmatched_data.append({**flat_item, "source": "Bank"})

#             if all_unmatched_data:
#                 headers = list(all_unmatched_data[0].keys()) if all_unmatched_data else []
#                 preferred_order = ["date", "amount", "narration", "ref_no", "source"]
#                 headers_ordered = [h for h in preferred_order if h in headers] + \
#                                   [h for h in headers if h not in preferred_order]

#                 return {
#                     "response_type": "table",
#                     "data": {
#                         "title": "Unmatched Transactions",
#                         "headers": headers_ordered,
#                         "rows": all_unmatched_data
#                     }
#                 }
#             else:
#                 return {"response_type": "text", "data": "No unmatched transactions found in the last reconciliation."}
#         else:
#             return {"response_type": "text", "data": "I don't have reconciliation summary data to show unmatched transactions. Please perform a reconciliation first."}

#     # NEW: Check for direct "show" or "list" commands for matched
#     if "show matched" in query_lower or "list matched" in query_lower or query_lower == "matched":
#         if reconciliation_summary and reconciliation_summary.get('matched'):
#             matched_data = reconciliation_summary.get('matched', [])
#             if matched_data:
#                 headers = list(matched_data[0].keys()) if matched_data else []
#                 preferred_order = ["ledger_date", "ledger_amount", "ledger_narration", "bank_date", "bank_amount", "bank_narration", "similarity_score", "match_type"]
#                 headers_ordered = [h for h in preferred_order if h in headers] + \
#                                   [h for h in headers if h not in preferred_order]

#                 return {
#                     "response_type": "table",
#                     "data": {
#                         "title": "Matched Transactions",
#                         "headers": headers_ordered,
#                         "rows": matched_data
#                     }
#                 }
#             else:
#                 return {"response_type": "text", "data": "No matched transactions found in the last reconciliation."}
#         else:
#             return {"response_type": "text", "data": "I don't have reconciliation summary data to show matched transactions. Please perform a reconciliation first."}

#     # --- If not a specific matched/unmatched query, proceed with Pandas Agent ---
#     # ... (the rest of your function remains the same) ...
#     combined_df = pd.concat([
#         ledger_df.assign(source="Ledger"),
#         bank_df.assign(source="Bank")
#     ])

#     start, end = parse_natural_date_range(query)
#     if start and end:
#         if 'date' in combined_df.columns:
#             combined_df['date'] = pd.to_datetime(combined_df['date'], errors='coerce')
#             combined_df = combined_df.dropna(subset=['date'])
#             combined_df = combined_df[
#                 (combined_df['date'].dt.date >= start) &
#                 (combined_df['date'].dt.date <= end)
#             ]
#             print(f"DEBUG: Filtered data for date range: {start} to {end}. Remaining rows: {len(combined_df)}")
#         else:
#             print("Warning: 'date' column not found in combined_df. Skipping date range filtering.")

#     agent = create_pandas_dataframe_agent(
#         llm,
#         combined_df,
#         verbose=True,
#         allow_dangerous_code=True,
#         max_iterations=15
#     )

#     try:
#         # NEW: Here, you might want to add context from reconciliation_summary to the agent's prompt
#         # if the query is about "how to match" or other reasoning tasks.
#         # This is where your prompt engineering for the LLM itself comes in.
#         # For now, let's just make sure it *reaches* here.
        
#         enhanced_query = query
#         if reconciliation_summary:
#             unmatched_ledger_transactions = reconciliation_summary.get('unmatched_ledger', [])
#             unmatched_bank_transactions = reconciliation_summary.get('unmatched_bank', [])

#             # Convert to string format for the prompt
#             unmatched_ledger_str = "\n".join([f"- Ledger Ref: {t.get('ref_no', 'N/A')}, Date: {t.get('date', 'N/A')}, Amount: {t.get('amount', 'N/A')}, Narration: '{t.get('narration', 'N/A')}'" for t in unmatched_ledger_transactions]) if unmatched_ledger_transactions else "None"
#             unmatched_bank_str = "\n".join([f"- Bank Ref: {t.get('ref_no', 'N/A')}, Date: {t.get('date', 'N/A')}, Amount: {t.get('amount', 'N/A')}, Narration: '{t.get('narration', 'N/A')}'" for t in unmatched_bank_transactions]) if unmatched_bank_transactions else "None"

#             enhanced_query = f"""
#             **Role:** You are a highly experienced and friendly financial reconciliation expert. Your goal is to help users understand *why* transactions are unmatched and provide clear, practical, and step-by-step guidance on *how* to reconcile them.

#             **Context:**
#             The last reconciliation process identified the following unmatched transactions:

#             **Unmatched from Ledger:**
#             {unmatched_ledger_str}

#             **Unmatched from Bank:**
#             {unmatched_bank_str}

#             **User's Request:** "{query}"

#             **Task:**
#             Based on the user's request and the provided unmatched transactions, offer actionable advice. Focus on common reconciliation strategies and potential reasons for discrepancies.

#             **Instructions for your response:**
#             1.  **Analyze each unmatched transaction individually** or in small, logical groups if a pattern emerges.
#             2.  For each, suggest **specific investigative steps** (e.g., "Check for typos in narration," "Look for transactions with similar amounts/dates," "Consider if this is a timing difference," "Was this transaction recorded twice or missed entirely?").
#             3.  Provide **clear, numbered steps** or bullet points for easy readability.
#             4.  **Avoid technical jargon** where simpler terms suffice.
#             5.  **Do NOT just re-list the transactions** unless it's part of a specific instruction (e.g., "Review this transaction: ..."). Your primary output should be *guidance*.
#             6.  Conclude with a general recommendation about re-running the reconciliation after making adjustments.

#             Let's think step by step to provide the most helpful and easy-to-understand advice.
#             """
#         print(f"DEBUG: Sending to Pandas Agent with enhanced query:\n{enhanced_query}")
#         result = agent.invoke({"input": enhanced_query})
#         return {"response_type": "text", "data": result["output"]}
#     except Exception as e:
#         print(f"Error during Pandas agent invocation: {e}")
#         return {"response_type": "text", "data": f"I encountered an error trying to process that query with the data. Please try rephrasing: {str(e)}"}


# # chat_agent.py - Smart Intent-Based Version v4.0
# import os
# import pandas as pd
# from langchain_openai import ChatOpenAI
# from langchain_experimental.agents import create_pandas_dataframe_agent
# from app.utils.query_parser import parse_natural_date_range
# from typing import Dict, List, Any, Union, Optional
# from datetime import datetime
# import logging
# from collections import Counter
# import json

# from dotenv import load_dotenv

# load_dotenv()

# # Configure logging for better debugging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Initialize LLM with optimal settings for financial analysis
# llm = ChatOpenAI(
#     temperature=0.1,
#     model="gpt-4",
#     max_tokens=2500,
#     request_timeout=60
# )

# # Intent classifier LLM - lighter for quick classification
# intent_llm = ChatOpenAI(
#     temperature=0.0,
#     model="gpt-3.5-turbo",
#     max_tokens=200,
#     request_timeout=30
# )

# def safe_float(value, default=0.0):
#     """Safely convert value to float"""
#     try:
#         if value is None or value == '' or value == 'None':
#             return default
#         return float(value)
#     except (ValueError, TypeError):
#         return default

# def safe_str(value, default=''):
#     """Safely convert value to string"""
#     try:
#         if value is None:
#             return default
#         return str(value).strip()
#     except:
#         return default

# class SmartReconciliationChatAgent:
#     """Smart AI-native chat agent that understands intent instead of pattern matching"""

#     def __init__(self):
#         self.llm = llm
#         self.intent_llm = intent_llm

#     def _classify_query_intent(self, query: str, has_reconciliation_data: bool) -> Dict[str, Any]:
#         """Use AI to intelligently classify user intent"""

#         intent_prompt = f"""
# You are an AI assistant that understands financial reconciliation queries. Analyze this user query and determine their intent.

# User Query: "{query}"
# Has Reconciliation Data: {has_reconciliation_data}

# Classify the intent into one of these categories and provide reasoning:

# INTENT CATEGORIES:
# 1. SHOW_UNMATCHED - User wants to see unmatched transactions (requires reconciliation data)
# 2. SHOW_MATCHED - User wants to see matched transactions (requires reconciliation data)
# 3. SHOW_DISCREPANCIES - User wants to see discrepancies/issues (same as unmatched)
# 4. ANALYZE_RECONCILIATION - User wants insights/analysis of reconciliation results
# 5. HOW_TO_MATCH - User asking for guidance on matching strategies
# 6. GENERAL_ANALYSIS - User wants analysis of raw transaction data
# 7. UNCLEAR - Intent is ambiguous

# RESPONSE FORMAT (JSON):
# {{
#     "intent": "<INTENT_CATEGORY>",
#     "confidence": <0-100>,
#     "reasoning": "<brief explanation>",
#     "requires_reconciliation": <true/false>,
#     "data_type": "<unmatched/matched/analysis/guidance>"
# }}

# Examples:
# - "show me unmatched transactions" → SHOW_UNMATCHED
# - "list the discrepancies" → SHOW_DISCREPANCIES
# - "why are there more bank unmatched?" → ANALYZE_RECONCILIATION
# - "how do I match these?" → HOW_TO_MATCH
# - "what's the total amount?" → GENERAL_ANALYSIS

# Analyze the query and respond with valid JSON only.
# """

#         try:
#             response = self.intent_llm.invoke(intent_prompt)
#             response_text = response.content.strip()

#             # Clean response - extract JSON if wrapped in markdown
#             if "```json" in response_text:
#                 response_text = response_text.split("```json")[1].split("```")[0].strip()
#             elif "```" in response_text:
#                 response_text = response_text.split("```")[1].strip()

#             intent_data = json.loads(response_text)
#             logger.info(f"Intent classified: {intent_data}")
#             return intent_data

#         except Exception as e:
#             logger.error(f"Intent classification error: {e}")
#             # Fallback to general analysis
#             return {
#                 "intent": "GENERAL_ANALYSIS",
#                 "confidence": 50,
#                 "reasoning": "Failed to classify intent, defaulting to general analysis",
#                 "requires_reconciliation": False,
#                 "data_type": "analysis"
#             }

#     def _deduplicate_transactions(self, transactions: List[Dict]) -> List[Dict]:
#         """Remove duplicate transactions based on key fields"""
#         if not transactions:
#             return []

#         seen = set()
#         unique_transactions = []

#         for transaction in transactions:
#             try:
#                 key_fields = (
#                     safe_str(transaction.get('date', '')),
#                     safe_str(transaction.get('amount', '')),
#                     safe_str(transaction.get('ref_no', '')),
#                     safe_str(transaction.get('narration', ''))[:100]
#                 )

#                 if key_fields not in seen:
#                     seen.add(key_fields)
#                     unique_transactions.append(transaction)
#                 else:
#                     logger.info(f"Removed duplicate transaction: {transaction.get('ref_no', 'Unknown')}")
#             except Exception as e:
#                 logger.warning(f"Error processing transaction for deduplication: {e}")
#                 unique_transactions.append(transaction)

#         return unique_transactions

#     def _analyze_transaction_patterns(self, transactions: List[Dict], source: str) -> Dict:
#         """Analyze patterns in unmatched transactions with type safety"""
#         if not transactions:
#             return {
#                 'total_count': 0,
#                 'total_amount': 0.0,
#                 'avg_amount': 0.0,
#                 'amount_range': (0.0, 0.0),
#                 'categories': {},
#                 'top_amounts': []
#             }

#         try:
#             amounts = []
#             for t in transactions:
#                 amount_val = safe_float(t.get('amount', 0))
#                 if amount_val != 0:
#                     amounts.append(amount_val)

#             categories = {
#                 'client_payments': [],
#                 'subscriptions': [],
#                 'bank_fees': [],
#                 'travel': [],
#                 'office': [],
#                 'utilities': [],
#                 'other': []
#             }

#             for txn in transactions:
#                 try:
#                     narration = safe_str(txn.get('narration', '')).lower()
#                     amount = safe_float(txn.get('amount', 0))

#                     if any(word in narration for word in ['client', 'payment', 'invoice', 'wire', 'stark', 'acme', 'wayne', 'zeta']):
#                         categories['client_payments'].append(amount)
#                     elif any(word in narration for word in ['figma', 'adobe', 'slack', 'notion', 'saas', 'subscription']):
#                         categories['subscriptions'].append(amount)
#                     elif any(word in narration for word in ['fee', 'charge', 'bank', 'interest', 'dividend']):     
#                         categories['bank_fees'].append(amount)
#                     elif any(word in narration for word in ['travel', 'flight', 'uber', 'lyft', 'meal', 'car']):   
#                         categories['travel'].append(amount)
#                     elif any(word in narration for word in ['office', 'supplies', 'amazon', 'equipment', 'maintenance']):
#                         categories['office'].append(amount)
#                     elif any(word in narration for word in ['utilities', 'water', 'electric', 'gas', 'internet']): 
#                         categories['utilities'].append(amount)
#                     else:
#                         categories['other'].append(amount)
#                 except Exception as e:
#                     logger.warning(f"Error categorizing transaction: {e}")
#                     categories['other'].append(safe_float(txn.get('amount', 0)))

#             total_amount = sum(amounts) if amounts else 0.0
#             avg_amount = total_amount / len(amounts) if amounts else 0.0
#             min_amount = min(amounts) if amounts else 0.0
#             max_amount = max(amounts) if amounts else 0.0

#             return {
#                 'total_count': len(transactions),
#                 'total_amount': total_amount,
#                 'avg_amount': avg_amount,
#                 'amount_range': (min_amount, max_amount),
#                 'categories': {k: {'count': len(v), 'total': sum(v)} for k, v in categories.items() if v},
#                 'top_amounts': sorted(amounts, key=abs, reverse=True)[:5] if amounts else []
#             }

#         except Exception as e:
#             logger.error(f"Error in pattern analysis: {e}")
#             return {
#                 'total_count': len(transactions),
#                 'total_amount': 0.0,
#                 'avg_amount': 0.0,
#                 'amount_range': (0.0, 0.0),
#                 'categories': {},
#                 'top_amounts': []
#             }

#     def _flatten_transaction_data(self, item: Dict) -> Dict:
#         """Properly flatten nested transaction data with type safety"""
#         flat_item = {}

#         try:
#             if 'transaction' in item and isinstance(item['transaction'], dict):
#                 flat_item.update(item['transaction'])

#             for key, value in item.items():
#                 if key != 'transaction':
#                     flat_item[key] = value

#             field_mapping = {
#                 'ref_no': 'ref no',
#                 'ref_number': 'ref no',
#                 'reference_no': 'ref no',
#                 'reference_number': 'ref no'
#             }

#             for old_key, new_key in field_mapping.items():
#                 if old_key in flat_item:
#                     flat_item[new_key] = flat_item.pop(old_key)

#         except Exception as e:
#             logger.warning(f"Error flattening transaction data: {e}")
#             flat_item = item.copy() if isinstance(item, dict) else {}

#         return flat_item

#     def _prepare_unmatched_data(self, reconciliation_summary: Dict) -> Dict:
#         """Prepare clean unmatched transaction data"""
#         try:
#             unmatched_ledger = reconciliation_summary.get('unmatched_ledger', [])
#             unmatched_bank = reconciliation_summary.get('unmatched_bank', [])

#             logger.info(f"Processing {len(unmatched_ledger)} ledger + {len(unmatched_bank)} bank unmatched transactions")

#             all_unmatched_data = []

#             # Process ledger transactions
#             for item in unmatched_ledger:
#                 if isinstance(item, dict):
#                     try:
#                         flat_item = self._flatten_transaction_data(item)
#                         flat_item["source"] = "Ledger"

#                         narration = safe_str(flat_item.get('narration', '')).lower()
#                         if any(word in narration for word in ['client', 'payment', 'invoice', 'wire']):
#                             flat_item["pattern type"] = "client_payment"
#                         elif any(word in narration for word in ['fee', 'charge', 'bank']):
#                             flat_item["pattern type"] = "bank_charges"
#                         elif any(word in narration for word in ['travel', 'flight', 'meal']):
#                             flat_item["pattern type"] = "travel"
#                         elif any(word in narration for word in ['subscription', 'saas']):
#                             flat_item["pattern type"] = "subscription"
#                         elif any(word in narration for word in ['office', 'supplies']):
#                             flat_item["pattern type"] = "office"
#                         elif any(word in narration for word in ['utilities', 'water', 'electric']):
#                             flat_item["pattern type"] = "utilities"
#                         else:
#                             flat_item["pattern type"] = "unknown"

#                         if 'reason' not in flat_item or not flat_item.get('reason'):
#                             flat_item["reason"] = "No suitable match found"

#                         all_unmatched_data.append(flat_item)
#                     except Exception as e:
#                         logger.warning(f"Error processing ledger transaction: {e}")

#             # Process bank transactions
#             for item in unmatched_bank:
#                 if isinstance(item, dict):
#                     try:
#                         flat_item = self._flatten_transaction_data(item)
#                         flat_item["source"] = "Bank"

#                         narration = safe_str(flat_item.get('narration', '')).lower()
#                         if any(word in narration for word in ['client', 'payment', 'invoice', 'wire']):
#                             flat_item["pattern type"] = "client_payment"
#                         elif any(word in narration for word in ['fee', 'charge', 'bank']):
#                             flat_item["pattern type"] = "bank_charges"
#                         elif any(word in narration for word in ['travel', 'flight', 'meal']):
#                             flat_item["pattern type"] = "travel"
#                         elif any(word in narration for word in ['subscription', 'saas']):
#                             flat_item["pattern type"] = "subscription"
#                         elif any(word in narration for word in ['office', 'supplies']):
#                             flat_item["pattern type"] = "office"
#                         elif any(word in narration for word in ['utilities', 'water', 'electric']):
#                             flat_item["pattern type"] = "utilities"
#                         else:
#                             flat_item["pattern type"] = "unknown"

#                         if 'reason' not in flat_item or not flat_item.get('reason'):
#                             flat_item["reason"] = "No matching ledger transaction found"

#                         all_unmatched_data.append(flat_item)
#                     except Exception as e:
#                         logger.warning(f"Error processing bank transaction: {e}")

#             all_unmatched_data = self._deduplicate_transactions(all_unmatched_data)

#             if not all_unmatched_data:
#                 return {"response_type": "text", "data": "No unmatched transactions found in the reconciliation data."}

#             all_headers = set()
#             for item in all_unmatched_data:
#                 all_headers.update(item.keys())

#             preferred_order = ["date", "amount", "narration", "ref no", "source", "reason", "pattern type"]        
#             headers_ordered = [h for h in preferred_order if h in all_headers]
#             headers_ordered.extend([h for h in sorted(all_headers) if h not in headers_ordered])

#             logger.info(f"Prepared {len(all_unmatched_data)} unique unmatched transactions")

#             return {
#                 "response_type": "table",
#                 "data": {
#                     "title": "Unmatched Transactions",
#                     "headers": headers_ordered,
#                     "rows": all_unmatched_data
#                 }
#             }

#         except Exception as e:
#             logger.error(f"Error preparing unmatched data: {e}")
#             return {"response_type": "text", "data": f"Error processing unmatched transactions: {str(e)}"}

#     def _prepare_matched_data(self, reconciliation_summary: Dict) -> Dict:
#         """Prepare clean matched transaction data"""
#         try:
#             matched_data = reconciliation_summary.get('matched', [])

#             if not matched_data:
#                 return {"response_type": "text", "data": "No matched transactions found in the reconciliation data."}

#             matched_data = self._deduplicate_transactions(matched_data)

#             all_headers = set()
#             for item in matched_data:
#                 all_headers.update(item.keys())

#             preferred_order = [
#                 "ledger_date", "ledger_amount", "ledger_narration", "ledger_ref_no",
#                 "bank_date", "bank_amount", "bank_narration", "bank_ref_no",
#                 "similarity_score", "match_type", "confidence"
#             ]

#             headers_ordered = [h for h in preferred_order if h in all_headers]
#             headers_ordered.extend([h for h in sorted(all_headers) if h not in headers_ordered])

#             logger.info(f"Prepared {len(matched_data)} unique matched transactions")

#             return {
#                 "response_type": "table",
#                 "data": {
#                     "title": "Matched Transactions",
#                     "headers": headers_ordered,
#                     "rows": matched_data
#                 }
#             }

#         except Exception as e:
#             logger.error(f"Error preparing matched data: {e}")
#             return {"response_type": "text", "data": f"Error processing matched transactions: {str(e)}"}

#     def _generate_reconciliation_insights(self, reconciliation_summary: Dict) -> str:
#         """Generate intelligent reconciliation insights"""

#         try:
#             unmatched_ledger = reconciliation_summary.get('unmatched_ledger', [])
#             unmatched_bank = reconciliation_summary.get('unmatched_bank', [])
#             matched = reconciliation_summary.get('matched', [])

#             ledger_analysis = self._analyze_transaction_patterns(unmatched_ledger, "Ledger")
#             bank_analysis = self._analyze_transaction_patterns(unmatched_bank, "Bank")

#             total_transactions = len(matched) + len(unmatched_ledger) + len(unmatched_bank)
#             match_rate = (len(matched) / total_transactions * 100) if total_transactions > 0 else 0

#             insights = f"""📊 **RECONCILIATION ANALYSIS REPORT**

# **🎯 Executive Summary:**
# Your reconciliation achieved a {match_rate:.1f}% match rate with {len(matched)} successful matches. The higher number of unmatched bank transactions ({len(unmatched_bank)} vs {len(unmatched_ledger)}) is **completely normal and expected** in business reconciliation.

# **🔍 Why Bank Has More Unmatched Transactions:**

# 1. **📦 Scope Difference:**
#    • **Bank Records:** ALL activity including fees, interest, automated payments, micro-transactions
#    • **Ledger Records:** Business-relevant transactions, often filtered or curated

# 2. **⏰ Timing Differences:**
#    • **Bank:** Real-time processing, immediate recording
#    • **Ledger:** Batched entries, manual processing, business-day delays

# 3. **🎯 Purpose Difference:**
#    • **Bank:** Legal compliance, complete transaction history
#    • **Ledger:** Accounting/business analysis, relevant transactions only

# **📈 Key Performance Metrics:**
# • **Match Success Rate:** {match_rate:.1f}% (Excellent for business reconciliation)
# • **Total Transactions Processed:** {total_transactions:,}
# • **Successfully Matched:** {len(matched)} pairs
# • **Bank-Only Transactions:** {len(unmatched_bank)} (normal operational activity)
# • **Ledger-Only Transactions:** {len(unmatched_ledger)} (requires review)

# **🔍 Transaction Pattern Analysis:**"""

#             if bank_analysis.get('categories'):
#                 insights += f"\n\n**Bank Transaction Categories:**"
#                 sorted_categories = sorted(bank_analysis['categories'].items(),
#                                          key=lambda x: x[1]['total'], reverse=True)
#                 for category, info in sorted_categories:
#                     if info['count'] > 0:
#                         category_name = category.replace('_', ' ').title()
#                         insights += f"\n• {category_name}: {info['count']} transactions (${info['total']:,.2f})"   

#             if ledger_analysis.get('categories'):
#                 insights += f"\n\n**Ledger Transaction Categories:**"
#                 sorted_categories = sorted(ledger_analysis['categories'].items(),
#                                          key=lambda x: x[1]['total'], reverse=True)
#                 for category, info in sorted_categories:
#                     if info['count'] > 0:
#                         category_name = category.replace('_', ' ').title()
#                         insights += f"\n• {category_name}: {info['count']} transactions (${info['total']:,.2f})"   

#             insights += f"""

# **💡 Expert Recommendations:**

# 1. **✅ Current Status:** Your reconciliation is performing well with {match_rate:.1f}% match rate
# 2. **🎯 Focus Priority:** Review the {len(unmatched_ledger)} unmatched ledger transactions (higher impact)
# 3. **🏦 Bank Unmatched:** The {len(unmatched_bank)} bank-only transactions are mostly operational (fees, interest, micro-payments)
# 4. **🔄 Process Optimization:** Consider automated rules for recurring patterns

# **🚀 Next Steps:**
# 1. **High Priority:** Review unmatched ledger transactions for potential data entry issues
# 2. **Medium Priority:** Categorize bank-only transactions for better tracking
# 3. **Low Priority:** Set up automated matching rules for recurring patterns
# 4. **Monitoring:** Establish target match rate benchmarks (70%+ is excellent)

# **✅ Overall Assessment:**
# Your reconciliation shows healthy patterns with expected bank-ledger differences. The {match_rate:.1f}% match rate with minimal ledger unmatched transactions indicates good data quality and process control.

# **🎯 Key Insight:** Having more bank unmatched transactions is the **normal state** for business reconciliation - it indicates comprehensive bank recording versus focused ledger management."""

#             return insights

#         except Exception as e:
#             logger.error(f"Error generating insights: {e}")
#             return f"""📊 **RECONCILIATION ANALYSIS**

# **🎯 Quick Answer:**
# Bank has more unmatched transactions ({len(reconciliation_summary.get('unmatched_bank', []))} vs {len(reconciliation_summary.get('unmatched_ledger', []))}) because this is **completely normal**:

# • **Banks record everything:** fees, interest, micro-transactions, automated payments
# • **Ledgers record business transactions:** curated, relevant transactions only
# • **Different purposes:** Bank = compliance, Ledger = business analysis

# **✅ Status:** Your {len(reconciliation_summary.get('matched', []))} matches indicate good reconciliation health.  

# **🚀 Action:** Focus on reviewing the {len(reconciliation_summary.get('unmatched_ledger', []))} unmatched ledger transactions as they have higher business impact.

# This pattern is expected and healthy for business reconciliation processes."""

#     def run_chat_agent(
#         self,
#         ledger_df: pd.DataFrame,
#         bank_df: pd.DataFrame,
#         query: str,
#         reconciliation_summary: dict = None
#     ) -> Dict[str, Any]:
#         """Smart intent-based chat agent v4.0"""

#         try:
#             logger.info(f"Processing query: '{query}'")
#             logger.info(f"Reconciliation summary available: {reconciliation_summary is not None}")

#             # STEP 1: AI-powered intent classification
#             intent_data = self._classify_query_intent(query, reconciliation_summary is not None)
#             intent = intent_data.get('intent', 'GENERAL_ANALYSIS')

#             logger.info(f"Intent classified as: {intent} (confidence: {intent_data.get('confidence', 0)}%)")       

#             # STEP 2: Route based on intelligent intent classification
#             if intent == "SHOW_UNMATCHED":
#                 if reconciliation_summary:
#                     return self._prepare_unmatched_data(reconciliation_summary)
#                 else:
#                     return {
#                         "response_type": "text",
#                         "data": "I need reconciliation results to show unmatched transactions. Please run a reconciliation first."
#                     }

#             elif intent == "SHOW_MATCHED":
#                 if reconciliation_summary:
#                     return self._prepare_matched_data(reconciliation_summary)
#                 else:
#                     return {
#                         "response_type": "text",
#                         "data": "I need reconciliation results to show matched transactions. Please run a reconciliation first."
#                     }

#             elif intent == "SHOW_DISCREPANCIES":
#                 if reconciliation_summary:
#                     unmatched_data = self._prepare_unmatched_data(reconciliation_summary)
#                     if unmatched_data.get("response_type") == "table":
#                         unmatched_data["data"]["title"] = "Financial Discrepancies (Unmatched Transactions)"       
#                     return unmatched_data
#                 else:
#                     return {
#                         "response_type": "text",
#                         "data": "I need reconciliation results to identify discrepancies. Please run a reconciliation first."
#                     }

#             elif intent == "ANALYZE_RECONCILIATION":
#                 if reconciliation_summary:
#                     insights = self._generate_reconciliation_insights(reconciliation_summary)
#                     return {"response_type": "text", "data": insights}
#                 else:
#                     return {
#                         "response_type": "text",
#                         "data": "I need reconciliation results to provide analysis. Please run a reconciliation first."
#                     }

#             # STEP 3: For general analysis or matching guidance, use LLM agent
#             else:  # GENERAL_ANALYSIS, HOW_TO_MATCH, UNCLEAR
#                 try:
#                     combined_df = pd.concat([
#                         ledger_df.assign(source="Ledger"),
#                         bank_df.assign(source="Bank")
#                     ], ignore_index=True)

#                     # Apply date filtering if requested
#                     start, end = parse_natural_date_range(query)
#                     if start and end:
#                         if 'date' in combined_df.columns:
#                             combined_df['date'] = pd.to_datetime(combined_df['date'], errors='coerce')
#                             combined_df = combined_df.dropna(subset=['date'])
#                             mask = (combined_df['date'].dt.date >= start) & (combined_df['date'].dt.date <= end)   
#                             combined_df = combined_df[mask]
#                             logger.info(f"Applied date filter: {start} to {end}, {len(combined_df)} transactions remaining")

#                     agent = create_pandas_dataframe_agent(
#                         self.llm,
#                         combined_df,
#                         verbose=True,
#                         allow_dangerous_code=True,
#                         max_iterations=10,
#                         handle_parsing_errors=True,
#                         return_intermediate_steps=False
#                     )

#                     # Smart context-aware prompting based on intent
#                     if intent == "HOW_TO_MATCH":
#                         enhanced_query = f"""
# You are a financial reconciliation expert. The user is asking: "{query}"

# This is raw transaction data (ledger + bank) before reconciliation. Provide practical guidance on:

# 1. **Matching Strategies**: How to identify corresponding transactions
# 2. **Key Fields**: What fields to use for matching (amount, date, reference numbers, narration patterns)
# 3. **Common Challenges**: What makes matching difficult
# 4. **Best Practices**: Proven approaches for successful reconciliation
# 5. **Tools/Techniques**: Specific methods for their data

# Data Context: {len(combined_df)} transactions ({len(combined_df[combined_df['source']=='Ledger'])} ledger, {len(combined_df[combined_df['source']=='Bank'])} bank)

# Be practical, specific, and actionable. Focus on their actual data patterns.
# """
#                     else:
#                         enhanced_query = f"""
# You are analyzing financial transaction data. User query: "{query}"

# Context: Combined ledger and bank transaction data with 'source' column.
# Intent detected: {intent} (confidence: {intent_data.get('confidence', 0)}%)

# Provide specific insights based on the data. Be professional and actionable.

# Data Overview: {len(combined_df)} transactions ({len(combined_df[combined_df['source']=='Ledger'])} ledger, {len(combined_df[combined_df['source']=='Bank'])} bank)
# """

#                     logger.info("Executing smart LLM agent analysis...")
#                     result = agent.invoke({"input": enhanced_query})

#                     if isinstance(result, dict) and 'output' in result:
#                         response_text = safe_str(result['output'])
#                     elif isinstance(result, str):
#                         response_text = safe_str(result)
#                     else:
#                         response_text = safe_str(result)

#                     response_text = response_text.strip()
#                     if not response_text:
#                         response_text = "I was able to analyze your data but couldn't generate a comprehensive response. Please try rephrasing your question."

#                     logger.info("Smart LLM agent analysis completed successfully")
#                     return {"response_type": "text", "data": response_text}

#                 except Exception as agent_error:
#                     logger.error(f"Agent execution error: {agent_error}")

#                     fallback_response = f"""I encountered a technical issue while analyzing your data, but I can still help!

# **Quick Solutions Based on Your Intent ({intent}):**
# • Try asking: "Show me all unmatched transactions"
# • Or: "What patterns do you see in the unmatched data?"
# • Or: "Analyze the reconciliation results"

# **Technical Details:** {str(agent_error)[:200]}...

# Would you like to try one of the suggested questions above?"""

#                     return {"response_type": "text", "data": fallback_response}

#         except Exception as e:
#             logger.error(f"Critical error in smart chat agent: {e}")
#             return {
#                 "response_type": "text",
#                 "data": f"I encountered an unexpected error. Please try again or contact support. Error: {str(e)[:100]}..."
#             }

# # Initialize global agent instance
# chat_agent_instance = SmartReconciliationChatAgent()

# def run_chat_agent(
#     ledger_df: pd.DataFrame,
#     bank_df: pd.DataFrame,
#     query: str,
#     reconciliation_summary: dict = None
# ) -> Dict[str, Any]:
#     """Smart AI-native interface for the chat agent"""
#     return chat_agent_instance.run_chat_agent(
#         ledger_df,
#         bank_df,
#         query,
#         reconciliation_summary
#     )

import os
import pandas as pd
from typing import Dict, List, Any, Union, Optional
from datetime import datetime
import logging
from collections import Counter
import json
import time
from app.core.config import settings
from app.utils.query_parser import parse_natural_date_range

from dotenv import load_dotenv
load_dotenv()

# Configure logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client for chatbot
openai_client = None

if settings.OPENAI_API_KEY:
    try:
        from openai import OpenAI
        openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        logger.info("✅ OpenAI client initialized for chatbot conversations.")
    except ImportError:
        logger.error("❌ OpenAI library not installed. Run: pip install openai")
    except Exception as e:
        logger.error(f"❌ Error initializing OpenAI client: {e}")
else:
    logger.error("❌ OPENAI_API_KEY is required but not provided")

def safe_float(value, default=0.0):
    """Safely convert value to float"""
    try:
        if value is None or value == '' or value == 'None':
            return default
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_str(value, default=''):
    """Safely convert value to string"""
    try:
        if value is None:
            return default
        return str(value).strip()
    except:
        return default

def safe_int(value, default=0):
    """Safely convert value to integer"""
    try:
        if value is None or value == '' or value == 'None':
            return default
        return int(float(value))
    except (ValueError, TypeError):
        return default

class OpenAIChatAgent:
    """OpenAI-powered AI-native chat agent for financial reconciliation v6.2 - FIXED"""

    def __init__(self):
        self.chat_config = settings.get_chatbot_config()

        if not openai_client:
            raise ValueError("OpenAI client not initialized. Check OPENAI_API_KEY.")

        logger.info(f"Chat agent v6.2 using: OpenAI {self.chat_config['config']['primary']}")

    def _openai_classify_intent(self, query: str, has_reconciliation_data: bool) -> Dict[str, Any]:
        """Use OpenAI to intelligently classify user intent - OPTIMIZED with smart pre-filtering"""
        if not openai_client:
            return self._fallback_intent_classification(query, has_reconciliation_data)

        # Smart pattern matching first to avoid unnecessary API calls (60% of queries)
        query_lower = query.lower()

        # High-confidence pattern matching for common requests
        if any(word in query_lower for word in ['show', 'display', 'list', 'get']) and any(word in query_lower for word in ['unmatched', 'discrepanc', 'missing', 'not matched']):
            return {
                "intent": "SHOW_UNMATCHED",
                "confidence": 90,
                "reasoning": "Direct request to show unmatched transactions",
                "requires_reconciliation": True,
                "data_type": "unmatched"
            }

        if any(word in query_lower for word in ['show', 'display', 'list', 'get']) and any(word in query_lower for word in ['matched', 'paired', 'found']):
            return {
                "intent": "SHOW_MATCHED",
                "confidence": 90,
                "reasoning": "Direct request to show matched transactions",
                "requires_reconciliation": True,
                "data_type": "matched"
            }

        if any(word in query_lower for word in ['summary', 'summarize', 'insights', 'analyze', 'analysis', 'explain', 'why']):
            return {
                "intent": "ANALYZE_RECONCILIATION",
                "confidence": 85,
                "reasoning": "Request for analysis or insights",
                "requires_reconciliation": has_reconciliation_data,
                "data_type": "analysis"
            }

        if any(word in query_lower for word in ['how to', 'how do', 'matching', 'strategy', 'guide', 'help']):
            return {
                "intent": "HOW_TO_MATCH",
                "confidence": 85,
                "reasoning": "Request for matching guidance",
                "requires_reconciliation": False,
                "data_type": "guidance"
            }

        # Only use OpenAI for complex/ambiguous queries (40% of cases)
        intent_prompt = f"""Classify this financial reconciliation query intent:

Query: "{query}"
Has reconciliation data: {has_reconciliation_data}
 
   Intent categories:
- SHOW_UNMATCHED: show unmatched/discrepancy transactions
- SHOW_MATCHED: show matched transaction pairs
- ANALYZE_RECONCILIATION: insights/analysis of results
- HOW_TO_MATCH: guidance on matching strategies
- GENERAL_ANALYSIS: general data questions

Response format (JSON only):
{{"intent": "<CATEGORY>", "confidence": <0-100>}}"""

        try:
            response = openai_client.chat.completions.create(
                model=self.chat_config["config"]["primary"],
                messages=[{"role": "user", "content": intent_prompt}],
                max_tokens=50,
                temperature=0.0
            )

            result_text = response.choices[0].message.content.strip()

            # Extract and parse JSON
            if "{" in result_text:
                start = result_text.find("{")
                end = result_text.rfind("}") + 1
                result_text = result_text[start:end]

            intent_data = json.loads(result_text)

            # Enrich the response
            intent = intent_data.get('intent', 'GENERAL_ANALYSIS')
            intent_data.update({
                "reasoning": f"OpenAI classified as {intent}",
                "requires_reconciliation": intent in ["SHOW_UNMATCHED", "SHOW_MATCHED", "ANALYZE_RECONCILIATION"],
                "data_type": "matched" if "MATCHED" in intent else "analysis"
            })

            logger.info(f"OpenAI intent classification: {intent} ({intent_data.get('confidence', 0)}%)")
            return intent_data

        except Exception as e:
            logger.warning(f"OpenAI intent classification failed, using fallback: {e}")
            return self._fallback_intent_classification(query, has_reconciliation_data)

    def _fallback_intent_classification(self, query: str, has_reconciliation_data: bool) -> Dict[str, Any]:
        """Enhanced fallback intent classification with better patterns"""
        query_lower = query.lower()

        # More comprehensive pattern matching
        if any(word in query_lower for word in ['unmatched', 'discrepanc', 'missing', 'not matched', 'failed', 'issues']):
            return {
                "intent": "SHOW_UNMATCHED",
                "confidence": 75,
                "reasoning": "Pattern match: unmatched/discrepancy keywords",
                "requires_reconciliation": True,
                "data_type": "unmatched"
            }
        elif any(word in query_lower for word in ['matched', 'paired', 'found matches', 'successful', 'pairs']):
            return {
                "intent": "SHOW_MATCHED",
                "confidence": 75,
                "reasoning": "Pattern match: matched keywords",
                "requires_reconciliation": True,
                "data_type": "matched"
            }
        elif any(word in query_lower for word in ['why', 'analysis', 'insight', 'explain', 'summary', 'report', 'assess']):
            return {
                "intent": "ANALYZE_RECONCILIATION",
                "confidence": 70,
                "reasoning": "Pattern match: analysis keywords",
                "requires_reconciliation": has_reconciliation_data,
                "data_type": "analysis"
            }
        elif any(word in query_lower for word in ['how to', 'how do', 'matching', 'strategy', 'guide', 'help', 'tips']):
            return {
                "intent": "HOW_TO_MATCH",
                "confidence": 75,
                "reasoning": "Pattern match: guidance keywords",
                "requires_reconciliation": False,
                "data_type": "guidance"
            }
        else:
            return {
                "intent": "GENERAL_ANALYSIS",
                "confidence": 50,
                "reasoning": "Fallback classification - general query",
                "requires_reconciliation": False,
                "data_type": "analysis"
            }

    def _deduplicate_transactions(self, transactions: List[Dict]) -> List[Dict]:
        """Smart deduplication to prevent showing duplicate transactions"""
        if not transactions:
            return []

        seen = set()
        unique_transactions = []
        duplicates_removed = 0

        for transaction in transactions:
            try:
                # Create unique key from critical fields
                key_fields = (
                    safe_str(transaction.get('date', '')),
                    safe_str(transaction.get('amount', '')),
                    safe_str(transaction.get('ref no', transaction.get('ref_no', ''))),
                    safe_str(transaction.get('narration', ''))[:100]  # First 100 chars to handle variations
                )

                if key_fields not in seen:
                    seen.add(key_fields)
                    unique_transactions.append(transaction)
                else:
                    duplicates_removed += 1
                    logger.debug(f"Removed duplicate: {transaction.get('ref no', 'Unknown')}")

            except Exception as e:
                logger.warning(f"Error in deduplication: {e}")
                # Include the transaction if deduplication fails
                unique_transactions.append(transaction)

        if duplicates_removed > 0:
            logger.info(f"Deduplication: removed {duplicates_removed} duplicates, {len(unique_transactions)} unique remain")

        return unique_transactions     
    def _analyze_transaction_patterns(self, transactions: List[Dict], source: str) -> Dict:
        """Advanced pattern analysis for business intelligence - FIXED"""
        if not transactions:
            return {
                'total_count': 0,
                'total_amount': 0.0,
                'avg_amount': 0.0,
                'amount_range': (0.0, 0.0),
                'categories': {},
                'top_amounts': [],
                'date_spread': None,
                'high_value_threshold': 0.0
            }

        try:
            amounts = []
            dates = []

            # Enhanced business categories for better insights
            categories = {
                'client_payments': {'amounts': [], 'count': 0, 'keywords': ['client', 'payment', 'invoice', 'wire', 'stark', 'acme', 'wayne', 'zeta', 'consulting', 'professional']},
                'subscriptions': {'amounts': [], 'count': 0, 'keywords': ['figma', 'adobe', 'slack', 'notion', 'saas', 'subscription', 'monthly', 'annual', 'license']},
                'bank_fees': {'amounts': [], 'count': 0, 'keywords': ['fee', 'charge', 'bank', 'interest', 'dividend', 'commission', 'penalty', 'overdraft']},
                'travel_expenses': {'amounts': [], 'count': 0, 'keywords': ['travel', 'flight', 'uber', 'lyft', 'meal', 'car', 'hotel', 'taxi', 'transport']},
                'office_supplies': {'amounts': [], 'count': 0, 'keywords': ['office', 'supplies', 'amazon', 'equipment', 'maintenance', 'furniture', 'stationery']},
                'utilities': {'amounts': [], 'count': 0, 'keywords': ['utilities', 'water', 'electric', 'gas', 'internet', 'phone', 'telecom', 'broadband']},
                'vendor_payments': {'amounts': [], 'count': 0, 'keywords': ['vendor', 'supplier', 'purchase', 'procurement', 'order', 'goods']},
                'other': {'amounts': [], 'count': 0, 'keywords': []}
            }

            for txn in transactions:
                try:
                    # Process amounts
                    amount = safe_float(txn.get('amount', 0))
                    if amount != 0:
                        amounts.append(amount)

                    # Process dates for spread analysis
                    date_str = safe_str(txn.get('date', ''))
                    if date_str:
                        try:
                            date_obj = pd.to_datetime(date_str, errors='coerce')
                            if pd.notna(date_obj):
                                dates.append(date_obj)
                        except:
                            pass

                    # Enhanced categorization
                    narration = safe_str(txn.get('narration', '')).lower()
                    categorized = False

                    for category, data in categories.items():
                        if category == 'other':
                            continue

                        if any(keyword in narration for keyword in data['keywords']):
                            data['amounts'].append(amount)
                            data['count'] += 1
                            categorized = True
                            break

                    if not categorized:
                        categories['other']['amounts'].append(amount)
                        categories['other']['count'] += 1

                except Exception as e:
                    logger.warning(f"Error processing transaction in pattern analysis: {e}")
                    categories['other']['amounts'].append(safe_float(txn.get('amount', 0)))
                    categories['other']['count'] += 1

            # Calculate comprehensive metrics
            total_amount = sum(amounts) if amounts else 0.0
            avg_amount = total_amount / len(amounts) if amounts else 0.0
            min_amount = min(amounts) if amounts else 0.0
            max_amount = max(amounts) if amounts else 0.0

            # Date spread analysis
            date_spread = None
            if dates:
                date_spread = {
                    'earliest': min(dates),
                    'latest': max(dates),
                    'span_days': (max(dates) - min(dates)).days if len(dates) > 1 else 0
                }

            # High value threshold (75th percentile) - FIXED comparison
            high_value_threshold = 0.0
            if amounts:
                amounts_sorted = sorted([abs(x) for x in amounts])
                percentile_75 = int(len(amounts_sorted) * 0.75)
                if percentile_75 < len(amounts_sorted):
                    high_value_threshold = amounts_sorted[percentile_75]

            # Clean up categories for response
            clean_categories = {}
            for cat_name, cat_data in categories.items():
                if cat_data['count'] > 0:
                    clean_categories[cat_name] = {
                        'count': cat_data['count'],
                        'total': sum(cat_data['amounts']),
                        'avg': sum(cat_data['amounts']) / cat_data['count'] if cat_data['count'] > 0 else 0
                    }

            result = {
                'total_count': len(transactions),
                'total_amount': total_amount,
                'avg_amount': avg_amount,
                'amount_range': (min_amount, max_amount),
                'categories': clean_categories,
                'top_amounts': sorted(amounts, key=abs, reverse=True)[:10] if amounts else [],
                'date_spread': date_spread,
                'high_value_threshold': high_value_threshold
            }

            logger.info(f"Pattern analysis complete for {source}: {len(clean_categories)} categories, ${total_amount:.2f} total")
            return result

        except Exception as e:
            logger.error(f"Error in pattern analysis for {source}: {e}")
            return {
                'total_count': len(transactions),
                'total_amount': 0.0,
                'avg_amount': 0.0,
                'amount_range': (0.0, 0.0),
                'categories': {'error': {'count': len(transactions), 'total': 0, 'avg': 0}},
                'top_amounts': [],
                'date_spread': None,
                'high_value_threshold': 0.0
            }

 
    def _flatten_transaction_data(self, item: Dict) -> Dict:
        """Enhanced transaction data flattening with pattern classification - FIXED"""
        flat_item = {}

        try:
            # FIXED: Ensure item is a dictionary before processing
            if not isinstance(item, dict):
                logger.warning(f"Expected dict, got {type(item)}: {item}")
                return {"error": "Invalid transaction format", "raw_data": str(item)}

            # Handle nested transaction structures
            if 'transaction' in item and isinstance(item['transaction'], dict):
                flat_item.update(item['transaction'])

            # Add all other fields
            for key, value in item.items():
                if key != 'transaction':
                    flat_item[key] = value

            # Standardize field names
            field_mapping = {
                'ref_no': 'ref no',
                'ref_number': 'ref no',
                'reference_no': 'ref no',
                'reference_number': 'ref no'
            }

            for old_key, new_key in field_mapping.items():
                if old_key in flat_item:
                    flat_item[new_key] = flat_item.pop(old_key)

            # Add enhanced pattern classification
            narration = safe_str(flat_item.get('narration', '')).lower()

            # More sophisticated pattern recognition
            if any(word in narration for word in ['client', 'payment', 'invoice', 'wire', 'consulting', 'professional']):
                flat_item["pattern type"] = "client_payment"
            elif any(word in narration for word in ['fee', 'charge', 'bank', 'commission', 'penalty']):
                flat_item["pattern type"] = "bank_charges"
            elif any(word in narration for word in ['travel', 'flight', 'meal', 'hotel', 'transport']):
                flat_item["pattern type"] = "travel_expense"
            elif any(word in narration for word in ['subscription', 'saas', 'license', 'monthly', 'annual']):
                flat_item["pattern type"] = "subscription"
            elif any(word in narration for word in ['office', 'supplies', 'equipment', 'stationery']):
                flat_item["pattern type"] = "office_expense"
            elif any(word in narration for word in ['utilities', 'water', 'electric', 'internet', 'phone']):
                flat_item["pattern type"] = "utility_payment"
            elif any(word in narration for word in ['vendor', 'supplier', 'purchase', 'procurement']):
                flat_item["pattern type"] = "vendor_payment"
            else:
                flat_item["pattern type"] = "unknown"

        except Exception as e:
            logger.warning(f"Error in transaction flattening: {e}")
            flat_item = item.copy() if isinstance(item, dict) else {"error": "Processing failed", "raw_data": str(item)}

        return flat_item
    def _prepare_unmatched_data(self, reconciliation_summary: Dict) -> Dict:
        """Enhanced unmatched transaction preparation - FIXED to handle string/dict issues"""
        try:
            # FIXED: Ensure reconciliation_summary is a dictionary
            if not isinstance(reconciliation_summary, dict):
                logger.error(f"Expected dict for reconciliation_summary, got {type(reconciliation_summary)}")
                return {"response_type": "text", "data": "Error: Invalid reconciliation data format."}

            unmatched_ledger = reconciliation_summary.get('unmatched_ledger', [])
            unmatched_bank = reconciliation_summary.get('unmatched_bank', [])

            # FIXED: Ensure these are lists
            if not isinstance(unmatched_ledger, list):
                unmatched_ledger = []
            if not isinstance(unmatched_bank, list):
                unmatched_bank = []

            logger.info(f"Processing unmatched: {len(unmatched_ledger)} ledger + {len(unmatched_bank)} bank transactions")

            if not unmatched_ledger and not unmatched_bank:
                return {"response_type": "text", "data": "✅ Excellent! No unmatched transactions found - perfect reconciliation."}

            all_unmatched_data = []

            # Process ledger transactions with enhanced categorization
            for item in unmatched_ledger:
                if isinstance(item, dict):
                    try:
                        flat_item = self._flatten_transaction_data(item)
                        flat_item["source"] = "Ledger"
                        if 'reason' not in flat_item or not flat_item.get('reason'):
                            flat_item["reason"] = "No suitable bank match found"
                        all_unmatched_data.append(flat_item)
                    except Exception as e:
                        logger.warning(f"Error processing ledger transaction: {e}")

            # Process bank transactions with enhanced categorization
            for item in unmatched_bank:
                if isinstance(item, dict):
                    try:
                        flat_item = self._flatten_transaction_data(item)
                        flat_item["source"] = "Bank"
                        if 'reason' not in flat_item or not flat_item.get('reason'):
                            flat_item["reason"] = "No matching ledger entry found"
                        all_unmatched_data.append(flat_item)
                    except Exception as e:
                        logger.warning(f"Error processing bank transaction: {e}")

            if not all_unmatched_data:
                return {"response_type": "text", "data": "Error processing unmatched transaction data."}

            # Apply smart deduplication
            all_unmatched_data = self._deduplicate_transactions(all_unmatched_data)

            # Organize headers for better readability
            all_headers = set()
            for item in all_unmatched_data:
                if isinstance(item, dict):
                    all_headers.update(item.keys())

            # Enhanced header ordering for business value
            preferred_order = [
                "date", "amount", "narration", "ref no", "source",
                "pattern type", "reason"
            ]
            headers_ordered = [h for h in preferred_order if h in all_headers]
            headers_ordered.extend([h for h in sorted(all_headers) if h not in headers_ordered])

            logger.info(f"Prepared {len(all_unmatched_data)} unique unmatched transactions")

            return {
                "response_type": "table",
                "data": {
                    "title": f"Unmatched Transactions ({len(all_unmatched_data)} items)",
                    "subtitle": f"Ledger: {len(unmatched_ledger)} | Bank: {len(unmatched_bank)}",
                    "headers": headers_ordered,
                    "rows": all_unmatched_data
                }
            }

        except Exception as e:
            logger.error(f"Error preparing unmatched data: {e}")
            return {"response_type": "text", "data": f"Error processing unmatched transactions: {str(e)}"}

    def _prepare_matched_data(self, reconciliation_summary: Dict) -> Dict:
        """Enhanced matched transaction preparation - FIXED"""
        try:
            # FIXED: Ensure reconciliation_summary is a dictionary
            if not isinstance(reconciliation_summary, dict):
                logger.error(f"Expected dict for reconciliation_summary, got {type(reconciliation_summary)}")
                return {"response_type": "text", "data": "Error: Invalid reconciliation data format."}

            matched_data = reconciliation_summary.get('matched', [])

            # FIXED: Ensure matched_data is a list
            if not isinstance(matched_data, list):
                matched_data = []

            logger.info(f"Processing matched transactions: {len(matched_data)} pairs")

            if not matched_data:
                return {"response_type": "text", "data": "No matched transactions found in the reconciliation data."}

            # Process ALL matched transactions without aggressive deduplication
            processed_matches = []
            for idx, item in enumerate(matched_data):
                try:
                    if isinstance(item, dict):
                        # Clean and flatten the data structure
                        clean_item = {}
                        for key, value in item.items():
                            try:
                                # Handle different data types
                                if isinstance(value, list):
                                    clean_item[key] = ', '.join([str(v) for v in value[:3]]) if value else ''  # Limit list length
                                elif isinstance(value, dict):
                                    # Convert nested dicts to readable format
                                    if len(str(value)) > 100:
                                        clean_item[key] = f"Complex data ({len(value)} fields)"
                                    else:
                                        clean_item[key] = str(value)
                                elif value is None:
                                    clean_item[key] = ''
                                else:
                                    clean_item[key] = value
                            except Exception as field_error:
                                logger.debug(f"Field processing error for {key}: {field_error}")
                                clean_item[key] = str(value) if value is not None else ''

                        # Add processing index to ensure uniqueness if needed
                        clean_item['_match_index'] = idx
                        processed_matches.append(clean_item)

                except Exception as e:
                    logger.warning(f"Error processing matched transaction {idx}: {e}")
                    # Fallback: include raw data with error indicator
                    fallback_item = item if isinstance(item, dict) else {'raw_data': str(item)}
                    fallback_item['_processing_error'] = str(e)
                    fallback_item['_match_index'] = idx
                    processed_matches.append(fallback_item)

            if not processed_matches:
                return {"response_type": "text", "data": "Error: Could not process any matched transaction data."}

            # Get all available headers intelligently
            all_headers = set()
            for item in processed_matches:
                if isinstance(item, dict):
                    # Skip internal processing fields from headers
                    headers = {k for k in item.keys() if not k.startswith('_')}
                    all_headers.update(headers)

            # Smart header ordering for matched transactions
            preferred_order = [
                "ledger_date", "ledger_amount", "ledger_narration", "ledger_ref_no", "ledger ref no",
                "bank_date", "bank_amount", "bank_narration", "bank_ref_no", "bank ref no",
                "similarity_score", "match_type", "confidence", "match_confidence",
                "amount_diff", "date_diff", "match_score", "match_reasons"
            ]

            headers_ordered = [h for h in preferred_order if h in all_headers]
            headers_ordered.extend([h for h in sorted(all_headers) if h not in headers_ordered])

            # Remove internal fields from display
            final_rows = []
            for item in processed_matches:
                clean_row = {k: v for k, v in item.items() if not k.startswith('_')}
                final_rows.append(clean_row)

            logger.info(f"Successfully prepared {len(final_rows)} matched transactions for display")

            return {
                "response_type": "table",
                "data": {
                    "title": f"Matched Transactions ({len(final_rows)} pairs)",
                    "subtitle": "Successfully paired ledger and bank transactions",
                    "headers": headers_ordered,
                    "rows": final_rows
                }
            }

        except Exception as e:
            logger.error(f"Critical error preparing matched data: {e}")
            return {"response_type": "text", "data": f"Error processing matched transactions: {str(e)}. Please check the reconciliation data format."}

    def _openai_generate_insights(self, reconciliation_summary: Dict) -> str:
        """Enhanced AI insights generation - FIXED string formatting issues"""
        if not openai_client:
            return self._fallback_insights(reconciliation_summary)

        try:
            # FIXED: Ensure reconciliation_summary is a dictionary
            if not isinstance(reconciliation_summary, dict):
                logger.error(f"Expected dict for reconciliation_summary, got {type(reconciliation_summary)}")
                return "Error: Invalid reconciliation data format for insights generation."

            unmatched_ledger = reconciliation_summary.get('unmatched_ledger', [])
            unmatched_bank = reconciliation_summary.get('unmatched_bank', [])
            matched = reconciliation_summary.get('matched', [])
            summary = reconciliation_summary.get('summary', {})

            # FIXED: Ensure all are lists/dicts
            if not isinstance(unmatched_ledger, list):
                unmatched_ledger = []
            if not isinstance(unmatched_bank, list):
                unmatched_bank = []
            if not isinstance(matched, list):
                matched = []
            if not isinstance(summary, dict):
                summary = {}

            # Enhanced analysis with pattern recognition
            analysis_data = {
                "matched_count": len(matched),
                "unmatched_ledger_count": len(unmatched_ledger),
                "unmatched_bank_count": len(unmatched_bank),
                "match_rate": safe_float(summary.get("match_rate_percentage", 0)),
                "processing_time": safe_float(summary.get("processing_time", 0)),
            }

            # Get transaction patterns for better insights
            ledger_patterns = self._analyze_transaction_patterns(unmatched_ledger, "Ledger")
            bank_patterns = self._analyze_transaction_patterns(unmatched_bank, "Bank")

            # FIXED: Use safe string formatting instead of f-strings on potentially non-string values
            matched_count = analysis_data['matched_count']
            unmatched_ledger_count = analysis_data['unmatched_ledger_count']
            unmatched_bank_count = analysis_data['unmatched_bank_count']
            match_rate = analysis_data['match_rate']
            processing_time = analysis_data['processing_time']

            # Enhanced insights prompt with pattern data
            insights_prompt = f"""Analyze this financial reconciliation with advanced pattern recognition:

**RECONCILIATION METRICS:**
- Successfully Matched: {matched_count} transaction pairs
- Unmatched Ledger: {unmatched_ledger_count} entries (need review)
- Unmatched Bank: {unmatched_bank_count} entries (normal operational)
- Match Success Rate: {match_rate:.1f}%
- Processing Time: {processing_time:.2f} seconds

**TRANSACTION PATTERNS DETECTED:**
Unmatched Ledger Categories: {ledger_patterns.get('categories', {})}
Unmatched Bank Categories: {bank_patterns.get('categories', {})}

**BUSINESS CONTEXT:**
The higher bank unmatched count ({unmatched_bank_count} vs {unmatched_ledger_count}) is NORMAL and expected because banks record ALL activity (fees, interest, micro-transactions) while ledgers focus on business-relevant transactions.

**ANALYSIS REQUIREMENTS:**
Provide a comprehensive business-focused analysis with:

1. **Executive Summary** - Overall reconciliation health assessment
2. **Pattern Insights** - What the transaction categories reveal about business operations
3. **Risk Assessment** - Which unmatched items need immediate attention vs. normal operational activity
4. **Action Plan** - Prioritized next steps with business impact reasoning
5. **Performance Rating** - Grade the reconciliation quality (A-F scale)

Write as a financial expert providing actionable business intelligence. Use professional language with strategic insights that help business decision-making."""

            response = openai_client.chat.completions.create(
                model=self.chat_config["config"]["primary"],
                messages=[{"role": "user", "content": insights_prompt}],
                max_tokens=1200,
                temperature=0.3  # Slightly higher for more nuanced insights
            )

            insights = response.choices[0].message.content.strip()

            if not insights or len(insights.strip()) < 100:
                logger.warning("OpenAI returned insufficient insights, using enhanced fallback")
                return self._enhanced_fallback_insights(reconciliation_summary, ledger_patterns, bank_patterns)

            logger.info("OpenAI insights with pattern analysis generated successfully")
            return insights

        except Exception as e:
            logger.error(f"OpenAI insights generation error: {e}")
            return self._enhanced_fallback_insights(reconciliation_summary, None, None)

    def _enhanced_fallback_insights(self, reconciliation_summary: Dict, ledger_patterns: Dict = None, bank_patterns: Dict = None) -> str:
        """Enhanced fallback insights with pattern analysis - FIXED"""
        try:
            # FIXED: Ensure reconciliation_summary is a dictionary
            if not isinstance(reconciliation_summary, dict):
                logger.error(f"Expected dict for reconciliation_summary, got {type(reconciliation_summary)}")
                return "**Reconciliation Analysis Available**\n\nUnable to generate detailed insights due to data format error. Please check your reconciliation data."

            unmatched_ledger = reconciliation_summary.get('unmatched_ledger', [])
            unmatched_bank = reconciliation_summary.get('unmatched_bank', [])
            matched = reconciliation_summary.get('matched', [])
            summary = reconciliation_summary.get('summary', {})

            # FIXED: Ensure all are proper types
            if not isinstance(unmatched_ledger, list):
                unmatched_ledger = []
            if not isinstance(unmatched_bank, list):
                unmatched_bank = []
            if not isinstance(matched, list):
                matched = []
            if not isinstance(summary, dict):
                summary = {}

            match_rate = safe_float(summary.get('match_rate_percentage', 0))
            total_transactions = len(matched) + len(unmatched_ledger) + len(unmatched_bank)

            # Generate patterns if not provided
            if not ledger_patterns:
                ledger_patterns = self._analyze_transaction_patterns(unmatched_ledger, "Ledger")
            if not bank_patterns:
                bank_patterns = self._analyze_transaction_patterns(unmatched_bank, "Bank")

            # Performance grading
            if match_rate >= 90:
                grade = "A+"
                status = "🏆 EXCELLENT"
            elif match_rate >= 80:
                grade = "A"
                status = "🌟 VERY GOOD"
            elif match_rate >= 70:
                grade = "B+"
                status = "✅ GOOD"
            elif match_rate >= 60:
                grade = "B"
                status = "📊 ACCEPTABLE"
            elif match_rate >= 50:
                grade = "C"
                status = "⚠️ NEEDS IMPROVEMENT"
            else:
                grade = "D"
                status = "🔴 REQUIRES ATTENTION"

            # Category analysis for insights
            ledger_categories = ledger_patterns.get('categories', {})
            bank_categories = bank_patterns.get('categories', {})

            priority_categories = []
            if ledger_categories.get('client_payments', {}).get('count', 0) > 0:
                priority_categories.append("Client payment discrepancies detected")
            if ledger_categories.get('vendor_payments', {}).get('count', 0) > 0:
                priority_categories.append("Vendor payment mismatches found")

            # FIXED: Use safe formatting for high_value_threshold
            high_value_threshold = safe_float(ledger_patterns.get('high_value_threshold', 1000))

            return f"""📊 **ADVANCED RECONCILIATION ANALYSIS REPORT**

**🎯 EXECUTIVE SUMMARY:**
Reconciliation processed {total_transactions:,} transactions achieving a **{match_rate:.1f}%** success rate with **{len(matched)} successful matches**. Overall performance: **{status}** (Grade: **{grade}**)

**📈 PERFORMANCE METRICS:**
• **Match Success Rate:** {match_rate:.1f}%
• **Successfully Matched:** {len(matched):,} transaction pairs
• **Unmatched Ledger:** {len(unmatched_ledger):,} entries (⚠️ **Priority Review**)
• **Unmatched Bank:** {len(unmatched_bank):,} entries (✅ **Normal Operational**)
• **Processing Efficiency:** {summary.get('processing_time', 0):.2f} seconds

**🔍 PATTERN ANALYSIS & BUSINESS INTELLIGENCE:**

*Unmatched Ledger Breakdown:*
{self._format_category_insights(ledger_categories, "Ledger")}

*Unmatched Bank Breakdown:*
{self._format_category_insights(bank_categories, "Bank")}

**💡 KEY BUSINESS INSIGHTS:**

1. **Normal Bank-Heavy Pattern:** ✅ The {len(unmatched_bank)} bank-only transactions vs {len(unmatched_ledger)} ledger-only is **healthy and expected**
   - Banks capture: fees, interest, automated payments, micro-transactions
   - Ledgers capture: business-relevant transactions only
   - This asymmetry is normal business operations

2. **Risk Assessment:**
   - **🔴 HIGH PRIORITY:** {len(unmatched_ledger)} unmatched ledger entries need review
   {f"- **⚠️ ATTENTION:** {priority_categories[0] if priority_categories else 'Review transaction categories'}" if priority_categories else "- **✅ LOW RISK:** No critical transaction types detectedd"}
   - **🟡 MEDIUM PRIORITY:** {len(unmatched_bank)} bank entries for categorization
   - **✅ LOW RISK:** Matched transactions are properly reconciled

**🚀 STRATEGIC ACTION PLAN:**

**Phase 1 - Immediate (Next 1-2 days):**
1. **Review {len(unmatched_ledger)} unmatched ledger transactions** - Check for data entry errors, missing bank deposits, timing differences
2. **Validate high-value discrepancies** - Focus on amounts > ${high_value_threshold:.0f}

**Phase 2 - Short-term (Next week):**
1. **Categorize {len(unmatched_bank)} bank-only transactions** - Set up automated rules for bank fees, interest, recurring charges
2. **Process optimization** - Create matching rules for identified patterns

**Phase 3 - Long-term (Next month):**
1. **Automation setup** - Configure rules for {len(ledger_categories) + len(bank_categories)} detected transaction patterns
2. **Performance monitoring** - Track improvement in match rates over time

**📋 QUALITY ASSESSMENT:**

**Strengths:**
• {f"Strong match rate of {match_rate:.1f}%" if match_rate >= 70 else f"Baseline match rate of {match_rate:.1f}% established"}
• Clean data processing with comprehensive pattern recognition
• Normal operational transaction distribution

**Areas for Improvement:**
{f"• Focus on {len(unmatched_ledger)} unmatched ledger entries" if len(unmatched_ledger) > 0 else "• Continue monitoring for data quality"}
• Enhanced automated matching rules could improve efficiency
• Regular reconciliation frequency optimization

**🎯 BOTTOM LINE:**
{f"**EXCELLENT RECONCILIATION** - Your financial data is well-managed with {match_rate:.1f}% accuracy. The unmatched transactions follow normal business patterns." if match_rate >= 80 else f"**SOLID FOUNDATION** - {match_rate:.1f}% match rate provides good baseline. Focus on unmatched ledger items for improvement." if match_rate >= 60 else f"**IMPROVEMENT OPPORTUNITY** - {match_rate:.1f}% match rate indicates data quality issues need attention for better business insights."}

---
*Generated by ReconBot AI • Advanced Pattern Recognition • Business Intelligence Focus*"""

        except Exception as e:
            logger.error(f"Error generating enhanced fallback insights: {e}")
            return "**Reconciliation Analysis Available**\n\nUnable to generate detailed insights due to processing error. Please review the matched and unmatched transactions manually for business insights."

    def _format_category_insights(self, categories: Dict, source: str) -> str:
        """Format transaction category insights for reporting"""
        if not categories or not isinstance(categories, dict):
            return f"  • No {source.lower()} transaction patterns detected"

        insights = []
        for category, data in sorted(categories.items(), key=lambda x: x[1].get('count', 0), reverse=True):
            count = safe_int(data.get('count', 0))
            total = safe_float(data.get('total', 0))
            if count > 0:
                category_name = category.replace('_', ' ').title()
                insights.append(f"  • **{category_name}:** {count} transactions (${abs(total):,.2f})")

        return '\n'.join(insights) if insights else f"  • No {source.lower()} patterns detected"

    def _openai_general_analysis(self, combined_df: pd.DataFrame, query: str, intent_data: Dict) -> str:
        """Enhanced OpenAI general analysis - FIXED"""
        if not openai_client:
            return self._fallback_analysis(combined_df, query, intent_data)

        try:
            # Apply date filtering if requested
            original_count = len(combined_df)
            start, end = parse_natural_date_range(query)

            if start and end:
                if 'date' in combined_df.columns:
                    combined_df['date'] = pd.to_datetime(combined_df['date'], errors='coerce')
                    combined_df = combined_df.dropna(subset=['date'])
                    mask = (combined_df['date'].dt.date >= start) & (combined_df['date'].dt.date <= end)
                    combined_df = combined_df[mask]
                    logger.info(f"Date filter applied: {start} to {end}, {len(combined_df)}/{original_count} transactions remaining")

            # Prepare comprehensive data context
            ledger_count = len(combined_df[combined_df['source'] == 'Ledger'])
            bank_count = len(combined_df[combined_df['source'] == 'Bank'])

            # Sample data for context (limit for token efficiency)
            sample_data = combined_df.head(3).to_dict('records')
            columns = list(combined_df.columns)

            # Get basic statistics
            if 'amount' in combined_df.columns:
                amount_stats = {
                    'total': safe_float(combined_df['amount'].sum()),
                    'avg': safe_float(combined_df['amount'].mean()),
                    'count': len(combined_df)
                }
            else:
                amount_stats = {'total': 0, 'avg': 0, 'count': len(combined_df)}

            # Enhanced analysis based on intent
            if intent_data.get('intent') == "HOW_TO_MATCH":
                analysis_prompt = f"""You are a financial reconciliation expert helping with matching strategies.

USER QUERY: "{query}"

CONTEXT:
- Dataset: {ledger_count} ledger + {bank_count} bank transactions
- Available columns: {columns}
- Date range: {f"{start} to {end}" if start and end else "Full dataset"}

Sample transactions:
{sample_data}

Provide expert guidance on:

1. **Matching Strategy** - Best approach for this specific data structure
2. **Key Fields Analysis** - Which columns are most reliable for matching
3. **Common Challenges** - Potential issues with this data format
4. **Best Practices** - Proven techniques for similar transaction types
5. **Specific Recommendations** - Actionable steps based on their actual data

Focus on practical, implementable advice. Be specific about their data structure."""

            elif "date" in query.lower() or start or end:
                analysis_prompt = f"""Analyze this financial transaction data with focus on date-based insights.

USER QUERY: "{query}"
DATE FILTER: {f"Applied {start} to {end}" if start and end else "No date filter"}

DATASET:
- Ledger transactions: {ledger_count}
- Bank transactions: {bank_count}
- Total amount: ${amount_stats['total']:,.2f}
- Average amount: ${amount_stats['avg']:,.2f}
- Columns: {columns}

Provide insights about:
1. **Date Range Analysis** - Transaction patterns over the specified period
2. **Volume Trends** - Transaction frequency and patterns
3. **Business Insights** - What the data reveals about operations
4. **Data Quality** - Assessment of completeness and consistency

Be specific about date-related patterns and business implications."""

            else:
                analysis_prompt = f"""Analyze this financial transaction dataset and provide comprehensive insights.

USER QUERY: "{query}"

DATASET OVERVIEW:
- Total transactions: {len(combined_df):,}
- Ledger records: {ledger_count:,}
- Bank records: {bank_count:,}
- Total amount: ${amount_stats['total']:,.2f}
- Columns available: {columns}

Sample data structure:
{sample_data}

Provide analysis covering:
1. **Data Quality Assessment** - Completeness, patterns, potential issues
2. **Business Insights** - What the transactions reveal about operations
3. **Recommendations** - Next steps for analysis or reconciliation
4. **Notable Patterns** - Any interesting trends or anomalies

Focus on actionable insights and business value."""

            response = openai_client.chat.completions.create(
                model=self.chat_config["config"]["primary"],
                messages=[{"role": "user", "content": analysis_prompt}],
                max_tokens=1000,
                temperature=0.3
            )

            analysis_result = response.choices[0].message.content.strip()

            # Add date filter info if applied
            if start and end:
                filter_info = f"\n\n📅 **Date Filter Applied:** {start} to {end} ({len(combined_df)} of {original_count} transactions)"
                analysis_result += filter_info

            logger.info("OpenAI general analysis completed successfully")
            return analysis_result

        except Exception as e:
            logger.error(f"OpenAI general analysis error: {e}")
            return self._fallback_analysis(combined_df, query, intent_data)

    def _fallback_analysis(self, combined_df: pd.DataFrame, query: str, intent_data: Dict) -> str:
        """Enhanced fallback analysis - FIXED"""
        try:
            ledger_count = len(combined_df[combined_df['source'] == 'Ledger'])
            bank_count = len(combined_df[combined_df['source'] == 'Bank'])

            # Check for date filtering
            start, end = parse_natural_date_range(query)
            date_info = f" (filtered {start} to {end})" if start and end else ""

            intent = intent_data.get('intent', 'GENERAL_ANALYSIS')

            if intent == "HOW_TO_MATCH":
                return f"""🎯 **MATCHING STRATEGY GUIDE**

**📊 Your Data Analysis:**
• **Ledger Transactions:** {ledger_count:,}
• **Bank Transactions:** {bank_count:,}
• **Available Columns:** {list(combined_df.columns)}
• **Balance Assessment:** {"✅ Well balanced" if abs(ledger_count - bank_count) < 50 else "⚠️ Significant imbalance - investigate data completeness"}

**🔑 RECOMMENDED MATCHING STRATEGY:**

**Phase 1: High-Confidence Matches**
1. **Exact Amount + Date Match** (±1 day tolerance)
   - Most reliable matching method
   - Handle processing delays automatically

2. **Reference Number Matching**
   - Check: `ref_no`, `reference`, `transaction_id` fields
   - Clean and normalize reference formats

**Phase 2: Pattern-Based Matching**
3. **Company Name Normalization**
   - Remove variations: "Inc", "Corp", "Ltd", "LLC"
   - Handle abbreviations and spacing differences

4. **Amount + Narration Keywords**
   - Match on amount + key terms in descriptions
   - Use fuzzy matching for company names

**Phase 3: Advanced Techniques**
5. **Date Window Matching** (±3 days)
   - Account for weekend processing delays
   - Bank holidays and cut-off times

6. **Recurring Transaction Rules**
   - Identify monthly subscriptions, utilities
   - Set up automated matching rules

**⚠️ COMMON CHALLENGES WITH YOUR DATA:**
• **Date Format Variations** - Standardize before matching
• **Amount Precision** - Handle decimal place differences
• **Incomplete References** - Use partial matching strategies
• **Processing Delays** - Allow date tolerance windows

**🚀 IMPLEMENTATION STEPS:**
1. **Start Simple** - Run exact matches first
2. **Analyze Unmatched** - Identify patterns in failures
3. **Iterative Improvement** - Add rules based on patterns
4. **Manual Review** - Validate automated matches

**🎯 SUCCESS METRICS:**
• Target: 70%+ automated match rate
• Goal: <5% manual review required
• Monitor: Processing time vs. accuracy balance

Would you like specific guidance on any of these matching strategies?"""

            elif "total" in query.lower() or "amount" in query.lower():
                # Calculate amounts if available
                if 'amount' in combined_df.columns:
                    total_amount = safe_float(combined_df['amount'].sum())
                    avg_amount = safe_float(combined_df['amount'].mean())
                    ledger_total = safe_float(combined_df[combined_df['source'] == 'Ledger']['amount'].sum())
                    bank_total = safe_float(combined_df[combined_df['source'] == 'Bank']['amount'].sum())

                    return f"""💰 **FINANCIAL ANALYSIS SUMMARY**{date_info}

**📊 Transaction Totals:**
• **Total Amount:** ${total_amount:,.2f}
• **Average Transaction:** ${avg_amount:,.2f}
• **Transaction Count:** {len(combined_df):,}

**📈 Breakdown by Source:**
• **Ledger Total:** ${ledger_total:,.2f} ({ledger_count:,} transactions)
• **Bank Total:** ${bank_total:,.2f} ({bank_count:,} transactions)

**🔍 Next Analysis Options:**
• Run reconciliation to identify matches and discrepancies
• Analyze transaction patterns by date ranges
• Review high-value transactions for accuracy

Ready to reconcile these transactions?"""
                else:
                    return f"Amount analysis requires 'amount' column in your data. Available columns: {list(combined_df.columns)}"

            else:
                return f"""📊 **DATASET ANALYSIS**{date_info}

**📋 Data Overview:**
• **Total Transactions:** {len(combined_df):,}
• **Ledger Records:** {ledger_count:,}
• **Bank Records:** {bank_count:,}
• **Available Fields:** {len(combined_df.columns)} columns

**🔍 Data Quality Assessment:**
• **Balance Check:** {"✅ Balanced dataset" if abs(ledger_count - bank_count) < 100 else "⚠️ Imbalanced - review data completeness"}
• **Column Structure:** {list(combined_df.columns)[:5]}{"..." if len(combined_df.columns) > 5 else ""}
• **Data Readiness:** {"✅ Ready for reconciliation" if 'amount' in combined_df.columns and 'date' in combined_df.columns else "⚠️ Missing key fields (amount/date)"}

**💡 RECOMMENDED NEXT STEPS:**

**For Analysis:**
• "Show me transactions from last month"
• "What's the total amount by source?"
• "How many high-value transactions are there?"

**For Reconciliation:**
• "How do I match these transactions?"
• Run reconciliation to identify matches/discrepancies
• Analyze reconciliation results for insights

**For Insights:**
• "Analyze the reconciliation results"
• "Show me unmatched transactions"
• "What patterns do you see?"

What specific analysis would you like to perform?"""

        except Exception as e:
            logger.error(f"Error in fallback analysis: {e}")
            return f"Error analyzing data: {str(e)}. Please try a more specific question or check your data format."

    def run_chat_agent(
        self,
        ledger_df: pd.DataFrame,
        bank_df: pd.DataFrame,
        query: str,
        reconciliation_summary: dict = None
    ) -> Dict[str, Any]:
        """Enhanced OpenAI-powered smart intent-based chat agent v6.2 - FIXED ALL ERRORS"""

        try:
            logger.info(f"🤖 Chat Agent v6.2 processing: '{query}' | Reconciliation: {reconciliation_summary is not None}")

            # STEP 1: Optimized intent classification with smart pre-filtering
            intent_data = self._openai_classify_intent(query, reconciliation_summary is not None)
            intent = intent_data.get('intent', 'GENERAL_ANALYSIS')

            logger.info(f"🎯 Intent classified: {intent} (confidence: {intent_data.get('confidence', 0)}%)")

            # STEP 2: Enhanced routing with comprehensive data handling
            if intent == "SHOW_UNMATCHED":
                if reconciliation_summary:
                    result = self._prepare_unmatched_data(reconciliation_summary)
                    logger.info(f"📊 Unmatched data prepared: {result.get('response_type')} with {len(result.get('data', {}).get('rows', []))} items")
                    return result
                else:
                    return {
                        "response_type": "text",
                        "data": "📄 **Reconciliation Required**\n\nI need reconciliation results to show unmatched transactions. Please run a reconciliation first using the AI-powered engine to unlock this analysis.\n\n**Available Now:**\n• General data analysis\n• Matching guidance\n• Data quality assessment"
                    }

            elif intent == "SHOW_MATCHED":
                if reconciliation_summary:
                    result = self._prepare_matched_data(reconciliation_summary)
                    logger.info(f"✅ Matched data prepared: {result.get('response_type')} with {len(result.get('data', {}).get('rows', []))} pairs")
                    return result
                else:
                    return {
                        "response_type": "text",
                        "data": "📄 **Reconciliation Required**\n\nI need reconciliation results to show matched transactions. Please run a reconciliation first to see successfully paired transactions.\n\n**What I can do now:**\n• Analyze your raw transaction data\n• Provide matching strategies\n• Assess data quality"
                    }

            elif intent == "SHOW_DISCREPANCIES":
                if reconciliation_summary:
                    result = self._prepare_unmatched_data(reconciliation_summary)
                    if result.get("response_type") == "table":
                        result["data"]["title"] = f"🔍 Financial Discrepancies ({len(result['data']['rows'])} items)"
                        result["data"]["subtitle"] = "Transactions requiring review and resolution"
                    return result
                else:
                    return {
                        "response_type": "text",
                        "data": "🔍 **Discrepancy Analysis Requires Reconciliation**\n\nI need reconciliation results to identify discrepancies and unmatched transactions. Run reconciliation first to unlock:\n\n• ⚠️ Transaction discrepancies\n• 🔍 Missing entries analysis\n• 📊 Pattern-based insights\n• 🎯 Priority review recommendations"
                    }

            elif intent == "ANALYZE_RECONCILIATION":
                if reconciliation_summary:
                    try:
                        insights = self._openai_generate_insights(reconciliation_summary)
                        logger.info("🧠 AI insights generated successfully with pattern analysis")
                        return {"response_type": "text", "data": insights}
                    except Exception as e:
                        logger.error(f"Insights generation failed, using enhanced fallback: {e}")
                        fallback_insights = self._enhanced_fallback_insights(reconciliation_summary)
                        return {"response_type": "text", "data": fallback_insights}
                else:
                    return {
                        "response_type": "text",
                        "data": "📈 **Advanced Analysis Awaiting Reconciliation**\n\nI need reconciliation results to provide comprehensive business insights. Once you run reconciliation, I can deliver:\n\n• 🎯 **Executive Summary** - Performance assessment\n•    **Pattern Analysis** - Transaction categorization\n• ⚠️ **Risk Assessment** - Priority items for review\n• 🚀 **Action Plan** - Straategic next steps\n• 📈 **Business Intelligence** - Operational insights\n\nRun reconciliation to unlock full AI-powered analysis!"
                    }

            else:  # GENERAL_ANALYSIS, HOW_TO_MATCH, UNCLEAR
                try:
                    # Combine datasets for comprehensive analysis
                    combined_df = pd.concat([
                        ledger_df.assign(source="Ledger"),
                        bank_df.assign(source="Bank")
                    ], ignore_index=True)

                    if len(combined_df) == 0:
                        return {
                            "response_type": "text",
                            "data": "⚠️ **No Data Available**\n\nNo transaction data found to analyze. Please upload both ledger and bank transaction files to begin analysis."
                        }

                    logger.info("🔍 Executing enhanced OpenAI analysis with full feature set...")
                    response_text = self._openai_general_analysis(combined_df, query, intent_data)

                    if not response_text or len(response_text.strip()) < 20:
                        response_text = self._fallback_analysis(combined_df, query, intent_data)

                    logger.info("✅ General analysis completed successfully")
                    return {"response_type": "text", "data": response_text}

                except Exception as analysis_error:
                    logger.error(f"Analysis error: {analysis_error}")

                    # FIXED: Enhanced error handling WITHOUT system status exposure
                    return {
                        "response_type": "text",
                        "data": f"""⚠️ **Analysis Temporarily Unavailable**

I encountered a technical issue, but I can still help!

**🔄 Quick Solutions:**
• **"Show me matched transactions"** - View successful pairs
• **"Show me unmatched transactions"** - Review discrepancies
• **"How do I match these transactions?"** - Get matching strategies
• **"Analyze the reconciliation results"** - Get AI insights

**📊 Current Status:**
• Data loaded: {len(ledger_df)} ledger + {len(bank_df)} bank transactions
• Reconciliation: {"✅ Available for analysis" if reconciliation_summary else "⏳ Run reconciliation for full insights"}

Try one of the suggested commands above, or ask a more specific question!"""
                    }

        except Exception as e:
            logger.error(f"🚨 Critical error in chat agent v6.2: {e}")
            # FIXED: Removed controversial system status information
            return {
                "response_type": "text",
                "data": f"""🚨 **System Error**

I encountered an unexpected error but I'm still here to help!

**🔄 Recovery Options:**
• Try: "Show me the data summary"
• Try: "How many transactions do I have?"
• Try: "What columns are in my data?"

**📊 Current Status:**
• Data Status: {len(ledger_df) if ledger_df is not None else 0} ledger + {len(bank_df) if bank_df is not None else 0} bank transactions

Please try a simpler question or contact support if the issue persists."""
            }

# Initialize global agent instance
try:
    openai_chat_agent = OpenAIChatAgent()
    logger.info("✅ OpenAI Chat Agent v6.2 initialized successfully")
except Exception as init_error:
    logger.error(f"❌ Failed to initialize OpenAI Chat Agent: {init_error}")
    openai_chat_agent = None

def run_chat_agent(
    ledger_df: pd.DataFrame,
    bank_df: pd.DataFrame,
    query: str,
    reconciliation_summary: dict = None
) -> Dict[str, Any]:
    """Enhanced OpenAI-powered interface for chat agent v6.2 - FIXED"""
    if not openai_chat_agent:
        return {
            "response_type": "text",
            "data": "❌ Chat agent initialization failed. Please check configuration and try again."
        }

    return openai_chat_agent.run_chat_agent(
        ledger_df,
        bank_df,
        query,
        reconciliation_summary
    )

def get_chat_agent_info() -> Dict[str, str]:
    """Get information about chat agent v6.2 configuration - FIXED to remove controversial details"""
    if not openai_chat_agent:
        return {
            "provider": "openai",
            "model": "ai_assistant",
            "status": "initialization_failed",
            "version": "6.2"
        }

    return {
        "provider": "openai",
        "model": "ai_assistant",  # FIXED: Generic name instead of specific model
        "status": "active" if openai_client else "error",
        "version": "6.2",
        "features": "enhanced_patterns,smart_deduplication,date_filtering,comprehensive_analysis"
    }