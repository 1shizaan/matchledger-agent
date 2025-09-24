# app/utils/column_mapping.py - COMPLETE COLUMN ALIASING SOLUTION
import pandas as pd
import re
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class SmartColumnMapper:
    """
    Enhanced column mapper that handles multiple variations of column names
    """
    
    def __init__(self):
        # Define comprehensive column aliases
        self.column_aliases = {
            'date': [
                'date', 'transaction_date', 'txn_date', 'trans_date', 'posting_date',
                'value_date', 'effective_date', 'created_date', 'timestamp', 'created',
                'dt', 'transaction_dt', 'posting_dt'
            ],

            'amount': [
                'amount', 'transaction_amount', 'txn_amount', 'value', 'sum', 'total',
                'balance', 'credit', 'debit', 'price', 'cost', 'payment_amount',
                'transfer_amount', 'withdrawal', 'deposit'
            ],

            'narration': [
                'narration', 'description', 'details', 'memo', 'note', 'remark',
                'comment', 'particular', 'particulars', 'narrative', 'desc',
                'transaction_description', 'txn_desc', 'payment_description',
                'reference', 'purpose', 'reason'
            ],

            'ref_no': [
                # Primary reference variations
                'ref_no', 'reference_no', 'reference_number', 'ref', 'reference',

                # Transaction ID variations
                'transaction_id', 'txn_id', 'trans_id', 'transaction_ref',
                'txn_ref', 'trans_ref', 'transaction_reference',

                # Document variations
                'voucher_no', 'voucher', 'voucher_number',
                'receipt_no', 'receipt', 'receipt_number',
                'invoice_no', 'invoice', 'invoice_number',
                'bill_no', 'bill_number',

                # Payment variations
                'payment_id', 'payment_ref', 'payment_reference',
                'transfer_id', 'transfer_ref', 'transfer_reference',
                'batch_id', 'batch_ref', 'batch_reference',

                # Check variations
                'check_no', 'check_number', 'cheque_no', 'cheque_number',

                # Other variations
                'serial_no', 'serial_number', 'document_no', 'doc_no',
                'confirmation_no', 'tracking_no', 'trace_no'
            ]
        }

        # Confidence scoring for each match
        self.confidence_weights = {
            'exact_match': 100,
            'case_insensitive': 95,
            'space_normalized': 90,
            'underscore_normalized': 85,
            'partial_match': 75,
            'keyword_match': 60
        }

    def normalize_column_name(self, column_name: str) -> str:
        """Normalize column name for comparison"""
        if not isinstance(column_name, str):
            return str(column_name)

        # Convert to lowercase and normalize spaces/underscores
        normalized = column_name.lower().strip()
        normalized = re.sub(r'[\s\-_]+', '_', normalized)
        normalized = re.sub(r'[^\w]', '_', normalized)
        normalized = re.sub(r'_+', '_', normalized)
        normalized = normalized.strip('_')

        return normalized

    def find_column_match(self, available_columns: List[str], target_field: str) -> Dict[str, Any]:
        """
        Find best matching column for target field
        """
        if target_field not in self.column_aliases:
            return {'column': None, 'confidence': 0, 'match_type': 'none'}

        target_aliases = self.column_aliases[target_field]
        best_match = {'column': None, 'confidence': 0, 'match_type': 'none'}

        for col in available_columns:
            col_normalized = self.normalize_column_name(col)

            for alias in target_aliases:
                alias_normalized = self.normalize_column_name(alias)
                confidence = 0
                match_type = 'none'

                # Exact match (highest confidence)
                if col.lower() == alias.lower():
                    confidence = self.confidence_weights['exact_match']
                    match_type = 'exact_match'

                # Case insensitive match
                elif col.lower() == alias.lower():
                    confidence = self.confidence_weights['case_insensitive']
                    match_type = 'case_insensitive'

                # Normalized match (handles spaces, underscores)
                elif col_normalized == alias_normalized:
                    confidence = self.confidence_weights['space_normalized']
                    match_type = 'space_normalized'

                # Partial match (one contains the other)
                elif alias_normalized in col_normalized or col_normalized in alias_normalized:
                    confidence = self.confidence_weights['partial_match']
                    match_type = 'partial_match'

                # Keyword match (contains key parts)
                elif any(part in col_normalized for part in alias_normalized.split('_')):
                    confidence = self.confidence_weights['keyword_match']
                    match_type = 'keyword_match'

                # Update best match if this is better
                if confidence > best_match['confidence']:
                    best_match = {
                        'column': col,
                        'confidence': confidence,
                        'match_type': match_type,
                        'matched_alias': alias
                    }

        return best_match

    def map_dataframe_columns(self, df: pd.DataFrame, source_name: str = "") -> Dict[str, Any]:
        """
        Map DataFrame columns to standard names with detailed reporting
        """
        if df is None or len(df.columns) == 0:
            return {'mapped_df': df, 'mapping': {}, 'confidence': {}}

        logger.info(f"🔍 Mapping columns for {source_name}")
        logger.info(f"  Available columns: {list(df.columns)}")

        # Find matches for each target field
        field_mappings = {}
        confidence_scores = {}

        for field in ['date', 'amount', 'narration', 'ref_no']:
            match_result = self.find_column_match(list(df.columns), field)

            if match_result['column']:
                field_mappings[field] = match_result['column']
                confidence_scores[field] = match_result['confidence']

                logger.info(f"  ✅ {field}: '{match_result['column']}' (confidence: {match_result['confidence']}, type: {match_result['match_type']})")
            else:
                field_mappings[field] = None
                confidence_scores[field] = 0
                logger.warning(f"  ❌ {field}: No suitable column found")

        # Create mapped DataFrame
        mapped_df = df.copy()
        column_rename_map = {}

        for field, source_col in field_mappings.items():
            if source_col and source_col != field:
                column_rename_map[source_col] = field

        # Rename columns
        if column_rename_map:
            mapped_df = mapped_df.rename(columns=column_rename_map)
            logger.info(f"  �� Column renames: {column_rename_map}")

        # Ensure all expected columns exist (create empty ones if missing)
        for field in ['date', 'amount', 'narration', 'ref_no']:
            if field not in mapped_df.columns:
                mapped_df[field] = None
                logger.info(f"  ➕ Added missing column: {field}")

        return {
            'mapped_df': mapped_df,
            'mapping': field_mappings,
            'confidence': confidence_scores,
            'renames': column_rename_map
        }

# Global instance
column_mapper = SmartColumnMapper()

def smart_map_columns(df: pd.DataFrame, source_name: str = "") -> pd.DataFrame:
    """
    Main function to intelligently map DataFrame columns
    """
    try:
        result = column_mapper.map_dataframe_columns(df, source_name)
        return result['mapped_df']
    except Exception as e:
        logger.error(f"❌ Error mapping columns for {source_name}: {e}")
        return df

def analyze_column_mapping_quality(df: pd.DataFrame, source_name: str = "") -> Dict[str, Any]:
    """
    Analyze the quality of column mapping
    """
    try:
        result = column_mapper.map_dataframe_columns(df, source_name)

        quality_report = {
            'source_name': source_name,
            'total_columns': len(df.columns),
            'mapped_fields': sum(1 for conf in result['confidence'].values() if conf > 0),
            'confidence_scores': result['confidence'],
            'mapping_details': result['mapping'],
            'overall_quality': 'good' if result['confidence'].get('ref_no', 0) > 80 else 'fair' if result['confidence'].get('ref_no', 0) > 50 else 'poor'
        }

        # Special focus on ref_no
        if result['confidence'].get('ref_no', 0) > 0:
            mapped_df = result['mapped_df']
            ref_stats = {
                'total_rows': len(mapped_df),
                'non_null_refs': mapped_df['ref_no'].notna().sum() if 'ref_no' in mapped_df.columns else 0,
                'sample_values': mapped_df['ref_no'].head(5).tolist() if 'ref_no' in mapped_df.columns else [],
                'coverage_percentage': (mapped_df['ref_no'].notna().sum() / len(mapped_df) * 100) if 'ref_no' in mapped_df.columns and len(mapped_df) > 0 else 0
            }
            quality_report['ref_no_analysis'] = ref_stats

        return quality_report

    except Exception as e:
        logger.error(f"❌ Error analyzing column mapping for {source_name}: {e}")
        return {'error': str(e)}