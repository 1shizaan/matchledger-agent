# backend/app/utils/json_utils.py

import pandas as pd
import numpy as np
from datetime import datetime

def convert_df_for_json(obj):
    """
    Enhanced JSON serializer with security checks.
    """
    if isinstance(obj, dict):
        return {k: convert_df_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_df_for_json(elem) for elem in obj]
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    elif isinstance(obj, (pd.DataFrame, pd.Series)):
        return obj.where(pd.notnull(obj), None).to_dict(orient='records')
    elif pd.isna(obj):
        return None
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif hasattr(obj, '__dict__'):
        return convert_df_for_json(obj.__dict__)
    return str(obj)