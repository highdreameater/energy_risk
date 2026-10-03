import pandas as pd
import numpy as np

def clean_financial_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans financial statements, treats negative/zero denominators,
    and removes duplicate records.
    """
    data = df.copy()
    data = data.drop_duplicates(subset=["counterparty_id"])
    
    # Financial sanity bounds
    data["interest_expense"] = np.maximum(data["interest_expense"].fillna(1.0), 0.1)
    data["ebitda"] = data["ebitda"].fillna(1.0)
    data["total_debt"] = np.maximum(data["total_debt"].fillna(0.0), 0.0)
    data["cash_and_equiv"] = np.maximum(data["cash_and_equiv"].fillna(0.0), 0.0)
    data["operating_cf"] = data["operating_cf"].fillna(0.0)
    data["total_revenue"] = np.maximum(data["total_revenue"].fillna(1.0), 1.0)
    data["commodity_vol_252d"] = np.clip(data["commodity_vol_252d"].fillna(0.25), 0.01, 1.5)
    
    return data

def impute_missing_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes missing numeric features using sector-level medians.
    """
    data = df.copy()
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in ["target"]:
            continue
        data[col] = data.groupby("sector")[col].transform(lambda x: x.fillna(x.median()))
        data[col] = data[col].fillna(data[col].median())
    return data
