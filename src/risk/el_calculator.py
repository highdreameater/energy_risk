import os
import pandas as pd
import numpy as np
from typing import Optional

def compute_expected_loss(
    df: pd.DataFrame,
    exposures_path: Optional[str] = "data/portfolio/exposures_limits.csv"
) -> pd.DataFrame:
    """
    Computes counterparty Expected Loss (EL):
    EL_i = PD_i * LGD_i * EAD_i
    
    If exposures_path is provided and columns are not yet merged, merges on counterparty_id.
    """
    data = df.copy()
    
    if exposures_path and os.path.exists(exposures_path):
        df_exp = pd.read_csv(exposures_path)
        # Avoid duplicate column conflicts
        cols_to_merge = [c for c in df_exp.columns if c == "counterparty_id" or c not in data.columns]
        data = pd.merge(data, df_exp[cols_to_merge], on="counterparty_id", how="left")
        
    # Default fallbacks if missing
    if "ead" not in data.columns:
        data["ead"] = 50.0
    if "lgd" not in data.columns:
        data["lgd"] = 0.45
    if "credit_limit" not in data.columns:
        data["credit_limit"] = data["ead"] * 1.2
    if "collateral_posted" not in data.columns:
        data["collateral_posted"] = 0.0
        
    pd_series = data.get("pd_calibrated", data.get("target", 0.05))
    data["expected_loss"] = np.round(pd_series * data["lgd"] * data["ead"], 4)
    data["net_exposure"] = np.maximum(0.0, data["ead"] - data["collateral_posted"])
    
    return data

def aggregate_expected_loss(df: pd.DataFrame, group_col: str = "sector") -> pd.DataFrame:
    """
    Aggregates Expected Loss, Total EAD, and Weighted Average PD across dimensions.
    """
    if group_col not in df.columns:
        raise ValueError(f"Grouping column '{group_col}' not found in dataframe.")
        
    grouped = df.groupby(group_col).agg(
        counterparty_count=("counterparty_id", "count"),
        total_ead=("ead", "sum"),
        total_expected_loss=("expected_loss", "sum"),
        avg_lgd=("lgd", "mean"),
        total_credit_limit=("credit_limit", "sum")
    ).reset_index()
    
    grouped["weighted_avg_pd"] = np.round(grouped["total_expected_loss"] / np.maximum(grouped["total_ead"] * grouped["avg_lgd"], 1e-5), 4)
    grouped["portfolio_loss_rate_bps"] = np.round((grouped["total_expected_loss"] / np.maximum(grouped["total_ead"], 1e-5)) * 10000.0, 1)
    
    return grouped.sort_values(by="total_expected_loss", ascending=False)
