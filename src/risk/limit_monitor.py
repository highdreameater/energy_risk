import pandas as pd
import numpy as np
from typing import Dict, Any

def evaluate_portfolio_limits(
    df: pd.DataFrame,
    warning_thresh: float = 85.0,
    breach_thresh: float = 100.0
) -> pd.DataFrame:
    """
    Evaluates credit limit utilization and detects warning/breach status.
    CEM Mandate Compliance.
    
    Status Rules:
    - Normal:  Utilization < 85%
    - Warning: 85% <= Utilization <= 100%
    - BREACH:  Utilization > 100%
    """
    portfolio = df.copy()
    
    # Calculate Utilization Percentage
    portfolio["utilization_pct"] = np.round((portfolio["ead"] / np.maximum(portfolio["credit_limit"], 1e-5)) * 100.0, 2)
    
    # Determine Status
    conditions = [
        portfolio["utilization_pct"] < warning_thresh,
        (portfolio["utilization_pct"] >= warning_thresh) & (portfolio["utilization_pct"] <= breach_thresh),
        portfolio["utilization_pct"] > breach_thresh
    ]
    choices = ["Normal", "Warning", "BREACH"]
    
    portfolio["status"] = np.select(conditions, choices, default="Unknown")
    portfolio["excess_exposure"] = np.round(np.maximum(0.0, portfolio["ead"] - portfolio["credit_limit"]), 2)
    
    return portfolio

def summarize_limit_monitoring(portfolio: pd.DataFrame) -> Dict[str, Any]:
    """
    Produces executive summary metrics for credit limit monitoring.
    """
    total_cps = len(portfolio)
    breach_df = portfolio[portfolio["status"] == "BREACH"]
    warning_df = portfolio[portfolio["status"] == "Warning"]
    normal_df = portfolio[portfolio["status"] == "Normal"]
    
    return {
        "total_counterparties": total_cps,
        "breach_count": len(breach_df),
        "warning_count": len(warning_df),
        "normal_count": len(normal_df),
        "total_excess_exposure": round(breach_df["excess_exposure"].sum(), 2),
        "max_utilization_pct": round(portfolio["utilization_pct"].max(), 2),
        "avg_utilization_pct": round(portfolio["utilization_pct"].mean(), 2)
    }
