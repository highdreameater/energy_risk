import pytest
import pandas as pd
from src.risk.limit_monitor import evaluate_portfolio_limits, summarize_limit_monitoring

def test_evaluate_portfolio_limits():
    df = pd.DataFrame([
        {"counterparty_id": "CP-1", "ead": 70.0, "credit_limit": 100.0},   # 70% -> Normal
        {"counterparty_id": "CP-2", "ead": 90.0, "credit_limit": 100.0},   # 90% -> Warning
        {"counterparty_id": "CP-3", "ead": 120.0, "credit_limit": 100.0},  # 120% -> BREACH
    ])
    
    res = evaluate_portfolio_limits(df)
    
    assert res.loc[res["counterparty_id"] == "CP-1", "status"].iloc[0] == "Normal"
    assert res.loc[res["counterparty_id"] == "CP-2", "status"].iloc[0] == "Warning"
    assert res.loc[res["counterparty_id"] == "CP-3", "status"].iloc[0] == "BREACH"
    
    assert res.loc[res["counterparty_id"] == "CP-1", "excess_exposure"].iloc[0] == 0.0
    assert res.loc[res["counterparty_id"] == "CP-3", "excess_exposure"].iloc[0] == 20.0
    
    summary = summarize_limit_monitoring(res)
    assert summary["breach_count"] == 1
    assert summary["warning_count"] == 1
    assert summary["normal_count"] == 1
    assert summary["total_excess_exposure"] == 20.0
