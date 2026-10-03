import pytest
import numpy as np
import pandas as pd
from src.features.ratios import compute_commodity_ratios, compute_trailing_volatility

def test_compute_commodity_ratios():
    df = pd.DataFrame([{
        "counterparty_id": "CP-TEST-1",
        "counterparty_name": "Test Energy Corp",
        "sector": "Oil E&P",
        "ebitda": 100.0,
        "interest_expense": 20.0,
        "total_debt": 300.0,
        "cash_and_equiv": 50.0,
        "operating_cf": 80.0,
        "total_revenue": 500.0,
        "commodity_vol_252d": 0.20
    }])
    
    res = compute_commodity_ratios(df)
    
    # Raw ICR = 100 / 20 = 5.0
    assert pytest.approx(res["raw_icr"].iloc[0], rel=1e-3) == 5.0
    # Vol Adj ICR = 5.0 * (1 - 0.20) = 4.0
    assert pytest.approx(res["vol_adj_icr"].iloc[0], rel=1e-3) == 4.0
    # Leverage = 300 / 100 = 3.0
    assert pytest.approx(res["leverage_ratio"].iloc[0], rel=1e-3) == 3.0
    # Liquidity = 50 / (300 * 0.25) = 50 / 75 = 0.6667
    assert pytest.approx(res["liquidity_ratio"].iloc[0], rel=1e-3) == 50.0 / 75.0
    # Cash Margin = 80 / 500 = 0.16
    assert pytest.approx(res["cash_margin_ratio"].iloc[0], rel=1e-3) == 0.16

def test_trailing_volatility_calculation():
    # Constant price has 0 volatility
    constant_prices = pd.Series([100.0] * 260)
    vol_zero = compute_trailing_volatility(constant_prices)
    assert pytest.approx(vol_zero, abs=1e-5) == 0.0
    
    # Simulated log-normal returns with known annual vol ~0.30
    np.random.seed(42)
    daily_returns = np.random.normal(0, 0.30 / np.sqrt(252), 300)
    sim_prices = pd.Series(100.0 * np.exp(np.cumsum(daily_returns)))
    est_vol = compute_trailing_volatility(sim_prices)
    assert 0.22 <= est_vol <= 0.38
