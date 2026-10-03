import pytest
import numpy as np
import pandas as pd
from src.models.pd_engine import train_and_calibrate_pd_model
from src.models.validation import calculate_gini, calculate_ks_stat, calculate_brier_score, calculate_psi

def test_pd_engine_and_calibration():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "vol_adj_icr": np.random.uniform(1.0, 8.0, n),
        "leverage_ratio": np.random.uniform(1.0, 7.0, n),
        "liquidity_ratio": np.random.uniform(0.1, 1.5, n),
        "cash_margin_ratio": np.random.uniform(0.05, 0.40, n),
        "commodity_vol_252d": np.random.uniform(0.15, 0.50, n),
        "target": np.random.binomial(1, 0.15, n)
    })
    
    scored_df, model = train_and_calibrate_pd_model(df, save_path=None)
    
    assert "pd_calibrated" in scored_df.columns
    assert "pd_uncalibrated" in scored_df.columns
    
    # Check bounds
    assert (scored_df["pd_calibrated"] >= 0.0).all()
    assert (scored_df["pd_calibrated"] <= 1.0).all()
    
    # Validation metrics
    y_true = scored_df["target"].values
    y_prob = scored_df["pd_calibrated"].values
    
    gini = calculate_gini(y_true, y_prob)
    ks = calculate_ks_stat(y_true, y_prob)
    brier = calculate_brier_score(y_true, y_prob)
    
    assert -1.0 <= gini <= 1.0
    assert 0.0 <= ks <= 100.0
    assert 0.0 <= brier <= 1.0

def test_psi_identical_distributions():
    np.random.seed(42)
    dist = np.random.uniform(0, 1, 500)
    # Identical distributions should have near zero PSI (< 0.05)
    psi = calculate_psi(dist, dist)
    assert psi < 0.02
