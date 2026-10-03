import pytest
import pandas as pd
from src.risk.stress_tester import run_portfolio_stress_test, run_counterparty_stress_simulation, compute_rating_migration_matrix
from src.risk.rating_engine import assign_rating_bands

def test_stress_tester_pipeline():
    risk_df = pd.DataFrame([
        {
            "counterparty_id": "CP-1",
            "sector": "Oil E&P",
            "pd_calibrated": 0.015,
            "lgd": 0.40,
            "ead": 100.0,
            "expected_loss": 0.60
        },
        {
            "counterparty_id": "CP-2",
            "sector": "Commodity Trading",
            "pd_calibrated": 0.035,
            "lgd": 0.50,
            "ead": 200.0,
            "expected_loss": 3.50
        }
    ])
    
    # Assign baseline ratings
    risk_df = assign_rating_bands(risk_df, pd_col="pd_calibrated")
    
    results = run_portfolio_stress_test(risk_df, scenario_config_path="config/stress_scenarios.yaml")
    
    assert len(results) >= 4
    # Baseline delta EL must be 0
    assert results.loc[results["scenario_key"] == "baseline", "delta_el"].iloc[0] == 0.0
    
    # Severe Oil crash should have strictly positive delta EL
    oil_res = results.loc[results["scenario_key"] == "oil_crash_2008_style"]
    assert len(oil_res) == 1
    assert oil_res["delta_el"].iloc[0] > 0.0
    assert oil_res["portfolio_el"].iloc[0] > results.loc[results["scenario_key"] == "baseline", "portfolio_el"].iloc[0]

def test_counterparty_stress_migration():
    risk_df = pd.DataFrame([
        {
            "counterparty_id": "CP-1",
            "sector": "Oil E&P",
            "pd_calibrated": 0.002,  # Baseline Rating A
            "lgd": 0.40,
            "ead": 100.0,
            "expected_loss": 0.08
        }
    ])
    risk_df = assign_rating_bands(risk_df, pd_col="pd_calibrated")
    assert risk_df["rating_band"].iloc[0] == "A"
    
    stressed_cp = run_counterparty_stress_simulation(
        risk_df,
        scenario_key="oil_crash_2008_style",
        scenario_config_path="config/stress_scenarios.yaml"
    )
    
    assert "stressed_pd" not in stressed_cp.columns or "pd_stressed" in stressed_cp.columns
    assert "stressed_rating_band" in stressed_cp.columns
    assert stressed_cp["stressed_expected_loss"].iloc[0] > risk_df["expected_loss"].iloc[0]
    
    matrix = compute_rating_migration_matrix(stressed_cp)
    assert matrix.shape == (7, 7)
