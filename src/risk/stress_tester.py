import os
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from .rating_engine import assign_rating_bands, get_rating_scale

def load_stress_scenarios(scenario_config_path: str = "config/stress_scenarios.yaml") -> Dict[str, Any]:
    """
    Loads stress scenarios dictionary from YAML config file.
    """
    if not os.path.exists(scenario_config_path):
        raise FileNotFoundError(f"Stress scenario config not found at: {scenario_config_path}")
    with open(scenario_config_path, "r") as f:
        data = yaml.safe_load(f)
        return data.get("scenarios", {})

def run_portfolio_stress_test(
    risk_df: pd.DataFrame,
    scenario_config_path: str = "config/stress_scenarios.yaml"
) -> pd.DataFrame:
    """
    Applies macroeconomic shocks to baseline PD/LGD and evaluates portfolio loss deltas.
    RMG Mandate Compliance.
    """
    scenarios = load_stress_scenarios(scenario_config_path)
    results = []
    
    # Baseline
    base_el = (risk_df["pd_calibrated"] * risk_df["lgd"] * risk_df["ead"]).sum()
    total_ead = risk_df["ead"].sum()
    
    results.append({
        "scenario_key": "baseline",
        "scenario": "Baseline Portfolio",
        "portfolio_el": round(base_el, 2),
        "delta_el": 0.0,
        "pct_increase": 0.0,
        "loss_rate_bps": round((base_el / max(total_ead, 1e-5)) * 10000.0, 1)
    })
    
    for key, spec in scenarios.items():
        stressed_df = risk_df.copy()
        
        # Apply Sector-Specific PD Multipliers
        multipliers = stressed_df["sector"].map(spec.get("pd_multipliers", {})).fillna(1.0)
        stressed_df["pd_stressed"] = np.minimum(1.0, stressed_df["pd_calibrated"] * multipliers)
        stressed_df["lgd_stressed"] = np.minimum(1.0, stressed_df["lgd"] + spec.get("lgd_shift", 0.10))
        
        # Compute Stressed Expected Loss
        stressed_el = (stressed_df["pd_stressed"] * stressed_df["lgd_stressed"] * stressed_df["ead"]).sum()
        delta = stressed_el - base_el
        pct_change = (delta / max(base_el, 1e-5)) * 100.0
        
        results.append({
            "scenario_key": key,
            "scenario": spec.get("name", key),
            "portfolio_el": round(stressed_el, 2),
            "delta_el": round(delta, 2),
            "pct_increase": round(pct_change, 2),
            "loss_rate_bps": round((stressed_el / max(total_ead, 1e-5)) * 10000.0, 1)
        })
        
    return pd.DataFrame(results)

def run_counterparty_stress_simulation(
    risk_df: pd.DataFrame,
    scenario_key: str,
    scenario_config_path: str = "config/stress_scenarios.yaml",
    rating_config_path: str = "config/rating_bands.yaml"
) -> pd.DataFrame:
    """
    Computes counterparty-level stressed PDs, stressed EL, and re-maps stressed rating bands.
    Produces migration comparison from baseline rating to stressed rating.
    """
    scenarios = load_stress_scenarios(scenario_config_path)
    if scenario_key not in scenarios:
        raise ValueError(f"Scenario key '{scenario_key}' not found in {list(scenarios.keys())}")
        
    spec = scenarios[scenario_key]
    stressed_df = risk_df.copy()
    
    multipliers = stressed_df["sector"].map(spec.get("pd_multipliers", {})).fillna(1.0)
    stressed_df["pd_stressed"] = np.minimum(1.0, stressed_df["pd_calibrated"] * multipliers)
    stressed_df["lgd_stressed"] = np.minimum(1.0, stressed_df["lgd"] + spec.get("lgd_shift", 0.10))
    stressed_df["stressed_expected_loss"] = np.round(stressed_df["pd_stressed"] * stressed_df["lgd_stressed"] * stressed_df["ead"], 4)
    stressed_df["delta_expected_loss"] = np.round(stressed_df["stressed_expected_loss"] - stressed_df["expected_loss"], 4)
    
    # Assign Stressed Rating Bands
    stressed_rated = assign_rating_bands(stressed_df, pd_col="pd_stressed", config_path=rating_config_path)
    stressed_df["stressed_rating_band"] = stressed_rated["rating_band"]
    stressed_df["stressed_risk_category"] = stressed_rated["risk_category"]
    stressed_df["stressed_rating_tier"] = stressed_rated["rating_tier"]
    
    # Downgrade flag
    order = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]
    band_to_num = {b: i for i, b in enumerate(order)}
    stressed_df["downgrade_notches"] = stressed_df["stressed_rating_band"].map(band_to_num) - stressed_df["rating_band"].map(band_to_num)
    
    return stressed_df

def compute_rating_migration_matrix(stressed_cp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs an 8x8 migration transition matrix from baseline rating to stressed rating.
    """
    order = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]
    existing_base = [b for b in order if b in stressed_cp_df["rating_band"].values]
    existing_stress = [b for b in order if b in stressed_cp_df["stressed_rating_band"].values]
    
    matrix = pd.crosstab(
        stressed_cp_df["rating_band"],
        stressed_cp_df["stressed_rating_band"],
        rownames=["Baseline Rating"],
        colnames=["Stressed Rating"]
    )
    # Reindex to standardized order
    matrix = matrix.reindex(index=order, columns=order, fill_value=0)
    return matrix
