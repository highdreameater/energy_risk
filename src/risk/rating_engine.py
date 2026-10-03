import os
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

DEFAULT_RATING_SCALE = {
    "AAA": {"min_pd": 0.0000, "max_pd": 0.0005, "risk_category": "Minimal Risk", "investment_grade": True},
    "AA":  {"min_pd": 0.0005, "max_pd": 0.0015, "risk_category": "Low Risk", "investment_grade": True},
    "A":   {"min_pd": 0.0015, "max_pd": 0.0050, "risk_category": "Low-Moderate Risk", "investment_grade": True},
    "BBB": {"min_pd": 0.0050, "max_pd": 0.0200, "risk_category": "Moderate Risk", "investment_grade": True},
    "BB":  {"min_pd": 0.0200, "max_pd": 0.0600, "risk_category": "Sub-Investment Grade", "investment_grade": False},
    "B":   {"min_pd": 0.0600, "max_pd": 0.1500, "risk_category": "High Risk", "investment_grade": False},
    "CCC": {"min_pd": 0.1500, "max_pd": 1.0000, "risk_category": "Vulnerable / Default Imminent", "investment_grade": False}
}

def get_rating_scale(config_path: Optional[str] = "config/rating_bands.yaml") -> Dict[str, Any]:
    """
    Loads rating scale mapping from YAML config or returns default institutional scale.
    """
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
            return cfg.get("rating_scale", DEFAULT_RATING_SCALE)
    return DEFAULT_RATING_SCALE

def pd_to_rating(pd_val: float, rating_scale: Dict[str, Any]) -> str:
    """
    Maps a single PD probability into an internal rating band.
    """
    for band, params in rating_scale.items():
        min_p = params.get("min_pd", 0.0)
        max_p = params.get("max_pd", 1.0)
        if min_p <= pd_val <= max_p or (band == "CCC" and pd_val >= min_p):
            return band
    return "CCC"

def assign_rating_bands(
    df: pd.DataFrame,
    pd_col: str = "pd_calibrated",
    config_path: str = "config/rating_bands.yaml"
) -> pd.DataFrame:
    """
    Assigns internal rating bands (AAA to CCC), risk categories,
    and investment grade tierings to counterparties based on calibrated PD.
    """
    data = df.copy()
    scale = get_rating_scale(config_path)
    
    if pd_col not in data.columns:
        raise ValueError(f"PD column '{pd_col}' not present in dataframe.")
        
    ratings = []
    categories = []
    ig_flags = []
    
    for val in data[pd_col]:
        band = pd_to_rating(val, scale)
        ratings.append(band)
        categories.append(scale[band].get("risk_category", "Standard"))
        ig_flags.append(scale[band].get("investment_grade", False))
        
    data["rating_band"] = ratings
    data["risk_category"] = categories
    data["investment_grade"] = ig_flags
    data["rating_tier"] = np.where(data["investment_grade"], "Investment Grade (IG)", "High Yield (HY)")
    
    return data
