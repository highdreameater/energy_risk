import numpy as np
import pandas as pd

def compute_commodity_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes commodity-adjusted financial solvency, leverage, and liquidity metrics.
    
    Formulations per PRD:
    - vol_adj_icr = (ebitda * (1.0 - commodity_vol_252d)) / max(interest_expense, 1e-5)
    - leverage_ratio = total_debt / max(ebitda, 1e-5)
    - liquidity_ratio = cash_and_equiv / max(total_debt * 0.25, 1e-5)
    - cash_margin_ratio = operating_cf / max(total_revenue, 1e-5)
    """
    data = df.copy()
    
    # Volatility Adjustment to ICR
    raw_icr = data["ebitda"] / np.maximum(data["interest_expense"], 1e-5)
    data["raw_icr"] = raw_icr
    data["vol_adj_icr"] = raw_icr * (1.0 - data["commodity_vol_252d"])
    
    # Structural Leverage Ratio
    data["leverage_ratio"] = data["total_debt"] / np.maximum(data["ebitda"], 1e-5)
    
    # Liquidity & Margin Ratios
    data["liquidity_ratio"] = data["cash_and_equiv"] / np.maximum(data["total_debt"] * 0.25, 1e-5)
    data["cash_margin_ratio"] = data["operating_cf"] / np.maximum(data["total_revenue"], 1e-5)
    
    # Debt Service Coverage Proxy
    data["debt_service_proxy"] = data["operating_cf"] / np.maximum(data["interest_expense"], 1e-5)
    
    return data

def compute_trailing_volatility(price_series: pd.Series, window: int = 252) -> float:
    """
    Computes annualized log-return volatility over trailing trading days:
    sigma = std(ln(P_t / P_{t-1})) * sqrt(252)
    """
    log_returns = np.log(price_series / price_series.shift(1)).dropna()
    if len(log_returns) < 10:
        return 0.25
    recent_returns = log_returns.tail(window)
    annualized_vol = float(recent_returns.std() * np.sqrt(252))
    return annualized_vol

def compute_margin_sensitivity(collateral_delta: pd.Series, commodity_price_delta: pd.Series) -> pd.Series:
    """
    Margin call sensitivity coefficient: beta_margin = Delta Collateral Outflow / Delta P_Commodity
    """
    return collateral_delta / np.maximum(commodity_price_delta.abs(), 1e-5)
