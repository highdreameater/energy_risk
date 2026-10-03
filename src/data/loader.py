import os
import pandas as pd

def load_raw_data(file_path: str = "data/raw/counterparty_financials.csv") -> pd.DataFrame:
    """
    Ingests counterparty financial statement data from CSV or Parquet.
    Validates mandatory columns per FR-1 schema.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found at: {file_path}")
        
    if file_path.endswith(".parquet"):
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path)
        
    required_cols = [
        "counterparty_id", "counterparty_name", "sector", "ebitda",
        "interest_expense", "total_debt", "cash_and_equiv",
        "operating_cf", "total_revenue", "commodity_vol_252d"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing mandatory columns in raw financials: {missing}")
        
    return df

def load_commodity_prices(file_path: str = "data/raw/commodity_prices.csv") -> pd.DataFrame:
    """
    Ingests daily commodity price history.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Commodity prices file not found at: {file_path}")
    df = pd.read_csv(file_path, parse_dates=["date"])
    return df

def load_portfolio_exposures(file_path: str = "data/portfolio/exposures_limits.csv") -> pd.DataFrame:
    """
    Ingests portfolio credit exposures, limits, and collateral balances.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Exposures file not found at: {file_path}")
    df = pd.read_csv(file_path)
    required_cols = ["counterparty_id", "ead", "lgd", "credit_limit"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing mandatory columns in exposures data: {missing}")
    return df
