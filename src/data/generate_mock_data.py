import os
import numpy as np
import pandas as pd

def generate_market_data(output_path: str = "data/raw/commodity_prices.csv", num_days: int = 300):
    """
    Generates realistic daily commodity price timeseries over 252+ trading days.
    """
    np.random.seed(42)
    dates = pd.date_range(end="2026-10-01", periods=num_days, freq="B")
    
    # Simulate geometric brownian motion for key energy commodities
    commodities = {
        "WTI_Crude": {"s0": 78.0, "mu": 0.05, "sigma": 0.32},
        "Brent_Crude": {"s0": 82.5, "mu": 0.04, "sigma": 0.30},
        "Henry_Hub_Gas": {"s0": 2.75, "mu": 0.08, "sigma": 0.48},
        "TTF_Gas": {"s0": 34.0, "mu": 0.06, "sigma": 0.52},
        "PJM_Power": {"s0": 42.0, "mu": 0.03, "sigma": 0.38},
        "EUA_Carbon": {"s0": 68.0, "mu": 0.02, "sigma": 0.28}
    }
    
    df_prices = pd.DataFrame({"date": dates})
    dt = 1.0 / 252.0
    
    for comm, params in commodities.items():
        shocks = np.random.normal(0, 1, num_days)
        returns = (params["mu"] - 0.5 * params["sigma"]**2) * dt + params["sigma"] * np.sqrt(dt) * shocks
        price_path = params["s0"] * np.cumprod(np.exp(returns))
        df_prices[comm] = np.round(price_path, 2)
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_prices.to_csv(output_path, index=False)
    print(f"Generated commodity prices: {output_path} ({len(df_prices)} days)")
    return df_prices

def generate_counterparty_data(
    financials_path: str = "data/raw/counterparty_financials.csv",
    portfolio_path: str = "data/portfolio/exposures_limits.csv",
    defaults_path: str = "data/raw/default_events.csv",
    n_samples: int = 250
):
    """
    Generates institutional-grade counterparty financials, portfolio exposures, and default events.
    """
    np.random.seed(42)
    
    sectors = [
        "Oil E&P", "Midstream", "Utilities", "Commodity Trading", "LNG & Gas", "Renewables"
    ]
    sector_weights = [0.28, 0.20, 0.22, 0.15, 0.10, 0.05]
    
    counterparty_names_prefix = [
        "Apex Energy Resources", "Vortex Midstream Partners", "Meridian Power & Utility",
        "Pacific Coast Commodities", "Titan Energy Holdings", "Nordic Gas & Power",
        "Pinnacle Petroleum Corp", "Continental Pipeline Infrastructure", "Aegis Energy Trading",
        "Atlas Offshore Exploration", "Solaria Clean Energy", "Frontier Refining & Marketing",
        "Helios Power Systems", "Blue Horizon LNG", "Sterling Energy Capital",
        "Orion Energy Logistics", "Crestwood Natural Gas", "Equinox Renewable Generation",
        "Vanguard Petrochemical", "Zenith Commodity Merchants"
    ]
    
    assigned_sectors = np.random.choice(sectors, size=n_samples, p=sector_weights)
    
    cp_ids = [f"CP-{1000 + i}" for i in range(n_samples)]
    names = [
        f"{np.random.choice(counterparty_names_prefix)} #{i+1:03d}"
        for i in range(n_samples)
    ]
    
    # Financial metrics generation calibrated by sector
    data_list = []
    portfolio_list = []
    defaults_list = []
    
    for i in range(n_samples):
        cp_id = cp_ids[i]
        name = names[i]
        sector = assigned_sectors[i]
        
        # Base scale factor (Small, Mid, Large institutional players)
        size_tier = np.random.choice(["Mid", "Large", "Mega"], p=[0.45, 0.40, 0.15])
        size_mult = {"Mid": 1.0, "Large": 4.5, "Mega": 15.0}[size_tier]
        
        # Sector-specific commodity volatility profile
        vol_profiles = {
            "Oil E&P": (0.28, 0.07),
            "Midstream": (0.18, 0.04),
            "Utilities": (0.16, 0.03),
            "Commodity Trading": (0.35, 0.09),
            "LNG & Gas": (0.32, 0.08),
            "Renewables": (0.14, 0.04)
        }
        mean_vol, std_vol = vol_profiles[sector]
        commodity_vol = float(np.clip(np.random.normal(mean_vol, std_vol), 0.08, 0.65))
        
        # Revenue and EBITDA margins
        revenue = float(np.round(np.random.uniform(500, 4000) * size_mult, 1))
        
        margin_map = {
            "Oil E&P": (0.30, 0.08),
            "Midstream": (0.25, 0.05),
            "Utilities": (0.22, 0.04),
            "Commodity Trading": (0.08, 0.03),
            "LNG & Gas": (0.26, 0.06),
            "Renewables": (0.32, 0.07)
        }
        m_mean, m_std = margin_map[sector]
        ebitda_margin = np.clip(np.random.normal(m_mean, m_std), 0.02, 0.55)
        ebitda = float(np.round(revenue * ebitda_margin, 1))
        
        # Debt and Leverage
        lev_multiplier = np.random.gamma(shape=3.0, scale=1.1)  # average ~3.3x
        total_debt = float(np.round(ebitda * lev_multiplier, 1))
        
        # Interest expense (debt cost ~5% to 9%)
        interest_rate = np.random.uniform(0.05, 0.09)
        interest_expense = float(np.round(total_debt * interest_rate, 1))
        if interest_expense < 1.0:
            interest_expense = 1.0
            
        # Cash & Equivalents
        cash_and_equiv = float(np.round(np.random.uniform(0.04, 0.20) * total_debt + 10.0, 1))
        
        # Operating Cash Flow (OCF ~ 70% to 105% of EBITDA)
        operating_cf = float(np.round(ebitda * np.random.uniform(0.70, 1.05) - np.random.uniform(5, 50), 1))
        
        # Underlying credit health latent score
        # Vol-Adjusted ICR: ebitda * (1 - vol) / interest
        vol_adj_icr = (ebitda * (1.0 - commodity_vol)) / interest_expense
        leverage = total_debt / max(ebitda, 1.0)
        liquidity = cash_and_equiv / max(total_debt * 0.25, 1.0)
        cash_margin = operating_cf / max(revenue, 1.0)
        
        # Latent default propensity: high leverage, low ICR, low liquidity, high vol increases default
        z = (
            -3.2
            - 0.45 * vol_adj_icr
            + 0.55 * (leverage - 3.0)
            - 0.85 * (liquidity - 0.4)
            - 3.50 * cash_margin
            + 2.80 * commodity_vol
        )
        true_pd = 1.0 / (1.0 + np.exp(-z))
        true_pd = float(np.clip(true_pd, 0.0001, 0.85))
        
        # Default event indicator (target)
        default_indicator = int(np.random.binomial(1, true_pd))
        
        data_list.append({
            "counterparty_id": cp_id,
            "counterparty_name": name,
            "sector": sector,
            "ebitda": ebitda,
            "interest_expense": interest_expense,
            "total_debt": total_debt,
            "cash_and_equiv": cash_and_equiv,
            "operating_cf": operating_cf,
            "total_revenue": revenue,
            "commodity_vol_252d": round(commodity_vol, 4),
            "target": default_indicator
        })
        
        # Portfolio exposure generation
        # Credit limit assigned based on size and sector
        base_limit = np.round(size_mult * np.random.uniform(25.0, 180.0), 1)
        
        # Realistic utilization distribution with some normal, some warnings (85-100%), some breaches (>100%)
        util_dist = np.random.choice(["normal", "warning", "breach"], p=[0.74, 0.18, 0.08])
        if util_dist == "normal":
            util_rate = np.random.uniform(0.35, 0.84)
        elif util_dist == "warning":
            util_rate = np.random.uniform(0.85, 0.99)
        else:
            util_rate = np.random.uniform(1.02, 1.35)
            
        ead = float(np.round(base_limit * util_rate, 1))
        
        # Sector specific typical LGD
        sector_lgd = {
            "Oil E&P": 0.45,
            "Midstream": 0.38,
            "Utilities": 0.35,
            "Commodity Trading": 0.50,
            "LNG & Gas": 0.42,
            "Renewables": 0.40
        }[sector]
        lgd = float(np.clip(np.random.normal(sector_lgd, 0.06), 0.15, 0.75))
        
        collateral = float(np.round(ead * np.random.uniform(0.05, 0.40), 1))
        
        portfolio_list.append({
            "counterparty_id": cp_id,
            "ead": ead,
            "lgd": round(lgd, 4),
            "credit_limit": base_limit,
            "collateral_posted": collateral
        })
        
        if default_indicator == 1:
            defaults_list.append({
                "counterparty_id": cp_id,
                "counterparty_name": name,
                "default_date": "2025-11-14",
                "default_type": np.random.choice(["Missed Coupon", "Distressed Debt Exchange", "Bankruptcy Ch. 11"]),
                "resolved_recovery_rate": round(1.0 - lgd, 4)
            })

    # Save CSVs
    df_fin = pd.DataFrame(data_list)
    df_port = pd.DataFrame(portfolio_list)
    df_def = pd.DataFrame(defaults_list)
    
    os.makedirs(os.path.dirname(financials_path), exist_ok=True)
    os.makedirs(os.path.dirname(portfolio_path), exist_ok=True)
    
    df_fin.to_csv(financials_path, index=False)
    df_port.to_csv(portfolio_path, index=False)
    df_def.to_csv(defaults_path, index=False)
    
    print(f"Generated Financials: {financials_path} ({len(df_fin)} rows, Defaults: {df_fin['target'].sum()})")
    print(f"Generated Portfolio: {portfolio_path} ({len(df_port)} rows)")
    print(f"Generated Default Events: {defaults_path} ({len(df_def)} events)")

if __name__ == "__main__":
    generate_market_data()
    generate_counterparty_data()
