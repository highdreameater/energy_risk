import json
import os

def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    }

def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")]
    }

os.makedirs("notebooks", exist_ok=True)

# -------------------------------------------------------------------------
# Notebook 1: EDA & Commodity Volatility Analysis
# -------------------------------------------------------------------------
nb1_cells = [
    md_cell("# Enterprise Commodities & Energy Credit Risk & Capital Engine (ECCRE)\n## 01: Exploratory Data Analysis & Volatility Feature Engineering\n\nThis notebook analyzes financial statement distributions and trailing 252-day commodity price volatility across wholesale energy counterparties."),
    code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.loader import load_raw_data, load_commodity_prices
from src.features.ratios import compute_commodity_ratios, compute_trailing_volatility

sns.set_theme(style="darkgrid")
%matplotlib inline"""),
    md_cell("### 1. Ingest Raw Counterparty Financials & Daily Commodity Prices"),
    code_cell("""df_financials = load_raw_data("data/raw/counterparty_financials.csv")
df_prices = load_commodity_prices("data/raw/commodity_prices.csv")
print(f"Loaded {len(df_financials)} counterparties.")
df_financials.head()"""),
    md_cell("### 2. Commodity Log-Return Volatility Calculation (252-Day Trailing)"),
    code_cell("""price_cols = [c for c in df_prices.columns if c != "date"]
vol_summary = {}

for col in price_cols:
    vol = compute_trailing_volatility(df_prices[col], window=252)
    vol_summary[col] = vol

df_vol = pd.DataFrame(list(vol_summary.items()), columns=["Commodity", "Annualized Volatility (252d)"])
df_vol"""),
    md_cell("### 3. Volatility-Adjusted Solvency & Coverage Ratios"),
    code_cell("""df_features = compute_commodity_ratios(df_financials)
df_features[["counterparty_name", "sector", "raw_icr", "commodity_vol_252d", "vol_adj_icr", "leverage_ratio", "liquidity_ratio"]].head(10)"""),
    md_cell("### 4. Correlation Analysis: Leverage, ICR, Volatility, and Default Events"),
    code_cell("""plt.figure(figsize=(10, 6))
numeric_feats = ["vol_adj_icr", "leverage_ratio", "liquidity_ratio", "cash_margin_ratio", "commodity_vol_252d", "target"]
corr = df_features[numeric_feats].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlation Matrix: Engineered Ratios vs. Default Target")
plt.tight_layout()
plt.show()""")
]

with open("notebooks/01_eda_and_volatility_analysis.ipynb", "w") as f:
    json.dump(make_notebook(nb1_cells), f, indent=2)

# -------------------------------------------------------------------------
# Notebook 2: PD Model Training & Platt Calibration
# -------------------------------------------------------------------------
nb2_cells = [
    md_cell("# ECCRE | 02: Probability of Default (PD) Scorecard & Calibration Engine\n\nTrains Logistic Scorecard with Platt Scaling to generate institutional-grade 1-year PDs [0, 1]."),
    code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.data.loader import load_raw_data
from src.features.ratios import compute_commodity_ratios
from src.models.pd_engine import train_and_calibrate_pd_model
from src.models.validation import generate_validation_report, compute_calibration_curve_data

%matplotlib inline"""),
    md_cell("### 1. Train Scorecard and Calibrate via Platt Scaling"),
    code_cell("""df_raw = load_raw_data("data/raw/counterparty_financials.csv")
df_features = compute_commodity_ratios(df_raw)
df_scored, model = train_and_calibrate_pd_model(df_features, save_path="models/pd_model.joblib")
df_scored[["counterparty_name", "vol_adj_icr", "leverage_ratio", "pd_uncalibrated", "pd_calibrated", "target"]].head(10)"""),
    md_cell("### 2. RMG Validation Benchmark Table"),
    code_cell("""report = generate_validation_report(df_scored["target"], df_scored["pd_calibrated"])
report"""),
    md_cell("### 3. Platt Calibration Curve (Predicted PD vs. Observed Default Frequencies)"),
    code_cell("""calib_df = compute_calibration_curve_data(df_scored["target"].values, df_scored["pd_calibrated"].values)

plt.figure(figsize=(8, 5))
plt.plot(calib_df["pred_pd"], calib_df["obs_default_rate"], marker='o', label="Platt Calibrated Model", color="crimson")
plt.plot([0, 0.8], [0, 0.8], linestyle="--", color="gray", label="Perfect Calibration")
plt.xlabel("Mean Predicted Probability of Default (PD)")
plt.ylabel("Observed Default Rate")
plt.title("Scorecard Reliability Curve (Calibration Deciles)")
plt.legend()
plt.tight_layout()
plt.show()""")
]

with open("notebooks/02_pd_model_training_calibration.ipynb", "w") as f:
    json.dump(make_notebook(nb2_cells), f, indent=2)

# -------------------------------------------------------------------------
# Notebook 3: Macro Stress-Testing & RMG Validation
# -------------------------------------------------------------------------
nb3_cells = [
    md_cell("# ECCRE | 03: Macro Stress-Testing & Rating Migration Analytics\n\nExecutes macro scenario shifts across wholesale energy sectors and measures capital depletion (Delta EL)."),
    code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.risk.rating_engine import assign_rating_bands
from src.risk.el_calculator import compute_expected_loss
from src.risk.limit_monitor import evaluate_portfolio_limits
from src.risk.stress_tester import run_portfolio_stress_test, run_counterparty_stress_simulation, compute_rating_migration_matrix

%matplotlib inline"""),
    md_cell("### 1. Load Scored Portfolio and Compute Baseline Expected Loss"),
    code_cell("""df_scored = pd.read_parquet("data/processed/model_predictions.parquet")
print(f"Total Counterparties: {len(df_scored)}")
print(f"Total Baseline EAD: ${df_scored['ead'].sum():,.2f}M")
print(f"Total Baseline EL:  ${df_scored['expected_loss'].sum():,.2f}M")"""),
    md_cell("### 2. Multi-Scenario Portfolio Stress Loss Deltas"),
    code_cell("""stress_summary = run_portfolio_stress_test(df_scored, "config/stress_scenarios.yaml")
stress_summary"""),
    md_cell("### 3. Rating Migration Matrix under Oil Crash (-50%)"),
    code_cell("""stressed_cp = run_counterparty_stress_simulation(
    df_scored,
    scenario_key="oil_crash_2008_style",
    scenario_config_path="config/stress_scenarios.yaml",
    rating_config_path="config/rating_bands.yaml"
)

migration_matrix = compute_rating_migration_matrix(stressed_cp)

plt.figure(figsize=(9, 7))
sns.heatmap(migration_matrix, annot=True, fmt="d", cmap="Blues", cbar=True)
plt.title("Credit Rating Migration Matrix: Baseline vs. Oil Crash Scenario")
plt.xlabel("Stressed Rating Band")
plt.ylabel("Baseline Rating Band")
plt.tight_layout()
plt.show()""")
]

with open("notebooks/03_stress_testing_validation.ipynb", "w") as f:
    json.dump(make_notebook(nb3_cells), f, indent=2)

print("Created 3 institutional Jupyter notebooks in notebooks/")
