import os
import sys
import pandas as pd
from src.data.loader import load_raw_data
from src.features.ratios import compute_commodity_ratios
from src.models.pd_engine import train_and_calibrate_pd_model
from src.models.validation import calculate_gini, calculate_ks_stat, calculate_psi, calculate_brier_score, generate_validation_report
from src.risk.rating_engine import assign_rating_bands
from src.risk.el_calculator import compute_expected_loss
from src.risk.limit_monitor import evaluate_portfolio_limits
from src.risk.stress_tester import run_portfolio_stress_test

def main():
    print("================================================================================")
    print("  ENTERPRISE COMMODITIES & ENERGY CREDIT RISK & CAPITAL ENGINE (ECCRE) PIPELINE ")
    print("================================================================================")
    
    print("\n=== [1/6] Ingesting Financial Statements & Market Data ===")
    raw_path = "data/raw/counterparty_financials.csv"
    if not os.path.exists(raw_path):
        from src.data.generate_mock_data import generate_market_data, generate_counterparty_data
        print("Raw data not detected. Generating institutional-grade sample dataset...")
        generate_market_data()
        generate_counterparty_data()
        
    df_raw = load_raw_data(raw_path)
    print(f"Loaded {len(df_raw)} counterparties across {df_raw['sector'].nunique()} energy sectors.")
    
    print("\n=== [2/6] Engineering Volatility-Adjusted Ratios ===")
    df_features = compute_commodity_ratios(df_raw)
    os.makedirs("data/processed", exist_ok=True)
    df_features.to_parquet("data/processed/features_engineered.parquet", index=False)
    print("Features engineered: Vol-Adjusted ICR, Structural Leverage, Liquidity Ratio, Cash Margin.")
    print("Saved feature dataset to data/processed/features_engineered.parquet")
    
    print("\n=== [3/6] Scoring & Calibrating Probability of Default (PD) ===")
    df_scored, pd_model = train_and_calibrate_pd_model(df_features, save_path="models/pd_model.joblib")
    print("Fitted Logistic Scorecard and executed Platt Scaling calibration into 1-year PD.")
    
    print("\n=== [4/6] Executing RMG Model Validation Suite ===")
    gini = calculate_gini(df_scored['target'], df_scored['pd_calibrated'])
    ks = calculate_ks_stat(df_scored['target'], df_scored['pd_calibrated'])
    brier = calculate_brier_score(df_scored['target'], df_scored['pd_calibrated'])
    val_report = generate_validation_report(df_scored['target'], df_scored['pd_calibrated'])
    print(f"Validation Results -> Gini: {gini:.4f} | KS Stat: {ks:.2f}% | Brier Score: {brier:.4f}")
    print("\nRMG Model Validation Scorecard:")
    print(val_report.to_string(index=False))
    
    print("\n=== [5/6] Assigning Ratings, Computing EL, & Checking Credit Limits ===")
    df_rated = assign_rating_bands(df_scored, config_path="config/rating_bands.yaml")
    df_el = compute_expected_loss(df_rated, exposures_path="data/portfolio/exposures_limits.csv")
    df_final = evaluate_portfolio_limits(df_el)
    
    # Save output parquet
    df_final.to_parquet("data/processed/model_predictions.parquet", index=False)
    print("Saved final risk predictions to data/processed/model_predictions.parquet")
    
    # Portfolio EL and Exposure summary
    total_ead = df_final['ead'].sum()
    total_el = df_final['expected_loss'].sum()
    breach_count = (df_final['status'] == 'BREACH').sum()
    warning_count = (df_final['status'] == 'Warning').sum()
    print(f"\nPortfolio Exposure Summary:")
    print(f"  Total Portfolio EAD:      ${total_ead:,.2f}M")
    print(f"  Total Expected Loss (EL): ${total_el:,.2f}M ({total_el/total_ead*10000:.1f} bps)")
    print(f"  Breach Status Counterparties: {breach_count}")
    print(f"  Warning Status Counterparties: {warning_count}")
    
    print("\n=== [6/6] Running Macro Stress-Testing Engine ===")
    stress_results = run_portfolio_stress_test(df_final, scenario_config_path="config/stress_scenarios.yaml")
    print(stress_results.to_string(index=False))
    print("\n[SUCCESS] Pipeline execution complete.")

if __name__ == "__main__":
    main()
