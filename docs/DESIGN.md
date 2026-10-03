# System Design & Architecture (DESIGN.md)

## 1. Architecture Overview
ECCRE is architected as a modular, production-ready quantitative credit risk pipeline:

```
[Raw Financials & Market Volatility]
                  │
                  ▼
       [Data Ingestion & Cleaning]
        (src/data/loader.py, preprocessor.py)
                  │
                  ▼
       [Feature Engineering Engine]
        (src/features/ratios.py: Vol-Adj ICR, Leverage, Liquidity)
                  │
                  ▼
       [PD Calibration Engine] ──────────┐
        (src/models/pd_engine.py)        │
                  │                      ▼
                  │           [RMG Model Validation]
                  │           (src/models/validation.py: Gini, KS, Brier, PSI)
                  ▼
       [Internal Rating Engine]
        (src/risk/rating_engine.py: AAA to CCC Bands)
                  │
                  ▼
       [Exposure & Limit Monitor]
        (src/risk/limit_monitor.py: Utilization & Breach Detection)
                  │
                  ▼
       [Expected Loss Engine]
        (src/risk/el_calculator.py: EL = PD * LGD * EAD)
                  │
                  ▼
       [Macroeconomic Stress Testing Engine]
        (src/risk/stress_tester.py: Multi-Scenario Shock Shifter)
                  │
                  ▼
       [Interactive Streamlit Dashboard]
        (src/dashboard/app.py)
```

## 2. Quantitative Formulations
* **Vol-Adjusted ICR**:
  $$\text{ICR}_{\text{adj}} = \frac{\text{EBITDA} \times (1 - \sigma_{\text{commodity}})}{\max(\text{Interest Expense}, 10^{-5})}$$
* **Structural Leverage**:
  $$\text{Leverage} = \frac{\text{Total Debt}}{\max(\text{EBITDA}, 10^{-5})}$$
* **Liquidity Ratio**:
  $$\text{Liquidity} = \frac{\text{Cash \& Equivalents}}{\max(\text{Total Debt} \times 0.25, 10^{-5})}$$
* **Cash Margin Ratio**:
  $$\text{Cash Margin} = \frac{\text{Operating Cash Flow}}{\max(\text{Total Revenue}, 10^{-5})}$$
* **Platt Scaling**:
  $$PD_i = \frac{1}{1 + e^{-(A \cdot z_i + B)}}$$
* **Gini & KS**:
  $$\text{Gini} = 2 \cdot \text{AUC} - 1, \quad KS = \max_t |F_{\text{def}}(t) - F_{\text{non-def}}(t)| \times 100$$
* **Expected Loss**:
  $$EL_i = PD_i \times LGD_i \times EAD_i$$
