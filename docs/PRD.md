# Enterprise Commodities & Energy Credit Risk & Capital Engine (ECCRE)
## Product Requirement Document (PRD)

### 1. Executive Summary & Vision
The Enterprise Commodities & Energy Credit Risk & Capital Engine (ECCRE) is an institutional-grade wholesale credit risk analytics platform. Designed specifically for commodity trading desks, energy producers, and utility counterparties, ECCRE automates the end-to-end credit workflow: ingesting complex financial statements, engineering volatility-adjusted solvency metrics, calibrating scorecards, enforcing credit limits, and running macroeconomic stress tests.

### 2. Target Stakeholders & Nomura JD Mapping
* **Credit Risk Management (CRM)**: Assigning credit ratings, financial statement analysis (P&L, Balance Sheet, Cash Flow), and conducting counterparty credit reviews.
* **Credit Risk Exposure Management (CEM)**: Portfolio exposure monitoring (EAD), tracking margin call sensitivity, managing credit limits, and investigating breaches.
* **Risk Methodologies Group (RMG)**: Model development, validation (ROC-AUC, Gini, KS, PSI, Brier Score), and macroeconomic stress-testing framework design.

### 3. Functional Requirements
* **FR-1**: Financial statement ingestion & 252-day commodity price volatility-adjusted solvency ratios (Vol-adjusted ICR, Leverage, Liquidity, Cash Margin).
* **FR-2**: Probability of Default (PD) scorecard modeling with Logistic Regression / Tree models, calibrated via Platt Scaling into rigorous 1-year PDs [0, 1].
* **FR-3**: Quantitative Model Validation Engine computing ROC-AUC, Gini (2*AUC - 1), Kolmogorov-Smirnov (KS >= 45%), Brier Score (<= 0.08), and Population Stability Index (PSI < 0.10).
* **FR-4**: Internal Rating Assignment Engine mapping PDs to 8 credit rating bands (AAA through CCC) with Investment Grade vs. High Yield tiers.
* **FR-5**: Portfolio Expected Loss (EL) Calculation: $EL_i = PD_i \times LGD_i \times EAD_i$ and sector-level aggregation.
* **FR-6**: Credit Exposure & Limit Monitoring: Utilization % ($EAD_i / Limit_i \times 100$) with status flags (`Normal < 85%`, `Warning 85-100%`, `BREACH > 100%`).
* **FR-7**: Macroeconomic & Commodity Stress Testing: 4 scenarios (Oil Crash, Gas Price Spike, Stagflation, Energy Transition) simulating sector PD multipliers, LGD shifts, and portfolio delta EL.
* **FR-8**: Executive Risk Dashboard: Streamlit interactive web interface with 4 dedicated consoles for Portfolio Overview, RMG Validation, CEM Exposure Monitoring, and Stress Testing.
