import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve

def calculate_gini(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Computes Gini Coefficient = 2 * AUC - 1
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if len(np.unique(y_true)) < 2:
        return 0.0
    auc = roc_auc_score(y_true, y_prob)
    return float(2.0 * auc - 1.0)

def calculate_ks_stat(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Computes Kolmogorov-Smirnov (KS) statistic (in percentage [0, 100]).
    KS = max_t | F_Default(t) - F_Non-Default(t) | * 100
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if len(np.unique(y_true)) < 2 or y_true.sum() == 0 or (1 - y_true).sum() == 0:
        return 0.0
        
    df = pd.DataFrame({'target': y_true, 'prob': y_prob})
    df = df.sort_values(by='prob', ascending=False).reset_index(drop=True)
    
    df['default'] = df['target']
    df['non_default'] = 1 - df['target']
    
    cum_default = df['default'].cumsum() / df['default'].sum()
    cum_non_default = df['non_default'].cumsum() / df['non_default'].sum()
    
    ks_stat = float(np.max(np.abs(cum_default - cum_non_default)) * 100.0)
    return ks_stat

def calculate_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Computes Brier Score: (1 / N) * sum((y_prob - y_true)^2)
    """
    return float(brier_score_loss(y_true, y_prob))

def calculate_psi(expected_probs: np.ndarray, actual_probs: np.ndarray, num_bins: int = 10) -> float:
    """
    Calculates Population Stability Index (PSI) between baseline and monitoring periods.
    PSI = sum((Actual_b - Expected_b) * ln(Actual_b / Expected_b))
    """
    expected_probs = np.asarray(expected_probs)
    actual_probs = np.asarray(actual_probs)
    
    quantiles = np.linspace(0, 100, num_bins + 1)
    bins = np.percentile(expected_probs, quantiles)
    bins[0] = 0.0
    bins[-1] = 1.0 + 1e-7
    # Ensure monotonic bins
    bins = np.unique(bins)
    if len(bins) < 3:
        bins = np.linspace(0.0, 1.0 + 1e-7, num_bins + 1)
        
    expected_counts, _ = np.histogram(expected_probs, bins=bins)
    actual_counts, _ = np.histogram(actual_probs, bins=bins)
    
    expected_pct = np.maximum(expected_counts / max(len(expected_probs), 1), 1e-4)
    actual_pct = np.maximum(actual_counts / max(len(actual_probs), 1), 1e-4)
    
    # Normalize percentages
    expected_pct = expected_pct / expected_pct.sum()
    actual_pct = actual_pct / actual_pct.sum()
    
    psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi_val)

def compute_roc_curve_data(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Returns (fpr, tpr, roc_auc)
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = float(roc_auc_score(y_true, y_prob))
    return fpr, tpr, auc

def compute_ks_curve_data(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    """
    Generates cumulative distribution data for KS plot.
    """
    df = pd.DataFrame({'target': y_true, 'prob': y_prob})
    df = df.sort_values(by='prob', ascending=False).reset_index(drop=True)
    df['cum_pop_pct'] = (np.arange(len(df)) + 1) / len(df) * 100.0
    df['cum_default_pct'] = df['target'].cumsum() / max(df['target'].sum(), 1) * 100.0
    df['cum_non_default_pct'] = (1 - df['target']).cumsum() / max((1 - df['target']).sum(), 1) * 100.0
    df['separation'] = np.abs(df['cum_default_pct'] - df['cum_non_default_pct'])
    return df

def compute_calibration_curve_data(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """
    Computes calibration deciles comparing predicted PD vs observed default rates.
    """
    df = pd.DataFrame({'target': y_true, 'prob': y_prob})
    df['bin'] = pd.qcut(df['prob'], q=min(n_bins, df['prob'].nunique()), duplicates='drop')
    grouped = df.groupby('bin', observed=False).agg(
        pred_pd=('prob', 'mean'),
        obs_default_rate=('target', 'mean'),
        sample_count=('target', 'count'),
        default_count=('target', 'sum')
    ).reset_index()
    return grouped

def generate_validation_report(y_true: np.ndarray, y_prob: np.ndarray, psi_ref: np.ndarray = None) -> pd.DataFrame:
    """
    Generates RMG Model Validation Scorecard comparing metrics against institutional thresholds.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    
    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    gini = calculate_gini(y_true, y_prob)
    ks = calculate_ks_stat(y_true, y_prob)
    brier = calculate_brier_score(y_true, y_prob)
    
    if psi_ref is None:
        # Standard RMG baseline vs monitoring partition using seeded split
        np.random.seed(42)
        indices = np.random.permutation(len(y_prob))
        split_idx = len(y_prob) // 2
        base_sample = y_prob[indices[:split_idx]]
        mon_sample = y_prob[indices[split_idx:]]
        psi_val = calculate_psi(base_sample, mon_sample)
    else:
        psi_val = calculate_psi(psi_ref, y_prob)
        
    metrics = [
        {
            "Metric Class": "Discrimination",
            "Metric Name": "ROC-AUC",
            "Model Value": round(auc, 4),
            "Operational Target": ">= 0.80",
            "Fail Trigger": "< 0.72",
            "Status": "PASS (Green)" if auc >= 0.80 else ("AMBER" if auc >= 0.72 else "FAIL (Red)")
        },
        {
            "Metric Class": "Discrimination",
            "Metric Name": "Gini Coefficient",
            "Model Value": round(gini, 4),
            "Operational Target": ">= 0.60",
            "Fail Trigger": "< 0.44",
            "Status": "PASS (Green)" if gini >= 0.60 else ("AMBER" if gini >= 0.44 else "FAIL (Red)")
        },
        {
            "Metric Class": "Separation",
            "Metric Name": "KS Statistic (%)",
            "Model Value": round(ks, 2),
            "Operational Target": ">= 45.0%",
            "Fail Trigger": "< 35.0%",
            "Status": "PASS (Green)" if ks >= 45.0 else ("AMBER" if ks >= 35.0 else "FAIL (Red)")
        },
        {
            "Metric Class": "Calibration",
            "Metric Name": "Brier Score",
            "Model Value": round(brier, 4),
            "Operational Target": "<= 0.08",
            "Fail Trigger": "> 0.15",
            "Status": "PASS (Green)" if brier <= 0.08 else ("AMBER" if brier <= 0.15 else "FAIL (Red)")
        },
        {
            "Metric Class": "Stability",
            "Metric Name": "PSI (Stability Index)",
            "Model Value": round(psi_val, 4),
            "Operational Target": "< 0.10 (Stable)",
            "Fail Trigger": "> 0.25 (Action)",
            "Status": "PASS (Green)" if psi_val < 0.10 else ("AMBER" if psi_val <= 0.25 else "ACTION REQUIRED (Red)")
        }
    ]
    return pd.DataFrame(metrics)
