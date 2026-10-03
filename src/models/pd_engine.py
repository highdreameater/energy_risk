import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

FEATURE_COLUMNS = [
    "vol_adj_icr",
    "leverage_ratio",
    "liquidity_ratio",
    "cash_margin_ratio",
    "commodity_vol_252d"
]

class ScorecardPDModel:
    """
    Wholesale Credit Risk Scorecard & Probability of Default (PD) Engine.
    Combines feature scaling, Logistic Regression / XGBoost, and Platt calibration.
    """
    def __init__(self, model_type: str = "logistic", random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state
        self.pipeline = None
        self.calibrated_model = None
        self.feature_names = FEATURE_COLUMNS
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        X_features = X[self.feature_names].copy()
        
        # Base classifier
        if self.model_type == "xgboost" and HAS_XGBOOST:
            base_clf = XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                eval_metric="logloss",
                random_state=self.random_state
            )
            self.pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", base_clf)
            ])
        else:
            base_clf = LogisticRegression(
                C=1.0,
                solver="lbfgs",
                max_iter=1000,
                random_state=self.random_state
            )
            self.pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", base_clf)
            ])
            
        # Fit base model
        self.pipeline.fit(X_features, y)
        
        # Platt Scaling (Sigmoid calibration): PRD Module 2 / Section 3.B
        # Fit secondary logistic regression on logit or use cv=5 CalibratedClassifierCV
        try:
            self.calibrated_model = CalibratedClassifierCV(
                estimator=self.pipeline,
                method="sigmoid",
                cv=5
            )
            self.calibrated_model.fit(X_features, y)
        except Exception:
            # Fallback direct pipeline
            self.calibrated_model = self.pipeline
            
        return self
        
    def predict_proba(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (uncalibrated_pd, calibrated_pd)
        """
        X_features = X[self.feature_names].copy()
        uncalibrated_pd = self.pipeline.predict_proba(X_features)[:, 1]
        if hasattr(self.calibrated_model, "predict_proba"):
            calibrated_pd = self.calibrated_model.predict_proba(X_features)[:, 1]
        else:
            calibrated_pd = uncalibrated_pd
        return uncalibrated_pd, calibrated_pd
        
    def save(self, file_path: str = "models/pd_model.joblib"):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(self, file_path)
        
    @classmethod
    def load(cls, file_path: str = "models/pd_model.joblib"):
        return joblib.load(file_path)

def train_and_calibrate_pd_model(
    df: pd.DataFrame,
    target_col: str = "target",
    model_type: str = "logistic",
    save_path: Optional[str] = "models/pd_model.joblib"
) -> Tuple[pd.DataFrame, ScorecardPDModel]:
    """
    Trains and calibrates PD scorecard on input dataset.
    Returns scored DataFrame with 'pd_calibrated' and 'pd_uncalibrated' columns,
    along with the fitted model.
    """
    data = df.copy()
    
    # If target is missing, synthesize based on risk metrics
    if target_col not in data.columns:
        icr = data.get("vol_adj_icr", 2.0)
        lev = data.get("leverage_ratio", 3.0)
        latent_z = -2.5 - 0.4 * icr + 0.5 * lev
        prob = 1.0 / (1.0 + np.exp(-latent_z))
        np.random.seed(42)
        data[target_col] = np.random.binomial(1, np.clip(prob, 0.01, 0.95))
        
    model = ScorecardPDModel(model_type=model_type)
    model.fit(data, data[target_col])
    
    uncal_pd, cal_pd = model.predict_proba(data)
    data["pd_uncalibrated"] = np.round(uncal_pd, 6)
    data["pd_calibrated"] = np.round(cal_pd, 6)
    
    if save_path:
        model.save(save_path)
        
    return data, model
