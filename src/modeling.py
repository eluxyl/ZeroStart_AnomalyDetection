"""
Algorithmic Modeling Layer.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter
from config.settings import PipelineSettings

logger = logging.getLogger("ZeroStartFraud.Modeling")

class UnsupervisedAnomalyDetector:
    """ Cold-start isolation engine that learns topological variance without labels. """
    
    def __init__(self, settings: PipelineSettings):
        self.settings = settings
        self.model = IsolationForest(
            n_estimators=150,
            contamination=self.settings.CONTAMINATION_RATE,
            random_state=self.settings.RANDOM_STATE,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self._is_fitted = False

    def fit_normal(self, X: pd.DataFrame) -> "UnsupervisedAnomalyDetector":
        """ Trains structural invariants strictly on background unlabelled records. """
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self._is_fitted = True
        return self

    def score_anomalies(self, X: pd.DataFrame) -> np.ndarray:
        """ Returns inverted score profiles where high values mean extreme anomaly risk. """
        if not self._is_fitted:
            raise ValueError("Execution Error: Model instance must be fit before calling inference.")
        X_scaled = self.scaler.transform(X)
        raw_scores = self.model.score_samples(X_scaled)
        return -raw_scores


class SurvivalImpactAnalyzer:
    """ Quantifies model error side effects on product retention parameters. """
    
    def __init__(self, settings: PipelineSettings):
        self.settings = settings

    def run_survival_impact(self, df_results: pd.DataFrame) -> CoxPHFitter:
        """ Builds a simulated client history framework to calculate proportional hazards. """
        logger.info("Aggregating transactional errors to generate cohort behavior profiles...")
        np.random.seed(self.settings.RANDOM_STATE)
        
        cust_ids = [f"CUST_{str(i).zfill(5)}" for i in range(1, self.settings.CUSTOMER_BASE_SIZE + 1)]
        df_results['Cust_ID'] = np.random.choice(cust_ids, size=len(df_results))
        
        cust_profile = df_results.groupby('Cust_ID').agg(
            Total_Trans=('Amount', 'count'),
            Max_Amount=('Amount', 'max'),
            False_Positives=('FP_Flag', 'sum'),
            False_Negatives=('FN_Flag', 'sum'),
            True_Positives=('TP_Flag', 'sum')
        ).reset_index()
        
        cust_profile['High_Value_Customer'] = (cust_profile['Max_Amount'] > 250).astype(int)
        
        survival_durations = []
        churn_events = []
        
        for _, row in cust_profile.iterrows():
            hazard = (0.05 + 0.15 * row['False_Positives'] + 0.25 * row['False_Negatives'] - 0.02 * row['True_Positives'])
            hazard = min(max(hazard, 0.01), 0.95)
            
            days_active = int(np.random.geometric(p=hazard))
            days_active = min(days_active, self.settings.EVALUATION_WINDOW_DAYS)
            churned = 1 if days_active < self.settings.EVALUATION_WINDOW_DAYS else 0
            
            survival_durations.append(days_active)
            churn_events.append(churned)
            
        cust_profile['Tenure_Days'] = survival_durations
        cust_profile['Churn_Event'] = churn_events
        
        survival_df = cust_profile[[
            'Tenure_Days', 'Churn_Event', 'False_Positives', 
            'False_Negatives', 'True_Positives', 'High_Value_Customer'
        ]].copy()
        
        cph = CoxPHFitter()
        cph.fit(survival_df, duration_col='Tenure_Days', event_col='Churn_Event')
        return cph