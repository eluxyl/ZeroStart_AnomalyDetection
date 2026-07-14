"""
ZeroStart-Fraud: Unsupervised Anomaly Detection & Customer Survival Impact Pipeline.
Author: Gongyao Xu
License: MIT
"""

import os
import sys
import logging
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.datasets import fetch_openml
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, auc, f1_score, confusion_matrix
from lifelines import CoxPHFitter

# =====================================================================
# 1. LOGGING & GLOBAL CONFIGURATION
# =====================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ZeroStartFraud")

class PipelineConfig:
    """Config class storing hyper-parameters and run settings."""
    OPENML_DATA_ID = 1597  # Classic Credit Card Fraud dataset
    RANDOM_STATE = 42
    TRAIN_RATIO = 0.80
    CONTAMINATION_RATE = 0.0017  # True ratio of anomaly class in raw data
    CUSTOMER_BASE_SIZE = 5000    # Cohort size for survival simulation


# =====================================================================
# 2. DATA INGESTION COMPONENT
# =====================================================================
# =====================================================================
# 2. DATA INGESTION COMPONENT
# =====================================================================
class DataIngestion:
    """Handles fetching and local caching of open-source OpenML datasets."""
    
    @staticmethod
    def load_fraud_data() -> pd.DataFrame:
        """
        Fetches the Credit Card Fraud dataset from OpenML.
        Falls back to generating highly realistic synthetic data if OpenML is offline.
        """
        try:
            logger.info(f"Attempting to download OpenML dataset ID: {PipelineConfig.OPENML_DATA_ID}...")
            raw_data = fetch_openml(data_id=PipelineConfig.OPENML_DATA_ID, as_frame=True, parser='auto')
            df = raw_data.frame
            
            # --- THE FIX: Force unify column casing to Title Case ---
            rename_map = {}
            for col in df.columns:
                if col.lower() == 'time':
                    rename_map[col] = 'Time'
                elif col.lower() == 'amount':
                    rename_map[col] = 'Amount'
                elif col.lower() == 'class':
                    rename_map[col] = 'Class'
            
            if rename_map:
                df.rename(columns=rename_map, inplace=True)
            
            # Enforce explicit integer type on Target Class
            df['Class'] = df['Class'].astype(int)
                
            logger.info(f"Successfully loaded dataset with shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.warning(f"OpenML download failed: {str(e)}. Generating fallback synthetic dataset...")
            return DataIngestion._generate_synthetic_fallback()

    @staticmethod
    def _generate_synthetic_fallback() -> pd.DataFrame:
        """Generates a representative imbalanced dataset mimicking Credit Card PCA structure."""
        np.random.seed(PipelineConfig.RANDOM_STATE)
        n_samples = 150000
        n_fraud = int(n_samples * PipelineConfig.CONTAMINATION_RATE)
        
        features = np.random.normal(loc=0, scale=1, size=(n_samples, 28))
        features[:n_fraud] += np.random.normal(loc=1.5, scale=2.0, size=(n_fraud, 28))
        
        df = pd.DataFrame(features, columns=[f"V{i}" for i in range(1, 29)])
        df['Time'] = np.sort(np.random.uniform(0, 172800, size=n_samples))
        df['Amount'] = np.random.exponential(scale=88.0, size=n_samples)
        
        classes = np.zeros(n_samples, dtype=int)
        classes[:n_fraud] = 1
        df['Class'] = classes
        
        df = df.sample(frac=1, random_state=PipelineConfig.RANDOM_STATE).reset_index(drop=True)
        return df

# =====================================================================
# 3. TEMPORAL FEATURE ENGINEERING PIPELINE
# =====================================================================
# =====================================================================
# 3. TEMPORAL FEATURE ENGINEERING PIPELINE
# =====================================================================
class TemporalPipeline:
    """Generates streaming/temporal features using rolling calculation techniques."""
    
    @staticmethod
    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates transactional velocity, deviations, and cumulative profiles.
        Prevents look-ahead contamination using strict forward chronological execution.
        """
        logger.info("Initializing chronological temporal feature pipeline...")
        
        # --- THE FIX: Failsafe if OpenML dropped the 'Time' column ---
        if 'Time' not in df.columns:
            logger.warning("'Time' column is missing from the source dataset. Synthesizing realistic sequential timestamps...")
            # Simulate a 48-hour transactional period (172800 seconds) to keep temporal logic intact
            df['Time'] = np.sort(np.random.uniform(0, 172800, size=len(df)))
            
        df_processed = df.sort_values(by="Time").copy()
        
        # Convert absolute relative time into continuous hours
        df_processed['Time_Hours'] = df_processed['Time'] / 3600.0
        
        # Simulated rolling windows - we bin transactions to calculate rolling profiles
        df_processed['Amount_Log'] = np.log1p(df_processed['Amount'])
        
        logger.info("Engineering transaction velocity and rolling deviation features...")
        df_processed['Rolling_Count_100t'] = df_processed['Amount'].rolling(window=100, min_periods=1).count()
        df_processed['Rolling_Mean_100t'] = df_processed['Amount'].rolling(window=100, min_periods=1).mean()
        df_processed['Amount_to_Rolling_Mean_Ratio'] = df_processed['Amount'] / (df_processed['Rolling_Mean_100t'] + 1e-5)
        
        # Clean any infinite numerical anomalies or NaNs from the sequence
        df_processed.fillna(method='bfill', inplace=True)
        return df_processed
# =====================================================================
# 4. UNSUPERVISED ANOMALY DETECTION ENGINE
# =====================================================================
class UnsupervisedAnomalyDetector:
    """Isolation Forest framework representing zero-historical label dependency (Cold-Start)."""
    
    def __init__(self, contamination: float = PipelineConfig.CONTAMINATION_RATE):
        self.model = IsolationForest(
            n_estimators=150,
            max_samples='auto',
            contamination=contamination,
            random_state=PipelineConfig.RANDOM_STATE,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self._is_fitted = False

    def fit_normal(self, X: pd.DataFrame) -> "UnsupervisedAnomalyDetector":
        """Fits the anomaly model on unlabelled background transactions."""
        logger.info("Normalizing features for the isolation forest...")
        X_scaled = self.scaler.fit_transform(X)
        logger.info("Training Unsupervised Isolation Forest model...")
        self.model.fit(X_scaled)
        self._is_fitted = True
        return self

    def score_anomalies(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns anomaly decision score arrays.
        Lower scores indicate highly isolated samples (severe anomaly risk).
        We invert this so higher values correspond to a higher anomaly likelihood.
        """
        if not self._is_fitted:
            raise ValueError("The detector instance must be fit first before calling scoring functions.")
        X_scaled = self.scaler.transform(X)
        # score_samples returns negative anomaly scores
        raw_scores = self.model.score_samples(X_scaled)
        # Transform scores so high values = higher fraud risk (0 to 1 scale roughly)
        return -raw_scores


# =====================================================================
# 5. BUSINESS AND SEVERITY METRICS EVALUATOR
# =====================================================================
class PerformanceEvaluator:
    """Evaluates unsupervised classification targets over high imbalance thresholds."""
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_scores: np.ndarray) -> Dict[str, Any]:
        """Runs precision-recall evaluations, KS statistics, and optimal F1 analysis."""
        logger.info("Calculating operational pipeline performance statistics...")
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        
        # Calculate optimal operating threshold utilizing custom F1-score maximizers
        f1_scores = []
        # Protect against empty arrays
        eps = 1e-9
        for p, r in zip(precision, recall):
            f1_scores.append(2 * (p * r) / (p + r + eps))
            
        opt_idx = np.argmax(f1_scores)
        opt_threshold = thresholds[min(opt_idx, len(thresholds)-1)]
        max_f1 = f1_scores[opt_idx]
        
        # Generate hard binary classifications at optimal threshold boundary
        y_pred = (y_scores >= opt_threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate Kolmogorov-Smirnov (KS) Statistics
        benign_scores = y_scores[y_true == 0]
        anomaly_scores = y_scores[y_true == 1]
        ks_stat, p_value = ks_2samp(benign_scores, anomaly_scores)
        
        return {
            "PR_AUC": pr_auc,
            "Optimal_Threshold": opt_threshold,
            "Max_F1": max_f1,
            "KS_Statistic": ks_stat,
            "KS_P_Value": p_value,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn
        }


# =====================================================================
# 6. COHORT SIMULATION AND COAL SURVIVAL IMPACT ANALYZER
# =====================================================================
class SurvivalImpactAnalyzer:
    """Quantifies business churn risks using Cox Proportional Hazard modeling."""
    
    @staticmethod
    def run_survival_impact(df_results: pd.DataFrame, num_customers: int = PipelineConfig.CUSTOMER_BASE_SIZE) -> CoxPHFitter:
        """
        Groups metrics by simulated users and models survival decay rates.
        Quantifies how false positives vs. false negatives drive customer churn.
        """
        logger.info(f"Simulating Customer Experience cohort for {num_customers} account holders...")
        np.random.seed(PipelineConfig.RANDOM_STATE)
        
        # Generate simulated Customer IDs for the transaction history
        cust_ids = [f"CUST_{str(i).zfill(5)}" for i in range(1, num_customers + 1)]
        df_results['Cust_ID'] = np.random.choice(cust_ids, size=len(df_results))
        
        # Group metrics to build a per-customer experience history profile
        cust_profile = df_results.groupby('Cust_ID').agg(
            Total_Trans=('Amount', 'count'),
            Max_Amount=('Amount', 'max'),
            False_Positives=('FP_Flag', 'sum'),
            False_Negatives=('FN_Flag', 'sum'),
            True_Positives=('TP_Flag', 'sum')
        ).reset_index()
        
        # Normalize variables for modeling stability
        cust_profile['High_Value_Customer'] = (cust_profile['Max_Amount'] > 250).astype(int)
        
        # Create simulated survival tenure (days active) and active event parameters
        # Hazard logic modeling:
        # Base Churn Rate (hazard) is scaled heavily by False Positives (customer friction) 
        # and False Negatives (actual theft loss)
        survival_durations = []
        churn_events = []
        
        for idx, row in cust_profile.iterrows():
            # Linear combination of hazards
            hazard = (
                0.05 + 
                0.15 * row['False_Positives'] + 
                0.25 * row['False_Negatives'] - 
                0.02 * row['True_Positives']
            )
            hazard = min(max(hazard, 0.01), 0.95) # Bound probabilities
            
            # Draw duration and event status
            days_active = int(np.random.geometric(p=hazard))
            days_active = min(days_active, 365) # Clamp to 1-year evaluation window
            
            # If they lasted less than a year, they churned
            churned = 1 if days_active < 365 else 0
            
            survival_durations.append(days_active)
            churn_events.append(churned)
            
        cust_profile['Tenure_Days'] = survival_durations
        cust_profile['Churn_Event'] = churn_events
        
        # Drop raw high-cardinality fields before statistical estimation
        survival_df = cust_profile[[
            'Tenure_Days', 'Churn_Event', 'False_Positives', 
            'False_Negatives', 'True_Positives', 'High_Value_Customer'
        ]].copy()
        
        logger.info("Fitting Cox Proportional Hazards Model on Customer Cohorts...")
        cph = CoxPHFitter()
        cph.fit(survival_df, duration_col='Tenure_Days', event_col='Churn_Event')
        return cph


# =====================================================================
# 7. EXECUTION ORCHESTRATION PIPELINE
# =====================================================================
def main():
    logger.info("Initializing ZeroStart-Fraud Production Analysis Pipeline...")
    
    # Step 1: Ingestion
    df_raw = DataIngestion.load_fraud_data()
    
    # Step 2: Feature Engineering
    df_engineered = TemporalPipeline.engineer_features(df_raw)
    
    # Split into chronological datasets (Out-Of-Time Validation)
    split_idx = int(len(df_engineered) * PipelineConfig.TRAIN_RATIO)
    train_df = df_engineered.iloc[:split_idx]
    test_df = df_engineered.iloc[split_idx:].copy()
    
    # Define features for training
    feature_cols = [c for c in df_engineered.columns if c not in ['Class', 'Time', 'Cust_ID']]
    
    # Step 3: Model Invariant Learning (Unsupervised)
    # Fit ONLY on clean transaction traces within the training partition
    normal_train = train_df[train_df['Class'] == 0][feature_cols]
    
    detector = UnsupervisedAnomalyDetector()
    detector.fit_normal(normal_train)
    
    # Step 4: Out-Of-Time Scoring & Metric Calculation
    test_scores = detector.score_anomalies(test_df[feature_cols])
    test_df['Anomaly_Score'] = test_scores
    
    y_test_true = test_df['Class'].values
    eval_metrics = PerformanceEvaluator.calculate_metrics(y_test_true, test_scores)
    
    # Output metrics to console
    print("\n" + "="*50)
    print("        UNSUPERVISED DETECTOR METRICS")
    print("="*50)
    print(f"PR-AUC (AUPRC)        : {eval_metrics['PR_AUC']:.5f}")
    print(f"Optimal Threshold    : {eval_metrics['Optimal_Threshold']:.5f}")
    print(f"Maximized F1 Score   : {eval_metrics['Max_F1']:.5f}")
    print(f"KS-Statistic (Score) : {eval_metrics['KS_Statistic']:.5f} (p-val: {eval_metrics['KS_P_Value']:.2e})")
    print(f"True Positives (TP)  : {eval_metrics['TP']}")
    print(f"False Positives (FP) : {eval_metrics['FP']}")
    print(f"False Negatives (FN) : {eval_metrics['FN']}")
    print(f"True Negatives (TN)  : {eval_metrics['TN']}")
    print("="*50 + "\n")
    
    # Map predictions back to the test dataframe
    opt_t = eval_metrics['Optimal_Threshold']
    test_df['Prediction'] = (test_df['Anomaly_Score'] >= opt_t).astype(int)
    
    # Tag transactional classifications
    test_df['TP_Flag'] = ((test_df['Prediction'] == 1) & (test_df['Class'] == 1)).astype(int)
    test_df['FP_Flag'] = ((test_df['Prediction'] == 1) & (test_df['Class'] == 0)).astype(int)
    test_df['FN_Flag'] = ((test_df['Prediction'] == 0) & (test_df['Class'] == 1)).astype(int)
    
    # Step 5: Downstream Survival Metric Integration
    cph_model = SurvivalImpactAnalyzer.run_survival_impact(test_df)
    
    print("\n" + "="*50)
    print("     COHORT HAZARDS & CLTV SURVIVAL SUMMARY")
    print("="*50)
    cph_model.print_summary()
    print("="*50 + "\n")
    
    logger.info("ZeroStart-Fraud pipeline completed successfully!")


if __name__ == "__main__":
    main()