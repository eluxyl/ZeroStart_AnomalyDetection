"""
Data Ingestion Component.
"""

import logging
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from config.settings import PipelineSettings

logger = logging.getLogger("ZeroStartFraud.Ingestion")

class DataIngestion:
    """Handles data pulling from OpenML with resilient synthetic data failovers."""
    
    def __init__(self, settings: PipelineSettings):
        self.settings = settings

    def load_fraud_data(self) -> pd.DataFrame:
        """ Fetches the Credit Card Fraud dataset from OpenML or initiates fallback. """
        try:
            logger.info(f"Connecting to OpenML server for dataset ID: {self.settings.OPENML_DATA_ID}...")
            raw_data = fetch_openml(data_id=self.settings.OPENML_DATA_ID, as_frame=True, parser='auto')
            df = raw_data.frame
            
            # 1. Unify column casing to Title Case (Time, Amount, Class)
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
            
            # 2. Enforce explicit integer type on Target Class
            df['Class'] = df['Class'].astype(int)
                
            logger.info(f"Ingestion successful and columns standardized. Data shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.warning(f"Network ingestion failed: {str(e)}. Triggering high-fidelity synthetic fallback generation...")
            return self._generate_synthetic_fallback()
            
         

    def _generate_synthetic_fallback(self) -> pd.DataFrame:
        """ Generates a representative imbalanced dataset mimicking PCA vector profiles. """
        np.random.seed(self.settings.RANDOM_STATE)
        n_samples = 150000
        n_fraud = int(n_samples * self.settings.CONTAMINATION_RATE)
        
        features = np.random.normal(loc=0, scale=1, size=(n_samples, 28))
        features[:n_fraud] += np.random.normal(loc=1.8, scale=2.0, size=(n_fraud, 28)) # clear separation bias
        
        df = pd.DataFrame(features, columns=[f"V{i}" for i in range(1, 29)])
        df['Time'] = np.sort(np.random.uniform(0, 172800, size=n_samples))
        df['Amount'] = np.random.exponential(scale=88.0, size=n_samples)
        
        classes = np.zeros(n_samples, dtype=int)
        classes[:n_fraud] = 1
        df['Class'] = classes
        
        return df.sample(frac=1, random_state=self.settings.RANDOM_STATE).reset_index(drop=True)