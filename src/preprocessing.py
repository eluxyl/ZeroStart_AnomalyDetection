"""
Temporal Feature Engineering and Data Preprocessing Component.
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger("ZeroStartFraud.Preprocessing")

class TemporalPipeline:
    """Constructs rolling historical velocity statistics for transaction flows."""

    @staticmethod
    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """ Transforms raw transactions into continuous rolling profile metrics. """
        logger.info("Initializing transaction ordering and temporal velocity extractions...")
        df_processed = df.sort_values(by="Time").copy()
        
        # Convert relative timeline metrics into logical operational hour intervals
        df_processed['Time_Hours'] = df_processed['Time'] / 3600.0
        df_processed['Amount_Log'] = np.log1p(df_processed['Amount'])
        
        # Simulated feature engine rolling windows
        df_processed['Rolling_Count_100t'] = df_processed['Amount'].rolling(window=100, min_periods=1).count()
        df_processed['Rolling_Mean_100t'] = df_processed['Amount'].rolling(window=100, min_periods=1).mean()
        df_processed['Amount_to_Rolling_Mean_Ratio'] = df_processed['Amount'] / (df_processed['Rolling_Mean_100t'] + 1e-5)
        
        # Backward fill any rolling boundary initialization empty metrics
        df_processed.fillna(method='bfill', inplace=True)
        return df_processed