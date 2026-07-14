"""
Configuration settings for the ZeroStart-Fraud pipeline.
"""

from dataclasses import dataclass, field
from typing import List

@dataclass(frozen=True)
class PipelineSettings:
    # Ingestion Core
    OPENML_DATA_ID: int = 1597
    RANDOM_STATE: int = 42
    TRAIN_RATIO: float = 0.80
    CONTAMINATION_RATE: float = 0.0017
    
    # Simulation Parameters
    CUSTOMER_BASE_SIZE: int = 5000
    EVALUATION_WINDOW_DAYS: int = 365
    
    # Feature Engineering Column Layouts
    EXCLUDE_COLUMNS: List[str] = field(default_factory=lambda: [
        'Class', 'Time', 'Cust_ID', 'Prediction', 
        'TP_Flag', 'FP_Flag', 'FN_Flag', 'Tenure_Days', 'Churn_Event'
    ])