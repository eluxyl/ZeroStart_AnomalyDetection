"""
Advanced Validation Metrics Component.
"""

import logging
from typing import Dict, Any
import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix

logger = logging.getLogger("ZeroStartFraud.Evaluation")

class PerformanceEvaluator:
    """ Computes business performance metrics for imbalanced distributions. """
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_scores: np.ndarray) -> Dict[str, Any]:
        """ Extracts AUPRC, optimal decision indices, and score separation metrics. """
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        
        eps = 1e-9
        f1_scores = [2 * (p * r) / (p + r + eps) for p, r in zip(precision, recall)]
        opt_idx = np.argmax(f1_scores)
        opt_threshold = thresholds[min(opt_idx, len(thresholds)-1)]
        max_f1 = f1_scores[opt_idx]
        
        y_pred = (y_scores >= opt_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
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