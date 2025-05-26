"""
utils/metrics.py

Signal and classification metrics for cavitation detection.
"""
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

class Metrics:
    """
    Collection of signal-level and classification-level metrics.
    """

    @staticmethod
    def rms(signal: np.ndarray) -> float:
        """
        Compute root mean square of a signal.
        """
        return float(np.sqrt(np.mean(signal.astype(np.float64) ** 2)))

    @staticmethod
    def snr_db(clean: np.ndarray, noise: np.ndarray) -> float:
        """
        Compute signal-to-noise ratio in decibels.
        """
        ps = np.mean(clean.astype(np.float64) ** 2)
        pn = np.mean(noise.astype(np.float64) ** 2)
        if pn <= 0:
            return float('inf')
        return 10.0 * np.log10(ps / pn)

    @staticmethod
    def classification_report(y_true, y_pred, average: str = 'macro') -> dict:
        """
        Generate a dictionary of classification metrics.

        Returns keys: accuracy, precision, recall, f1, confusion_matrix.
        """
        report = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
            'f1': f1_score(y_true, y_pred, average=average, zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        return report
