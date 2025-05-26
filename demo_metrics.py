#!/usr/bin/env python3
"""
demo_metrics.py

Complete demonstration of the Metrics utility:
 - Computes RMS of a clean signal
 - Computes SNR between clean signal and noise
 - Generates a synthetic multiclass classification example
 - Prints out accuracy, precision, recall, F1 and confusion matrix
"""

import numpy as np
from utils.metrics import Metrics

def main():
    # 1) Generate a clean sine wave and additive white noise
    sr = 1000
    t = np.linspace(0, 1, sr, endpoint=False)
    clean = np.sin(2 * np.pi * 5 * t)          # 5 Hz sine
    noise = 0.2 * np.random.randn(sr)          # white noise, σ=0.2
    mixed = clean + noise

    # 2) Compute signal‐level metrics
    rms_clean = Metrics.rms(clean)
    rms_noise = Metrics.rms(noise)
    snr_value = Metrics.snr_db(clean, noise)

    print("=== Signal‐level Metrics ===")
    print(f"RMS (clean): {rms_clean:.4f}")
    print(f"RMS (noise): {rms_noise:.4f}")
    print(f"SNR (clean vs noise): {snr_value:.2f} dB")
    print()

    # 3) Create a toy classification problem
    #    Suppose cavitation levels 0–2, 10 samples each
    rng = np.random.RandomState(42)
    y_true = np.repeat([0, 1, 2], 10)
    # Simulate imperfect predictions
    y_pred = y_true.copy()
    # flip a few at random
    idx = rng.choice(len(y_true), size=5, replace=False)
    y_pred[idx] = rng.choice([0, 1, 2], size=5)

    # 4) Compute classification metrics
    report = Metrics.classification_report(y_true, y_pred, average='macro')

    print("=== Classification Metrics ===")
    print(f"Accuracy : {report['accuracy']:.3f}")
    print(f"Precision: {report['precision']:.3f}")
    print(f"Recall   : {report['recall']:.3f}")
    print(f"F1‐score : {report['f1']:.3f}")
    print("Confusion matrix:")
    print(report['confusion_matrix'])

if __name__ == "__main__":
    main()
