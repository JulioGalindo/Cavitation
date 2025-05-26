#!/usr/bin/env python3
"""
demo_timeseries.py

Demonstration of the TimeSeries utility:
 - Construction from data & sample_rate
 - Assigning a label
 - Inspecting metadata (num_samples, duration, label)
 - Indexing and slicing
 - Simple plotting
"""

import numpy as np
import matplotlib.pyplot as plt
from utils.timeseries import TimeSeries

def main():
    # 1) Generate a 50 Hz sine wave, 0.1 s at 1 kHz
    sr = 1000
    duration_seconds = 0.1
    num_samples = int(sr * duration_seconds)
    t = np.linspace(0, duration_seconds, num_samples, endpoint=False)
    data = 0.5 * np.sin(2 * np.pi * 50 * t)

    # 2) Instantiate TimeSeries (only data + sample_rate)
    ts = TimeSeries(data, sr)
    ts.label = "50 Hz tone"

    # 3) Inspect basic properties
    print("TimeSeries repr:", ts)
    print("  Label       :", ts.label)
    print("  Num samples :", len(ts))
    # Compute duration from number of samples and sample rate
    duration = len(ts) / ts.sample_rate
    print("  Duration    : {:.4f} seconds".format(duration))

    # 4) Indexing and slicing
    print("\nFirst sample:", ts[0])
    sub = ts[100:200]
    sub_duration = len(sub) / sub.sample_rate
    print(f"Slice ts[100:200] → samples: {len(sub)}, duration: {sub_duration:.4f}s")

    # 5) Plot the waveform
    fig, ax = plt.subplots()
    ax.plot(ts.time, ts.data, label=ts.label)
    ax.set_title("TimeSeries Demo")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()
