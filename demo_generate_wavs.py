#!/usr/bin/env python3
"""
demo_generate_wavs.py

Generate WAV files and metadata.csv for all built-in generators and a set of states.
 - WAVs are only created if they don't exist.
 - CSV is only appended with rows for newly seen (generator, state) pairs.
"""

import os
import csv

from generators.rule_based import RuleBasedGenerator
from generators.physical import PhysicalModelGenerator
from generators.cgan import CGANGenerator
from generators.cvae import CVAEGenerator
from generators.diffusion import DiffusionGenerator

# Configuration
cfg = {
    'sample_rate':      8000,
    'duration_seconds': 0.1,
    # ...other defaults...
}

generators = {
    'rule_based': RuleBasedGenerator(cfg),
    'physical':   PhysicalModelGenerator(cfg),
    'cgan':       CGANGenerator(cfg),
    'cvae':       CVAEGenerator(cfg),
    'diffusion':  DiffusionGenerator(cfg),
}

states = [0, 1, 2, 3]

out_dir = "output"
wav_dir = os.path.join(out_dir, "wavs")
os.makedirs(wav_dir, exist_ok=True)

csv_path = os.path.join(out_dir, "metadata.csv")
fieldnames = ['generator', 'wav_file', 'time_start', 'time_end', 'cavitation_level']

# 1) Load existing metadata keys to avoid duplicates
existing = set()
if os.path.exists(csv_path):
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # key by (generator, cavitation_level)
            existing.add((row['generator'], int(row['cavitation_level'])))

# 2) Open CSV in append mode
write_header = not os.path.exists(csv_path)
with open(csv_path, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    new_rows = 0
    for name, gen in generators.items():
        gen_dir = os.path.join(wav_dir, name)
        os.makedirs(gen_dir, exist_ok=True)

        for state in states:
            key = (name, state)
            # Skip if we've already logged this (generator,state)
            if key in existing:
                continue

            # Determine wav path and generate if missing
            wav_path = os.path.join(gen_dir, f"state_{state}.wav")
            if not os.path.exists(wav_path):
                ts, _ = gen.generate(state)
                ts.stream_to_wav(wav_path)
            else:
                # regenerate ts so we know its length
                ts, _ = gen.generate(state)

            start_sec = 0.0
            end_sec   = len(ts) / ts.sample_rate

            writer.writerow({
                'generator':        name,
                'wav_file':         wav_path,
                'time_start':       start_sec,
                'time_end':         end_sec,
                'cavitation_level': state
            })
            new_rows += 1
            existing.add(key)

print(f"Appended {new_rows} new rows to {csv_path}")
