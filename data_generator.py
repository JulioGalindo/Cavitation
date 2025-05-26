import os
import yaml
from datetime import datetime
from generators.rule_based import RuleBasedGenerator
# Future: import other generators
from utils.timeseries import TimeSeries

class SyntheticDataGenerator:
    """
    Front-end for synthetic cavitation data generation.
    Parses configuration, dispatches to the chosen GenerativeModel,
    and handles WAV + metadata CSV output.
    """
    def __init__(self, config_path: str):
        # Load YAML config
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        self.cfg = cfg.get('data_generator', cfg)
        # Output settings
        self.output_dir = self.cfg.get('default_output_dir', 'output')
        self.csv_path = self.cfg.get('default_csv_path', os.path.join(self.output_dir, 'metadata.csv'))
        os.makedirs(self.output_dir, exist_ok=True)
        # Instantiate generator based on method
        method = self.cfg.get('method', 'rule_based')
        if method == 'rule_based':
            self.generator = RuleBasedGenerator(self.cfg)
        else:
            raise ValueError(f"Unsupported generation method: {method}")

    def generate_and_save(self, state: int):
        """
        Generate a signal for the given cavitation state, save WAV and metadata.

        Parameters
        ----------
        state : int
            Cavitation severity level.

        Returns
        -------
        wav_path : str
            Path to the saved WAV file.
        csv_path : str
            Path to the metadata CSV file (appended).
        """
        # Generate signal and metadata
        series, metadata = self.generator.generate(state)
        # Prepare WAV path with timestamp
        timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
        wav_filename = f'signal_state{state}_{timestamp}.wav'
        wav_path = os.path.join(self.output_dir, wav_filename)
        # Save WAV in chunks
        series.stream_to_wav(wav_path, chunk_size=self.cfg.get('wav_chunk_size', None))
        # Enrich metadata
        enriched = []
        for m in metadata:
            enriched.append({
                'wav_file': wav_path,
                'time_start': m['time_start'],
                'time_end': m['time_end'],
                'cavitation_level': m['state']
            })
        # Save metadata CSV
        series.stream_to_csv(self.csv_path, enriched, chunk_size=self.cfg.get('csv_chunk_size', None))
        return wav_path, self.csv_path

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Synthetic Cavitation Data Generator')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to YAML configuration')
    parser.add_argument('--state', type=int, required=True, help='Cavitation state to generate')
    args = parser.parse_args()
    gen = SyntheticDataGenerator(args.config)
    wav, csv = gen.generate_and_save(args.state)
    print(f'Generated WAV: {wav}\nUpdated CSV: {csv}')
