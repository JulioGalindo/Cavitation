import numpy as np
import soundfile as sf
import csv
import os

class TimeSeries:
    """
    Encapsulates a time-domain signal with its sample rate and provides utilities
    for unit-safe slicing and chunked streaming to WAV or CSV.
    """
    def __init__(self, data: np.ndarray, sample_rate: int):
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a numpy.ndarray")
        if data.ndim != 1:
            raise ValueError("data must be a 1D array")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self.data = data
        self.sample_rate = sample_rate
        self._time = None

    @property
    def time(self) -> np.ndarray:
        """
        Time vector in seconds corresponding to each sample.
        """
        if self._time is None or len(self._time) != len(self.data):
            self._time = np.arange(len(self.data)) / self.sample_rate
        return self._time

    def slice_time(self, start_sec: float, end_sec: float) -> 'TimeSeries':
        """
        Returns a new TimeSeries with data between start_sec and end_sec.

        Parameters
        ----------
        start_sec : float
            Start time in seconds.
        end_sec : float
            End time in seconds.

        Returns
        -------
        TimeSeries
            Sliced segment.
        """
        if start_sec < 0 or end_sec < 0:
            raise ValueError("start_sec and end_sec must be non-negative")
        if end_sec <= start_sec:
            raise ValueError("end_sec must be greater than start_sec")
        start_idx = int(np.floor(start_sec * self.sample_rate))
        end_idx = int(np.ceil(end_sec * self.sample_rate))
        segment = self.data[start_idx:end_idx]
        return TimeSeries(segment, self.sample_rate)

    def stream_to_wav(self, path: str, chunk_size: int = None):
        """
        Write the TimeSeries to a WAV file in chunks to control memory usage.

        Parameters
        ----------
        path : str
            Output WAV file path.
        chunk_size : int, optional
            Number of samples per write chunk. Defaults to full signal.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if chunk_size is None or chunk_size >= len(self.data):
            sf.write(path, self.data, self.sample_rate, format='WAV', subtype='FLOAT')
        else:
            with sf.SoundFile(path, mode='w', samplerate=self.sample_rate,
                              channels=1, format='WAV', subtype='FLOAT') as f:
                for start in range(0, len(self.data), chunk_size):
                    end = min(start + chunk_size, len(self.data))
                    f.write(self.data[start:end])

    def stream_to_csv(self, path: str, metadata: list, chunk_size: int = None):
        """
        Append metadata rows to a CSV file in a memory-efficient manner.

        Parameters
        ----------
        path : str
            CSV file path.
        metadata : list of dict
            Each dict should have keys matching CSV header, e.g.,
            {'wav_file': str, 'time_start': float, 'time_end': float, 'cavitation_level': int}.
        chunk_size : int, optional
            Number of rows to write per flush. Defaults to all.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Determine header: use default if no metadata
        if not metadata:
            header = ['wav_file', 'time_start', 'time_end', 'cavitation_level']
        else:
            header = list(metadata[0].keys())
        write_header = not os.path.exists(path)
        with open(path, mode='a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header)
            if write_header:
                writer.writeheader()
            # If no metadata rows, header only
            if not metadata:
                return
            # Write rows in chunks if requested
            if chunk_size is None:
                for row in metadata:
                    writer.writerow(row)
            else:
                for i in range(0, len(metadata), chunk_size):
                    batch = metadata[i:i+chunk_size]
                    for row in batch:
                        writer.writerow(row)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Allows indexing by sample or by slice.

        If idx is a slice of ints: returns a new TimeSeries.
        If idx is an int: returns the sample value.
        """
        if isinstance(idx, slice):
            segment = self.data[idx]
            return TimeSeries(segment, self.sample_rate)
        return self.data[idx]
