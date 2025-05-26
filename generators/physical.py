import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import butter, filtfilt
from generators.base import GenerativeModel
from utils.timeseries import TimeSeries

class PhysicalModelGenerator(GenerativeModel):
    """
    Physics-based cavitation signal generator using the Rayleigh–Plesset equation.

    Generates an acoustic proxy signal from bubble dynamics and optionally applies
    a band-pass filter and rotor-tone baseline.
    """
    def __init__(self, config: dict):
        # Core signal parameters
        self.sample_rate = config['sample_rate']
        self.duration_seconds = config['duration_seconds']
        self.num_samples = int(self.sample_rate * self.duration_seconds)
        self.t = np.linspace(0, self.duration_seconds, self.num_samples)

        # Physical model parameters
        phys = config.get('physical_model', {})
        self.rho = phys.get('rho', 998.0)
        self.sigma = phys.get('sigma', 0.072)
        self.mu = phys.get('mu', 0.001)
        self.P_v = phys.get('P_v', 2300.0)
        self.R0 = phys.get('R0', 1.0)
        self.dR0 = phys.get('dR0', 0.0)
        self.scale_factor = phys.get('scale_factor', 1.0)
        self.max_state = phys.get('max_state', config.get('max_state', 6))

        # Rotor-tone baseline parameters
        self.rotor_freq = phys.get('rotor_freq', config.get('rotor_freq', 0.0))

        # Optional band-pass filter
        filt = config.get('filter_type', None)
        if filt == 'butter':
            cutoff = config.get('filter_cutoff', [3000.0, 20000.0])
            order = config.get('filter_order', 3)
            nyq = self.sample_rate / 2.0
            low, high = np.array(cutoff) / nyq
            self._b, self._a = butter(order, [low, high], btype='band')
        else:
            self._b = self._a = None

    def train(self, *args, **kwargs):
        # No training for physics-based generator
        return None

    def generate(self, state: int):
        """
        Generate a synthetic acoustic signal for a given cavitation state.

        Parameters
        ----------
        state : int
            Cavitation severity level (0=no cavitation to max_state).

        Returns
        -------
        ts : TimeSeries
            Generated signal with time vector.
        metadata : list
            Empty list (no discrete events).
        """
        # Rayleigh-Plesset ODE system
        def rp_ode(t, y):
            R, dR = y
            P_inf = self.P_v
            ddR = (1.0/self.rho) * (P_inf - self.P_v - 2*self.sigma/R - 4*self.mu*dR/R)
            ddR -= 1.5 * (dR/R)**2
            return [dR, ddR]

        sol = solve_ivp(
            rp_ode,
            t_span=(0, self.duration_seconds),
            y0=[self.R0, self.dR0],
            t_eval=self.t,
            method='RK45'
        )

        R = sol.y[0]
        dR = sol.y[1]
        # Compute acoustic proxy
        ddR = (1.0/self.rho) * (self.P_v - self.P_v - 2*self.sigma/R - 4*self.mu*dR/R) - 1.5*(dR/R)**2
        ddR = np.array(ddR, dtype=np.float32)

        # Normalize acoustic signal
        if np.max(np.abs(ddR)) > 0:
            ac = ddR / np.max(np.abs(ddR))
        else:
            ac = ddR

        # Rotor-tone baseline
        if self.rotor_freq > 0:
            tone = np.sin(2 * np.pi * self.rotor_freq * self.t)
            if np.max(np.abs(tone)) > 0:
                tone = tone / np.max(np.abs(tone))
        else:
            tone = np.zeros_like(ac)

        # Combine and scale by state severity
        signal = tone + self.scale_factor * ac
        if state > 0:
            signal = signal * (state / self.max_state)

                # Optional filtering
        if self._b is not None and self._a is not None:
            # filtfilt padlen requirement: ntaps*3 where ntaps = max(len(a), len(b))
            ntaps = max(len(self._a), len(self._b))
            padlen = ntaps * 3
            if len(signal) > padlen:
                signal = filtfilt(self._b, self._a, signal)
            # else skip filtering for too-short signals

        ts = TimeSeries(data=signal, sample_rate=self.sample_rate)
        metadata = []
        return ts, metadata
