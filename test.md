# audio_kernels

High-performance LibTorch extensions for artificial generation and processing of 32-bit cavitation audio signals, developed and maintained by Julio Galindo.

Designed specifically for the synthetic generation of cavitation phenomena in hydraulic turbines and their subsequent analysis.

All operations use `torch::Tensor<float>` on-device, with coefficients and metadata as tensors. Kernels leverage vectorization and multi-threading via `at::parallel_for` for minimal host–device transfers.

This version replaces the previous **heavy_compute**/**dtensor_bridge** modules with **audio_kernels**/**audio_kernels_bridge**.

---

## Table of Contents


- [Theoretical Background and Mathematical Formulations](#theoretical-background-and-mathematical-formulations)  
- [Overview](#overview)  
- [API Reference](#api-reference)  
  - [`stft_forward`](#stft_forward)  
  - [`generate_pink_noise_fft`](#generate_pink_noise_fft)  
  - [`generate_pink_noise_pooling`](#generate_pink_noise_pooling)  
  - [`generate_noise`](#generate_noise)  
  - [`generate_burst`](#generate_burst)  
  - [`generate_tone`](#generate_tone)  
  - [`simulate_rp`](#simulate_rp)  
  - [`butterworth_sos`](#butterworth_sos)  
  - [`apply_butterworth`](#apply_butterworth)  
  - [`framing_batch`](#framing_batch)  
  - [`spec_augment_batch`](#spec_augment_batch)  
- [Examples](#examples)  
- [CPU/GPU Performance Comparison (Mac M4 Processor)](#cpu-gpu-performance-comparison-mac-m4-processor)  
- [References](#references)  

---

## Theoretical Background and Mathematical Formulations

### Precision and Dynamic Range: 32-bit vs. 64-bit Floating Point

IEEE-754 defines:

float32: 24-bit significand (~7 decimal digits), ~144 dB quantization range, ~1500 dB exponent range [10].

float64: 53-bit significand (~16 digits), ~318 dB quantization range [10].

Human hearing spans ~120 dB; ML audio models tolerate <3 % accuracy loss even at lower bit-depths [11]. Thus float32 provides ample precision for cavitation signal synthesis and training, halving memory and compute vs. float64 without sacrificing fidelity.

---

### Noise Generation

#### Gaussian White Noise

Independent samples drawn from  
```math
x[n] \sim \mathcal{N}(0,\,\sigma^2),
````

where σ is the noise standard deviation.

#### Pink (1/f) Noise

Pink noise exhibits a power spectral density (PSD)

```math
S(f)\propto \frac{1}{f}\,,
```

i.e. each octave carries equal energy. Two principal synthesis methods are used:

---

### Octave-Pooling Synthesis

1. Generate \$O\$ white-noise octaves \$w\_i\[n]\in\mathcal N(0,1)\$, \$i=0,\dots,O-1\$.
2. Downsample octave \$i\$ by factor \$2^i\$ via average-pooling:

```math
   d_i[n] = \mathrm{avg\_pool}\bigl(w_i[n],\,2^i\bigr)\,,
   ```

   yielding length \$N\_i=N/2^i\$.
3. Upsample by repeat-interleave back to \$N\$:

```math
u_i[n] = \mathrm{repeat\_interleave}(d_i,\;2^i)\,,
```

   then truncate or zero-pad to length \$N\$.
4. Weight each octave by \$2^{-i/2}\$ and sum:

```math
   y_{\mathrm{pool}}[n]
   = \sum_{i=0}^{O-1} 2^{-i/2}\,u_i[n]\,.
   ```

This method is purely tensor-based and executes entirely on device with cost \$O(O,N)\$, but yields PSD that approximates \$1/f\$ only in octave bands.

---

### Spectral FFT-Based Shaping (SOTA)

1. Generate white noise \$w\[n]\in\mathcal N(0,1)\$.
2. Compute real FFT:

```math
   W[k] = \sum_{n=0}^{N-1} w[n]\,e^{-j2\pi kn/N},\quad k=0,\dots,\tfrac{N}{2}.
   ```
3. Build a frequency-domain filter

```math
   H[k] = \frac{1}{\sqrt{f_k+\epsilon}},
   \quad f_k = \frac{k}{N}\,f_s,
   ```

   where \$f\_s\$ is the sampling rate and \$\epsilon\$ avoids division by zero.
4. Multiply and invert:

```math
   \widetilde W[k] = W[k]\,H[k],\qquad
   y_{\mathrm{fft}}[n] = \mathrm{IRFFT}\{\widetilde W[k]\}.
   ```
5. Normalize RMS to unity:

```math
   y_{\mathrm{fft}}[n] \;\leftarrow\; \frac{y_{\mathrm{fft}}[n]}{\sqrt{\tfrac{1}{N}\sum_n y_{\mathrm{fft}}[n]^2}}.
   ```

This spectral-shaping approach is \$O(N\log N)\$, yields a precise \$1/f\$ PSD across all bins, and is the state-of-the-art for batch ML pipelines on GPU/MPS.

---

### Short-Time Fourier Transform (STFT)

Given \$x\[n]\$ and window \$w\[m]\$,

```math
\mathrm{STFT}\{x\}(t,k)
= \sum_{m=0}^{L-1} x[tH+m]\,w[m]\,e^{-j2\pi k m/N},
```

with window length \$L\$, hop \$H\$, FFT size \$N\$, producing a complex spectrogram of shape \$\bigl(N/2+1\bigr)\times T\$ \[1,2].

---

### Butterworth IIR Filter (Cascade-Biquad)

Analog \$N\$-th order Butterworth:

```math
|H(j\Omega)|^2 = \frac{1}{1+\Omega^{2N}}.
```

Bilinear transform \$s=\tfrac{2}{T}\frac{1-z^{-1}}{1+z^{-1}}\$ yields

```math
H(z)=\prod_{i=1}^{N/2}\frac{b_{0,i}+b_{1,i}z^{-1}+b_{2,i}z^{-2}}
                          {1+a_{1,i}z^{-1}+a_{2,i}z^{-2}},
```

with SOS coefficients per section \[2,5,9].

---

### Rayleigh–Plesset Bubble Dynamics

Bubble radius \$R(t)\$ obeys

```math
\rho\Bigl(R\ddot R+\tfrac{3}{2}\dot R^2\Bigr)
+4\mu\frac{\dot R}{R}+2\frac{\sigma}{R^2}
=\frac{P_\infty-P_v}{R},
```

integrated by 4th-order Runge–Kutta \[6].

---

### Rotor-Tone Synthesis

Frequency-modulated harmonic tone:

```math
x(t)=\sum_{k=1}^K A_k\sin\bigl(2\pi k f_0 t + \beta \sin(2\pi f_0 t)\bigr),
```

with amplitudes \$A\_k\$, base \$f\_0\$, modulation \$\beta\$.

---

### Framing and Windowing

Frame \$x\[n]\$ into length \$L\$, hop \$H\$:

```math
X_t[m]=x[tH+m]\,w[m],\quad m=0,\dots,L-1,
```

yielding $\[B,T\_f,L]\$ with \$T\_f=1+\lfloor(T-L)/H\rfloor\$.

---

### SpecAugment Masking

Zero out random frequency/time bands:

```math
F\sim U(0,F_M),\quad T\sim U(0,T_M),
```

applied in-place via `index_put_` \[7].

---

## Overview

This module provides signal-processing kernels optimized for real-time cavitation detection:

* **Noise Generation**: White & pink (FFT-based SOTA or octave pooling)
* **STFT**: Hann-windowed spectrogram
* **Burst Simulation**: Vectorized impulsive events + metadata
* **Rotor-Tone**: Harmonic FM synthesis
* **Rayleigh–Plesset**: RK4 bubble proxy
* **Butterworth IIR**: Cascade biquad via sequential `conv1d`
* **Framing**: Sliding-window segmentation
* **SpecAugment**: On-device masking

All operate in float32 for maximum throughput on GPU/MPS.

---

## API Reference

### stft\_forward

```cpp
torch::Tensor stft_forward(
    const Tensor& input,
    int32_t       n_fft,
    int32_t       hop_length,
    int32_t       win_length,
    bool          center,
    const std::string& pad_mode,
    bool          normalized,
    bool          onesided,
    Device        device
);
```

**Objective:** Compute Short-Time Fourier Transform (STFT) of the input signals.

**Returns:** Complex tensor `[batch_size, n_fft/2+1, n_frames]` and dtype `torch::kComplexFloat`, representing the STFT spectrogram.

---

### generate\_pink\_noise\_fft

```cpp
torch::Tensor generate_pink_noise_fft(
    int32_t batch_size,
    int32_t num_samples,
    int32_t sample_rate,
    Device  device
);
```

**Objective:** Generate pink (1/f) noise using FFT-based spectral shaping.

**Returns:** Float32 tensor `[batch_size, num_samples]` with pink noise normalized to unit RMS.

---

### generate\_pink\_noise\_pooling

```cpp
torch::Tensor generate_pink_noise_pooling(
    int32_t batch_size,
    int32_t num_samples,
    int32_t octaves,
    Device  device
);
```

**Objective:** Generate pink noise via octave pooling method.

**Returns:** Float32 tensor `[batch_size, num_samples]` containing pink noise.

---

### generate\_noise

```cpp
torch::Tensor generate_noise(
    int32_t batch_size,
    int32_t num_samples,
    int32_t sample_rate,
    float   snr_db,
    bool    add_pink,
    Device  device
);
```

**Objective:** Generate Gaussian white noise, optionally adding pink component to achieve desired SNR.

**Returns:** Float32 tensor `[batch_size, num_samples]` of noise with specified SNR (if `add_pink=true`, includes pink noise component).

---

### generate\_burst

```cpp
std::tuple<Tensor,Tensor> generate_burst(
    int32_t batch_size,
    int32_t num_samples,
    const Tensor& states,
    float count_mul,
    float width_min,
    float width_max,
    const std::vector<float>& amp_range,
    const std::string& envelope,
    int32_t max_state,
    Device device
);
```

**Objective:** Simulate cavitation burst events with vectorized operations.
**Returns:**

* `bursts`: Float32 tensor `[batch_size, num_samples]` of burst signals.

* `metadata`: Int32 tensor `[batch_size, max_bursts, 3]` containing indices/durations of bursts per example.

---

### generate\_tone

```cpp
torch::Tensor generate_tone(
    int32_t batch_size,
    int32_t num_samples,
    float   rotor_freq,
    int32_t num_harmonics,
    const std::vector<float>& tone_amps,
    float   mod_index,
    int32_t sample_rate,
    Device  device
);
```

**Objective:** Synthesize a multi-harmonic, frequency-modulated rotor tone.

**Returns:** Float32 tensor `[batch_size, num_samples]` of generated tone signal (normalized ±1).

---

### simulate\_rp

```cpp
torch::Tensor simulate_rp(
    const Tensor& states,
    int32_t num_samples,
    int32_t sample_rate,
    float   rho,
    float   sigma,
    float   mu,
    float   P_inf,
    float   P_v,
    float   R0,
    float   dR0,
    float   scale,
    Device device
);
```

**Objective:** Integrate Rayleigh–Plesset ODE via RK4 to simulate bubble dynamics.

**Returns:** Float32 tensor `[batch_size, num_samples]` of bubble radius (or related acceleration) per timestep.

---

### butterworth\_sos

```cpp
torch::Tensor butterworth_sos(
    int32_t sample_rate,
    const std::string& filter_type,
    float   cutoff_low,
    float   cutoff_high,
    int32_t order,
    Device  device
);
```

**Objective:** Compute second-order-section (SOS) coefficients for Butterworth IIR filter design.

**Returns:** Float32 tensor `[n_sections, 6]` of SOS coefficients `{b0,b1,b2,a0,a1,a2}`.

---

### apply\_butterworth

```cpp
torch::Tensor apply_butterworth(
    const Tensor& signals,
    int32_t sample_rate,
    const std::string& filter_type,
    float   cutoff_low,
    float   cutoff_high,
    int32_t order,
    Device  device
);
```

**Objective:** Apply cascade-biquad Butterworth IIR filtering to input signals.

**Returns:** Float32 tensor `[batch_size, num_samples]` of filtered signals.

---

### framing\_batch

```cpp
torch::Tensor framing_batch(
    const Tensor& signals,
    int32_t frame_length,
    int32_t hop_length
);
```

**Objective:** Segment input signals into overlapping frames.

**Returns:**

* If `signals` is `[batch_size, time]`, returns `[batch_size, n_frames, frame_length]`.
* If `signals` is `[batch_size, channels, time]`, returns `[batch_size, channels, n_frames, frame_length]`.

---

### spec\_augment\_batch

```cpp
torch::Tensor spec_augment_batch(
    Tensor  specs,
    int32_t freq_mask_param,
    int32_t time_mask_param,
    int32_t num_freq_masks,
    int32_t num_time_masks,
    bool    inplace
);
```

**Objective:** Apply SpecAugment masking (frequency and time) to a batch of spectrograms.

**Returns:** Float32 tensor `[batch_size, n_mels, time]` with masks applied. If `inplace=true`, masks `specs` in place; otherwise returns a masked clone.

---

## Examples

```cpp
#include "audio_kernels.h"
#include "audio_kernels_bridge.h"

auto noise = audio::generate_noise(
    /*batch_size=*/8,
    /*num_samples=*/48000,
    /*sample_rate=*/48000,
    /*snr_db=*/30.0f,
    /*add_pink=*/true,
    /*device=*/torch::kMPS
);
auto spec = audio::stft_forward(
    noise, 1024, 256, 1024,
    /*center=*/true, "reflect",
    /*normalized=*/true, /*onesided=*/true,
    torch::kMPS
);
```

---

## CPU/GPU Performance Comparison (Mac M4 Processor)

### **Small-scale tests**

| Test                                      | Size (elements)                               | Memory Footprint (MiB) | CPU time (s) | MPS time (s) |
| ----------------------------------------- | ---------------------------------------------- | ---------------------- | ------------ | ------------ |
| test\_generate\_noise                     | 512 × 1 000 000 = 512 000 000                   | 1953.13                | 3.511834     | 0.221977     |
| test\_generate\_noise\_diff\_sample\_rate  | 256 × 512 000 = 131 072 000                    | 500.00                 | 4.487107     | 0.681162     |
| test\_spec\_augment\_batch                 | 64 × 128 × 512 = 4 194 304                     | 16.00                  | 0.001181     | 0.124467     |
| test\_stft\_forward                        | 8 × 16 384 = 131 072                            | 0.50                   | 0.000463     | 0.029659     |
| test\_generate\_pink\_noise\_fft           | 16 × 16 384 = 262 144                           | 1.00                   | 0.003208     | 0.035720     |
| test\_generate\_pink\_noise\_pooling       | 16 × 16 384 = 262 144                           | 1.00                   | 0.014538     | 0.006774     |
| test\_generate\_burst                      | 16 × 16 384 = 262 144                           | 1.00                   | 0.001296     | 0.035956     |
| test\_generate\_tone                       | 16 × 16 384 = 262 144                           | 1.00                   | 0.005534     | 0.000968     |
| test\_simulate\_rp                         | 16 × 8 192 = 131 072                            | 0.50                   | 0.392570     | 1.182594     |

### **Large-scale tests**

| Test                                      | Size (elements)                               | Memory Footprint (MiB) | CPU time (s) | MPS time (s) |
| ----------------------------------------- | ---------------------------------------------- | ---------------------- | ------------ | ------------ |
| test\_generate\_noise                     | 512 × 500 000 = 256 000 000                     | 976.56                 | 1.789196     | 0.071655     |
| test\_generate\_noise\_diff\_sample\_rate  | 256 × 200 000 = 51 200 000                      | 195.31                 | 1.714811     | 0.281120     |
| test\_spec\_augment\_batch\_equal          | 128 × 256 × 1 024 = 33 554 432                  | 128.00                 | 0.008994     | 0.405727     |
| test\_stft\_forward                        | 32 × 131 072 = 4 194 304                        | 16.00                  | 0.010533     | 0.001465     |
| test\_generate\_pink\_noise\_fft           | 64 × 1 048 576 = 67 108 864                     | 256.00                 | 0.738784     | 0.007923     |
| test\_generate\_pink\_noise\_pooling       | 64 × 262 144 = 16 777 216                       | 64.00                  | 0.966833     | 0.004687     |
| test\_generate\_burst                      | 64 × 1 000 000 = 64 000 000                     | 244.14                 | 0.300093     | 0.151337     |
| test\_generate\_tone                       | 64 × 262 144 = 16 777 216                       | 64.00                  | 0.772285     | 0.004031     |
| test\_simulate\_rp                         | 32 × 8 192 = 262 144                            | 1.00                   | 0.777904     | 1.230409     |
| test\_butterworth\_sos                     | (order = 16; negligible coeffs, ~48 elements)   | 0.00018                | 0.000025     | 0.000266     |
| test\_apply\_butterworth                   | 32 × 16 384 = 524 288                           | 2.00                   | 4.132841     | 2.849894     |
| test\_framing\_batch                       | 512 × 131 072 = 67 108 864                       | 256.00                 | 0.193148     | 0.182961     |

---

## References

[1] Proakis, J. G., & Manolakis, D. G. (2006). *Digital Signal Processing*. Pearson.  
[2] Oppenheim, A. V., & Schafer, R. W. (2010). *Discrete-Time Signal Processing* (2nd ed.). Prentice Hall.  
[3] Rabiner, L. R., & Gold, B. (1975). *Theory and Application of Digital Signal Processing*. Prentice-Hall.  
[4] Voss, R. F., & Clarke, J. (1978). “1/f noise in music and speech.” *Nature*, 258(5533), 317–318.  
[5] Butterworth, S. (1930). “On the theory of filter amplifiers.” *Experimental Wireless and the Wireless Engineer*.  
[6] Brennen, C. E. (2013). *Cavitation and Bubble Dynamics*. Oxford University Press.  
[7] Park, D. S., Chan, W., Zhang, Y., Chiu, C., Zoph, B., Cubuk, E. D., & Le, Q. V. (2019). “SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition.” *Interspeech*.  
[8] Lee, K., & Jo, E. (2022). “Multi-task cavitation classification in hydraulic turbines.” arXiv:2203.01118v2.  
[9] SciPy Signal Documentation: “Butterworth filter design.” *SciPy.org*.  
[10] IEEE Standards Association. (2008). “IEEE Standard for Floating-Point Arithmetic.” IEEE Std 754-2008.  
[11] Cubuk, E. D., Zoph, B., Shlens, J., & Le, Q. V. (2018). “AutoAugment: Learning Augmentation Policies from Data.” *CVPR*.

