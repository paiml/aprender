# Audio Processing Theory

Audio processing is fundamental to speech recognition (ASR), text-to-speech (TTS), and voice applications. This chapter covers the signal processing theory behind Aprender's audio module.

## The Audio Processing Pipeline

Modern ASR systems like Whisper process audio through a standardized pipeline:

```
┌─────────────┐    ┌──────────┐    ┌─────────────┐    ┌───────────┐
│ Raw Audio   │───▸│ Resample │───▸│ Mel         │───▸│ Neural    │
│ (44.1kHz)   │    │ (16kHz)  │    │ Spectrogram │    │ Network   │
└─────────────┘    └──────────┘    └─────────────┘    └───────────┘
```

Each stage transforms the audio into a representation more suitable for machine learning.

## Mel Scale and Human Perception

The **mel scale** is a perceptual scale of pitches that models how humans perceive frequency. It's based on the observation that humans perceive equal intervals between low frequencies (e.g., 100-200 Hz) as larger than equal intervals at high frequencies (e.g., 8000-8100 Hz).

### Hz to Mel Conversion

```
mel = 2595 * log₁₀(1 + f/700)
```

And the inverse:

```
f = 700 * (10^(mel/2595) - 1)
```

| Frequency (Hz) | Mel Scale |
|----------------|-----------|
| 0 | 0 |
| 500 | 607 |
| 1000 | 1000 |
| 2000 | 1548 |
| 4000 | 2146 |
| 8000 | 2840 |

Notice how 0-1000 Hz spans 1000 mels, but 4000-8000 Hz only spans ~700 mels.

## Mel Filterbank

A **mel filterbank** is a set of triangular filters that convert the linear frequency spectrum to mel scale:

```
Filterbank
  ▲
  │     △      △       △        △         △
  │    / \    / \     / \      / \       / \
  │   /   \  /   \   /   \    /   \     /   \
  │  /     \/     \ /     \  /     \   /     \
  └─────────────────────────────────────────────▸
    0     500    1000    2000    4000    8000  Hz
```

Each triangular filter:
1. Is centered at a mel-spaced frequency
2. Overlaps with adjacent filters (50%)
3. Sums the power spectrum within its bandwidth

### Slaney Normalization

Aprender uses **Slaney area normalization**, which ensures each filter has unit area:

```
normalization_factor = 2 / (f_high - f_low)
```

This matches librosa's `norm='slaney'` and OpenAI Whisper's filterbank, ensuring consistent outputs across implementations.

## Mel Spectrogram Computation

The mel spectrogram computation follows these steps:

### 1. Frame the Audio

Divide audio into overlapping frames using a Hann window:

```
Frame 0: samples[0:400]      ← Apply Hann window
Frame 1: samples[160:560]    ← Hop by 160 samples
Frame 2: samples[320:720]
...
```

For Whisper at 16kHz:
- Frame size (n_fft): 400 samples = 25ms
- Hop length: 160 samples = 10ms
- Overlap: 60%

### 2. Apply FFT

Transform each windowed frame to frequency domain:

```
X[k] = Σₙ x[n] · e^(-j2πkn/N)
```

This produces a complex spectrum with N/2+1 frequency bins.

### 3. Compute Power Spectrum

```
P[k] = |X[k]|² = Re(X[k])² + Im(X[k])²
```

### 4. Apply Mel Filterbank

Matrix multiply the power spectrum by the filterbank:

```
mel_energies = filterbank @ power_spectrum
```

This reduces 201 frequency bins (for n_fft=400) to 80 mel channels.

### 5. Log Compression

Apply logarithmic compression for dynamic range:

```
log_mel = log₁₀(max(mel_energy, 1e-10))
```

The floor value (1e-10) prevents log(0).

### 6. Normalize

Whisper-style normalization:

```
normalized = (log_mel.max(max - 8.0) + 4.0) / 4.0
```

## Sample Rate Conversion

### Why Resample?

Different audio sources have different sample rates:
- CD quality: 44,100 Hz
- Professional audio: 48,000 Hz
- Whisper requirement: 16,000 Hz
- Telephone: 8,000 Hz

### Resampling Algorithm

Aprender uses linear interpolation for basic resampling:

```
For each output sample i:
    src_pos = i * (from_rate / to_rate)
    src_idx = floor(src_pos)
    frac = src_pos - src_idx

    output[i] = samples[src_idx] * (1 - frac)
              + samples[src_idx + 1] * frac
```

For higher quality, windowed-sinc interpolation minimizes aliasing.

## Audio Validation

### Clipping Detection

Properly normalized audio samples should be in the range [-1.0, 1.0]. Clipping occurs when samples exceed this range:

```
Clipped Audio
  ▲
1 │──────┬─────────────────────
  │     /│\         /│\
  │    / │ \       / │ \
  │   /  │  \     /  │  \
  │  /   │   \   /   │   \
──┼─/────┼────\─/────┼────\───▸
  │/     │     V     │     \
-1│──────┴───────────┴───────
```

Clipping causes:
- Distortion in reconstructed audio
- Poor ASR accuracy
- Incorrect mel spectrogram values

### NaN and Infinity Detection

Invalid floating-point values can propagate through the pipeline:
- **NaN**: Often from 0/0 or sqrt(-1)
- **Infinity**: From division by very small numbers

Aprender validates audio before processing to catch these early.

## Stereo to Mono Conversion

Most ASR models expect mono audio. Stereo conversion averages the channels:

```
mono[i] = (left[i] + right[i]) / 2
```

For interleaved stereo audio [L₀, R₀, L₁, R₁, ...]:

```rust
let mono: Vec<f32> = stereo
    .chunks(2)
    .map(|chunk| (chunk[0] + chunk[1]) / 2.0)
    .collect();
```

## Streaming and Chunking

Real-time ASR requires processing audio in chunks as it arrives:

```
┌─────────────────────────────────────────────────────┐
│  Chunk 1 (30s)      │  Chunk 2 (30s)     │ ...     │
│                     │                     │         │
│ ◀──────Overlap(1s)──▶                     │         │
└─────────────────────────────────────────────────────┘
```

### Overlap Handling

Chunks overlap to avoid boundary artifacts:
1. Process chunk 1, get transcription
2. Keep last 1 second of chunk 1
3. Prepend to chunk 2 for context
4. Merge transcriptions, removing duplicates

### Configuration

| Parameter | Default (Batch) | Real-time |
|-----------|-----------------|-----------|
| Chunk size | 30 seconds | 5 seconds |
| Overlap | 1 second | 0.5 seconds |
| Latency | N/A | ~5 seconds |

## Platform-Specific Audio Capture

### Backend Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   AudioCapture API                       │
├─────────────────────────────────────────────────────────┤
│  Linux    │  macOS      │  Windows   │  WASM            │
│  (ALSA)   │  (CoreAudio)│  (WASAPI)  │  (WebAudio API)  │
└─────────────────────────────────────────────────────────┘
```

Each backend implements the `CaptureBackend` trait:

```rust
pub trait CaptureBackend {
    fn open(device: Option<&str>, config: &CaptureConfig) -> Result<Self, AudioError>;
    fn read(&mut self, buffer: &mut [f32]) -> Result<usize, AudioError>;
    fn close(&mut self) -> Result<(), AudioError>;
}
```

### ALSA (Linux)

ALSA provides low-latency audio on Linux:
- Requires `libasound2-dev` package
- Enable with `audio-alsa` feature
- Captures in S16_LE format, converts to f32

## Configuration Presets

### Whisper (ASR)

```rust
MelConfig {
    n_mels: 80,          // 80 mel channels
    n_fft: 400,          // 25ms window at 16kHz
    hop_length: 160,     // 10ms hop
    sample_rate: 16000,  // 16kHz required
    fmin: 0.0,
    fmax: 8000.0,        // Nyquist frequency
}
```

### TTS (VITS-style)

```rust
MelConfig {
    n_mels: 80,
    n_fft: 1024,         // 46ms window at 22kHz
    hop_length: 256,     // 11.6ms hop
    sample_rate: 22050,  // CD-quality
    fmin: 0.0,
    fmax: 11025.0,
}
```

## Mathematical Foundations

### Hann Window

The Hann window reduces spectral leakage:

```
w[n] = 0.5 * (1 - cos(2πn / N))
```

It smoothly tapers to zero at the edges, preventing discontinuities.

### Short-Time Fourier Transform (STFT)

The STFT captures both time and frequency information:

```
X[m, k] = Σₙ x[n + m·H] · w[n] · e^(-j2πkn/N)
```

Where:
- m = frame index
- k = frequency bin
- H = hop length
- w[n] = window function

## References

- Radford, A. et al. (2023). "Robust Speech Recognition via Large-Scale Weak Supervision" (Whisper paper)
- Stevens, S., Volkmann, J., & Newman, E. (1937). "A Scale for the Measurement of the Psychological Magnitude Pitch"
- Slaney, M. (1998). "Auditory Toolbox" Technical Report #1998-010
