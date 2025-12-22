# Speech and Voice Processing Theory

Speech and voice processing enables machines to understand, generate, and manipulate human speech. This chapter covers ASR, TTS, VAD, diarization, and voice cloning.

## Speech Processing Pipeline

```
┌──────────┐    ┌─────┐    ┌─────────────┐    ┌──────────┐
│  Audio   │───▶│ VAD │───▶│ ASR/Speaker │───▶│  Output  │
│  Input   │    │     │    │ Recognition │    │  Text/ID │
└──────────┘    └─────┘    └─────────────┘    └──────────┘
```

## Voice Activity Detection (VAD)

Detect when speech is present in audio:

### Energy-Based VAD

Simple threshold on frame energy:

```
energy[t] = Σ(samples[t:t+frame]²)
is_speech[t] = energy[t] > threshold
```

**Pros:** Fast, no model needed
**Cons:** Sensitive to noise

### Neural VAD (Silero-style)

```
Audio → Mel Spectrogram → LSTM/Conv → [0.0, 1.0]
                                         Speech probability
```

**Pros:** Robust to noise
**Cons:** Requires model inference

### VAD Parameters

| Parameter | Typical Value | Effect |
|-----------|---------------|--------|
| Frame length | 20-30ms | Resolution |
| Threshold | 0.5 | Sensitivity |
| Min speech | 250ms | Filter noise |
| Min silence | 300ms | Merge segments |

## Automatic Speech Recognition (ASR)

Convert speech to text:

### Traditional Pipeline

```
Audio → MFCC → Acoustic Model → HMM → Language Model → Text
```

### End-to-End (Whisper-style)

```
Audio → Mel Spectrogram → Encoder → Decoder → Text
              │               │          │
              └──────────────────────────┘
                  Transformer Architecture
```

### Whisper Architecture

```
Audio (30s max)
      │
      ▼
Mel Spectrogram (80 mel, 3000 frames)
      │
      ▼
┌─────────────────────┐
│  Encoder            │ (Transformer)
│  - Conv stem        │
│  - Positional enc   │
│  - N layers         │
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  Decoder            │ (Transformer)
│  - Text tokens      │
│  - Cross-attention  │
│  - Autoregressive   │
└─────────────────────┘
      │
      ▼
Text tokens → Text
```

### Word-Level Timestamps

Cross-attention alignment:

```
For each word:
  1. Find decoder step that generated word
  2. Extract cross-attention weights
  3. Find peak attention position
  4. Map to audio timestamp
```

## Speaker Diarization

"Who spoke when?"

### Pipeline

```
Audio → VAD → Embedding → Clustering → Timeline
              │               │
              ▼               ▼
        Speaker Vectors   Speakers
```

### Speaker Embeddings

**X-Vector:**
```
Audio → Frame features → Statistics pooling → DNN → 512-dim
```

**ECAPA-TDNN:**
```
Audio → SE-Res2Net → Attentive Stats → 192-dim
```

### Clustering Methods

| Method | Requires K? | Notes |
|--------|-------------|-------|
| K-Means | Yes | Simple, fast |
| Spectral | Yes | Better for non-spherical |
| Agglomerative | No | Can auto-detect speakers |
| VBx | No | Bayesian, state-of-the-art |

## Text-to-Speech (TTS)

Convert text to speech:

### Two-Stage Pipeline

```
Text → Acoustic Model → Mel Spectrogram → Vocoder → Waveform
           │                                  │
           ▼                                  ▼
    Tacotron/FastSpeech              HiFi-GAN/WaveGlow
```

### FastSpeech 2

Non-autoregressive for fast synthesis:

```
Phonemes → Encoder → Variance Adaptor → Mel Decoder → Mel
                           │
              Duration, Pitch, Energy predictors
```

**Variance Adaptor:**
- Duration: How long each phoneme
- Pitch: F0 contour
- Energy: Loudness

### Vocoders

Convert mel spectrogram to waveform:

| Vocoder | Quality | Speed |
|---------|---------|-------|
| Griffin-Lim | Low | Fast |
| WaveNet | High | Very slow |
| HiFi-GAN | High | Fast |
| WaveGlow | High | Moderate |

## Voice Cloning

Clone a voice from samples:

### Zero-Shot Cloning (YourTTS)

```
Reference Audio → Speaker Encoder → Style Vector
                                          │
                                          ▼
Text → TTS Model ─────────────────────▶ Cloned Speech
```

Only needs 3-5 seconds of reference audio.

### Fine-Tuning Based

1. Pre-train TTS on large corpus
2. Fine-tune on target speaker (15-30 min audio)
3. Generate with fine-tuned model

**Trade-off:** Better quality, more data needed

## Voice Conversion

Change voice identity while preserving content:

### PPG-Based

```
Source Audio → ASR → PPG (Content) ─────┐
                                        │
Target Speaker → Embedding ────────────▶│───▶ Converted
                                        │
Prosody extraction ────────────────────┘
```

PPG = Phonetic Posteriorgram (content representation)

### Autoencoder-Based

```
Audio → Content Encoder → Content ─────┐
                                       │
Audio → Speaker Encoder → Speaker ────▶│───▶ Decoder → Audio'
                                       │
Audio → Prosody Encoder → Prosody ────┘
```

## Voice Isolation

Separate voice from background:

### Spectral Subtraction

```
Y(f) = Speech(f) + Noise(f)
Speech(f) ≈ Y(f) - E[Noise(f)]
```

Estimate noise from silent segments.

### Neural Source Separation

```
Mixture → U-Net/Conv-TasNet → Separated Sources
               │
          Mask estimation per source
```

## Speaker Verification

"Is this the claimed speaker?"

### Pipeline

```
Enrollment:  Audio → Embedding Model → Reference Vector
                                              │
                                              ▼
Verification: Audio → Embedding Model → Query Vector
                                              │
                                              ▼
                                       Cosine Similarity
                                              │
                                              ▼
                                      Accept/Reject
```

### Metrics

| Metric | Description |
|--------|-------------|
| EER | Equal Error Rate (FAR = FRR) |
| minDCF | Detection cost function |
| TAR@FAR | True accept at fixed false accept |

## Prosody Transfer

Transfer speaking style:

```
Source Audio → Style Encoder → Style Vector
                                     │
                    ┌────────────────┘
                    ▼
Target Audio → TTS → New Audio with Source Style
```

Style includes:
- Speaking rate
- Pitch patterns
- Emphasis
- Emotion

## Quality Metrics

| Metric | Measures | Range |
|--------|----------|-------|
| WER | ASR accuracy | 0-∞ (lower=better) |
| MOS | Subjective quality | 1-5 |
| PESQ | Perceptual quality | -0.5 to 4.5 |
| STOI | Intelligibility | 0-1 |

## References

- Radford, A., et al. (2023). "Robust Speech Recognition via Large-Scale Weak Supervision." (Whisper)
- Ren, Y., et al. (2020). "FastSpeech 2." ICLR.
- Kong, J., et al. (2020). "HiFi-GAN." NeurIPS.
- Desplanques, B., et al. (2020). "ECAPA-TDNN." Interspeech.
