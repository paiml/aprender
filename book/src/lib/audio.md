<!-- PCU: lib-audio | contract: contracts/apr-page-lib-audio-v1.yaml -->

# Module: `aprender::audio`

Public module of the `aprender-core` crate.

## Source

[`crates/aprender-core/src/audio/mod.rs`](https://github.com/paiml/aprender/blob/main/crates/aprender-core/src/audio/mod.rs) or directory.

## Example

```rust
use aprender::audio::{MelConfig, MelFilterbank, validate_audio};
// See `cargo doc -p aprender-core --open` for full API reference.
```

## Module summary

`aprender::audio` is the speech / audio I/O and feature-extraction layer.
It provides mel-spectrogram filterbanks (used by Whisper-style models),
resampling, audio-validation helpers that detect clipping / NaN / Inf,
stereo-to-mono conversion, capture / playback abstractions, and streaming
support. Audio codec encode/decode is currently a stub (see the comment in
`audio/codec.rs`).

## Key types

| Type | Description |
|------|-------------|
| `MelConfig` | Builder for mel-filterbank parameters. Presets: `whisper()`, `tts()`, `custom(...)`. |
| `MelFilterbank` | The actual mel-frequency triangular filterbank used to convert STFT → mel-spectrogram. |
| `ClippingReport` | Result of `detect_clipping`; has `clipping_percentage`. |
| `DecodedAudio`, `AudioError` | Generic decoded audio container + error type. |
| `validate_audio`, `has_nan`, `has_inf`, `stereo_to_mono`, `resample` | Audio-validation and shape utilities. |
| `audio::capture`, `audio::playback`, `audio::stream`, `audio::noise` | Sub-modules for live I/O and noise. |

## Usage patterns

### Pattern 1: Build a Whisper-style mel filterbank

```rust
use aprender::audio::{MelConfig, MelFilterbank};

let cfg = MelConfig::whisper();
let filterbank = MelFilterbank::new(&cfg);
println!("n_freqs = {}, n_mels = {}", cfg.n_freqs(), cfg.n_mels);
```

### Pattern 2: Validate a chunk of decoded audio

```rust
use aprender::audio::{validate_audio, detect_clipping, has_nan, stereo_to_mono};

let stereo = vec![0.1_f32, -0.1, 0.2, -0.2, 0.5, -0.5];
let mono = stereo_to_mono(&stereo);
assert_eq!(mono.len(), stereo.len() / 2);

assert!(!has_nan(&mono));
validate_audio(&mono).expect("clean audio");
let report = detect_clipping(&mono);
println!("clipping = {:.2}%", report.clipping_percentage());
```

## See also

- [`voice`](voice.md) — higher-level voice / speech pipelines
- [`speech`](speech.md) — speech-recognition specific helpers
- [`format`](format.md) — model formats for audio models (e.g. Whisper)
- [`models`](models.md) — audio-capable transformer implementations

## Full API

Run `cargo doc -p aprender-core --open` for the rendered rustdoc, or browse
[docs.rs/aprender](https://docs.rs/aprender) for the published version.
