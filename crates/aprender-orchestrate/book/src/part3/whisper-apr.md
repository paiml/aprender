# Whisper.apr: Pure Rust Speech Recognition

**whisper.apr** is a pure Rust implementation of OpenAI's Whisper automatic speech recognition model, designed for the Sovereign AI Stack with WASM-first deployment and APR v2 model format.

## Overview

whisper.apr delivers:
- **Pure Rust**: No Python, no C++ dependencies
- **WASM-First**: Browser deployment with full functionality
- **APR v2 Format**: LZ4/ZSTD compressed models
- **Quantization**: Int4/Int8 for reduced memory footprint
- **Streaming**: Real-time transcription support
- **Multilingual**: 99+ languages

```
┌─────────────────────────────────────────────────────────────┐
│                    whisper.apr                              │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ APR v2 Model│  │  Streaming  │  │   Quantization      │  │
│  │ LZ4/ZSTD    │  │ Transcriber │  │   Int4/Int8         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  trueno (SIMD)  │  aprender (ML)  │  realizar (inference)  │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```toml
[dependencies]
whisper-apr = "0.1"

# With GPU acceleration
whisper-apr = { version = "0.1", features = ["gpu"] }

# WASM-only (smaller bundle)
whisper-apr = { version = "0.1", default-features = false, features = ["wasm"] }
```

## Quick Start

```rust
use whisper_apr::{WhisperModel, Transcriber, TranscribeOptions};

// Load model (APR v2 format with compression)
let model = WhisperModel::load_apr("whisper-small-int8.apr")?;
let transcriber = Transcriber::new(model);

// Transcribe audio file
let result = transcriber.transcribe_file(
    "audio.wav",
    TranscribeOptions::default(),
)?;

println!("Text: {}", result.text);
println!("Language: {}", result.language);

// With timestamps
for segment in result.segments {
    println!("[{:.2}s - {:.2}s] {}",
        segment.start, segment.end, segment.text);
}
```

## Model Sizes

| Model | FP32 | Int8 | Int4 | Languages |
|-------|------|------|------|-----------|
| Tiny | 150 MB | 40 MB | 22 MB | 99+ |
| Base | 290 MB | 75 MB | 40 MB | 99+ |
| Small | 970 MB | 250 MB | 130 MB | 99+ |
| Medium | 3.0 GB | 780 MB | 400 MB | 99+ |
| Large | 6.2 GB | 1.6 GB | 820 MB | 99+ |

## Streaming Transcription

Real-time transcription from audio stream:

```rust
use whisper_apr::{StreamingTranscriber, AudioChunk};

let mut streamer = StreamingTranscriber::new(model);

// Process audio chunks as they arrive
while let Some(chunk) = audio_source.next_chunk().await {
    if let Some(partial) = streamer.process_chunk(&chunk)? {
        print!("\r{}", partial.text);  // Live update
    }
}

// Finalize and get complete transcription
let final_result = streamer.finalize()?;
```

## WASM Deployment

Browser-compatible transcription:

```rust
use whisper_apr::wasm::{WasmWhisper, init_wasm};

#[wasm_bindgen]
pub async fn transcribe_audio(audio_data: &[u8]) -> String {
    init_wasm().await;

    let whisper = WasmWhisper::load_from_bytes(MODEL_BYTES).await?;
    let result = whisper.transcribe(audio_data)?;
    result.text
}
```

Bundle sizes (gzipped):

| Model | WASM Runtime | Total |
|-------|--------------|-------|
| Tiny Int4 | 200 KB | 22 MB |
| Base Int4 | 200 KB | 40 MB |
| Small Int4 | 200 KB | 130 MB |

## Language Detection

```rust
use whisper_apr::LanguageDetector;

let detector = LanguageDetector::new(&model);
let detection = detector.detect(&audio)?;

println!("Detected: {} ({:.1}% confidence)",
    detection.language, detection.confidence * 100.0);

// Top 5 candidates
for (lang, prob) in detection.top_languages(5) {
    println!("  {}: {:.1}%", lang, prob * 100.0);
}
```

## Stack Integration

whisper.apr integrates with the Sovereign AI Stack:

| Dependency | Version | Purpose |
|------------|---------|---------|
| trueno | 0.10+ | SIMD tensor operations |
| aprender | 0.20+ | ML primitives, APR v2 format |
| realizar | 0.4+ | Inference runtime (optional) |

## Running the Example

```bash
cargo run --example whisper_apr_demo
```

## References

- [whisper-apr on crates.io](https://crates.io/crates/whisper-apr)
- [OpenAI Whisper Paper](https://arxiv.org/abs/2212.04356)
- [APR v2 Format Specification](./aprender.md#apr-v2-format)

---

**Navigate:** [Table of Contents](../SUMMARY.md) | [Previous: Realizar](./realizar.md) | [Next: trueno-zram](./trueno-zram.md)
