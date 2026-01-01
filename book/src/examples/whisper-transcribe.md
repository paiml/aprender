# Whisper Transcribe - Audio Transcription Example

This example demonstrates audio transcription using OpenAI Whisper models.

**Run command:**
```bash
cargo run --example whisper_transcribe --release --features inference
```

**Topics covered:**
- Whisper model architecture (encoder-decoder)
- Audio preprocessing (mel spectrogram)
- Beam search decoding
- Timestamp extraction
- Multi-language transcription

**Supported models:**
- whisper-tiny (39M parameters)
- whisper-base (74M parameters)
- whisper-small (244M parameters)
- whisper-medium (769M parameters)

**See also:**
- [Audio Mel Spectrogram](./audio-mel-spectrogram.md)
- [Examples Reference](./examples-reference.md)
