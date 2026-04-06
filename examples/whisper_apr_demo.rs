//! Whisper.apr Speech Recognition Demo
//!
//! Demonstrates the whisper-apr crate for pure Rust automatic speech recognition.
//!
//! # Features Demonstrated
//!
//! - **APR v2 Format**: Compressed model loading with LZ4/ZSTD
//! - **Streaming Transcription**: Real-time audio processing
//! - **Multilingual Support**: 99+ languages
//! - **WASM-First**: Browser-compatible design
//!
//! # Running
//!
//! ```bash
//! # This example shows the API - actual whisper-apr is a separate crate
//! cargo run --example whisper_apr_demo
//! ```
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    whisper.apr                              │
//! ├─────────────────────────────────────────────────────────────┤
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
//! │  │ APR v2 Model│  │  Streaming  │  │   Quantization      │  │
//! │  │ LZ4/ZSTD    │  │ Transcriber │  │   Int4/Int8         │  │
//! │  └─────────────┘  └─────────────┘  └─────────────────────┘  │
//! ├─────────────────────────────────────────────────────────────┤
//! │  trueno (SIMD)  │  aprender (ML)  │  realizar (inference)  │
//! └─────────────────────────────────────────────────────────────┘
//! ```

fn main() {
    println!("=== whisper.apr Speech Recognition Demo ===\n");

    // =========================================================================
    // Section 1: Model Loading
    // =========================================================================
    println!("1. Model Loading (APR v2 Format)");
    println!("   ─────────────────────────────────────────");
    println!("   whisper.apr uses the APR v2 format for efficient model storage:\n");

    println!("   ```rust");
    println!("   use whisper_apr::{{WhisperModel, ModelSize, Compression}};");
    println!();
    println!("   // Load a quantized model with LZ4 compression");
    println!("   let model = WhisperModel::load_apr(");
    println!("       \"whisper-small-int8.apr\",");
    println!("       Compression::Lz4,");
    println!("   )?;");
    println!();
    println!("   // Or load from Hugging Face");
    println!("   let model = WhisperModel::from_hf(");
    println!("       \"paiml/whisper-small-apr\",");
    println!("       ModelSize::Small,");
    println!("   ).await?;");
    println!("   ```\n");

    println!("   Model sizes and memory footprint:");
    println!("   ┌─────────────┬──────────┬───────────┬───────────┐");
    println!("   │ Model       │ FP32     │ Int8      │ Int4      │");
    println!("   ├─────────────┼──────────┼───────────┼───────────┤");
    println!("   │ Tiny        │ 150 MB   │ 40 MB     │ 22 MB     │");
    println!("   │ Base        │ 290 MB   │ 75 MB     │ 40 MB     │");
    println!("   │ Small       │ 970 MB   │ 250 MB    │ 130 MB    │");
    println!("   │ Medium      │ 3.0 GB   │ 780 MB    │ 400 MB    │");
    println!("   │ Large       │ 6.2 GB   │ 1.6 GB    │ 820 MB    │");
    println!("   └─────────────┴──────────┴───────────┴───────────┘\n");

    // =========================================================================
    // Section 2: Transcription API
    // =========================================================================
    println!("2. Transcription API");
    println!("   ─────────────────────────────────────────");
    println!("   Simple API for transcribing audio files:\n");

    println!("   ```rust");
    println!("   use whisper_apr::{{Transcriber, TranscribeOptions}};");
    println!();
    println!("   let transcriber = Transcriber::new(model);");
    println!();
    println!("   // Transcribe a file");
    println!("   let result = transcriber.transcribe_file(");
    println!("       \"audio.wav\",");
    println!("       TranscribeOptions::default(),");
    println!("   )?;");
    println!();
    println!("   println!(\"Text: {{}}\", result.text);");
    println!("   println!(\"Language: {{}}\", result.language);");
    println!("   println!(\"Confidence: {{:.2}}\", result.confidence);");
    println!();
    println!("   // With timestamps");
    println!("   for segment in result.segments {{");
    println!("       println!(\"[{{:.2}}s - {{:.2}}s] {{}}\",");
    println!("           segment.start, segment.end, segment.text);");
    println!("   }}");
    println!("   ```\n");

    // =========================================================================
    // Section 3: Streaming Transcription
    // =========================================================================
    println!("3. Streaming Transcription");
    println!("   ─────────────────────────────────────────");
    println!("   Real-time transcription from audio stream:\n");

    println!("   ```rust");
    println!("   use whisper_apr::{{StreamingTranscriber, AudioChunk}};");
    println!();
    println!("   let mut streamer = StreamingTranscriber::new(model);");
    println!();
    println!("   // Process audio chunks as they arrive");
    println!("   while let Some(chunk) = audio_source.next_chunk().await {{");
    println!("       if let Some(partial) = streamer.process_chunk(&chunk)? {{");
    println!("           print!(\"\\r{{}}\", partial.text);  // Live update");
    println!("       }}");
    println!("   }}");
    println!();
    println!("   // Finalize and get complete transcription");
    println!("   let final_result = streamer.finalize()?;");
    println!("   ```\n");

    // =========================================================================
    // Section 4: WASM Deployment
    // =========================================================================
    println!("4. WASM Deployment (Browser)");
    println!("   ─────────────────────────────────────────");
    println!("   whisper.apr is designed for WASM-first deployment:\n");

    println!("   ```rust");
    println!("   // In wasm32 target");
    println!("   use whisper_apr::wasm::{{WasmWhisper, init_wasm}};");
    println!();
    println!("   #[wasm_bindgen]");
    println!("   pub async fn transcribe_audio(audio_data: &[u8]) -> String {{");
    println!("       init_wasm().await;");
    println!();
    println!("       let whisper = WasmWhisper::load_from_bytes(MODEL_BYTES).await?;");
    println!("       let result = whisper.transcribe(audio_data)?;");
    println!("       result.text");
    println!("   }}");
    println!("   ```\n");

    println!("   Bundle sizes (gzipped):");
    println!("   ┌─────────────┬──────────┬───────────┐");
    println!("   │ Model       │ WASM     │ Total     │");
    println!("   ├─────────────┼──────────┼───────────┤");
    println!("   │ Tiny Int4   │ 200 KB   │ 22 MB     │");
    println!("   │ Base Int4   │ 200 KB   │ 40 MB     │");
    println!("   │ Small Int4  │ 200 KB   │ 130 MB    │");
    println!("   └─────────────┴──────────┴───────────┘\n");

    // =========================================================================
    // Section 5: Language Detection
    // =========================================================================
    println!("5. Automatic Language Detection");
    println!("   ─────────────────────────────────────────");
    println!("   Detect language before transcription:\n");

    println!("   ```rust");
    println!("   use whisper_apr::LanguageDetector;");
    println!();
    println!("   let detector = LanguageDetector::new(&model);");
    println!();
    println!("   // Detect from first 30 seconds");
    println!("   let detection = detector.detect(&audio)?;");
    println!();
    println!("   println!(\"Detected: {{}} ({{:.1}}% confidence)\",");
    println!("       detection.language, detection.confidence * 100.0);");
    println!();
    println!("   // Top 5 candidates");
    println!("   for (lang, prob) in detection.top_languages(5) {{");
    println!("       println!(\"  {{}}: {{:.1}}%\", lang, prob * 100.0);");
    println!("   }}");
    println!("   ```\n");

    // =========================================================================
    // Section 6: Integration with Stack
    // =========================================================================
    println!("6. Sovereign AI Stack Integration");
    println!("   ─────────────────────────────────────────");
    println!("   whisper.apr integrates with the full stack:\n");

    println!("   Dependencies:");
    println!("   • trueno (0.10+) - SIMD tensor operations");
    println!("   • aprender (0.20+) - ML primitives");
    println!("   • realizar - Inference runtime (optional)\n");

    println!("   ```toml");
    println!("   [dependencies]");
    println!("   whisper-apr = \"0.1\"");
    println!();
    println!("   # With GPU acceleration");
    println!("   whisper-apr = {{ version = \"0.1\", features = [\"gpu\"] }}");
    println!();
    println!("   # WASM-only (smaller bundle)");
    println!(
        "   whisper-apr = {{ version = \"0.1\", default-features = false, features = [\"wasm\"] }}"
    );
    println!("   ```\n");

    println!("=== Demo Complete ===");
    println!("\nFor actual usage, add whisper-apr to your Cargo.toml:");
    println!("  whisper-apr = \"0.1\"");
}
