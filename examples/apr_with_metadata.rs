#![allow(clippy::disallowed_methods)]
//! APR Format with JSON Metadata Example
//!
//! Demonstrates how to embed arbitrary metadata (vocab, config, tokenizer settings)
//! alongside tensors in a single WASM-deployable file.
//!
//! Also shows how to embed binary data like mel filterbanks using named tensors.
//!
//! Run with: `cargo run --example apr_with_metadata`

use aprender::serialization::apr::{AprReader, AprWriter};
use serde_json::json;

fn main() -> Result<(), String> {
    println!("=== APR Format with JSON Metadata ===\n");

    // Create a writer
    let mut writer = AprWriter::new();

    // Add model configuration metadata
    writer.set_metadata("model_type", json!("whisper-tiny"));
    writer.set_metadata("n_vocab", json!(51865));
    writer.set_metadata("n_audio_ctx", json!(1500));
    writer.set_metadata("n_audio_state", json!(384));
    writer.set_metadata("n_layers", json!(4));

    // Add vocabulary as metadata (for BPE tokenization)
    let vocab_sample = json!({
        "tokens": ["<|endoftext|>", "<|startoftranscript|>", "the", "a", "is"],
        "merges": [["t", "h"], ["th", "e"], ["a", "n"]],
        "special_tokens": {
            "eot": 50256,
            "sot": 50257,
            "transcribe": 50358
        }
    });
    writer.set_metadata("tokenizer", vocab_sample);

    // Add model tensors
    println!("Adding tensors...");
    writer.add_tensor_f32(
        "encoder.conv1.weight",
        vec![384, 80, 3],
        &vec![0.01; 384 * 80 * 3],
    );
    writer.add_tensor_f32("encoder.conv1.bias", vec![384], &vec![0.0; 384]);
    writer.add_tensor_f32(
        "decoder.embed_tokens.weight",
        vec![51865, 384],
        &vec![0.001; 51865 * 384],
    );

    // Add mel filterbank as a named tensor (critical for audio models!)
    // This stores the exact filterbank used during training to avoid
    // the "rererer" hallucination bug from filterbank mismatches.
    println!("Adding mel filterbank...");
    let (n_mels, n_freqs) = (80, 201);
    let filterbank = create_slaney_filterbank(n_mels, n_freqs);
    writer.add_tensor_f32("audio.mel_filterbank", vec![n_mels, n_freqs], &filterbank);

    // Store audio config in metadata
    writer.set_metadata(
        "audio",
        json!({
            "sample_rate": 16000,
            "n_fft": 400,
            "hop_length": 160,
            "n_mels": n_mels
        }),
    );

    // Write to bytes (could also write to file)
    let bytes = writer.to_bytes()?;
    println!(
        "Total file size: {} bytes ({:.2} MB)",
        bytes.len(),
        bytes.len() as f64 / 1_000_000.0
    );

    // Read it back
    println!("\nReading APR file...");
    let reader = AprReader::from_bytes(bytes)?;

    // Access metadata
    println!("\n--- Metadata ---");
    if let Some(model_type) = reader.get_metadata("model_type") {
        println!("Model type: {}", model_type);
    }
    if let Some(n_vocab) = reader.get_metadata("n_vocab") {
        println!("Vocab size: {}", n_vocab);
    }
    if let Some(tokenizer) = reader.get_metadata("tokenizer") {
        println!(
            "Tokenizer config: {}",
            serde_json::to_string_pretty(tokenizer).unwrap_or_default()
        );
    }

    // Access tensors
    println!("\n--- Tensors ---");
    for tensor in &reader.tensors {
        println!(
            "  {} - shape: {:?}, dtype: {}, size: {} bytes",
            tensor.name, tensor.shape, tensor.dtype, tensor.size
        );
    }

    // Read specific tensor data
    let conv_weight = reader.read_tensor_f32("encoder.conv1.weight")?;
    println!(
        "\nFirst 5 values of encoder.conv1.weight: {:?}",
        &conv_weight[..5]
    );

    // Read mel filterbank
    println!("\n--- Mel Filterbank ---");
    let filterbank = reader.read_tensor_f32("audio.mel_filterbank")?;
    let audio_config = reader
        .get_metadata("audio")
        .expect("audio metadata should be present");
    let n_mels = audio_config["n_mels"].as_u64().unwrap_or(80) as usize;
    let n_freqs = filterbank.len() / n_mels;

    println!("  Filterbank shape: {}x{}", n_mels, n_freqs);
    println!("  Total values: {}", filterbank.len());

    // Verify slaney normalization (row sums should be ~0.025)
    let row_sum: f32 = filterbank[0..n_freqs].iter().sum();
    println!("  Row 0 sum: {:.6} (slaney: ~0.025)", row_sum);

    if row_sum < 0.1 {
        println!("  Status: Slaney-normalized filterbank detected");
    } else {
        println!("  Status: Peak-normalized filterbank (may cause issues)");
    }

    println!("\n=== Done ===");
    Ok(())
}

/// Create a slaney-normalized mel filterbank (simplified version)
fn create_slaney_filterbank(n_mels: usize, n_freqs: usize) -> Vec<f32> {
    let mut filters = vec![0.0f32; n_mels * n_freqs];
    let sample_rate = 16000.0f32;
    let n_fft = 400usize;

    // Mel scale boundaries
    let f_min = 0.0f32;
    let f_max = sample_rate / 2.0;
    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);

    // Create mel points
    let mel_points: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_min + (mel_max - mel_min) * (i as f32) / ((n_mels + 1) as f32))
        .collect();

    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
    let bin_points: Vec<usize> = hz_points
        .iter()
        .map(|&f| ((n_fft as f32 + 1.0) * f / sample_rate).floor() as usize)
        .collect();

    // Create triangular filters with slaney normalization
    for m in 0..n_mels {
        let f_m_minus = bin_points[m];
        let f_m = bin_points[m + 1];
        let f_m_plus = bin_points[m + 2];

        // Slaney normalization: scale by 2 / (f_high - f_low)
        let bandwidth = hz_points[m + 2] - hz_points[m];
        let norm = if bandwidth > 0.0 {
            2.0 / bandwidth
        } else {
            1.0
        };

        // Rising slope
        for k in f_m_minus..f_m {
            if k < n_freqs && f_m > f_m_minus {
                let slope = (k - f_m_minus) as f32 / (f_m - f_m_minus) as f32;
                filters[m * n_freqs + k] = slope * norm;
            }
        }

        // Falling slope
        for k in f_m..f_m_plus {
            if k < n_freqs && f_m_plus > f_m {
                let slope = (f_m_plus - k) as f32 / (f_m_plus - f_m) as f32;
                filters[m * n_freqs + k] = slope * norm;
            }
        }
    }

    filters
}

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}
