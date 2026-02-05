//! Pipeline step execution for showcase demo

use crate::error::{CliError, Result};
use colored::Colorize;
use std::process::Command;
use std::time::Instant;

use super::types::*;

/// Step A: Import from HuggingFace
pub(super) fn run_import(config: &ShowcaseConfig) -> Result<bool> {
    println!("{}", "═══ Step A: HuggingFace Import ═══".cyan().bold());
    println!();

    // Use tier-specific filename
    let gguf_filename = config.tier.gguf_filename();
    let gguf_path = config.model_dir.join(gguf_filename);

    // Check if model for this tier already exists
    if gguf_path.exists() {
        println!(
            "{} Model already exists at {}",
            "✓".green(),
            gguf_path.display()
        );
        return Ok(true);
    }

    // Ensure model directory exists
    std::fs::create_dir_all(&config.model_dir)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to create model dir: {e}")))?;

    println!(
        "Tier: {} ({})",
        format!("{:?}", config.tier).cyan(),
        config.tier.params()
    );
    println!("Model: {}", config.tier.model_path().cyan());
    println!("Size: ~{:.1} GB", config.tier.size_gb());
    println!("Output: {}", gguf_path.display());
    println!();

    // Use huggingface-cli to download the specific file
    let output = Command::new("huggingface-cli")
        .args([
            "download",
            config.tier.model_path(),
            gguf_filename,
            "--local-dir",
            config.model_dir.to_str().unwrap_or("."),
        ])
        .output();

    match output {
        Ok(out) if out.status.success() => {
            println!("{} Download complete", "✓".green());
            Ok(true)
        }
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            println!("{} Download failed: {}", "✗".red(), stderr);
            Ok(false)
        }
        Err(e) => {
            println!("{} huggingface-cli not found: {}", "✗".red(), e);
            println!("Install with: pip install huggingface_hub");
            Ok(false)
        }
    }
}

/// Step B: GGUF Inference with realizar
#[cfg(feature = "inference")]
pub(super) fn run_gguf_inference(config: &ShowcaseConfig) -> Result<bool> {
    #[cfg(feature = "cuda")]
    use realizar::gguf::OwnedQuantizedModelCuda;
    use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, QuantizedGenerateConfig};

    println!();
    println!("{}", "═══ Step B: GGUF Inference ═══".cyan().bold());
    println!();

    // Use tier-specific model path
    let gguf_path = config.model_dir.join(config.tier.gguf_filename());
    if !gguf_path.exists() {
        return Err(CliError::ValidationFailed(format!(
            "Model not found: {}. Run 'apr showcase --step import' first.",
            gguf_path.display()
        )));
    }
    println!("Model: {} ({})", gguf_path.display(), config.tier.params());
    println!(
        "  Mode: {}",
        if config.gpu {
            "GPU (CUDA)".green()
        } else {
            "CPU".yellow()
        }
    );

    let start = Instant::now();
    println!("Loading model with realizar...");

    // Load model using memory-mapped GGUF
    let mapped = MappedGGUFModel::from_path(&gguf_path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to load GGUF: {e}")))?;

    let model = OwnedQuantizedModel::from_mapped(&mapped)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to create model: {e}")))?;

    let load_time = start.elapsed();
    println!(
        "{} Model loaded in {:.2}s",
        "✓".green(),
        load_time.as_secs_f32()
    );

    // Model info
    println!("  Layers: {}", model.config.num_layers);
    println!("  Hidden dim: {}", model.config.hidden_dim);
    println!("  Heads: {}", model.config.num_heads);
    println!("  KV heads: {}", model.config.num_kv_heads);

    // DEBUG: Print actual embedding values from model BEFORE potential move to GPU
    let emb_token_0 = model.embed(&[0]);
    println!(
        "  Embed(token 0) first 8: {:?}",
        &emb_token_0[..8.min(emb_token_0.len())]
    );
    println!(
        "  Embed(token 0) sum: {:.6}",
        emb_token_0.iter().sum::<f32>()
    );
    let emb_token_9707 = model.embed(&[9707]);
    println!(
        "  Embed(token 9707 = 'Hello') first 8: {:?}",
        &emb_token_9707[..8.min(emb_token_9707.len())]
    );
    println!(
        "  Embed(token 9707) sum: {:.6}",
        emb_token_9707.iter().sum::<f32>()
    );

    // Run inference test
    println!();
    println!("Running inference test...");

    // Use ChatML format for Qwen2.5-Coder (required for proper generation)
    let user_message = "Hello, I am a coding assistant. Write a function";
    let test_prompt = format!(
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        user_message
    );
    let prompt_tokens: Vec<u32> = mapped.model.encode(&test_prompt).unwrap_or_else(|| {
        // Fallback to simple tokens if vocab not available
        println!(
            "  {} No vocabulary found, using fallback tokens",
            "⚠".yellow()
        );
        vec![1, 2, 3, 4, 5]
    });
    // DEBUG: Show what tokens the prompt was encoded to
    let prompt_decoded = mapped.model.decode(&prompt_tokens);
    println!("  Prompt token IDs: {:?}", prompt_tokens);
    println!("  Prompt decoded: \"{}\"", prompt_decoded);
    println!(
        "  User message: \"{}\" ({} tokens total with ChatML)",
        user_message,
        prompt_tokens.len()
    );

    let gen_config = QuantizedGenerateConfig {
        max_tokens: 16,
        temperature: 0.0, // Greedy for reproducibility
        top_k: 1,
        ..Default::default()
    };

    let infer_start = Instant::now();

    // PAR-054: Use GPU path when --gpu flag is set
    #[cfg(feature = "cuda")]
    let output = if config.gpu {
        // OwnedQuantizedModelCuda provides GPU-resident inference with CUDA graph capture
        let mut cuda_model = OwnedQuantizedModelCuda::new(model, 0)
            .map_err(|e| CliError::ValidationFailed(format!("CUDA init failed: {e}")))?;
        cuda_model
            .generate_gpu_resident(&prompt_tokens, &gen_config)
            .map_err(|e| CliError::ValidationFailed(format!("GPU inference failed: {e}")))?
    } else {
        model
            .generate_with_cache(&prompt_tokens, &gen_config)
            .map_err(|e| CliError::ValidationFailed(format!("Inference failed: {e}")))?
    };

    #[cfg(not(feature = "cuda"))]
    let output = {
        if config.gpu {
            println!(
                "  {} GPU requested but CUDA feature not enabled, falling back to CPU",
                "⚠".yellow()
            );
        }
        model
            .generate_with_cache(&prompt_tokens, &gen_config)
            .map_err(|e| CliError::ValidationFailed(format!("Inference failed: {e}")))?
    };

    let infer_time = infer_start.elapsed();
    let tokens_generated = output.len().saturating_sub(prompt_tokens.len());
    let tps = tokens_generated as f64 / infer_time.as_secs_f64();

    println!(
        "{} Generated {} tokens in {:.2}s ({:.1} tok/s)",
        "✓".green(),
        tokens_generated,
        infer_time.as_secs_f32(),
        tps
    );

    // DEBUG: Decode and print generated output to verify correctness
    let generated_tokens = &output[prompt_tokens.len()..];
    let generated_text = mapped.model.decode(generated_tokens);
    println!("  Output tokens: {:?}", generated_tokens);
    println!("  Generated text: \"{}\"", generated_text);

    // Check rope config from metadata
    println!(
        "  rope_theta from metadata: {:?}",
        mapped.model.rope_freq_base()
    );
    println!("  rope_type from metadata: {:?}", mapped.model.rope_type());

    // DEBUG: Verify tensor_data_start
    println!("  tensor_data_start: {}", mapped.model.tensor_data_start);
    println!("  Expected: 5948576");

    // DEBUG: Print first embedding values
    if let Some(tensor) = mapped
        .model
        .tensors
        .iter()
        .find(|t| t.name == "token_embd.weight")
    {
        let data = mapped.data();
        let offset = mapped.model.tensor_data_start + tensor.offset as usize;
        println!("  token_embd offset in file: {}", offset);
        println!(
            "  First 10 bytes at offset: {:02x?}",
            &data[offset..offset + 10]
        );
    }

    Ok(true)
}

#[cfg(not(feature = "inference"))]
pub(super) fn run_gguf_inference(config: &ShowcaseConfig) -> Result<bool> {
    println!();
    println!("{}", "═══ Step B: GGUF Inference ═══".cyan().bold());
    println!();
    println!(
        "{} Inference feature not enabled. Rebuild with --features inference",
        "⚠".yellow()
    );

    // Fallback: validate file exists using tier-specific path
    let gguf_path = config.model_dir.join(config.tier.gguf_filename());
    let file_size = std::fs::metadata(&gguf_path)
        .map_err(|e| CliError::ValidationFailed(format!("Cannot read model: {e}")))?
        .len();

    if file_size < 1_000_000 {
        return Err(CliError::ValidationFailed(
            "Model file too small".to_string(),
        ));
    }

    println!(
        "{} GGUF file validated: {} ({:.2} GB)",
        "✓".green(),
        gguf_path.display(),
        file_size as f64 / 1e9
    );

    Ok(true)
}

/// Step C: Convert GGUF to APR format
///
/// Uses the canonical `apr_import` path from aprender - the ONE way to convert GGUF to APR.
/// This preserves quantization (Q4_K/Q6_K) and embeds tokenizer+config.
#[cfg(feature = "inference")]
pub(super) fn run_convert(config: &ShowcaseConfig) -> Result<bool> {
    use aprender::format::{apr_import, ImportOptions};

    println!();
    println!("{}", "═══ Step C: APR Conversion ═══".cyan().bold());
    println!();

    let gguf_path = config.model_dir.join(config.tier.gguf_filename());
    let apr_basename = config.tier.gguf_filename().replace(".gguf", ".apr");
    let apr_path = config.model_dir.join(&apr_basename);

    if apr_path.exists() {
        println!(
            "{} APR model already exists at {}",
            "✓".green(),
            apr_path.display()
        );
        return Ok(true);
    }

    println!("Input: {}", gguf_path.display());
    println!("Output: {}", apr_path.display());
    println!(
        "Format: {} (preserves Q4_K/Q6_K quantization)",
        "APR v2".cyan()
    );
    println!();

    let start = Instant::now();
    println!("Converting GGUF to APR format (preserving quantization)...");

    // Use canonical apr_import path - the ONE way to convert GGUF to APR
    let gguf_size = std::fs::metadata(&gguf_path)
        .map(|m| m.len())
        .unwrap_or(0);

    let _report = apr_import(
        gguf_path.to_string_lossy().as_ref(),
        &apr_path,
        ImportOptions::default(),
    )
    .map_err(|e| CliError::ValidationFailed(format!("Conversion failed: {e}")))?;

    let elapsed = start.elapsed();
    let apr_size = std::fs::metadata(&apr_path)
        .map(|m| m.len())
        .unwrap_or(0);

    println!(
        "{} Conversion complete in {:.2}s",
        "✓".green(),
        elapsed.as_secs_f32()
    );
    println!(
        "  GGUF: {:.2} GB → APR: {:.2} GB (quantization preserved)",
        gguf_size as f64 / 1e9,
        apr_size as f64 / 1e9
    );

    Ok(true)
}

#[cfg(not(feature = "inference"))]
pub(super) fn run_convert(config: &ShowcaseConfig) -> Result<bool> {
    println!();
    println!("{}", "═══ Step C: APR Conversion ═══".cyan().bold());
    println!();
    println!(
        "{} Inference feature not enabled. Rebuild with --features inference",
        "⚠".yellow()
    );

    let gguf_path = config.model_dir.join(config.tier.gguf_filename());
    let apr_basename = config.tier.gguf_filename().replace(".gguf", ".apr");
    let apr_path = config.model_dir.join(&apr_basename);

    // Create placeholder for testing
    let gguf_size = std::fs::metadata(&gguf_path).map(|m| m.len()).unwrap_or(0);
    let placeholder = format!(
        "APR-PLACEHOLDER-V2\nsource: {}\nsize: {}\nstatus: STUB\n",
        gguf_path.display(),
        gguf_size
    );
    std::fs::write(&apr_path, placeholder)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to write placeholder: {e}")))?;

    println!(
        "{} Created placeholder APR file (real conversion requires --features inference)",
        "⚠".yellow()
    );

    Ok(true)
}

/// Step D: APR Inference
#[cfg(feature = "inference")]
pub(super) fn run_apr_inference(config: &ShowcaseConfig) -> Result<bool> {
    // BUG-FIX: Use GgufToAprConverter::from_apr_bytes which handles JSON tensor index
    // The AprTransformer::from_apr_bytes uses binary format which is incompatible
    use realizar::convert::GgufToAprConverter;

    println!();
    println!("{}", "═══ Step D: APR Inference ═══".cyan().bold());
    println!();

    // BUG-FIX: Use the converted file matching the tier, not hardcoded 32b
    let apr_basename = config.tier.gguf_filename().replace(".gguf", ".apr");
    let apr_path = config.model_dir.join(&apr_basename);

    if !apr_path.exists() {
        println!("{} APR model not found. Run conversion first.", "✗".red());
        return Ok(false);
    }

    println!("Model: {}", apr_path.display());
    if config.zram {
        // NOTE: ZRAM is not yet implemented in realizar - flag is captured but not active
        println!(
            "ZRAM: {} (not yet implemented in realizar)",
            "disabled".yellow()
        );
    }

    let start = Instant::now();
    println!("Loading APR model...");

    let model_bytes = std::fs::read(&apr_path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to read APR: {e}")))?;

    // Use GgufToAprConverter which writes/reads JSON tensor index format
    let transformer = GgufToAprConverter::from_apr_bytes(&model_bytes)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to load APR: {e}")))?;

    let load_time = start.elapsed();
    println!(
        "{} APR model loaded in {:.2}s",
        "✓".green(),
        load_time.as_secs_f32()
    );

    // Run inference
    println!("Running inference test...");

    // NOTE: APR format doesn't include tokenizer vocabulary
    // For proper tokenization, we'd need to load the source GGUF or store vocab in APR
    // Using Qwen2 BOS token (151643) + common word tokens for reasonable test
    let prompt_tokens: Vec<u32> = vec![151643, 9707, 11, 358, 1079, 264, 11761, 18328];
    println!(
        "  {} APR doesn't include vocabulary - using pre-tokenized Qwen2 tokens",
        "ℹ".cyan()
    );

    let infer_start = Instant::now();
    let output = transformer
        .generate(&prompt_tokens, 16)
        .map_err(|e| CliError::ValidationFailed(format!("APR inference failed: {e}")))?;

    let infer_time = infer_start.elapsed();
    let tokens_generated = output.len().saturating_sub(prompt_tokens.len());
    let tps = tokens_generated as f64 / infer_time.as_secs_f64();

    println!(
        "{} Generated {} tokens in {:.2}s ({:.1} tok/s)",
        "✓".green(),
        tokens_generated,
        infer_time.as_secs_f32(),
        tps
    );

    Ok(true)
}

#[cfg(not(feature = "inference"))]
pub(super) fn run_apr_inference(config: &ShowcaseConfig) -> Result<bool> {
    println!();
    println!("{}", "═══ Step D: APR Inference ═══".cyan().bold());
    println!();

    // BUG-FIX: Use the converted file matching the tier, not hardcoded 32b
    let apr_basename = config.tier.gguf_filename().replace(".gguf", ".apr");
    let apr_path = config.model_dir.join(&apr_basename);
    if !apr_path.exists() {
        println!("{} APR model not found. Run conversion first.", "✗".red());
        return Ok(false);
    }

    println!(
        "{} Inference feature not enabled. Rebuild with --features inference",
        "⚠".yellow()
    );
    println!("Model: {}", apr_path.display());
    if config.zram {
        // NOTE: ZRAM is not yet implemented in realizar - flag is captured but not active
        println!(
            "ZRAM: {} (not yet implemented in realizar)",
            "disabled".yellow()
        );
    }

    Ok(true)
}
