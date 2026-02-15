
/// Normalize tensor name for cross-format comparison (GH-202 fix)
///
/// Maps both GGUF and APR/HuggingFace naming conventions to a common canonical form.
/// This enables proper tensor matching when comparing models across formats.
///
/// GGUF convention: `blk.N.attn_q.weight`
/// APR/HF convention: `model.layers.N.self_attn.q_proj.weight`
/// Canonical form: `N.q_proj.weight`
fn normalize_tensor_name(name: &str) -> String {
    // Step 1: Remove format-specific prefixes
    let name = name
        .trim_start_matches("model.")
        .trim_start_matches("blk.")
        .trim_start_matches("layers.");

    // Step 2: Remove APR/HF intermediate prefixes (self_attn., mlp.)
    // These don't exist in GGUF naming
    let name = name.replace(".self_attn.", ".").replace(".mlp.", ".");

    // Step 3: Normalize GGUF tensor suffixes to HF convention
    // GGUF: attn_q → HF: q_proj
    let name = name
        .replace("attn_q", "q_proj")
        .replace("attn_k", "k_proj")
        .replace("attn_v", "v_proj")
        .replace("attn_output", "o_proj")
        .replace("ffn_gate", "gate_proj")
        .replace("ffn_up", "up_proj")
        .replace("ffn_down", "down_proj")
        .replace("attn_norm", "input_layernorm")
        .replace("ffn_norm", "post_attention_layernorm")
        .replace("token_embd", "embed_tokens")
        .replace("output_norm", "norm");

    // Step 4: Handle lm_head vs output naming
    // GGUF: output.weight → APR: lm_head.weight
    if name == "output.weight" {
        "lm_head.weight".to_string()
    } else {
        name
    }
}

/// Check if two shapes are transposed versions of each other
fn is_transposed_dims(shape_a: &[usize], shape_b: &[usize]) -> bool {
    if shape_a.len() != 2 || shape_b.len() != 2 {
        return false;
    }
    // Check if dims are swapped AND shapes are different (not square identical)
    // For [896, 896] vs [896, 896], this should return false (identical, not transposed)
    let is_swapped = shape_a[0] == shape_b[1] && shape_a[1] == shape_b[0];
    let is_different = shape_a != shape_b;
    is_swapped && is_different
}

/// Inference result with logit data
struct InferenceResult {
    tokens: Vec<u32>,
    logits: Vec<f32>,
    top5: Vec<Vec<u32>>,
    output_text: String,
}

/// Run a model and capture output
/// Parse PMAT-113-F trace lines for selected tokens, logits, and top-5 predictions.
/// Parse a "Selected token:" line into (token_id, logit).
fn parse_selected_token(line: &str) -> Option<(u32, Option<f32>)> {
    let token_part = line.split("Selected token:").nth(1)?.trim();
    let paren_pos = token_part.find(" (")?;
    let token_id = token_part[..paren_pos].parse::<u32>().ok()?;
    let logit = token_part.find("logit:").and_then(|start| {
        let logit_str = &token_part[start + 6..];
        let end = logit_str.find(')')?;
        logit_str[..end].trim().parse::<f32>().ok()
    });
    Some((token_id, logit))
}

/// Parse a "Top 5 tokens:" line into a vector of token IDs.
fn parse_top5_line(line: &str) -> Option<Vec<u32>> {
    let top5_part = line.split("Top 5 tokens:").nth(1)?;
    let ids: Vec<u32> = top5_part
        .split("),")
        .filter_map(|pair| {
            let inner = &pair[pair.find('(')? + 1..];
            inner[..inner.find(',')?].trim().parse().ok()
        })
        .collect();
    if ids.is_empty() {
        None
    } else {
        Some(ids)
    }
}

fn parse_trace_lines(combined: &str) -> (Vec<u32>, Vec<f32>, Vec<Vec<u32>>) {
    let mut tokens = Vec::new();
    let mut logits = Vec::new();
    let mut top5 = Vec::new();

    for line in combined.lines() {
        if let Some((token_id, logit)) = parse_selected_token(line) {
            tokens.push(token_id);
            if let Some(l) = logit {
                logits.push(l);
            }
        }
        if let Some(ids) = parse_top5_line(line) {
            top5.push(ids);
        }
    }
    (tokens, logits, top5)
}

/// Extract clean output text from realizar stdout, stripping noise and debug lines.
fn extract_clean_output(stdout_text: &str) -> String {
    strip_ansi(stdout_text)
        .chars()
        .filter(|c| !matches!(c, '⠋' | '⠙' | '⠹' | '⠸' | '⠼' | '⠴' | '⠦' | '⠧' | '⠇' | '⠏'))
        .collect::<String>()
        .lines()
        .filter(|l| {
            let t = l.trim();
            !t.is_empty()
                && !t.starts_with('[')
                && !t.starts_with("Loading")
                && !t.starts_with("Model loaded")
                && !t.starts_with("Prompt tokens")
                && !t.starts_with("Temperature:")
                && !t.starts_with("Generated (")
                && !t.contains("tok/s")
                && !t.contains("ERROR")
                && !t.contains("using greedy")
        })
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_string()
}

fn run_model_with_logits(
    model_path: &Path,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
) -> Result<InferenceResult> {
    use std::process::{Command, Stdio};

    let realizar_path = std::env::var("REALIZAR_PATH").unwrap_or_else(|_| "realizar".to_string());
    let output = Command::new(&realizar_path)
        .arg("run")
        .arg(model_path)
        .arg(prompt)
        .arg("--max-tokens")
        .arg(max_tokens.to_string())
        .arg("--temperature")
        .arg(temperature.to_string())
        .arg("--format")
        .arg("text")
        .env("NO_COLOR", "1")
        .env("TERM", "dumb")
        .env("REALIZE_DEBUG", "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| CliError::ValidationFailed(format!("Failed to run realizar: {e}")))?;

    let (stdout_text, stderr_text) = (
        String::from_utf8_lossy(&output.stdout).to_string(),
        String::from_utf8_lossy(&output.stderr).to_string(),
    );

    if std::env::var("ROSETTA_DEBUG").is_ok() {
        eprintln!("[ROSETTA] Model: {}", model_path.display());
        eprintln!("[ROSETTA] Exit code: {:?}", output.status.code());
        eprintln!(
            "[ROSETTA] STDOUT ({} bytes): {:?}",
            stdout_text.len(),
            &stdout_text[..stdout_text.len().min(200)]
        );
        eprintln!(
            "[ROSETTA] STDERR ({} bytes): {:?}",
            stderr_text.len(),
            &stderr_text[..stderr_text.len().min(200)]
        );
    }

    let combined = format!("{}\n{}", stdout_text, stderr_text);
    let (tokens, logits, top5) = parse_trace_lines(&combined);
    let output_text = extract_clean_output(&stdout_text);

    Ok(InferenceResult {
        tokens,
        logits,
        top5,
        output_text,
    })
}

/// Strip ANSI escape codes from text
fn strip_ansi(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '\x1b' {
            // Skip escape sequence
            if chars.peek() == Some(&'[') {
                chars.next(); // consume '['
                              // Skip until we hit a letter
                while let Some(&next) = chars.peek() {
                    chars.next();
                    if next.is_ascii_alphabetic() {
                        break;
                    }
                }
            }
        } else {
            result.push(c);
        }
    }

    result
}

/// Truncate path for display
fn truncate_path(path: String, max_len: usize) -> String {
    if path.len() <= max_len {
        path
    } else {
        format!("...{}", &path[path.len() - max_len + 3..])
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn print_inspection_report(report: &InspectionReport, hexdump: bool) {
    output::header("Rosetta Stone Inspection");

    let mut pairs: Vec<(&str, String)> = vec![
        ("Format", report.format.to_string()),
        ("File Size", output::format_size(report.file_size as u64)),
        ("Parameters", output::count_fmt(report.total_params)),
    ];
    if let Some(ref arch) = report.architecture {
        pairs.push(("Architecture", arch.clone()));
    }
    if let Some(ref quant) = report.quantization {
        pairs.push(("Quantization", quant.clone()));
    }
    println!("{}", output::kv_table(&pairs));

    // Metadata
    if !report.metadata.is_empty() {
        output::subheader(&format!("Metadata ({} keys)", report.metadata.len()));
        let meta_pairs: Vec<(&str, String)> = report
            .metadata
            .iter()
            .map(|(k, v)| {
                let display_v = if v.len() > 60 {
                    format!("{}...", &v[..60])
                } else {
                    v.clone()
                };
                (k.as_str(), display_v)
            })
            .collect();
        println!("{}", output::kv_table(&meta_pairs));
    }

    // Tensors
    output::subheader(&format!("Tensors ({} total)", report.tensors.len()));
    let mut rows: Vec<Vec<String>> = Vec::new();
    for (i, t) in report.tensors.iter().enumerate() {
        if i < 10 || i >= report.tensors.len().saturating_sub(2) {
            rows.push(vec![
                t.name.clone(),
                format!("{}", output::dtype_color(&t.dtype)),
                format!("{:?}", t.shape),
                output::format_size(t.size_bytes as u64),
            ]);
        } else if i == 10 {
            rows.push(vec![
                format!("... {} more ...", report.tensors.len().saturating_sub(12)),
                String::new(),
                String::new(),
                String::new(),
            ]);
        }
    }
    println!(
        "{}",
        output::table(&["Name", "DType", "Shape", "Size"], &rows)
    );

    if hexdump {
        output::subheader("Hexdump (first 64 bytes)");
        println!("  (Use 'apr hex <file>' for full hex dump)");
    }
}

fn print_inspection_summary(report: &InspectionReport) {
    let mut pairs: Vec<(&str, String)> = vec![
        ("Format", report.format.to_string()),
        ("File Size", output::format_size(report.file_size as u64)),
        ("Tensors", output::count_fmt(report.tensors.len())),
        ("Parameters", output::count_fmt(report.total_params)),
    ];
    if let Some(ref arch) = report.architecture {
        pairs.push(("Architecture", arch.clone()));
    }
    if let Some(ref quant) = report.quantization {
        pairs.push(("Quantization", quant.clone()));
    }
    println!("{}", output::kv_table(&pairs));
}

fn print_inspection_json(report: &InspectionReport) {
    // Simple JSON output
    println!("{{");
    println!("  \"format\": \"{}\",", report.format);
    println!("  \"file_size\": {},", report.file_size);
    println!("  \"total_params\": {},", report.total_params);
    println!("  \"tensor_count\": {},", report.tensors.len());
    if let Some(ref arch) = report.architecture {
        println!("  \"architecture\": \"{arch}\",");
    }
    if let Some(ref quant) = report.quantization {
        println!("  \"quantization\": \"{quant}\",");
    }
    println!("  \"metadata_keys\": {}", report.metadata.len());
    println!("}}");
}

fn print_conversion_json(
    path: &ConversionPath,
    source: &InspectionReport,
    target: &InspectionReport,
) {
    println!("{{");
    println!("  \"path\": \"{path}\",");
    println!("  \"source\": {{");
    println!("    \"format\": \"{}\",", source.format);
    println!("    \"tensors\": {}", source.tensors.len());
    println!("  }},");
    println!("  \"target\": {{");
    println!("    \"format\": \"{}\",", target.format);
    println!("    \"tensors\": {}", target.tensors.len());
    println!("  }}");
    println!("}}");
}

fn print_verification_json(report: &VerificationReport) {
    println!("{{");
    println!("  \"is_equivalent\": {},", report.is_equivalent);
    println!("  \"max_diff\": {},", report.max_diff);
    println!("  \"mean_diff\": {},", report.mean_diff);
    println!("  \"failed_tensors\": {}", report.failed_tensors.len());
    println!("}}");
}
