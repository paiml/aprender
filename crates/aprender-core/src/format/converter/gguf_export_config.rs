
/// Export to SafeTensors with optional companion files (config.json, tokenizer.json)
fn export_safetensors_with_companions(
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    input_path: &Path,
    output_path: &Path,
    options: &ExportOptions,
    original_dtypes: &BTreeMap<String, String>,
) -> Result<()> {
    // PMAT-223: Extract user metadata from APR custom field for round-trip
    let user_metadata = extract_user_metadata(input_path);

    // PMAT-260: Log dtype preservation when BF16/F16 tensors are present
    let non_f32_count = original_dtypes
        .values()
        .filter(|d| d.as_str() != "F32")
        .count();
    if non_f32_count > 0 {
        eprintln!(
            "[PMAT-260] Preserving original dtypes for {non_f32_count} non-F32 tensors (BF16/F16)"
        );
    }

    if user_metadata.is_empty() {
        save_safetensors_typed(output_path, tensors, original_dtypes).map_err(|e| {
            AprenderError::FormatError {
                message: format!("Failed to export to SafeTensors: {e}"),
            }
        })?;
    } else {
        eprintln!(
            "[PMAT-223] Restoring {} user metadata key(s) to SafeTensors __metadata__",
            user_metadata.len()
        );
        save_safetensors_with_metadata_typed(
            output_path,
            tensors,
            &user_metadata,
            original_dtypes,
        )
        .map_err(|e| AprenderError::FormatError {
            message: format!("Failed to export to SafeTensors: {e}"),
        })?;
    }

    // GH-182: Write companion files alongside SafeTensors
    let output_dir = output_path.parent().unwrap_or(Path::new("."));

    if options.include_config {
        let config = infer_model_config(tensors);
        let config_path = output_dir.join("config.json");
        if let Err(e) = fs::write(&config_path, config) {
            eprintln!("[GH-182] Warning: Failed to write config.json: {e}");
        }
    }

    if options.include_tokenizer {
        let tokenizer_json = infer_tokenizer_json(input_path);
        if !tokenizer_json.is_empty() {
            let tokenizer_path = output_dir.join("tokenizer.json");
            if let Err(e) = fs::write(&tokenizer_path, &tokenizer_json) {
                eprintln!("[GH-182] Warning: Failed to write tokenizer.json: {e}");
            }
        }
    }

    Ok(())
}

/// Export tensors to GGUF format (GGUF-EXPORT-001 fix)
///
/// Reads APR metadata to populate GGUF KV pairs and maps tensor names
/// from HuggingFace convention to GGUF convention.
///
/// BUG-1 FIX: Now supports Q4_K quantization for GGUF inference compatibility.
/// F32 GGUF files don't work with realizar's fused matmul kernels.
///
/// BUG-EXPORT-004 FIX: Now includes tokenizer metadata for realizar inference.
/// Without BOS/EOS token IDs, the model produces empty output.
/// Resolved GGUF export configuration (APR metadata with inferred fallbacks).
struct GgufExportConfig {
    arch: String,
    hidden_size: usize,
    num_layers: usize,
    num_heads: usize,
    num_kv_heads: usize,
    vocab_size: usize,
    intermediate_size: usize,
    max_pos: usize,
    rope_theta: f32,
    rms_norm_eps: f32,
    head_dim: usize,
    model_name: String,
}

/// Normalize an APR-side architecture string into the GGUF / llama.cpp
/// convention (lowercase family name).
///
/// APR metadata uses HuggingFace transformers convention (e.g.
/// `"LlamaForCausalLM"`, `"Qwen2ForCausalLM"`). GGUF / llama.cpp
/// expects lowercase family names (`"llama"`, `"qwen2"`).
///
/// SPEC-SHIP-TWO-001 §81 P0-F. Empirical surfacing on §78's MODEL-2
/// checkpoint: `apr export --format gguf` succeeded but `llama-cli`
/// refused to load with "unknown model architecture: 'LlamaForCausalLM'".
///
/// Strategy: explicit mapping table for known HF names; lowercase fallback
/// for anything else (preserves backward compatibility with already-correct
/// inputs).
pub(crate) fn normalize_arch_for_gguf(arch: &str) -> String {
    match arch {
        // HF "*ForCausalLM" suffixed family names → GGUF lowercase
        "LlamaForCausalLM" => "llama".to_string(),
        "Qwen2ForCausalLM" => "qwen2".to_string(),
        "Qwen2MoeForCausalLM" => "qwen2moe".to_string(),
        "Qwen3ForCausalLM" => "qwen3".to_string(),
        "Qwen3MoeForCausalLM" => "qwen3moe".to_string(),
        "MistralForCausalLM" => "llama".to_string(), // Mistral uses llama arch in GGUF
        "Phi3ForCausalLM" => "phi3".to_string(),
        "GPT2LMHeadModel" => "gpt2".to_string(),
        "BertForMaskedLM" => "bert".to_string(),
        // Already in GGUF convention → pass through unchanged
        "llama" | "qwen2" | "qwen2moe" | "qwen3" | "qwen3moe" | "phi3" | "gpt2" | "bert"
        | "unknown" => arch.to_string(),
        // Unknown HF name → lowercase fallback (preserves debuggability)
        other => other.to_lowercase(),
    }
}

/// Resolve GGUF export config from APR metadata + inferred fallbacks.
fn resolve_gguf_config(
    apr_metadata: Option<&crate::format::v2::AprV2Metadata>,
    inferred: Option<&crate::format::gguf::GgufModelConfig>,
) -> GgufExportConfig {
    /// Resolve a field: APR metadata → inferred → default.
    fn resolve<T: Copy>(
        apr: Option<&crate::format::v2::AprV2Metadata>,
        inf: Option<&crate::format::gguf::GgufModelConfig>,
        apr_f: impl Fn(&crate::format::v2::AprV2Metadata) -> Option<T>,
        inf_f: impl Fn(&crate::format::gguf::GgufModelConfig) -> Option<T>,
        default: T,
    ) -> T {
        apr.and_then(&apr_f)
            .or_else(|| inf.and_then(&inf_f))
            .unwrap_or(default)
    }

    // N-02 (Meyer DbC): Use 0 for missing dimensions — no silent LLaMA-7B defaults.
    let num_heads = resolve(apr_metadata, inferred, |m| m.num_heads, |c| c.num_heads, 0);
    let hidden_size = resolve(
        apr_metadata,
        inferred,
        |m| m.hidden_size,
        |c| c.hidden_size,
        0,
    );

    // N-01 (Meyer DbC): Resolve architecture, then use it for rope_theta default.
    let arch_raw = apr_metadata
        .and_then(|m| m.architecture.clone())
        .or_else(|| inferred.and_then(|c| c.architecture.clone()))
        .unwrap_or_else(|| "unknown".to_string());
    // §81 P0-F: APR metadata uses HuggingFace transformers convention
    // (e.g. "LlamaForCausalLM"); GGUF / llama.cpp expects lowercase
    // family names (e.g. "llama"). Map at the export boundary.
    let arch = normalize_arch_for_gguf(&arch_raw);

    GgufExportConfig {
        arch,
        hidden_size,
        num_layers: resolve(
            apr_metadata,
            inferred,
            |m| m.num_layers,
            |c| c.num_layers,
            0,
        ),
        num_heads,
        num_kv_heads: resolve(
            apr_metadata,
            inferred,
            |m| m.num_kv_heads,
            |c| c.num_kv_heads,
            num_heads,
        ),
        vocab_size: resolve(
            apr_metadata,
            inferred,
            |m| m.vocab_size,
            |c| c.vocab_size,
            0,
        ),
        intermediate_size: resolve(
            apr_metadata,
            inferred,
            |m| m.intermediate_size,
            |c| c.intermediate_size,
            0,
        ),
        max_pos: apr_metadata
            .and_then(|m| m.max_position_embeddings)
            .unwrap_or(0),
        rope_theta: apr_metadata
            .and_then(|m| m.rope_theta)
            .unwrap_or_else(|| {
                let a = apr_metadata.and_then(|m| m.architecture.as_deref()).unwrap_or("unknown");
                super::export::default_rope_theta_for_architecture(a)
            }),
        rms_norm_eps: apr_metadata.and_then(|m| m.rms_norm_eps).unwrap_or(1e-6),
        // N-02 (Meyer DbC): 0 when dimensions unknown, not hardcoded 128.
        head_dim: hidden_size.checked_div(num_heads).unwrap_or(0),
        model_name: apr_metadata
            .and_then(|m| m.name.clone())
            .unwrap_or_else(|| "model".to_string()),
    }
}

/// Build GGUF architecture metadata KV pairs from resolved config.
fn build_gguf_config_metadata(
    cfg: &GgufExportConfig,
) -> Vec<(String, crate::format::gguf::GgufValue)> {
    use crate::format::gguf::GgufValue;
    let arch = &cfg.arch;
    let mut metadata = vec![
        (
            "general.architecture".to_string(),
            GgufValue::String(arch.clone()),
        ),
        (
            "general.name".to_string(),
            GgufValue::String(cfg.model_name.clone()),
        ),
        (
            "general.quantization_version".to_string(),
            GgufValue::Uint32(2),
        ),
        ("general.file_type".to_string(), GgufValue::Uint32(0)),
        (
            format!("{arch}.context_length"),
            GgufValue::Uint32(cfg.max_pos as u32),
        ),
        (
            format!("{arch}.embedding_length"),
            GgufValue::Uint32(cfg.hidden_size as u32),
        ),
        (
            format!("{arch}.block_count"),
            GgufValue::Uint32(cfg.num_layers as u32),
        ),
        (
            format!("{arch}.feed_forward_length"),
            GgufValue::Uint32(cfg.intermediate_size as u32),
        ),
        (
            format!("{arch}.attention.head_count"),
            GgufValue::Uint32(cfg.num_heads as u32),
        ),
        (
            format!("{arch}.attention.head_count_kv"),
            GgufValue::Uint32(cfg.num_kv_heads as u32),
        ),
    ];

    // GH-277: GPT-2 uses standard LayerNorm, not RMSNorm
    if arch == "gpt2" {
        metadata.push((
            format!("{arch}.attention.layer_norm_epsilon"),
            GgufValue::Float32(cfg.rms_norm_eps),
        ));
    } else {
        metadata.push((
            format!("{arch}.attention.layer_norm_rms_epsilon"),
            GgufValue::Float32(cfg.rms_norm_eps),
        ));
    }

    // GH-277: Only emit RoPE keys for architectures that use RoPE
    if uses_rope(arch) {
        metadata.push((
            format!("{arch}.rope.dimension_count"),
            GgufValue::Uint32(cfg.head_dim as u32),
        ));
        metadata.push((
            format!("{arch}.rope.freq_base"),
            GgufValue::Float32(cfg.rope_theta),
        ));
    }

    metadata.push((
        format!("{arch}.vocab_size"),
        GgufValue::Uint32(cfg.vocab_size as u32),
    ));

    metadata
}

/// Build tokenizer metadata KV pairs for GGUF export.
///
/// P0-G: `vocab_size` is the model's `<arch>.vocab_size` (e.g. 151936 for Qwen2.5
/// with TP-alignment padding). When the tokenizer's real vocabulary is smaller,
/// the emitted `tokenizer.ggml.tokens` array is padded with placeholder entries
/// (`<|pad_N|>`) so that llama.cpp's `check_tensor_dims` accepts the matching
/// `token_embd.weight` first dim. Pass 0 to disable padding (back-compat).
fn build_tokenizer_gguf_metadata(
    tokenizer: &crate::format::gguf::GgufTokenizer,
    arch: &str,
    model_name: &str,
    vocab_size: usize,
) -> Vec<(String, crate::format::gguf::GgufValue)> {
    use crate::format::gguf::GgufValue;
    let mut metadata = Vec::new();
    let model_type = tokenizer.model_type.as_deref().unwrap_or("gpt2");

    metadata.push((
        "tokenizer.ggml.model".to_string(),
        GgufValue::String(model_type.to_lowercase()),
    ));
    // GH-277: Use pre-tokenizer type mapping, preferring round-trip preserved value
    let pre_type = tokenizer
        .pre_type
        .as_deref()
        .unwrap_or_else(|| resolve_pre_tokenizer_type(arch, model_name));
    metadata.push((
        "tokenizer.ggml.pre".to_string(),
        GgufValue::String(pre_type.to_string()),
    ));

    if let Some(bos) = tokenizer.bos_token_id {
        metadata.push((
            "tokenizer.ggml.bos_token_id".to_string(),
            GgufValue::Uint32(bos),
        ));
    }
    if let Some(eos) = tokenizer.eos_token_id {
        metadata.push((
            "tokenizer.ggml.eos_token_id".to_string(),
            GgufValue::Uint32(eos),
        ));
    }
    if !tokenizer.vocabulary.is_empty() {
        // GH-279: Dedup token table for llama.cpp compatibility.
        // HuggingFace tokenizers (Qwen3, etc.) may have multiple reserved tokens
        // mapped to "<unk>" — llama.cpp requires unique token strings.
        // Fix: append "_N" suffix to duplicates (same approach as convert.py).
        let mut seen = std::collections::HashMap::with_capacity(tokenizer.vocabulary.len());
        let mut deduped: Vec<String> = tokenizer
            .vocabulary
            .iter()
            .enumerate()
            .map(|(idx, tok)| {
                let count = seen.entry(tok.clone()).or_insert(0u32);
                *count += 1;
                if *count > 1 {
                    eprintln!(
                        "[GH-279] Dedup token id={idx}: {tok:?} → {tok}_{c}",
                        c = *count - 1
                    );
                    format!("{tok}_{}", *count - 1)
                } else {
                    tok.clone()
                }
            })
            .collect();

        // P0-G: pad to `vocab_size` so `len(tokenizer.ggml.tokens)` matches
        // `token_embd.weight` first dim. Qwen2.5 has 151643 real tokens but pads
        // embed_tokens to 151936 for TP-alignment. llama.cpp's check_tensor_dims
        // refuses to load when these disagree.
        let padded_len = if vocab_size > deduped.len() {
            let pad_count = vocab_size - deduped.len();
            eprintln!(
                "[P0-G] Padding tokenizer.ggml.tokens: {} real tokens + {} placeholders = {}",
                deduped.len(), pad_count, vocab_size
            );
            for i in deduped.len()..vocab_size {
                deduped.push(format!("<|pad_{i}|>"));
            }
            vocab_size
        } else {
            deduped.len()
        };

        metadata.push((
            "tokenizer.ggml.tokens".to_string(),
            GgufValue::ArrayString(deduped),
        ));
        eprintln!(
            "[BUG-EXPORT-004] Added tokenizer metadata: model={}, tokens_len={}, bos={:?}, eos={:?}",
            model_type, padded_len, tokenizer.bos_token_id, tokenizer.eos_token_id
        );
    }
    if !tokenizer.merges.is_empty() {
        metadata.push((
            "tokenizer.ggml.merges".to_string(),
            GgufValue::ArrayString(tokenizer.merges.clone()),
        ));
    }
    metadata
}

/// Determine if a tensor needs Conv1D-to-Linear transpose.
fn needs_conv1d_transpose(gguf_name: &str, name: &str, shape: &[usize], needs_transpose: bool) -> bool {
    if !needs_transpose {
        return false;
    }
    let is_weight_2d = shape.len() == 2 && gguf_name.ends_with(".weight");
    let is_embedding = gguf_name == "token_embd.weight" || name.contains("embed_tokens");
    let is_lm_head = gguf_name == "output.weight" || name.contains("lm_head");
    is_weight_2d && !is_embedding && !is_lm_head && !gguf_name.contains("_norm") && !gguf_name.contains("position_embd")
}

/// Transpose a 2D tensor from Conv1D [rows, cols] to Linear [cols, rows].
fn transpose_2d(data: &[f32], rows: usize, cols: usize) -> (Vec<f32>, Vec<usize>) {
    let mut transposed = vec![0.0f32; data.len()];
    for r in 0..rows {
        for c in 0..cols {
            transposed[c * rows + r] = data[r * cols + c];
        }
    }
    (transposed, vec![cols, rows])
}

/// Convert shape to GGUF format: [rows, cols] -> [ne0=cols, ne1=rows].
fn to_gguf_shape(shape: &[usize]) -> Vec<u64> {
    if shape.len() == 2 {
        vec![shape[1] as u64, shape[0] as u64]
    } else {
        shape.iter().map(|&d| d as u64).collect()
    }
}

/// Quantize or encode tensor data for GGUF output.
///
/// PMAT-690 P3-C-prep defect 2 (2026-05-17): Q4_K requires the inner
/// matmul dimension (K) to be divisible by 256 (the Q4_K block size).
/// llama.cpp rejects GGUF files where any Q4_K tensor has K % 256 != 0
/// with `tensor 'X' has N elements per row, not a multiple of block size (256)`.
/// This breaks llama-cli interop on architectures with hidden_size not
/// divisible by 256 — notably Qwen2 0.5B (hidden=896) where 7+ tensors
/// per layer hit this case.
///
/// Fix: when shape[1] (the K dim after GGUF's row/col swap) is not
/// divisible by 256, fall back to F32 for that tensor. Matches the
/// convention `llama.cpp/convert_hf_to_gguf.py` uses (F16 fallback;
/// we use F32 because our intermediate is already f32 — no precision
/// loss in the fallback path).
///
/// The fallback inflates file size for affected tensors by 8× vs Q4_K.
/// For Qwen2 0.5B this means the GGUF Q4_K export is ~2.1 GB instead of
/// the ~700 MB it would be if all tensors were Q4_K-encodable. Tradeoff
/// is acceptable for the v1 stack-existence-proof ship (SPEC §88) since
/// the alternative is "broken artifact." Future enhancement: investigate
/// Q4_0 (block=32) for these tensors — would give ~1.0 GB output.
fn encode_gguf_data(
    data: &[f32],
    shape: &[usize],
    gguf_name: &str,
    name: &str,
    use_q4k: bool,
) -> (crate::format::gguf::GgmlType, Vec<u8>) {
    use crate::format::gguf::GgmlType;

    let is_embedding = gguf_name == "token_embd.weight" || name.contains("embed_tokens");
    let is_lm_head = gguf_name == "output.weight" || name.contains("lm_head");

    let q4k_compatible = use_q4k
        && shape.len() == 2
        && data.len() >= 256
        && !is_embedding
        && !is_lm_head
        // Defect 2 fix: Q4_K block_size=256 requires K % 256 == 0
        && shape[1] % 256 == 0;

    if q4k_compatible {
        // PMAT-690 defect 3 (2026-05-17): pass APR-native row-major shape
        // [rows=out, cols=in=K] to quantize_q4_k_matrix. The function pads
        // along cols when cols is not 256-divisible. Previously we swapped
        // to [in, out] which made the function pad along the OUT dim and
        // slice data with wrong stride, producing transposed bytes with the
        // wrong total length. llama-cpp expects bytes laid out as
        // `out` super-block-rows of `in/256` blocks each → exactly what
        // we get when passing shape directly (no swap).
        let q4k_bytes = super::quantize_q4_k_matrix(data, shape);
        (GgmlType::Q4K, q4k_bytes)
    } else {
        if use_q4k && shape.len() == 2 && data.len() >= 256 && !is_embedding && !is_lm_head {
            // Q4_K was requested but tensor isn't Q4_K-compatible — log
            // the fallback so operators can see which tensors inflated
            // file size and why.
            eprintln!(
                "[GGUF-EXPORT-Q4K-FALLBACK] {} (shape [{}, {}]) — \
                 K={} not divisible by 256; falling back to F32 \
                 (llama.cpp Q4_K block-size requirement, defect 2 fix)",
                gguf_name, shape[0], shape[1], shape[1]
            );
        }
        let f32_bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        (GgmlType::F32, f32_bytes)
    }
}

fn export_to_gguf(
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    output: &Path,
    input: &Path,
    quantize: Option<&QuantizationType>,
) -> Result<()> {
    use crate::format::gguf::{export_tensors_to_gguf, GgufTensor};
    use crate::format::v2::AprV2Reader;
    use std::fs::File;
    use std::io::BufWriter;

    let tokenizer = super::import::load_tokenizer_from_json(input);

    let apr_metadata = if input.extension().and_then(|e| e.to_str()) == Some("apr") {
        fs::read(input)
            .ok()
            .and_then(|d| AprV2Reader::from_bytes(&d).ok())
            .map(|r| r.metadata().clone())
    } else {
        None
    };
    let inferred = super::import::infer_model_config_from_tensors(tensors);
    let cfg = resolve_gguf_config(apr_metadata.as_ref(), inferred.as_ref());

    let mut metadata = build_gguf_config_metadata(&cfg);
    append_tokenizer_to_metadata(
        &mut metadata,
        tokenizer.as_ref(),
        apr_metadata.as_ref(),
        &cfg.arch,
        &cfg.model_name,
        cfg.vocab_size,
        input,
    );

    eprintln!(
        "[GGUF-EXPORT-001] Writing {} metadata keys (arch={}, layers={}, heads={}/{}kv, hidden={})",
        metadata.len(), cfg.arch, cfg.num_layers, cfg.num_heads, cfg.num_kv_heads, cfg.hidden_size
    );

    let mapper = build_gguf_mapper(&cfg.arch);
    let use_q4k = matches!(quantize, Some(QuantizationType::Q4K | QuantizationType::Int4));
    let needs_transpose = mapper.needs_transpose();

    let gguf_tensors: Vec<GgufTensor> = tensors
        .iter()
        .filter_map(|(name, (data, shape))| {
            let gguf_name = mapper.map_name(name)?;

            let (effective_data, effective_shape) = if needs_conv1d_transpose(&gguf_name, name, shape, needs_transpose) {
                transpose_2d(data, shape[0], shape[1])
            } else {
                (data.clone(), shape.clone())
            };

            let gguf_shape = to_gguf_shape(&effective_shape);
            let (dtype, bytes) = encode_gguf_data(&effective_data, &effective_shape, &gguf_name, name, use_q4k);

            Some(GgufTensor { name: gguf_name, shape: gguf_shape, dtype, data: bytes })
        })
        .collect();

    let fused = build_fused_tensors_f32(&mapper, tensors, use_q4k);
    let mut gguf_tensors = gguf_tensors;
    gguf_tensors.extend(fused);

    let has_lm_head = gguf_tensors.iter().any(|t| t.name == "output.weight");
    if use_q4k && !has_lm_head {
        if let Some(tied) = build_tied_output_weight(tensors) {
            gguf_tensors.push(tied);
        }
    }

    super::export::dedup_token_table(&mut metadata);

    let file = File::create(output).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to create output file: {e}"),
    })?;
    let mut writer = BufWriter::new(file);

    export_tensors_to_gguf(&mut writer, &gguf_tensors, &metadata)
}


#[cfg(test)]
mod q4k_divisibility_tests {
    //! PMAT-690 P3-C-prep defect 2 (2026-05-17): Q4_K block_size=256
    //! requires K % 256 == 0 (llama.cpp llama-cli enforces this).
    //!
    //! These tests pin the fallback behaviour added in encode_gguf_data so
    //! a future refactor cannot silently regress and produce a GGUF that
    //! llama-cli rejects with
    //! `tensor 'X' of type 12 (q4_K) has N elements per row, not a multiple
    //! of block size (256)`.
    //!
    //! Real-world trigger: Qwen2 0.5B (hidden=896, intermediate=4864).
    //! 896 % 256 = 128 — most projections must fall back to F32. The 0.5B
    //! variant is unusually small for Qwen2; the 1.5B (hidden=1536) and
    //! 7B (hidden=3584) keep K % 256 == 0 throughout, so this defect did
    //! not surface until P2-E shipping.
    use super::encode_gguf_data;
    use crate::format::gguf::GgmlType;

    #[test]
    fn q4k_falls_back_to_f32_when_inner_dim_not_divisible_by_256() {
        // Qwen2 0.5B ffn_gate.weight: [intermediate=4864, hidden=896]
        // After GGUF mapping the inner (K) dim is hidden=896, NOT divisible.
        let shape = vec![4864, 896];
        let data = vec![0.0_f32; 4864 * 896];
        let (dtype, bytes) =
            encode_gguf_data(&data, &shape, "blk.0.ffn_gate.weight", "mlp.gate_proj.weight", true);
        assert_eq!(
            dtype,
            GgmlType::F32,
            "K=896 not divisible by 256 — must fall back to F32"
        );
        assert_eq!(
            bytes.len(),
            4864 * 896 * 4,
            "F32 byte count = elements * 4"
        );
    }

    #[test]
    fn q4k_applied_when_inner_dim_divisible_by_256() {
        // ffn_down: [hidden=896, intermediate=4864]. K=4864, 4864/256=19. Quantize.
        let shape = vec![896, 4864];
        let data = vec![0.0_f32; 896 * 4864];
        let (dtype, _bytes) = encode_gguf_data(
            &data,
            &shape,
            "blk.0.ffn_down.weight",
            "mlp.down_proj.weight",
            true,
        );
        assert_eq!(
            dtype,
            GgmlType::Q4K,
            "K=4864 divisible by 256 — must Q4_K encode"
        );
    }

    #[test]
    fn q4k_applied_on_exact_256_boundary() {
        // Exact block boundary [128, 256] — K=256 divides itself once.
        let shape = vec![128, 256];
        let data = vec![0.0_f32; 128 * 256];
        let (dtype, _bytes) =
            encode_gguf_data(&data, &shape, "blk.0.attn_q.weight", "self_attn.q_proj.weight", true);
        assert_eq!(
            dtype,
            GgmlType::Q4K,
            "K=256 exactly divisible — must Q4_K encode"
        );
    }

    #[test]
    fn q4k_falls_back_for_qwen2_0_5b_attn_projections() {
        // Qwen2 0.5B attention: q_proj=[896, 896], k_proj=[128, 896]
        // GQA-7:1 ratio means k/v are tiny. All have K=896, NOT divisible.
        for (name, shape) in &[
            ("self_attn.q_proj.weight", vec![896_usize, 896]),
            ("self_attn.k_proj.weight", vec![128_usize, 896]),
            ("self_attn.v_proj.weight", vec![128_usize, 896]),
            ("self_attn.o_proj.weight", vec![896_usize, 896]),
        ] {
            let data = vec![0.0_f32; shape[0] * shape[1]];
            let (dtype, _bytes) =
                encode_gguf_data(&data, shape, "blk.0.attn_q.weight", name, true);
            assert_eq!(
                dtype,
                GgmlType::F32,
                "Qwen2 0.5B {} (K={}) must fall back to F32",
                name,
                shape[1]
            );
        }
    }

    #[test]
    fn embedding_and_lm_head_always_f32_regardless_of_divisibility() {
        // Even when K is divisible, embedding and lm_head MUST stay F32
        // (special-cased — Q4_K of the vocab table breaks llama-cli too).
        let shape = vec![1024_usize, 1024]; // K=1024 divisible by 256 — but path excluded
        let data = vec![0.0_f32; 1024 * 1024];
        let (dtype, _) = encode_gguf_data(&data, &shape, "token_embd.weight", "embed_tokens", true);
        assert_eq!(dtype, GgmlType::F32, "embedding stays F32");
        let (dtype, _) = encode_gguf_data(&data, &shape, "output.weight", "lm_head", true);
        assert_eq!(dtype, GgmlType::F32, "lm_head stays F32");
    }

    #[test]
    fn use_q4k_false_always_returns_f32() {
        // When the user didn't ask for Q4_K, no quantization regardless of shape.
        let shape = vec![896, 4864]; // divisible — would Q4_K with use_q4k=true
        let data = vec![0.0_f32; 896 * 4864];
        let (dtype, _) =
            encode_gguf_data(&data, &shape, "blk.0.ffn_down.weight", "mlp.down_proj.weight", false);
        assert_eq!(dtype, GgmlType::F32, "use_q4k=false → always F32");
    }

    #[test]
    fn q4k_byte_count_matches_llama_cpp_expectation() {
        // PMAT-690 defect 3 (2026-05-17): the bytes per Q4_K tensor must
        // equal `(rows * cols / 256) * 144` (super-blocks × bytes/block).
        // Previously we swapped shape and the quantizer padded along the
        // wrong dim, inflating bytes from 2,451,456 → 2,801,664 for
        // ffn_down [896, 4864]. The +350,208 byte excess caused
        // `gguf_init_from_file_impl: tensor 'blk.0.ffn_gate.weight' has
        // offset N, expected M` in llama-cli (offset drift in subsequent
        // tensors).
        //
        // For Qwen2 0.5B ffn_down [out=896, in=4864]:
        //   - rows = 896, cols=4864 (in = K = 256-divisible)
        //   - super_blocks_per_row = 4864 / 256 = 19
        //   - total super-blocks = 896 * 19 = 17_024
        //   - bytes = 17_024 * 144 = 2_451_456
        let shape = vec![896_usize, 4864];
        let data = vec![0.0_f32; 896 * 4864];
        let (dtype, bytes) = encode_gguf_data(
            &data,
            &shape,
            "blk.0.ffn_down.weight",
            "mlp.down_proj.weight",
            true,
        );
        assert_eq!(dtype, GgmlType::Q4K);
        assert_eq!(
            bytes.len(),
            (896 * 4864 / 256) * 144,
            "Q4_K bytes = (total_elements / 256) * 144 — \
             llama-cpp-compatible layout"
        );
        assert_eq!(bytes.len(), 2_451_456, "exact byte count for ffn_down");
    }

    /// AUDIT-Q4K-SHAPE-001 — in-tree falsification of the pre-v0.34.0
    /// shape-swap bug for the 256-divisible-on-both-dims case. See
    /// `docs/specifications/audits/q4k-shape-swap-impact.md`.
    ///
    /// **Empirical finding**: when both `shape[0]` and `shape[1]` are
    /// 256-divisible, `quantize_q4_k_matrix(data, [a, b])` and
    /// `quantize_q4_k_matrix(data, [b, a])` produce **byte-identical**
    /// output. Therefore Qwen2 1.5B (hidden=1536, intermediate=8960) and
    /// Qwen2 7B (hidden=3584, intermediate=18944) Q4_K exports produced
    /// before the v0.34.0 fix are **bit-equivalent to a post-fix re-export**
    /// for the in-tree path — no re-export needed for correctness.
    ///
    /// **Why**: the function iterates `rows` times grabbing `cols`
    /// contiguous elements per iteration, then quantizes that 1D slice as
    /// fixed-size 256-element super-blocks via `quantize_q4_k`. When both
    /// dims are 256-multiples, the data is consumed in the same linear
    /// order with the same 256-aligned chunking either way — the "row"
    /// boundary is invisible to the quantizer because it sits on a
    /// super-block boundary.
    ///
    /// **The shape-swap bug bites only when `cols % 256 != 0`** because
    /// the function then pads `cols` up to the next 256-multiple,
    /// shifting subsequent super-blocks off-stride. That's Qwen2 0.5B
    /// territory (hidden=896 → cols=896 when swapped wrong) — handled
    /// by the defect-2 K-divisibility fallback (forces F32 instead of
    /// quantizing).
    ///
    /// This test pins the byte-equivalence finding. If it ever fails,
    /// trueno-quant changed its layout and the audit doc needs a revisit.
    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn audit_q4k_shape_swap_byte_identical_when_both_dims_divisible() {
        use super::super::quantize_q4_k_matrix;

        // 256 × 512: both 256-divisible, shape[0] != shape[1] so the swap
        // meaningfully differs. Small enough for a fast unit test.
        let rows = 256_usize;
        let cols = 512_usize;
        let n = rows * cols;

        // Heterogeneous per-row distribution: row r centered at r*0.01,
        // std 0.1. Adjacent rows differ enough that — IF the swap shifted
        // super-block boundaries — the resulting bytes would diverge.
        let mut data = vec![0.0_f32; n];
        for r in 0..rows {
            let row_mean = (r as f32) * 0.01;
            for c in 0..cols {
                let perturbation = ((r * 31 + c * 17) as f32).sin() * 0.1;
                data[r * cols + c] = row_mean + perturbation;
            }
        }

        // CORRECT (post-v0.34.0): APR-native shape.
        let correct_bytes = quantize_q4_k_matrix(&data, &[rows, cols]);
        // BUGGY (pre-v0.34.0): swap before passing.
        let buggy_bytes = quantize_q4_k_matrix(&data, &[cols, rows]);

        assert_eq!(
            correct_bytes.len(),
            buggy_bytes.len(),
            "shape-swap audit precondition: byte counts equal"
        );

        // **Central finding**: for the 256-divisible-on-both-dims case,
        // the bytes are IDENTICAL. The bug doesn't manifest here at all.
        assert_eq!(
            correct_bytes, buggy_bytes,
            "AUDIT-Q4K-SHAPE-001: when both dims are 256-divisible, the \
             pre-v0.34.0 shape-swap produces byte-identical output to the \
             post-v0.34.0 correct call. Falsification of any future divergence \
             would mean trueno-quant's layout changed — revisit the audit doc \
             and the v0.33.0 / earlier shipped Q4_K artifacts."
        );
    }
}

include!("export_include_01.rs");
