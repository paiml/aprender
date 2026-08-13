
/// Infer `num_layers` from APR tensor names by counting unique `blk.N.*` prefixes.
/// Returns `None` if the file uses a different naming convention (e.g. pre-mapping
/// HF names like `model.layers.N.*`).
///
/// Contract: apr-export-num-layers-v1 (#1865).
pub(crate) fn infer_num_layers_from_tensor_names(names: &[&str]) -> Option<usize> {
    let mut max_idx: Option<usize> = None;
    for name in names {
        // Match both `blk.N.*` (GGUF) and `model.layers.N.*` (HF) conventions.
        let stripped = name
            .strip_prefix("blk.")
            .or_else(|| name.strip_prefix("model.layers."));
        if let Some(rest) = stripped {
            if let Some(dot) = rest.find('.') {
                if let Ok(idx) = rest[..dot].parse::<usize>() {
                    max_idx = Some(max_idx.map_or(idx, |m| m.max(idx)));
                }
            }
        }
    }
    max_idx.map(|i| i + 1)
}

fn missing_dim_err(field: &str) -> AprenderError {
    AprenderError::FormatError {
        message: format!(
            "C-07: {field} required for GGUF export (missing in APR metadata). \
             Re-stamp the APR file with `apr stamp` populating model dimensions, \
             or convert from the original GGUF/SafeTensors source."
        ),
    }
}

/// PMAT-920: actionable error for the one dimension that is NOT inferable from
/// tensor shapes alone — `num_heads`.
///
/// `q_dim = num_heads * head_dim` cannot be factored without knowing `head_dim`
/// (a 1536-wide projection is 12×128 OR 24×64 OR 16×96 — all valid). The old
/// exporter GUESSED `head_dim` from a hardcoded `[64, 128, 96, 80]` list and
/// took the first divisor, which silently MIS-STAMPED real models: Qwen2-1.5B
/// (hidden=1536, head_dim=128, 12 heads) got `1536/64 = 24` heads written into
/// a valid-looking GGUF, no error. A silently-wrong head count is worse than an
/// honest failure, so when neither `num_heads` nor `head_dim` is present we
/// hard-fail and tell the user exactly how to supply the missing dimension.
fn missing_num_heads_err() -> AprenderError {
    AprenderError::FormatError {
        message:
            "C-07: num_heads required for GGUF export and NOT inferable from tensor shapes \
             alone — q_dim = num_heads × head_dim has no unique factorization without head_dim \
             (a 1536-wide projection is 12×128 OR 24×64 OR 16×96, all valid). \
             To supply the missing dimension, populate the APR metadata header so the export \
             can derive it exactly: set an explicit `head_dim` (then num_heads = q_dim / head_dim, \
             num_kv_heads = kv_dim / head_dim) or an explicit `num_heads`/`num_kv_heads` via \
             `apr stamp`, or re-`apr convert` from the original GGUF/SafeTensors source whose \
             config.json carries head_dim / num_attention_heads. \
             Refusing to guess head_dim (the old [64,128,96,80] first-divisor guess silently \
             mis-stamped models like Qwen2-1.5B as 24 heads instead of 12 — a wrong-but-valid \
             GGUF is worse than an honest failure)."
            .to_string(),
    }
}

/// PMAT-920: smaller dimension of the first 2D projection tensor matching any
/// of the given name patterns. For a row-major `[out, in]` projection this is
/// the model/projection width (`q_dim`/`kv_dim`), which together with an
/// EXPLICIT `head_dim` yields an exact, sound head count.
fn projection_dim_from_shapes(
    reader: &crate::format::v2::AprV2Reader,
    name_patterns: &[&str],
) -> Option<usize> {
    for name in reader.tensor_names() {
        if name_patterns.iter().any(|p| name.contains(p)) {
            if let Some(entry) = reader.get_tensor(name) {
                if entry.shape.len() == 2 {
                    return Some(entry.shape[0].min(entry.shape[1]));
                }
            }
        }
    }
    None
}

/// PMAT-920 (OBLIG-APR-GGUF-EXPORT-INFER-METADATA): fill missing GGUF-required
/// dimensions on `apr_metadata` for metadata-light `.apr` files.
///
/// CORRECTNESS CONTRACT — which dimensions are inferable from shapes:
///   - `hidden_size`, `vocab_size` — UNAMBIGUOUS from the embedding shape
///     `[vocab, hidden]`. Inferred from shapes.
///   - `intermediate_size` — UNAMBIGUOUS from the FFN gate/up shapes. Inferred.
///   - `num_heads` / `num_kv_heads` — **NOT** inferable from shapes alone:
///     `q_dim = num_heads × head_dim` has no unique factorization without
///     `head_dim`. We derive them ONLY from an EXPLICIT `head_dim`
///     (`num_heads = q_dim / head_dim`, EXACT and sound). If `head_dim` is
///     absent and `num_heads` is absent, we DO NOT guess — the export
///     hard-fails with an actionable error (`missing_num_heads_err`).
///
/// This deliberately drops the old `[64,128,96,80]` first-divisor head_dim
/// guess, which silently mis-stamped real models (e.g. Qwen2-1.5B as 24 heads
/// instead of 12). An honest error (or a user `--head-dim`/`--num-heads`
/// override) beats a silently-wrong head count.
///
/// Only fills fields that are currently `None`; explicit metadata always wins.
/// LAYOUT-001: APR tensor shapes are row-major.
fn infer_missing_gguf_dims_from_shapes(
    reader: &crate::format::v2::AprV2Reader,
    apr_metadata: &mut crate::format::v2::AprV2Metadata,
) {
    // Head counts: ONLY from an explicit head_dim. No shape guessing.
    fill_head_counts_from_explicit_head_dim(reader, apr_metadata);

    let needs_shape_inference = apr_metadata.hidden_size.is_none()
        || apr_metadata.vocab_size.is_none()
        || apr_metadata.intermediate_size.is_none();
    if !needs_shape_inference {
        return;
    }

    // Build a shape-only tensor map (empty data — the inference engine reads
    // shapes only) so we can reuse the SafeTensors shape-inference path for the
    // dimensions that ARE unambiguous (hidden/vocab/intermediate).
    let mut shape_map: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    for name in reader.tensor_names() {
        if let Some(entry) = reader.get_tensor(name) {
            shape_map.insert(name.to_string(), (Vec::new(), entry.shape.clone()));
        }
    }

    let Some(inferred) = super::import::infer_model_config_from_tensors(&shape_map) else {
        return;
    };

    if apr_metadata.hidden_size.is_none() {
        if let Some(v) = inferred.hidden_size {
            eprintln!("[PMAT-920] hidden_size missing from APR metadata — inferred {v} from embedding tensor shape");
            apr_metadata.hidden_size = Some(v);
        }
    }
    if apr_metadata.vocab_size.is_none() {
        if let Some(v) = inferred.vocab_size {
            eprintln!("[PMAT-920] vocab_size missing from APR metadata — inferred {v} from embedding tensor shape");
            apr_metadata.vocab_size = Some(v);
        }
    }
    if apr_metadata.intermediate_size.is_none() {
        if let Some(v) = inferred.intermediate_size {
            eprintln!("[PMAT-920] intermediate_size missing from APR metadata — inferred {v} from FFN tensor shapes");
            apr_metadata.intermediate_size = Some(v);
        }
    }
    // NOTE: deliberately NOT copying inferred.num_heads / inferred.num_kv_heads
    // here — those come from the [64,128,96,80] head_dim guess in
    // infer_head_counts and are silently-wrong. Head counts are filled only by
    // fill_head_counts_from_explicit_head_dim above.
}

/// PMAT-920: derive `num_heads` / `num_kv_heads` from an EXPLICIT `head_dim`.
///
/// SOUND because `head_dim` is given: `num_heads = q_dim / head_dim`,
/// `num_kv_heads = kv_dim / head_dim`. Only fills fields that are `None`;
/// explicit metadata always wins. Returns without touching head counts when
/// `head_dim` is absent — the caller then hard-fails via `missing_num_heads_err`
/// rather than guessing.
fn fill_head_counts_from_explicit_head_dim(
    reader: &crate::format::v2::AprV2Reader,
    apr_metadata: &mut crate::format::v2::AprV2Metadata,
) {
    if apr_metadata.num_heads.is_some() && apr_metadata.num_kv_heads.is_some() {
        return;
    }
    let Some(head_dim) = apr_metadata.head_dim else {
        // No explicit head_dim → NOT inferable. Do not guess.
        return;
    };
    if head_dim == 0 {
        return;
    }

    let q_dim =
        projection_dim_from_shapes(reader, &["q_proj.weight", "query.weight", "attn_q.weight"]);
    let kv_dim =
        projection_dim_from_shapes(reader, &["k_proj.weight", "key.weight", "attn_k.weight"]);

    if apr_metadata.num_heads.is_none() {
        if let Some(q) = q_dim {
            if q.is_multiple_of(head_dim) {
                let n = q / head_dim;
                eprintln!(
                    "[PMAT-920] num_heads missing — derived {n} = q_dim({q}) / explicit head_dim({head_dim})"
                );
                apr_metadata.num_heads = Some(n);
            }
        }
    }
    if apr_metadata.num_kv_heads.is_none() {
        if let Some(kv) = kv_dim {
            if kv.is_multiple_of(head_dim) {
                let n = kv / head_dim;
                eprintln!(
                    "[PMAT-920] num_kv_heads missing — derived {n} = kv_dim({kv}) / explicit head_dim({head_dim})"
                );
                apr_metadata.num_kv_heads = Some(n);
            }
        }
    }
}

/// Build GGUF architecture metadata from APR model metadata.
///
/// Returns `Err(AprenderError::FormatError)` instead of panicking when required
/// dimensions are missing, so `apr export` produces a clean exit-5 instead of
/// a panic-101 (#1865).
fn build_gguf_arch_metadata(
    apr_metadata: &crate::format::v2::AprV2Metadata,
) -> Result<Vec<(String, crate::format::gguf::GgufValue)>> {
    use crate::format::gguf::GgufValue;

    let arch = resolve_architecture(apr_metadata);
    let hidden_size = apr_metadata.hidden_size.ok_or_else(|| missing_dim_err("hidden_size"))?;
    let num_layers = apr_metadata.num_layers.ok_or_else(|| missing_dim_err("num_layers"))?;
    let num_heads = apr_metadata.num_heads.ok_or_else(missing_num_heads_err)?;
    let num_kv_heads = apr_metadata.num_kv_heads.unwrap_or(num_heads);
    let vocab_size = apr_metadata.vocab_size.ok_or_else(|| missing_dim_err("vocab_size"))?;
    let intermediate_size =
        apr_metadata.intermediate_size.ok_or_else(|| missing_dim_err("intermediate_size"))?;
    let max_pos = apr_metadata.max_position_embeddings.unwrap_or(0);
    // N-01 (Meyer DbC): rope_theta from metadata, or architecture-specific default.
    let rope_theta = apr_metadata.rope_theta.unwrap_or_else(||
        super::export::default_rope_theta_for_architecture(arch));
    let rms_norm_eps = apr_metadata.rms_norm_eps.unwrap_or(1e-6);
    let head_dim = hidden_size.checked_div(num_heads).unwrap_or(0);
    let model_name = apr_metadata
        .name
        .clone()
        .unwrap_or_else(|| "model".to_string());

    let mut metadata = vec![
        (
            "general.architecture".to_string(),
            GgufValue::String(arch.to_string()),
        ),
        ("general.name".to_string(), GgufValue::String(model_name)),
        (
            "general.quantization_version".to_string(),
            GgufValue::Uint32(2),
        ),
        ("general.file_type".to_string(), GgufValue::Uint32(0)),
        (
            format!("{arch}.context_length"),
            GgufValue::Uint32(max_pos as u32),
        ),
        (
            format!("{arch}.embedding_length"),
            GgufValue::Uint32(hidden_size as u32),
        ),
        (
            format!("{arch}.block_count"),
            GgufValue::Uint32(num_layers as u32),
        ),
        (
            format!("{arch}.feed_forward_length"),
            GgufValue::Uint32(intermediate_size as u32),
        ),
        (
            format!("{arch}.attention.head_count"),
            GgufValue::Uint32(num_heads as u32),
        ),
        (
            format!("{arch}.attention.head_count_kv"),
            GgufValue::Uint32(num_kv_heads as u32),
        ),
    ];

    // GH-277: GPT-2 uses standard LayerNorm, not RMSNorm
    if arch == "gpt2" {
        metadata.push((
            format!("{arch}.attention.layer_norm_epsilon"),
            GgufValue::Float32(rms_norm_eps),
        ));
    } else {
        metadata.push((
            format!("{arch}.attention.layer_norm_rms_epsilon"),
            GgufValue::Float32(rms_norm_eps),
        ));
    }

    // GH-277: Only emit RoPE keys for architectures that use RoPE
    if uses_rope(arch) {
        metadata.push((
            format!("{arch}.rope.dimension_count"),
            GgufValue::Uint32(head_dim as u32),
        ));
        metadata.push((
            format!("{arch}.rope.freq_base"),
            GgufValue::Float32(rope_theta),
        ));
    }

    metadata.push((
        format!("{arch}.vocab_size"),
        GgufValue::Uint32(vocab_size as u32),
    ));

    Ok(metadata)
}

/// Push a string array from APR custom fields to GGUF entries.
fn push_string_array(
    entries: &mut Vec<(String, crate::format::gguf::GgufValue)>,
    custom: &std::collections::HashMap<String, serde_json::Value>,
    src_key: &str,
    gguf_key: &str,
) {
    let arr = custom.get(src_key).and_then(|v| v.as_array());
    let Some(arr) = arr else { return };
    let strings: Vec<String> = arr
        .iter()
        .filter_map(|v| v.as_str().map(String::from))
        .collect();
    if !strings.is_empty() {
        entries.push((
            gguf_key.to_string(),
            crate::format::gguf::GgufValue::ArrayString(strings),
        ));
    }
}

/// Push a u32 value from APR custom fields to GGUF entries.
fn push_u32_field(
    entries: &mut Vec<(String, crate::format::gguf::GgufValue)>,
    custom: &std::collections::HashMap<String, serde_json::Value>,
    src_key: &str,
    gguf_key: &str,
) {
    if let Some(val) = custom.get(src_key).and_then(|v| v.as_u64()) {
        entries.push((
            gguf_key.to_string(),
            crate::format::gguf::GgufValue::Uint32(val as u32),
        ));
    }
}

/// Push an i32 array from APR custom fields to GGUF entries.
fn push_i32_array(
    entries: &mut Vec<(String, crate::format::gguf::GgufValue)>,
    custom: &std::collections::HashMap<String, serde_json::Value>,
    src_key: &str,
    gguf_key: &str,
) {
    let arr = custom.get(src_key).and_then(|v| v.as_array());
    let Some(arr) = arr else { return };
    let types: Vec<i32> = arr
        .iter()
        .filter_map(|v| v.as_i64().map(|n| n as i32))
        .collect();
    if !types.is_empty() {
        entries.push((
            gguf_key.to_string(),
            crate::format::gguf::GgufValue::ArrayInt32(types),
        ));
    }
}

/// Extract tokenizer metadata from APR custom fields for GGUF export (GH-253).
///
/// P0-G: `vocab_size` is the model's `<arch>.vocab_size`. When the embedded tokenizer
/// vocabulary is smaller than the model's vocab_size (Qwen2.5: 151643 vs 151936),
/// the emitted `tokenizer.ggml.tokens` array is padded with `<|pad_N|>` placeholders
/// so llama.cpp's `check_tensor_dims` accepts the corresponding `token_embd.weight`
/// first dim. Pass 0 to disable padding.
fn extract_apr_tokenizer_for_gguf(
    apr_metadata: &crate::format::v2::AprV2Metadata,
    vocab_size: usize,
) -> Vec<(String, crate::format::gguf::GgufValue)> {
    use crate::format::gguf::GgufValue;

    let mut entries = Vec::new();
    let custom = &apr_metadata.custom;
    let arch = resolve_architecture(apr_metadata);

    // Tokenizer model type: "gpt2" for byte-level BPE (Qwen, GPT-2), "llama" for SentencePiece
    // GH-253-3: APR stores raw model_type from GGUF which may be "bpe" — map to "gpt2"
    let raw_model_type = custom
        .get("tokenizer.model")
        .and_then(|v| v.as_str())
        .unwrap_or("gpt2");
    let model_type = match raw_model_type {
        "bpe" => "gpt2",
        other => other,
    };
    entries.push((
        "tokenizer.ggml.model".to_string(),
        GgufValue::String(model_type.to_string()),
    ));
    // GH-277: Use pre-tokenizer type mapping, preferring round-trip preserved value
    let model_name = apr_metadata.name.as_deref().unwrap_or("");
    let pre_type = custom
        .get("tokenizer.pre_type")
        .and_then(|v| v.as_str())
        .unwrap_or_else(|| resolve_pre_tokenizer_type(arch, model_name));
    entries.push((
        "tokenizer.ggml.pre".to_string(),
        GgufValue::String(pre_type.to_string()),
    ));

    push_string_array(
        &mut entries,
        custom,
        "tokenizer.vocabulary",
        "tokenizer.ggml.tokens",
    );
    // P0-G: pad tokens array to vocab_size if smaller (Qwen2.5: 151643 → 151936).
    if vocab_size > 0 {
        if let Some(tokens) = entries.iter_mut().find_map(|(k, v)| match v {
            GgufValue::ArrayString(arr) if k == "tokenizer.ggml.tokens" => Some(arr),
            _ => None,
        }) {
            if tokens.len() < vocab_size {
                let pad_count = vocab_size - tokens.len();
                eprintln!(
                    "[P0-G] Padding APR-fallback tokenizer.ggml.tokens: {} + {} placeholders = {}",
                    tokens.len(),
                    pad_count,
                    vocab_size
                );
                for i in tokens.len()..vocab_size {
                    tokens.push(format!("<|pad_{i}|>"));
                }
            }
        }
    }
    push_string_array(
        &mut entries,
        custom,
        "tokenizer.merges",
        "tokenizer.ggml.merges",
    );
    push_u32_field(
        &mut entries,
        custom,
        "tokenizer.bos_token_id",
        "tokenizer.ggml.bos_token_id",
    );
    push_u32_field(
        &mut entries,
        custom,
        "tokenizer.eos_token_id",
        "tokenizer.ggml.eos_token_id",
    );
    push_i32_array(
        &mut entries,
        custom,
        "tokenizer.token_type",
        "tokenizer.ggml.token_type",
    );
    push_u32_field(
        &mut entries,
        custom,
        "tokenizer.padding_token_id",
        "tokenizer.ggml.padding_token_id",
    );

    // GH-253-1: add_bos_token flag
    if let Some(add_bos) = custom
        .get("tokenizer.add_bos_token")
        .and_then(|v| v.as_bool())
    {
        entries.push((
            "tokenizer.ggml.add_bos_token".to_string(),
            GgufValue::Bool(add_bos),
        ));
    }

    // GH-253-1: Chat template (Jinja2)
    let chat_tmpl = apr_metadata.chat_template.as_deref().or_else(|| {
        custom
            .get("tokenizer.chat_template")
            .and_then(|v| v.as_str())
    });
    if let Some(tmpl) = chat_tmpl {
        entries.push((
            "tokenizer.chat_template".to_string(),
            GgufValue::String(tmpl.to_string()),
        ));
    }

    entries
}

/// GH-246: Export to MLX format (Apple Silicon).
///
/// MLX models are stored as a directory containing:
/// - `model.safetensors` — weights in SafeTensors format
/// - `config.json` — model configuration (HuggingFace-compatible)
/// - `tokenizer.json` — tokenizer (optional, from APR metadata)
///
/// This reuses the SafeTensors export path since MLX uses SafeTensors as its
/// underlying weight format. The key difference is the directory structure.
fn export_mlx(
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    input_path: &Path,
    output_path: &Path,
    options: &ExportOptions,
) -> Result<()> {
    // Output path is the directory
    fs::create_dir_all(output_path).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to create MLX output directory: {e}"),
    })?;

    // Write model.safetensors
    let weights_path = output_path.join("model.safetensors");
    let user_metadata = extract_user_metadata(input_path);
    if user_metadata.is_empty() {
        save_safetensors(&weights_path, tensors).map_err(|e| AprenderError::FormatError {
            message: format!("Failed to write MLX weights: {e}"),
        })?;
    } else {
        save_safetensors_with_metadata(&weights_path, tensors, &user_metadata).map_err(|e| {
            AprenderError::FormatError {
                message: format!("Failed to write MLX weights: {e}"),
            }
        })?;
    }

    // Write config.json
    let config = infer_model_config(tensors);
    let config_path = output_path.join("config.json");
    fs::write(&config_path, config).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to write MLX config.json: {e}"),
    })?;

    // Write tokenizer.json if available
    if options.include_tokenizer {
        let tokenizer_json = infer_tokenizer_json(input_path);
        if !tokenizer_json.is_empty() {
            let tokenizer_path = output_path.join("tokenizer.json");
            if let Err(e) = fs::write(&tokenizer_path, &tokenizer_json) {
                eprintln!("[GH-246] Warning: Failed to write tokenizer.json: {e}");
            }
        }
    }

    Ok(())
}

/// PMAT-252: Raw block passthrough for APR→GGUF export.
///
/// Reads raw tensor bytes directly from APR file (Q4K super-blocks, F32 vectors,
/// etc.) and writes them to GGUF without any dequantization/requantization.
/// This is LOSSLESS for quantized data — zero quality degradation.
///
/// The key insight: APR and GGUF both store Q4K blocks in the same binary format
/// (256-element super-blocks, 144 bytes each). The only differences are:
/// 1. Tensor names (HF convention in APR → GGML convention in GGUF)
/// 2. Shape representation (APR [rows, cols] → GGUF [ne0=cols, ne1=rows])
/// 3. File-level metadata (APR header → GGUF KV pairs)
fn export_apr_to_gguf_raw(input: &Path, output: &Path) -> Result<ExportReport> {
    use crate::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor};
    use crate::format::v2::{AprV2Reader, TensorDType};
    use std::fs::File;
    use std::io::BufWriter;

    let data = fs::read(input).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to read APR file: {e}"),
    })?;
    let original_size = data.len();

    let reader = AprV2Reader::from_bytes(&data).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to parse APR file: {e:?}"),
    })?;

    let mut apr_metadata = reader.metadata().clone();

    // #1865: infer `num_layers` from tensor names when metadata is silent. Older
    // APR files (and any produced without `apr stamp --num-layers`) leave this
    // field unset; the tensor layout always carries the layer count, so derive
    // it before raising a missing-dimension error.
    if apr_metadata.num_layers.is_none() {
        let names = reader.tensor_names();
        if let Some(inferred) = infer_num_layers_from_tensor_names(&names) {
            eprintln!(
                "[#1865] num_layers missing from APR metadata — inferred {} from blk.N.* tensor names",
                inferred
            );
            apr_metadata.num_layers = Some(inferred);
        }
    }

    // PMAT-920 (OBLIG-APR-GGUF-EXPORT-INFER-METADATA): when the APR metadata
    // block is "light" (no explicit num_heads / hidden_size / vocab_size /
    // intermediate_size — e.g. a .apr produced by training or `apr convert`
    // without a fully-populated header), the tensor shapes still carry these
    // GGUF-required dimensions unambiguously. Infer them from the shapes
    // before the C-07 missing-dimension error fires, so apr→gguf works on
    // arbitrary metadata-light .apr files (llama.cpp / ollama interop) instead
    // of hard-failing. Shapes are APR-native row-major (LAYOUT-001), exactly
    // what `infer_model_config_from_tensors` expects.
    infer_missing_gguf_dims_from_shapes(&reader, &mut apr_metadata);

    let arch = resolve_architecture(&apr_metadata);
    let num_layers = apr_metadata.num_layers.ok_or_else(|| missing_dim_err("num_layers"))?;
    let num_heads = apr_metadata.num_heads.ok_or_else(missing_num_heads_err)?;
    let num_kv_heads = apr_metadata.num_kv_heads.unwrap_or(num_heads);
    let hidden_size = apr_metadata.hidden_size.ok_or_else(|| missing_dim_err("hidden_size"))?;

    // Build metadata from architecture config + tokenizer custom fields
    let mut metadata = build_gguf_arch_metadata(&apr_metadata)?;
    let vocab_size = apr_metadata.vocab_size.unwrap_or(0);
    metadata.extend(extract_apr_tokenizer_for_gguf(&apr_metadata, vocab_size));

    // GH-253-4: Validate metadata completeness before writing
    let validated = ValidatedGgufMetadata::validate(metadata)?;

    eprintln!(
        "[PMAT-252] Writing {} metadata keys (arch={}, layers={}, heads={}/{}kv, hidden={})",
        validated.as_slice().len(),
        arch,
        num_layers,
        num_heads,
        num_kv_heads,
        hidden_size
    );

    // GH-277: Build contract-driven tensor name mapper
    let mapper = build_gguf_mapper(arch);

    // Build GGUF tensors with raw byte passthrough
    let tensor_names = reader.tensor_names();
    let mut gguf_tensors = Vec::with_capacity(tensor_names.len());

    for name in &tensor_names {
        // GH-277: Use contract-driven mapping; skip tensors that return None
        let Some(gguf_name) = mapper.map_name(name) else {
            eprintln!("[GH-277] Skipping tensor '{}' (not in GGUF contract)", name);
            continue;
        };

        let entry = reader
            .get_tensor(name)
            .ok_or_else(|| AprenderError::FormatError {
                message: format!("Tensor '{}' missing from index", name),
            })?;
        let raw_bytes = reader
            .get_tensor_data(name)
            .ok_or_else(|| AprenderError::FormatError {
                message: format!("Tensor '{}' data not found", name),
            })?;

        // Map APR dtype → GGUF dtype. NOTE: the discriminants are NOT shared —
        // APR-native quant types (AprQ8=129, AprQ4=128) have NO GGML equivalent
        // despite the similar names.
        // GH-439 (poka-yoke): Exhaustive match — no silent fallbacks.
        // Adding a new TensorDType variant forces a compile error here.
        let gguf_dtype = match entry.dtype {
            TensorDType::F32 => GgmlType::F32,
            TensorDType::F16 => GgmlType::F16,
            TensorDType::Q4K => GgmlType::Q4K,
            TensorDType::Q6K => GgmlType::Q6K,
            // AprQ8 is APR-native single-whole-tensor-scale 8-bit
            // ([scale: f32 (4B)] + [i8; N] = 4+N bytes). GGML Q8_0 is a totally
            // different per-32-block layout ([f16 scale (2B) + 32×i8] =
            // ceil(N/32)*34 bytes). Emitting the raw APR bytes under a Q8_0
            // label produces a CORRUPT GGUF, so reject — symmetric with the
            // AprQ4 arm below and the import-side Q8_0 rejection
            // (write_model_config.rs). A real AprQ8→Q8_0 requantize is a
            // separate feature, not a silent relabel.
            TensorDType::AprQ8 => {
                return Err(AprenderError::FormatError {
                    message: format!(
                        "Tensor '{}' has dtype AprQ8 (APR-native single-scale 8-bit, \
                         NOT GGML Q8_0) which has no GGUF equivalent. \
                         Convert to F32/F16 first with `apr convert`.",
                        name
                    ),
                });
            }
            TensorDType::BF16 | TensorDType::F64 | TensorDType::I32
            | TensorDType::I64 | TensorDType::I8 | TensorDType::U8
            | TensorDType::AprQ4 => {
                return Err(AprenderError::FormatError {
                    message: format!(
                        "Tensor '{}' has dtype {:?} which has no GGUF equivalent. \
                         Convert to F32/F16 first with `apr convert`.",
                        name, entry.dtype
                    ),
                });
            }
        };

        // Reverse shape for GGUF: [rows, cols] → [ne0=cols, ne1=rows]
        let gguf_shape = if entry.shape.len() == 2 {
            vec![entry.shape[1] as u64, entry.shape[0] as u64]
        } else {
            entry.shape.iter().map(|&d| d as u64).collect()
        };

        eprintln!(
            "[PMAT-252] '{}': {} bytes (dtype={:?})",
            gguf_name,
            raw_bytes.len(),
            entry.dtype
        );

        gguf_tensors.push(GgufTensor {
            name: gguf_name,
            shape: gguf_shape,
            dtype: gguf_dtype,
            data: raw_bytes.to_vec(),
        });
    }

    // GH-277: Add fused tensors (e.g., QKV fusion for GPT-2)
    let fused = build_fused_tensors_raw(&mapper, &reader);
    gguf_tensors.extend(fused);

    // Write to file
    let file = File::create(output).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to create output file: {e}"),
    })?;
    let mut writer = BufWriter::new(file);

    export_tensors_to_gguf(&mut writer, &gguf_tensors, validated.as_slice())?;

    let exported_size = fs::metadata(output).map(|m| m.len() as usize).unwrap_or(0);

    Ok(ExportReport {
        original_size,
        exported_size,
        tensor_count: gguf_tensors.len(),
        format: ExportFormat::Gguf,
        quantization: Some(QuantizationType::Q4K),
    })
}

/// Legacy mapper for test compatibility.
/// Uses the fallback legacy mapper (same behavior as old hardcoded function).
#[cfg(test)]
fn hf_to_gguf_name(name: &str) -> String {
    let mapper = build_legacy_mapper();
    mapper.map_name(name).unwrap_or_else(|| name.to_string())
}
