
/// Append tokenizer metadata to GGUF metadata, preferring tokenizer.json over APR fallback.
///
/// P0-G: `vocab_size` is the model's `<arch>.vocab_size` (e.g. 151936 for Qwen2.5 with
/// TP-alignment padding). The tokenizer's true vocabulary may be smaller (151643 for
/// Qwen2.5-Coder). llama.cpp uses `len(tokenizer.ggml.tokens)` as the expected first dim
/// of `token_embd.weight`, so the tokens array MUST be padded to `vocab_size` with
/// placeholder entries to match the actual tensor shape.
fn append_tokenizer_to_metadata(
    metadata: &mut Vec<(String, crate::format::gguf::GgufValue)>,
    tokenizer: Option<&crate::format::gguf::GgufTokenizer>,
    apr_metadata: Option<&crate::format::v2::AprV2Metadata>,
    arch: &str,
    model_name: &str,
    vocab_size: usize,
    input: &Path,
) {
    if let Some(tok) = tokenizer {
        metadata.extend(build_tokenizer_gguf_metadata(tok, arch, model_name, vocab_size));
        // #5254: this path wrote every tokenizer key but the chat template, so a
        // runtime that reads the GGUF alone (ollama) had nothing to chat with.
        let tpl = tok.chat_template.clone().filter(|t| !t.trim().is_empty());
        if let Some(tpl) = tpl.or_else(|| export_chat_template(apr_metadata, input)) {
            metadata.push((
                "tokenizer.chat_template".to_string(),
                crate::format::gguf::GgufValue::String(tpl),
            ));
        }
        return;
    }

    eprintln!(
        "[BUG-EXPORT-004] Warning: No tokenizer.json found near {}, GGUF may lack tokenizer metadata",
        input.display()
    );

    // GH-211: Fallback — extract tokenizer from APR metadata when no tokenizer.json
    let Some(apr_meta) = apr_metadata else {
        return;
    };
    let apr_tok_entries = extract_apr_tokenizer_for_gguf(apr_meta, vocab_size);
    if !apr_tok_entries.is_empty() {
        eprintln!(
            "[GH-211] Extracted {} tokenizer entries from APR metadata",
            apr_tok_entries.len()
        );
        metadata.extend(apr_tok_entries);
    }
}

/// Build a Q4K output.weight tensor from embedding data for tied-embedding models (BUG-4).
fn build_tied_output_weight(
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
) -> Option<crate::format::gguf::GgufTensor> {
    use crate::format::gguf::{GgmlType, GgufTensor};

    let (_, (data, shape)) = tensors
        .iter()
        .find(|(name, _)| name.contains("embed_tokens") || name.contains("token_embedding"))?;

    if shape.len() != 2 || data.len() < 256 {
        return None;
    }

    // PMAT-690 defects 2+3 (2026-05-17): only Q4_K when shape[1] (=K=ne0)
    // is 256-divisible, and pass APR-native shape directly (no swap) so
    // the quantizer pads/slices along the correct dim. When K isn't
    // 256-divisible (e.g., Qwen2 0.5B hidden=896), the tied output weight
    // must be F32 to keep the GGUF llama-cli-compatible.
    if shape[1] % 256 != 0 {
        eprintln!(
            "[BUG-4-FIX-Q4K-FALLBACK] tied output.weight shape {:?} — \
             K={} not divisible by 256; emitting F32 instead of Q4K",
            shape, shape[1]
        );
        let f32_bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let gguf_shape = vec![shape[1] as u64, shape[0] as u64];
        return Some(GgufTensor {
            name: "output.weight".to_string(),
            shape: gguf_shape,
            dtype: GgmlType::F32,
            data: f32_bytes,
        });
    }

    eprintln!("[BUG-4-FIX] Creating Q4K output.weight from embedding for tied embeddings");

    let q4k_bytes = super::quantize_q4_k_matrix(data, shape);
    let gguf_shape = vec![shape[1] as u64, shape[0] as u64];

    Some(GgufTensor {
        name: "output.weight".to_string(),
        shape: gguf_shape,
        dtype: GgmlType::Q4K,
        data: q4k_bytes,
    })
}

/// The model's own chat template for a GGUF export: APR metadata (field, then
/// custom key), else `chat_template.jinja` beside the input, else the
/// `chat_template` string in `tokenizer_config.json` beside it (#5254).
fn export_chat_template(
    apr_metadata: Option<&crate::format::v2::AprV2Metadata>,
    input: &Path,
) -> Option<String> {
    apr_metadata
        .and_then(|m| {
            m.chat_template.clone().or_else(|| {
                m.custom
                    .get("tokenizer.chat_template")
                    .and_then(|v| v.as_str())
                    .map(str::to_string)
            })
        })
        .or_else(|| std::fs::read_to_string(input.with_file_name("chat_template.jinja")).ok())
        .or_else(|| {
            let cfg = std::fs::read_to_string(input.with_file_name("tokenizer_config.json")).ok()?;
            serde_json::from_str::<serde_json::Value>(&cfg)
                .ok()?
                .get("chat_template")?
                .as_str()
                .map(str::to_string)
        })
        .filter(|t| !t.trim().is_empty())
}
