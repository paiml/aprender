
/// Detect GGUF model architecture for tensor name mapping (GH-200).
/// GH-236: Added GPT-2 recognition — was falling through to Qwen2 default,
/// causing metadata key mismatch (writes "qwen2.embedding_length" but reader
/// looks for "gpt2.embedding_length") → hidden_dim=0 on reimport.
fn detect_gguf_architecture(path: &Path) -> Architecture {
    GgufReader::from_file(path)
        .ok()
        .and_then(|r| r.architecture())
        .map(|a| Architecture::from_model_type(&a).unwrap_or(Architecture::Qwen2))
        .unwrap_or(Architecture::Qwen2)
}

/// Infer vocab_size and hidden_dim from tensor shapes.
///
/// Find the first 2D tensor matching any of the given name patterns.
fn find_2d_tensor_shape<'a>(
    tensors: &'a BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    patterns: &[&str],
) -> Option<&'a [usize]> {
    tensors.iter().find_map(|(name, (_, shape))| {
        if shape.len() == 2 && patterns.iter().any(|p| name.contains(p)) {
            Some(shape.as_slice())
        } else {
            None
        }
    })
}

/// Used for contract validation during export.
fn infer_vocab_hidden(tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>) -> (usize, usize) {
    // Try embedding, then lm_head for [vocab_size, hidden_dim]
    if let Some(shape) = find_2d_tensor_shape(tensors, &["embed_tokens", "token_embd"]) {
        return (shape[0], shape[1]);
    }
    if let Some(shape) = find_2d_tensor_shape(tensors, &["lm_head", "output.weight"]) {
        return (shape[0], shape[1]);
    }
    // Fallback: get hidden_dim from q_proj
    let hidden = find_2d_tensor_shape(tensors, &["q_proj"]).map_or(0, |s| s[1]);
    (0, hidden)
}

#[cfg(test)]
#[path = "export_tests.rs"]
mod tests;
