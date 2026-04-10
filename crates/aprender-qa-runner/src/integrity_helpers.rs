/// Extract layer number from tensor name
/// Matches patterns like "model.layers.23.self_attn.q_proj.weight" -> 23
fn extract_layer_number(name: &str) -> Option<usize> {
    // Try different layer naming conventions
    let patterns = ["layers.", "h.", "transformer.h."];

    for pattern in patterns {
        if let Some(idx) = name.find(pattern) {
            let rest = &name[idx + pattern.len()..];
            let num_str: String = rest.chars().take_while(char::is_ascii_digit).collect();
            if let Ok(num) = num_str.parse::<usize>() {
                return Some(num);
            }
        }
    }
    None
}

/// Gate IDs for G0 integrity checks
pub mod gate_ids {
    /// Config.json exists and is readable
    pub const CONFIG: &str = "G0-INTEGRITY-CONFIG";
    /// Layer count in config matches tensor count
    pub const LAYERS: &str = "G0-INTEGRITY-LAYERS";
    /// Hidden size in config matches tensor shape
    pub const HIDDEN: &str = "G0-INTEGRITY-HIDDEN";
    /// Vocab size in config matches tensor shape
    pub const VOCAB: &str = "G0-INTEGRITY-VOCAB";
}


#[cfg(test)]
#[path = "integrity_tests.rs"]
mod integrity_tests;
