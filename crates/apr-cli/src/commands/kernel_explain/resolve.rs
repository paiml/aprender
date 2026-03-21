//! Family resolution: alias tables, normalization, and family/architecture lookup.

use super::family::load_families;
use super::{FamilyInfo, FAMILY_ALIASES};
use std::path::Path;

/// Known HuggingFace architecture class names for common aliases.
/// Used to display the correct architecture instead of the raw alias string.
pub(crate) const ALIAS_ARCHITECTURES: &[(&str, &str)] = &[
    ("bloom", "BloomForCausalLM"),
    ("bloomz", "BloomForCausalLM"),
    ("bloom_560m", "BloomForCausalLM"),
    ("falcon", "FalconForCausalLM"),
    ("mixtral", "MixtralForCausalLM"),
    ("phi3", "Phi3ForCausalLM"),
    ("phi3small", "Phi3SmallForCausalLM"),
    ("codellama", "LlamaForCausalLM"),
    ("vicuna", "LlamaForCausalLM"),
    ("solar", "LlamaForCausalLM"),
    ("nemotron", "LlamaForCausalLM"),
    ("olmo", "OlmoForCausalLM"),
    ("olmo2", "Olmo2ForCausalLM"),
    ("granite", "GraniteForCausalLM"),
    ("internlm2", "InternLM2ForCausalLM"),
    ("yi", "LlamaForCausalLM"),
    ("baichuan", "BaichuanForCausalLM"),
    ("stablelm", "StableLmForCausalLM"),
    ("starcoder2", "Starcoder2ForCausalLM"),
    ("codestral", "MistralForCausalLM"),
    ("mathstral", "MistralForCausalLM"),
    ("pixtral", "LlavaMistralForCausalLM"),
    ("zephyr", "MistralForCausalLM"),
    ("openchat", "MistralForCausalLM"),
    ("openhermes", "MistralForCausalLM"),
    ("qwen2_moe", "Qwen2MoeForCausalLM"),
    ("qwen2moe", "Qwen2MoeForCausalLM"),
    ("qwen3_moe", "Qwen3MoeForCausalLM"),
    ("qwen3moe", "Qwen3MoeForCausalLM"),
    ("deepseek_v2", "DeepseekV2ForCausalLM"),
    ("deepseekv2", "DeepseekV2ForCausalLM"),
    ("qwq", "Qwen2ForCausalLM"),
    ("bigscience", "BloomForCausalLM"),
    ("qwen3_next", "Qwen3ForCausalLM"),
    ("qwen3next", "Qwen3ForCausalLM"),
    ("smollm", "LlamaForCausalLM"),
    ("smollm2", "LlamaForCausalLM"),
    // GPT variants
    ("gpt_neo", "GPTNeoForCausalLM"),
    ("gptneo", "GPTNeoForCausalLM"),
    ("gpt_neox", "GPTNeoXForCausalLM"),
    ("gptneox", "GPTNeoXForCausalLM"),
    ("gpt_j", "GPTJForCausalLM"),
    ("gptj", "GPTJForCausalLM"),
    ("gpt_bigcode", "GPTBigCodeForCausalLM"),
    ("gptbigcode", "GPTBigCodeForCausalLM"),
    ("starcoder1", "GPTBigCodeForCausalLM"),
    ("codegen", "CodeGenForCausalLM"),
    ("xglm", "XGLMForCausalLM"),
    ("opt", "OPTForCausalLM"),
    ("galactica", "OPTForCausalLM"),
    ("roberta", "RobertaForMaskedLM"),
    ("deberta", "DebertaV2ForMaskedLM"),
    ("electra", "ElectraForPreTraining"),
    ("distilbert", "DistilBertModel"),
    ("tinyllama", "LlamaForCausalLM"),
    ("phi4", "Phi3ForCausalLM"),
    ("codegemma", "CodeGemmaForCausalLM"),
    ("gemma2", "Gemma2ForCausalLM"),
    ("gemma3", "Gemma3ForCausalLM"),
];

/// Get all family aliases for display in help/error messages.
pub fn family_aliases() -> &'static [(&'static str, &'static str)] {
    FAMILY_ALIASES
}

/// Normalize input: lowercase, trim, replace hyphens/dots with underscores.
/// E.g., "falcon-h1" -> "falcon_h1", "qwen3.5" -> "qwen3_5"
pub(crate) fn normalize_input(input: &str) -> String {
    input
        .to_lowercase()
        .trim()
        .replace('-', "_")
        .replace('.', "_")
}

/// Secondary normalization: strip all separators between name and version.
/// E.g., "phi_3" -> "phi3", "gpt_2" -> "gpt2", "rwkv_7" -> "rwkv7"
/// Returns None if same as input.
pub(crate) fn compact_input(normalized: &str) -> Option<String> {
    let compact = normalized.replace('_', "");
    if compact != normalized {
        Some(compact)
    } else {
        None
    }
}

/// Resolve a family string or architecture string to `FamilyInfo`.
pub fn resolve_family(input: &str) -> Option<FamilyInfo> {
    let lower = input.to_lowercase();
    let lower = lower.trim();
    if lower.is_empty() {
        return None;
    }

    // Strip non-ASCII characters (emoji, CJK, etc.) — family names are ASCII-only
    let ascii_only: String = lower.chars().filter(|c| c.is_ascii()).collect();
    let ascii_only = ascii_only.trim();
    if ascii_only.is_empty() {
        return None;
    }
    // Use the ASCII-stripped version for all matching
    let lower = ascii_only;

    let families = load_families();
    // Normalized form for matching (hyphens/dots -> underscores)
    let normalized = normalize_input(lower);
    // Compact form for matching (all separators removed: phi_3 -> phi3)
    let compact = compact_input(&normalized);

    // Direct family name match (try raw, normalized, compact, and cross-compact)
    if let Some(f) = families.iter().find(|f| {
        f.family == lower
            || f.family == normalized
            || compact.as_deref().is_some_and(|c| f.family == c)
            // Cross-compact: compare compact forms of both sides
            // e.g., input "qwen-3-5" compact="qwen35", family "qwen3_5" compact="qwen35"
            || compact
                .as_deref()
                .is_some_and(|c| compact_input(&f.family).as_deref() == Some(c))
    }) {
        return Some(f.clone());
    }

    // Alias match (model types sharing kernel pipeline with existing family)
    // Try raw lowercase -> normalized -> compact forms
    let alias_match = FAMILY_ALIASES
        .iter()
        .find(|(alias, _)| *alias == lower)
        .or_else(|| {
            FAMILY_ALIASES
                .iter()
                .find(|(alias, _)| *alias == normalized.as_str())
        })
        .or_else(|| {
            compact
                .as_deref()
                .and_then(|c| FAMILY_ALIASES.iter().find(|(alias, _)| *alias == c))
        });
    if let Some((matched_alias, target)) = alias_match {
        if let Some(f) = families.iter().find(|f| f.family == *target) {
            let mut aliased = f.clone();
            aliased.display_name = format!("{} (via {} kernel pipeline)", matched_alias, f.family);
            return Some(aliased);
        }
    }

    // Architecture match (e.g., "Qwen2ForCausalLM")
    if let Some(f) = families.iter().find(|f| {
        f.architectures
            .iter()
            .any(|a| a.to_lowercase() == lower || a == input)
    }) {
        return Some(f.clone());
    }

    // Architecture string -> model_type extraction -> alias re-check
    // e.g., "GraniteForCausalLM" -> "granite" -> alias -> llama
    let stripped = strip_arch_suffix(lower);
    if stripped != lower {
        // Try alias with stripped name
        if let Some((_, target)) = FAMILY_ALIASES.iter().find(|(alias, _)| *alias == stripped) {
            if let Some(f) = families.iter().find(|f| f.family == *target) {
                let mut aliased = f.clone();
                aliased.display_name = format!("{stripped} (via {target} kernel pipeline)");
                return Some(aliased);
            }
        }
        // Try direct family match with stripped name
        if let Some(f) = families.iter().find(|f| f.family == stripped) {
            return Some(f.clone());
        }
    }

    // Partial match against families (e.g., "qwen" matches "qwen2")
    // Try both normalized (with underscores) and compact (without) forms
    // Each form must be >= 3 chars to avoid spurious matches (e.g., "ab" -> "stablelm")
    let search_forms: Vec<&str> = {
        let mut v = vec![];
        if normalized.len() >= 3 {
            v.push(normalized.as_str());
        }
        if let Some(ref c) = compact {
            if c.len() >= 3 {
                v.push(c.as_str());
            }
        }
        v
    };
    if !search_forms.is_empty() {
        // Two-pass partial matching:
        // Pass 1: search starts_with alias (search is MORE specific, e.g., "phi3mini" starts_with "phi3")
        //   -> Aliases win because they're exact subtypes (phi3 -> llama, not phi family)
        // Pass 2: family starts_with search (search is LESS specific, e.g., "qwen" prefix of "qwen2")
        //   -> Families win because they're direct matches, not accidental alias prefixes

        // Pass 1: search starts_with alias (search is longer/more specific than alias)
        for search in &search_forms {
            if let Some((matched_alias, target)) = FAMILY_ALIASES
                .iter()
                .find(|(alias, _)| search.starts_with(alias))
            {
                if let Some(f) = families.iter().find(|f| f.family == *target) {
                    let mut aliased = f.clone();
                    aliased.display_name =
                        format!("{} (via {} kernel pipeline)", matched_alias, f.family);
                    return Some(aliased);
                }
            }
        }
        // Pass 2: family starts_with search (search is a prefix of family name)
        for search in &search_forms {
            if let Some(f) = families
                .iter()
                .find(|f| f.family.starts_with(*search) || search.starts_with(f.family.as_str()))
            {
                return Some(f.clone());
            }
        }
        // Pass 3: alias starts_with search (search is a prefix of alias name)
        for search in &search_forms {
            if let Some((matched_alias, target)) = FAMILY_ALIASES
                .iter()
                .find(|(alias, _)| alias.starts_with(*search))
            {
                if let Some(f) = families.iter().find(|f| f.family == *target) {
                    let mut aliased = f.clone();
                    aliased.display_name =
                        format!("{} (via {} kernel pipeline)", matched_alias, f.family);
                    return Some(aliased);
                }
            }
        }
    }

    None
}

/// Strip HuggingFace architecture suffixes to get model type.
/// E.g., "graniteforCausalLM" -> "granite", "phi3smallforcausallm" -> "phi3small"
pub(crate) fn strip_arch_suffix(s: &str) -> &str {
    // All known suffixes (lowercase). Order matters: longest first.
    const SUFFIXES: &[&str] = &["forconditionalgeneration", "forcausallm", "model"];
    for suffix in SUFFIXES {
        if let Some(prefix) = s.strip_suffix(suffix) {
            if !prefix.is_empty() {
                return prefix;
            }
        }
    }
    s
}

/// Try to resolve family from a config.json file.
/// Returns None if model_type is absent/unresolvable. Returns Err for structural issues.
pub fn resolve_from_config_json(path: &Path) -> Option<FamilyInfo> {
    let content = std::fs::read_to_string(path).ok()?;

    // Reject JSON arrays — config.json must be an object
    let trimmed = content.trim();
    if trimmed.starts_with('[') {
        return None;
    }

    // Parse model_type from config.json
    let model_type = super::config::extract_json_string(&content, "model_type");

    // If no model_type, try architectures field as fallback
    let model_type = match model_type {
        Some(mt) => mt,
        None => {
            // Extract first architecture and strip suffix to get family name
            let arch =
                super::config::extract_json_string(&content, "architectures").or_else(|| {
                    // architectures is a JSON array — manually extract first element
                    let pos = content.find("\"architectures\"")?;
                    let after = &content[pos..];
                    let bracket = after.find('[')?;
                    let inner = &after[bracket + 1..];
                    let quote_start = inner.find('"')?;
                    let rest = &inner[quote_start + 1..];
                    let quote_end = rest.find('"')?;
                    Some(rest[..quote_end].to_string())
                })?;
            // Convert "LlamaForCausalLM" -> "llama"
            let lower = arch.to_lowercase();
            strip_arch_suffix(&lower).to_string()
        }
    };

    resolve_family(&model_type)
}
