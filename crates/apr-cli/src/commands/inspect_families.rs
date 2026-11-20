//! `apr inspect --no-weights` (#7330, parent #7318): name every tensor family of a
//! checkpoint from `config.json` + `model.safetensors.index.json` ONLY.
//!
//! Nothing here opens a shard. The 2.4T checkpoint is ~4.9 TB bf16 and fits no
//! fleet box, so the inspection must be answerable from the two small JSON files.
//! A tensor this table cannot name is reported as UNNAMED and fails the run: a
//! family list that silently drops what it does not recognise is a confident wrong
//! answer.

use crate::error::CliError;
use std::collections::BTreeMap;
use std::path::Path;

/// One named family: how many index entries it holds and one example name.
#[derive(Debug)]
pub(crate) struct FamilyEntry {
    pub family: String,
    pub count: usize,
    pub example: String,
}

#[derive(Debug)]
pub(crate) struct FamilyReport {
    pub model_type: String,
    pub num_hidden_layers: Option<u64>,
    pub num_experts: Option<u64>,
    pub total_size_bytes: Option<u64>,
    pub tensor_count: usize,
    pub families: Vec<FamilyEntry>,
    pub unnamed: Vec<String>,
}

/// Groups inside `layers.{n}.`, longest first so `self_attn.indexer` beats `self_attn`
/// and `mlp.shared_expert_gate` beats `mlp.shared_expert`.
const LAYER_GROUPS: &[&str] = &[
    "self_attn.indexer",
    "self_attn",
    "linear_attn",
    "mlp.experts",
    "mlp.shared_expert_gate",
    "mlp.shared_expert",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
    "mlp.gate",
    "attn_hyper_connection",
    "mlp_hyper_connection",
    "ple",
    "input_layernorm",
    "post_attention_layernorm",
];

const SCOPE_TOP: &[(&str, &str)] = &[
    ("embed_tokens", "embed_tokens"),
    ("norm", "final_norm"),
    ("hyper_connection_mixer", "hyper_connection_mixer"),
    ("fc_embedding", "fc_embedding"),
    ("fc_hidden", "fc_hidden"),
    ("pre_fc_norm_embedding", "pre_fc_norm_embedding"),
    ("pre_fc_norm_hidden", "pre_fc_norm_hidden"),
];

const VISION_TOP: &[&str] = &["blocks", "merger", "patch_embed", "pos_embed"];

fn first_seg(s: &str) -> &str {
    s.split('.').next().unwrap_or("")
}

fn in_group(rest: &str, group: &str) -> bool {
    rest == group || rest.strip_prefix(group).is_some_and(|t| t.starts_with('.'))
}

/// The family of one tensor name, or `None` when the table has no name for it.
pub(crate) fn classify(name: &str) -> Option<String> {
    if name == "lm_head.weight" {
        return Some("head.lm_head".to_string());
    }
    let (scope, rest) = if let Some(r) = name.strip_prefix("mtp.") {
        ("mtp", r)
    } else if let Some(r) = name.strip_prefix("model.visual.") {
        let top = first_seg(r);
        return VISION_TOP.contains(&top).then(|| format!("vision.{top}"));
    } else if let Some(r) = name
        .strip_prefix("model.language_model.")
        .or_else(|| name.strip_prefix("model."))
    {
        ("language", r)
    } else {
        return None;
    };
    if let Some(after) = rest.strip_prefix("layers.") {
        let (idx, inner) = after.split_once('.')?;
        if idx.is_empty() || !idx.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        let group = LAYER_GROUPS.iter().find(|g| in_group(inner, g))?;
        return Some(format!("{scope}.{group}"));
    }
    let top = first_seg(rest);
    SCOPE_TOP
        .iter()
        .find(|(k, _)| *k == top)
        .map(|(_, fam)| format!("{scope}.{fam}"))
}

fn bad(what: &str, e: impl std::fmt::Display) -> CliError {
    CliError::ValidationFailed(format!("{what}: {e}"))
}

fn text_config(cfg: &serde_json::Value) -> &serde_json::Value {
    cfg.get("text_config").unwrap_or(cfg)
}

/// Build the report from the two JSON documents (as text). Pure: no filesystem.
pub(crate) fn report(config: &str, index: &str) -> Result<FamilyReport, CliError> {
    let cfg: serde_json::Value =
        serde_json::from_str(config).map_err(|e| bad("config.json is not valid JSON", e))?;
    let idx: serde_json::Value = serde_json::from_str(index)
        .map_err(|e| bad("model.safetensors.index.json is not valid JSON", e))?;
    let weight_map = idx
        .get("weight_map")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| {
            CliError::ValidationFailed(
                "model.safetensors.index.json has no `weight_map` object".to_string(),
            )
        })?;
    let tc = text_config(&cfg);
    let model_type = tc
        .get("model_type")
        .or_else(|| cfg.get("model_type"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown")
        .to_string();
    let num = |keys: &[&str]| {
        keys.iter()
            .find_map(|k| tc.get(*k).and_then(serde_json::Value::as_u64))
    };
    let mut fam: BTreeMap<String, (usize, String)> = BTreeMap::new();
    let mut unnamed = Vec::new();
    for name in weight_map.keys() {
        match classify(name) {
            Some(f) => {
                let e = fam.entry(f).or_insert_with(|| (0, name.clone()));
                e.0 += 1;
            }
            None => unnamed.push(name.clone()),
        }
    }
    Ok(FamilyReport {
        model_type,
        num_hidden_layers: num(&["num_hidden_layers"]),
        num_experts: num(&["num_experts", "num_local_experts"]),
        total_size_bytes: idx
            .get("metadata")
            .and_then(|m| m.get("total_size"))
            .and_then(serde_json::Value::as_u64),
        tensor_count: weight_map.len(),
        families: fam
            .into_iter()
            .map(|(family, (count, example))| FamilyEntry {
                family,
                count,
                example,
            })
            .collect(),
        unnamed,
    })
}

pub(crate) fn render_json(r: &FamilyReport) -> serde_json::Value {
    serde_json::json!({
        "weights_read": false,
        "model_type": r.model_type,
        "num_hidden_layers": r.num_hidden_layers,
        "num_experts": r.num_experts,
        "total_size_bytes": r.total_size_bytes,
        "tensor_count": r.tensor_count,
        "families": r.families.iter().map(|f| serde_json::json!({
            "family": f.family, "count": f.count, "example": f.example,
        })).collect::<Vec<_>>(),
        "unnamed": r.unnamed,
    })
}

pub(crate) fn render_text(r: &FamilyReport) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    let _ = writeln!(
        s,
        "model_type {} | layers {} | experts {} | {} tensors | weights not read",
        r.model_type,
        r.num_hidden_layers.map_or("?".into(), |n| n.to_string()),
        r.num_experts.map_or("?".into(), |n| n.to_string()),
        r.tensor_count
    );
    for f in &r.families {
        let _ = writeln!(s, "  {:<40} {:>6}  e.g. {}", f.family, f.count, f.example);
    }
    if !r.unnamed.is_empty() {
        let _ = writeln!(s, "UNNAMED ({}):", r.unnamed.len());
        for n in &r.unnamed {
            let _ = writeln!(s, "  {n}");
        }
    }
    s
}

/// `apr inspect DIR --no-weights`: `DIR` holds `config.json` and
/// `model.safetensors.index.json` (a path to either file is accepted). Reads those
/// two files and nothing else.
///
/// # Errors
/// A missing or malformed file, or any UNNAMED tensor (printed first).
pub(crate) fn run_no_weights(path: &Path, json: bool) -> Result<(), CliError> {
    let dir = if path.is_file() {
        path.parent().unwrap_or_else(|| Path::new("."))
    } else {
        path
    };
    let read = |f: &str| {
        std::fs::read_to_string(dir.join(f))
            .map_err(|e| bad(&format!("--no-weights needs {f} in {}", dir.display()), e))
    };
    let r = report(
        &read("config.json")?,
        &read("model.safetensors.index.json")?,
    )?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&render_json(&r)).map_err(|e| bad("json", e))?
        );
    } else {
        print!("{}", render_text(&r));
    }
    if r.unnamed.is_empty() {
        Ok(())
    } else {
        Err(CliError::ValidationFailed(format!(
            "{} tensor(s) have no family name (#7330)",
            r.unnamed.len()
        )))
    }
}

#[cfg(test)]
#[path = "inspect_families_tests.rs"]
mod tests;
