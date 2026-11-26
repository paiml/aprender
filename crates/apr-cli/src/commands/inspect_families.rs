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
    /// Distinct language-model layer indices the index names.
    pub layers_in_index: usize,
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
    ("fc", "fc"),
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
            .and_then(byte_count),
        tensor_count: weight_map.len(),
        layers_in_index: weight_map
            .keys()
            .filter_map(|n| language_layer(n))
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
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
        "layers_in_index": r.layers_in_index,
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

/// A byte count written as an integer or as a whole float (`4892365451008.0`, which is
/// how the published 2.4T index writes `total_size`).
fn byte_count(v: &serde_json::Value) -> Option<u64> {
    v.as_u64().or_else(|| {
        let f = v.as_f64()?;
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        (f.is_finite() && f >= 0.0 && f.fract() == 0.0 && f < u64::MAX as f64).then_some(f as u64)
    })
}

/// The language-model layer a tensor belongs to (`model[.language_model].layers.{n}.`).
fn language_layer(name: &str) -> Option<u64> {
    let rest = name
        .strip_prefix("model.language_model.")
        .or_else(|| name.strip_prefix("model."))?;
    rest.strip_prefix("layers.")?
        .split_once('.')?
        .0
        .parse()
        .ok()
}

/// The two files `--no-weights` reads.
const NO_WEIGHTS_FILES: [&str; 2] = ["config.json", "model.safetensors.index.json"];

/// `org/repo` (optionally `hf://org/repo`): exactly two non-empty segments of repo-id
/// characters, and not a filesystem-looking path.
fn repo_id(arg: &str) -> Option<&str> {
    let id = arg.strip_prefix("hf://").unwrap_or(arg);
    let seg = |s: &str| {
        !s.is_empty()
            && s != "."
            && s != ".."
            && s.bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
    };
    let (org, repo) = id.split_once('/')?;
    (seg(org) && seg(repo)).then_some(id)
}

/// Where `--no-weights` reads its two files from: an existing directory (or the directory
/// of an existing file), else the `apr pull` cache directory of a bare repo id. A file the
/// cache lacks is fetched from the repo's `resolve/main` into the cache; a failed fetch is
/// an error naming the repo and the file, and leaves nothing behind.
///
/// # Errors
/// An argument that is neither an existing path nor `org/repo`; an uncached file that
/// cannot be fetched (`--offline`, no network, 404, gated).
pub(crate) fn no_weights_dir(
    path: &Path,
    cache_dir: &dyn Fn(&str) -> Result<std::path::PathBuf, CliError>,
    fetch: &dyn Fn(&str, &Path) -> Result<(), CliError>,
) -> Result<std::path::PathBuf, CliError> {
    if path.is_file() {
        return Ok(path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf());
    }
    if path.is_dir() {
        return Ok(path.to_path_buf());
    }
    let arg = path.to_string_lossy();
    let Some(repo) = repo_id(&arg) else {
        return Err(CliError::ValidationFailed(format!(
            "--no-weights: {arg} is neither an existing directory nor a Hugging Face repo id (org/repo)"
        )));
    };
    let dir = cache_dir(repo)?;
    for f in NO_WEIGHTS_FILES {
        if dir.join(f).is_file() {
            continue;
        }
        // Directories this call creates, outermost last, so a failure can remove them.
        let mut made: Vec<&Path> = dir.ancestors().take_while(|a| !a.exists()).collect();
        made.retain(|a| !a.as_os_str().is_empty());
        let part = dir.join(format!("{f}.part"));
        let url = format!("https://huggingface.co/{repo}/resolve/main/{f}");
        let got = std::fs::create_dir_all(&dir)
            .map_err(CliError::from)
            .and_then(|()| fetch(&url, &part))
            .and_then(|()| std::fs::rename(&part, dir.join(f)).map_err(CliError::from));
        if let Err(e) = got {
            let _ = std::fs::remove_file(&part);
            for d in made {
                let _ = std::fs::remove_dir(d);
            }
            let msg = format!(
                "--no-weights: {f} of {repo} is not in the cache ({}) and could not be fetched: {e}",
                dir.display()
            );
            return Err(match e {
                CliError::HttpNotFound(_) => CliError::HttpNotFound(msg),
                _ => CliError::NetworkError(msg),
            });
        }
    }
    Ok(dir)
}

/// `apr inspect DIR|ORG/REPO --no-weights`: a bare repo id is read from the `apr pull`
/// cache (see [`no_weights_dir`]). `DIR` holds `config.json` and
/// `model.safetensors.index.json` (a path to either file is accepted). Reads those
/// two files and nothing else.
///
/// # Errors
/// A missing or malformed file, or any UNNAMED tensor (printed first).
pub(crate) fn run_no_weights(path: &Path, json: bool) -> Result<(), CliError> {
    let dir = no_weights_dir(
        path,
        &crate::commands::pull::resolve_cache_dir_for_ref,
        &crate::commands::pull::download_file,
    )?;
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
