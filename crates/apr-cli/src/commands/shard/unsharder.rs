//! Reconstruct a single safetensors file from a sharded directory.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use safetensors::tensor::{SafeTensors, TensorView};

#[derive(Debug)]
pub enum UnshardError {
    Io(std::io::Error),
    SafeTensors(safetensors::SafeTensorError),
    Invalid(String),
}

impl std::fmt::Display for UnshardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UnshardError::Io(e) => write!(f, "i/o error: {e}"),
            UnshardError::SafeTensors(e) => write!(f, "safetensors error: {e}"),
            UnshardError::Invalid(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for UnshardError {}

impl From<std::io::Error> for UnshardError {
    fn from(e: std::io::Error) -> Self {
        UnshardError::Io(e)
    }
}

impl From<safetensors::SafeTensorError> for UnshardError {
    fn from(e: safetensors::SafeTensorError) -> Self {
        UnshardError::SafeTensors(e)
    }
}

/// Result of an unshard operation.
#[derive(Debug, Clone)]
pub struct UnshardReport {
    pub output: PathBuf,
    pub tensor_count: usize,
    pub shard_count: usize,
    pub total_size: u64,
}

/// Parsed `model.safetensors.index.json` (key → shard filename).
struct Index {
    weight_map: Vec<(String, String)>, // preserves insertion order from index.json
    total_size: Option<u64>,
}

/// Scan out `"total_size": <digits>` if the index declares one.
fn find_total_size(json: &str) -> Option<u64> {
    let pos = json.find("\"total_size\"")?;
    let after = &json[pos + 12..];
    let colon = after.find(':')?;
    let after_colon = after[colon + 1..].trim_start();
    let end = after_colon
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(after_colon.len());
    after_colon[..end].parse::<u64>().ok()
}

/// Return the body of the `weight_map` object, without its outer braces.
fn weight_map_body(json: &str) -> Result<&str, UnshardError> {
    let wm_start = json.find("\"weight_map\"").ok_or_else(|| {
        UnshardError::Invalid("missing 'weight_map' key in index.json".to_string())
    })?;
    let after_key = &json[wm_start + 12..];
    let obj_start = after_key.find('{').ok_or_else(|| {
        UnshardError::Invalid("malformed weight_map: missing opening brace".to_string())
    })?;
    let obj = &after_key[obj_start..];

    let obj_end = match_closing_brace(obj).ok_or_else(|| {
        UnshardError::Invalid("malformed weight_map: missing closing brace".to_string())
    })?;
    Ok(&obj[1..obj_end])
}

/// Index of the `}` closing the `{` that `s` starts with, or None if unbalanced.
fn match_closing_brace(s: &str) -> Option<usize> {
    let mut depth = 0i32;
    for (i, c) in s.char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    // A `}` at index 0 cannot close anything, so 0 stays a miss.
                    return (i != 0).then_some(i);
                }
            }
            _ => {}
        }
    }
    None
}

/// Split a `"key": "value"` comma list into pairs, skipping malformed entries.
fn parse_weight_map_entries(inner: &str) -> Vec<(String, String)> {
    let mut entries = Vec::new();
    for pair in inner.split(',') {
        let parts: Vec<&str> = pair.trim().splitn(2, ':').collect();
        if parts.len() != 2 {
            continue;
        }
        let key = unquote(parts[0].trim());
        let val = unquote(parts[1].trim());
        if !key.is_empty() && !val.is_empty() {
            entries.push((key, val));
        }
    }
    entries
}

fn parse_index_json(json: &str) -> Result<Index, UnshardError> {
    let json = json.trim();
    if !json.starts_with('{') || !json.ends_with('}') {
        return Err(UnshardError::Invalid(
            "index.json is not a JSON object".to_string(),
        ));
    }

    let total_size = find_total_size(json);
    let inner = weight_map_body(json)?;

    Ok(Index {
        weight_map: parse_weight_map_entries(inner),
        total_size,
    })
}

fn unquote(s: &str) -> String {
    let s = s.trim();
    if s.len() >= 2 && s.starts_with('"') && s.ends_with('"') {
        s[1..s.len() - 1].to_string()
    } else {
        s.to_string()
    }
}

/// Validate that a shard filename is a plain relative path with no parent
/// traversal — enforcement of B-05 invariant
/// "weight_map shard filenames are relative (no absolute paths, no ..)".
fn validate_shard_path(name: &str) -> Result<(), UnshardError> {
    let p = Path::new(name);
    if p.is_absolute() {
        return Err(UnshardError::Invalid(format!(
            "shard filename must be relative: {name}"
        )));
    }
    for comp in p.components() {
        use std::path::Component;
        match comp {
            Component::Normal(_) => {}
            Component::ParentDir => {
                return Err(UnshardError::Invalid(format!(
                    "shard filename contains '..': {name}"
                )));
            }
            Component::RootDir | Component::Prefix(_) => {
                return Err(UnshardError::Invalid(format!(
                    "shard filename has root component: {name}"
                )));
            }
            Component::CurDir => {}
        }
    }
    Ok(())
}

/// Read `model.safetensors.index.json` out of a shard directory.
fn read_index(input_dir: &Path) -> Result<Index, UnshardError> {
    if !input_dir.is_dir() {
        return Err(UnshardError::Invalid(format!(
            "input is not a directory: {}",
            input_dir.display()
        )));
    }
    let index_path = input_dir.join("model.safetensors.index.json");
    if !index_path.is_file() {
        return Err(UnshardError::Invalid(format!(
            "missing model.safetensors.index.json in {}",
            input_dir.display()
        )));
    }

    let index_text = fs::read_to_string(&index_path)?;
    let index = parse_index_json(&index_text)?;
    if index.weight_map.is_empty() {
        return Err(UnshardError::Invalid(
            "weight_map is empty in index.json".to_string(),
        ));
    }
    Ok(index)
}

/// Load each unique shard exactly once, preserving the first-seen order so that
/// within a shard tensors land in the order safetensors stores them.
fn load_shards(
    input_dir: &Path,
    weight_map: &[(String, String)],
) -> Result<(Vec<String>, HashMap<String, Vec<u8>>), UnshardError> {
    // Pass 1 — validate EVERY weight_map entry before touching the disk, so a
    // traversal attempt anywhere in the index is rejected before any file read.
    let mut shard_order: Vec<String> = Vec::new();
    for (_tensor, shard) in weight_map {
        validate_shard_path(shard)?;
        if !shard_order.iter().any(|s| s == shard) {
            shard_order.push(shard.clone());
        }
    }

    // Pass 2 — read each unique shard once.
    let mut shard_bytes: HashMap<String, Vec<u8>> = HashMap::new();
    for shard in &shard_order {
        let shard_path = input_dir.join(shard);
        if !shard_path.is_file() {
            return Err(UnshardError::Invalid(format!(
                "shard file missing on disk: {}",
                shard_path.display()
            )));
        }
        shard_bytes.insert(shard.clone(), fs::read(&shard_path)?);
    }

    Ok((shard_order, shard_bytes))
}

/// Gather every tensor named in the weight_map, in insertion order, verifying
/// each one is actually present in the shard the index claims holds it.
///
/// Returns the views alongside their total byte count.
fn collect_views<'a>(
    by_shard: &'a HashMap<&str, SafeTensors<'a>>,
    weight_map: &[(String, String)],
) -> Result<(Vec<(String, TensorView<'a>)>, u64), UnshardError> {
    let mut all_views: Vec<(String, TensorView<'a>)> = Vec::with_capacity(weight_map.len());
    let mut total_bytes: u64 = 0;

    for (tensor_name, shard_name) in weight_map {
        let st = by_shard
            .get(shard_name.as_str())
            .ok_or_else(|| UnshardError::Invalid(format!("shard not loaded: {shard_name}")))?;
        let view = st.tensor(tensor_name).map_err(|e| {
            UnshardError::Invalid(format!(
                "tensor '{tensor_name}' declared in weight_map but not present in shard {shard_name}: {e}"
            ))
        })?;
        total_bytes = total_bytes.saturating_add(view.data().len() as u64);
        all_views.push((tensor_name.clone(), view));
    }

    Ok((all_views, total_bytes))
}

/// Reconstruct a single safetensors file from a sharded directory.
pub fn unshard_safetensors_dir(
    input_dir: &Path,
    output: &Path,
) -> Result<UnshardReport, UnshardError> {
    let index = read_index(input_dir)?;
    let (shard_order, shard_bytes) = load_shards(input_dir, &index.weight_map)?;

    let mut by_shard: HashMap<&str, SafeTensors<'_>> = HashMap::new();
    for shard in &shard_order {
        let bytes = shard_bytes
            .get(shard)
            .ok_or_else(|| UnshardError::Invalid(format!("shard not loaded: {shard}")))?;
        by_shard.insert(shard.as_str(), SafeTensors::deserialize(bytes)?);
    }

    let (all_views, total_bytes) = collect_views(&by_shard, &index.weight_map)?;

    if let Some(declared) = index.total_size {
        if declared != total_bytes {
            return Err(UnshardError::Invalid(format!(
                "index.json total_size {declared} disagrees with shard contents {total_bytes}"
            )));
        }
    }

    let view_refs: Vec<(&str, TensorView<'_>)> = all_views
        .iter()
        .map(|(n, v)| (n.as_str(), v.clone()))
        .collect();

    let serialized = safetensors::serialize(view_refs, None).map_err(UnshardError::SafeTensors)?;

    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    fs::write(output, &serialized)?;

    Ok(UnshardReport {
        output: output.to_path_buf(),
        tensor_count: index.weight_map.len(),
        shard_count: shard_order.len(),
        total_size: total_bytes,
    })
}

#[cfg(test)]
mod parser_tests {
    use super::{parse_index_json, validate_shard_path};

    #[test]
    fn parses_minimal_index() {
        let json = r#"{
            "metadata": {"total_size": 1024},
            "weight_map": {
                "a.weight": "model-00001-of-00002.safetensors",
                "b.weight": "model-00002-of-00002.safetensors"
            }
        }"#;
        let idx = parse_index_json(json).expect("parse");
        assert_eq!(idx.total_size, Some(1024));
        assert_eq!(idx.weight_map.len(), 2);
        assert_eq!(idx.weight_map[0].0, "a.weight");
    }

    #[test]
    fn rejects_traversal() {
        assert!(validate_shard_path("../escape.safetensors").is_err());
    }

    #[test]
    fn rejects_absolute() {
        assert!(validate_shard_path("/etc/passwd").is_err());
    }

    #[test]
    fn accepts_relative_filename() {
        assert!(validate_shard_path("model-00001-of-00002.safetensors").is_ok());
    }
}
