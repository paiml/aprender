//! Split a single safetensors file into shards + weight-map index.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use safetensors::tensor::{Dtype, SafeTensors, TensorView};

#[derive(Debug)]
pub enum ShardError {
    ParseSize(String),
    Io(std::io::Error),
    SafeTensors(safetensors::SafeTensorError),
    Invalid(String),
}

impl std::fmt::Display for ShardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShardError::ParseSize(m) => write!(f, "invalid --max-shard-size: {m}"),
            ShardError::Io(e) => write!(f, "i/o error: {e}"),
            ShardError::SafeTensors(e) => write!(f, "safetensors error: {e}"),
            ShardError::Invalid(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for ShardError {}

impl From<std::io::Error> for ShardError {
    fn from(e: std::io::Error) -> Self {
        ShardError::Io(e)
    }
}

impl From<safetensors::SafeTensorError> for ShardError {
    fn from(e: safetensors::SafeTensorError) -> Self {
        ShardError::SafeTensors(e)
    }
}

/// Result of a shard operation.
#[derive(Debug, Clone)]
pub struct ShardReport {
    pub shard_files: Vec<PathBuf>,
    pub index_path: PathBuf,
    pub total_size: u64,
    pub tensor_count: usize,
}

/// Element-size in bytes for every safetensors dtype.
fn dtype_size(dt: Dtype) -> usize {
    match dt {
        Dtype::BOOL | Dtype::U8 | Dtype::I8 | Dtype::F8_E4M3 | Dtype::F8_E5M2 => 1,
        Dtype::U16 | Dtype::I16 | Dtype::F16 | Dtype::BF16 => 2,
        Dtype::U32 | Dtype::I32 | Dtype::F32 => 4,
        Dtype::U64 | Dtype::I64 | Dtype::F64 => 8,
        _ => 1, // future-proof fallback; safetensors crate adds new dtypes occasionally
    }
}

fn tensor_byte_size(view: &TensorView<'_>) -> u64 {
    let elems: u64 = view.shape().iter().map(|d| *d as u64).product();
    elems.saturating_mul(dtype_size(view.dtype()) as u64)
}

/// Group tensors into shards via single-pass greedy bin packing.
///
/// Walks tensors in the original on-disk order. A new shard starts when the
/// current one is non-empty and adding the next tensor would exceed
/// `max_shard_size`. Tensors larger than `max_shard_size` are placed in their
/// own shard alone (HF transformers behaviour — single tensors never split).
fn plan_shards<'a>(names: &'a [&'a str], sizes: &[u64], max_shard_size: u64) -> Vec<Vec<usize>> {
    debug_assert_eq!(names.len(), sizes.len());
    let mut shards: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    let mut current_size: u64 = 0;

    for (i, &sz) in sizes.iter().enumerate() {
        let would_overflow = current_size.saturating_add(sz) > max_shard_size;
        if !current.is_empty() && would_overflow {
            shards.push(std::mem::take(&mut current));
            current_size = 0;
        }
        current.push(i);
        current_size = current_size.saturating_add(sz);
    }
    if !current.is_empty() {
        shards.push(current);
    }
    shards
}

/// Build a `model-NNNNN-of-MMMMM.safetensors` filename per HF convention.
fn shard_filename(index: usize, total: usize) -> String {
    format!("model-{index:05}-of-{total:05}.safetensors")
}

/// Produce a sorted, deterministic `model.safetensors.index.json` payload.
fn build_index_json(weight_map: &BTreeMap<String, String>, total_size: u64) -> String {
    let mut out = String::with_capacity(weight_map.len() * 80 + 64);
    out.push_str("{\n");
    out.push_str("  \"metadata\": {\n");
    out.push_str(&format!("    \"total_size\": {total_size}\n"));
    out.push_str("  },\n");
    out.push_str("  \"weight_map\": {\n");

    let mut first = true;
    for (name, shard) in weight_map {
        if !first {
            out.push_str(",\n");
        }
        first = false;
        out.push_str("    ");
        out.push_str(&json_string(name));
        out.push_str(": ");
        out.push_str(&json_string(shard));
    }
    if !first {
        out.push('\n');
    }
    out.push_str("  }\n");
    out.push_str("}\n");
    out
}

/// Minimal JSON string encoder for shard names and tensor keys.
fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Split `input` into shards + emit `model.safetensors.index.json`.
pub fn shard_safetensors_file(
    input: &Path,
    max_shard_size: u64,
    output_dir: &Path,
) -> Result<ShardReport, ShardError> {
    if max_shard_size == 0 {
        return Err(ShardError::Invalid(
            "--max-shard-size must be positive".to_string(),
        ));
    }
    if !input.is_file() {
        return Err(ShardError::Invalid(format!(
            "input is not a file: {}",
            input.display()
        )));
    }

    let bytes = fs::read(input)?;
    let st = SafeTensors::deserialize(&bytes)?;
    // safetensors 0.7: `names()` yields `&str` directly (it yielded `&String` in 0.4).
    let names: Vec<&str> = st.names();
    if names.is_empty() {
        return Err(ShardError::Invalid("input has no tensors".to_string()));
    }

    let views: Vec<TensorView<'_>> = names
        .iter()
        .map(|n| st.tensor(n))
        .collect::<Result<Vec<_>, _>>()?;
    let sizes: Vec<u64> = views.iter().map(tensor_byte_size).collect();
    let total_size: u64 = sizes.iter().sum();

    let plan = plan_shards(&names, &sizes, max_shard_size);
    let total_shards = plan.len();

    fs::create_dir_all(output_dir)?;

    let mut weight_map = BTreeMap::new();
    let mut shard_files = Vec::with_capacity(total_shards);

    for (idx, group) in plan.iter().enumerate() {
        let file_name = shard_filename(idx + 1, total_shards);
        let shard_path = output_dir.join(&file_name);

        let shard_tensors: Vec<(&str, TensorView<'_>)> = group
            .iter()
            .map(|&i| (names[i], views[i].clone()))
            .collect();

        let serialized =
            safetensors::serialize(shard_tensors, None).map_err(ShardError::SafeTensors)?;
        fs::write(&shard_path, &serialized)?;

        for &i in group {
            weight_map.insert(names[i].to_string(), file_name.clone());
        }
        shard_files.push(shard_path);
    }

    let index_path = output_dir.join("model.safetensors.index.json");
    let index_json = build_index_json(&weight_map, total_size);
    fs::write(&index_path, index_json)?;

    Ok(ShardReport {
        shard_files,
        index_path,
        total_size,
        tensor_count: names.len(),
    })
}

#[cfg(test)]
mod plan_tests {
    use super::{plan_shards, shard_filename};

    #[test]
    fn single_shard_when_under_limit() {
        let names = vec!["a", "b", "c"];
        let sizes = vec![10u64, 20, 30];
        let plan = plan_shards(&names, &sizes, 1000);
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0], vec![0, 1, 2]);
    }

    #[test]
    fn splits_when_over_limit() {
        let names = vec!["a", "b", "c", "d"];
        let sizes = vec![60u64, 60, 60, 60];
        let plan = plan_shards(&names, &sizes, 100);
        assert_eq!(plan.len(), 4);
    }

    #[test]
    fn oversized_tensor_alone() {
        let names = vec!["a", "big", "c"];
        let sizes = vec![10u64, 5000, 10];
        let plan = plan_shards(&names, &sizes, 100);
        // a => shard 0; big => shard 1 (over-limit, alone); c => shard 2
        assert_eq!(plan.len(), 3);
        assert_eq!(plan[1], vec![1]);
    }

    #[test]
    fn preserves_insertion_order() {
        let names = vec!["x", "y", "z"];
        let sizes = vec![50u64, 50, 50];
        let plan = plan_shards(&names, &sizes, 100);
        let flat: Vec<usize> = plan.into_iter().flatten().collect();
        assert_eq!(flat, vec![0, 1, 2]);
    }

    #[test]
    fn shard_filename_format() {
        assert_eq!(shard_filename(1, 3), "model-00001-of-00003.safetensors");
        assert_eq!(shard_filename(42, 100), "model-00042-of-00100.safetensors");
    }
}
