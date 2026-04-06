//! BLAKE3 hashing for playbook cache keys (PB-003)
//!
//! Provides deterministic hashing for files, directories, parameters, and commands.
//! All hashes are formatted as `"blake3:{hex}"`.
//! Uses streaming I/O to avoid OOM on large files.

use super::types::yaml_value_to_string;
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

/// Hash a single file's contents via BLAKE3 (streaming)
pub fn hash_file(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("failed to open file for hashing: {}", path.display()))?;
    let mut hasher = blake3::Hasher::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = file
            .read(&mut buf)
            .with_context(|| format!("failed to read file: {}", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let hash = hasher.finalize();
    Ok(format!("blake3:{}", hash.to_hex()))
}

/// Result of hashing a directory
#[derive(Debug, Clone)]
pub struct DirHashResult {
    pub hash: String,
    pub file_count: u64,
    pub total_bytes: u64,
}

/// Stream a file's contents into a hasher, returning its byte size.
fn stream_into_hasher(path: &Path, hasher: &mut blake3::Hasher) -> Result<u64> {
    let meta =
        std::fs::metadata(path).with_context(|| format!("failed to stat: {}", path.display()))?;
    let mut file =
        std::fs::File::open(path).with_context(|| format!("failed to open: {}", path.display()))?;
    let mut buf = [0u8; 65536];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(meta.len())
}

/// Hash a directory by walking files in sorted order (streaming I/O)
pub fn hash_directory(path: &Path) -> Result<DirHashResult> {
    if !path.is_dir() {
        let mut hasher = blake3::Hasher::new();
        let size = stream_into_hasher(path, &mut hasher)?;
        let hash = hasher.finalize();
        return Ok(DirHashResult {
            hash: format!("blake3:{}", hash.to_hex()),
            file_count: 1,
            total_bytes: size,
        });
    }

    let mut entries: Vec<std::path::PathBuf> = Vec::new();
    collect_files_sorted(path, &mut entries)?;

    let mut hasher = blake3::Hasher::new();
    let mut file_count = 0u64;
    let mut total_bytes = 0u64;

    for entry in &entries {
        let rel = entry.strip_prefix(path).unwrap_or(entry);
        hasher.update(rel.to_string_lossy().as_bytes());
        total_bytes += stream_into_hasher(entry, &mut hasher)?;
        file_count += 1;
    }

    let hash = hasher.finalize();
    Ok(DirHashResult { hash: format!("blake3:{}", hash.to_hex()), file_count, total_bytes })
}

fn collect_files_sorted(dir: &Path, out: &mut Vec<std::path::PathBuf>) -> Result<()> {
    let mut entries: Vec<std::path::PathBuf> = Vec::new();

    for entry in
        std::fs::read_dir(dir).with_context(|| format!("failed to read dir: {}", dir.display()))?
    {
        let entry = entry?;
        let ft = entry.file_type()?;
        // Skip symlinks to avoid circular references and symlink attacks
        if ft.is_symlink() {
            continue;
        }
        entries.push(entry.path());
    }

    // Sort for deterministic ordering
    entries.sort();

    for entry in entries {
        if entry.is_dir() {
            collect_files_sorted(&entry, out)?;
        } else {
            out.push(entry);
        }
    }

    Ok(())
}

/// Hash a dependency (file or directory)
pub fn hash_dep(path: &Path) -> Result<DirHashResult> {
    hash_directory(path)
}

/// Hash the parameter set relevant to a stage
///
/// Uses the union of declared param keys and template-extracted refs.
/// Sorted by key for determinism.
pub fn hash_params(
    global_params: &HashMap<String, serde_yaml_ng::Value>,
    referenced_keys: &[String],
) -> Result<String> {
    let mut pairs: Vec<(String, String)> = Vec::new();

    for key in referenced_keys {
        if let Some(val) = global_params.get(key) {
            pairs.push((key.clone(), yaml_value_to_string(val)));
        }
    }

    pairs.sort_by(|a, b| a.0.cmp(&b.0));

    let mut hasher = blake3::Hasher::new();
    for (k, v) in &pairs {
        hasher.update(k.as_bytes());
        hasher.update(b"=");
        hasher.update(v.as_bytes());
        hasher.update(b"\n");
    }

    let hash = hasher.finalize();
    Ok(format!("blake3:{}", hash.to_hex()))
}

/// Extract param keys referenced in a command template (UTF-8 safe)
pub fn extract_param_refs(cmd: &str) -> Vec<String> {
    let mut keys = Vec::new();
    let mut pos = 0;

    while pos < cmd.len() {
        if cmd[pos..].starts_with("{{") {
            let start = pos + 2;
            if let Some(end_offset) = cmd[start..].find("}}") {
                let ref_str = cmd[start..start + end_offset].trim();
                if let Some(key) = ref_str.strip_prefix("params.") {
                    if !keys.contains(&key.to_string()) {
                        keys.push(key.to_string());
                    }
                }
                pos = start + end_offset + 2;
            } else {
                pos += 2;
            }
        } else {
            let ch = cmd[pos..].chars().next().expect("iterator empty");
            pos += ch.len_utf8();
        }
    }

    keys
}

/// Compute the effective param keys for a stage.
///
/// Union of explicitly declared `stage.params` keys and template-extracted refs.
/// This implements spec §2.3 granular param invalidation.
pub fn effective_param_keys(declared: &Option<Vec<String>>, cmd: &str) -> Vec<String> {
    let mut keys = extract_param_refs(cmd);
    if let Some(declared_keys) = declared {
        for k in declared_keys {
            if !keys.contains(k) {
                keys.push(k.clone());
            }
        }
    }
    keys
}

/// Hash a resolved command string
pub fn hash_cmd(resolved_cmd: &str) -> String {
    let hash = blake3::hash(resolved_cmd.as_bytes());
    format!("blake3:{}", hash.to_hex())
}

/// Compute composite cache key from component hashes
///
/// `cache_key = BLAKE3(cmd_hash || deps_hash || params_hash)`
pub fn compute_cache_key(cmd_hash: &str, deps_hash: &str, params_hash: &str) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(cmd_hash.as_bytes());
    hasher.update(deps_hash.as_bytes());
    hasher.update(params_hash.as_bytes());
    let hash = hasher.finalize();
    format!("blake3:{}", hash.to_hex())
}

/// Compute the combined deps hash from individual dependency hashes
pub fn combine_deps_hashes(hashes: &[String]) -> String {
    let mut hasher = blake3::Hasher::new();
    for h in hashes {
        hasher.update(h.as_bytes());
    }
    let hash = hasher.finalize();
    format!("blake3:{}", hash.to_hex())
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    fn make_params(pairs: &[(&str, &str)]) -> HashMap<String, serde_yaml_ng::Value> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), serde_yaml_ng::Value::String(v.to_string())))
            .collect()
    }

    #[test]
    fn test_PB003_hash_file_deterministic() {
        let dir = tempfile::tempdir().expect("tempdir creation failed");
        let file = dir.path().join("test.txt");
        std::fs::write(&file, b"hello world").expect("fs write failed");

        let h1 = hash_file(&file).expect("unexpected failure");
        let h2 = hash_file(&file).expect("unexpected failure");
        assert_eq!(h1, h2);
        assert!(h1.starts_with("blake3:"));
    }

    #[test]
    fn test_PB003_hash_file_changes_with_content() {
        let dir = tempfile::tempdir().expect("tempdir creation failed");
        let file = dir.path().join("test.txt");

        std::fs::write(&file, b"hello").expect("fs write failed");
        let h1 = hash_file(&file).expect("unexpected failure");

        std::fs::write(&file, b"world").expect("fs write failed");
        let h2 = hash_file(&file).expect("unexpected failure");

        assert_ne!(h1, h2);
    }

    #[test]
    fn test_PB003_hash_directory_sorted_walk() {
        let dir = tempfile::tempdir().expect("tempdir creation failed");
        std::fs::write(dir.path().join("b.txt"), b"content-b").expect("fs write failed");
        std::fs::write(dir.path().join("a.txt"), b"content-a").expect("fs write failed");

        let r1 = hash_directory(dir.path()).expect("unexpected failure");
        assert!(r1.hash.starts_with("blake3:"));
        assert_eq!(r1.file_count, 2);
        assert_eq!(r1.total_bytes, 18);

        let r2 = hash_directory(dir.path()).expect("unexpected failure");
        assert_eq!(r1.hash, r2.hash);
    }

    #[test]
    fn test_PB003_hash_directory_single_file() {
        let dir = tempfile::tempdir().expect("tempdir creation failed");
        let file = dir.path().join("only.txt");
        std::fs::write(&file, b"data").expect("fs write failed");

        let result = hash_directory(&file).expect("unexpected failure");
        assert_eq!(result.file_count, 1);
        assert_eq!(result.total_bytes, 4);
    }

    #[test]
    fn test_PB003_hash_params_sorted() {
        let global = make_params(&[("b", "2"), ("a", "1")]);
        let refs = vec!["a".to_string(), "b".to_string()];

        let h1 = hash_params(&global, &refs).expect("unexpected failure");

        // Different reference order, same result
        let refs2 = vec!["b".to_string(), "a".to_string()];
        let h2 = hash_params(&global, &refs2).expect("unexpected failure");

        assert_eq!(h1, h2);
        assert!(h1.starts_with("blake3:"));
    }

    #[test]
    fn test_PB003_hash_cmd() {
        let h1 = hash_cmd("echo hello");
        let h2 = hash_cmd("echo hello");
        let h3 = hash_cmd("echo world");

        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
        assert!(h1.starts_with("blake3:"));
    }

    #[test]
    fn test_PB003_compute_cache_key() {
        let key1 = compute_cache_key("blake3:aaa", "blake3:bbb", "blake3:ccc");
        let key2 = compute_cache_key("blake3:aaa", "blake3:bbb", "blake3:ccc");
        let key3 = compute_cache_key("blake3:xxx", "blake3:bbb", "blake3:ccc");

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_PB003_extract_param_refs() {
        let refs = extract_param_refs("run --model {{params.model}} --lang {{params.lang}} plain");
        assert_eq!(refs, vec!["model", "lang"]);
    }

    #[test]
    fn test_PB003_extract_param_refs_no_refs() {
        let refs = extract_param_refs("echo hello world");
        assert!(refs.is_empty());
    }

    #[test]
    fn test_PB003_extract_param_refs_dedup() {
        let refs = extract_param_refs("{{params.x}} and {{params.x}} again");
        assert_eq!(refs, vec!["x"]);
    }

    #[test]
    fn test_PB003_effective_param_keys() {
        // Template refs only
        let keys = effective_param_keys(&None, "echo {{params.model}}");
        assert_eq!(keys, vec!["model"]);

        // Declared + template: union
        let declared = Some(vec!["chunk_size".to_string()]);
        let keys = effective_param_keys(&declared, "echo {{params.model}}");
        assert_eq!(keys, vec!["model", "chunk_size"]);
    }

    #[test]
    fn test_PB003_combine_deps_hashes() {
        let h1 = combine_deps_hashes(&["blake3:aaa".to_string(), "blake3:bbb".to_string()]);
        let h2 = combine_deps_hashes(&["blake3:aaa".to_string(), "blake3:bbb".to_string()]);
        assert_eq!(h1, h2);

        // Order matters
        let h3 = combine_deps_hashes(&["blake3:bbb".to_string(), "blake3:aaa".to_string()]);
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_PB003_hash_file_missing() {
        let err = hash_file(Path::new("/nonexistent/file.txt")).unwrap_err();
        assert!(err.to_string().contains("failed to open"));
    }

    #[test]
    fn test_PB003_hash_directory_nested() {
        let dir = tempfile::tempdir().expect("tempdir creation failed");
        std::fs::create_dir(dir.path().join("sub")).expect("unexpected failure");
        std::fs::write(dir.path().join("top.txt"), b"top").expect("fs write failed");
        std::fs::write(dir.path().join("sub").join("nested.txt"), b"nested")
            .expect("fs write failed");

        let result = hash_directory(dir.path()).expect("unexpected failure");
        assert_eq!(result.file_count, 2);
        assert_eq!(result.total_bytes, 9);
    }

    #[test]
    fn test_PB003_extract_param_refs_unicode_safe() {
        let refs = extract_param_refs("echo {{params.model}} — résumé {{params.lang}}");
        assert_eq!(refs, vec!["model", "lang"]);
    }
}
