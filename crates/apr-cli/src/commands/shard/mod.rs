//! CRUX-B-05 — Safetensors shard/unshard via weight-map.
//!
//! Parity target: HuggingFace `model.safetensors.index.json` layout used by
//! `transformers.PreTrainedModel.save_pretrained` for big models.
//!
//! Surface:
//! - `apr shard FILE --max-shard-size SZ -o OUT/`   (split)
//! - `apr unshard DIR -o merged.safetensors`        (merge)
//!
//! See `contracts/crux-B-05-v1.yaml`.

use std::path::Path;

pub mod sharder;
pub mod unsharder;

#[cfg(test)]
mod tests;

/// Entry point for `apr shard`.
pub fn run_shard(
    file: &Path,
    max_shard_size: &str,
    output: &Path,
) -> Result<sharder::ShardReport, sharder::ShardError> {
    let limit = parse_size(max_shard_size).map_err(sharder::ShardError::ParseSize)?;
    sharder::shard_safetensors_file(file, limit, output)
}

/// Entry point for `apr unshard`.
pub fn run_unshard(
    input_dir: &Path,
    output: &Path,
) -> Result<unsharder::UnshardReport, unsharder::UnshardError> {
    unsharder::unshard_safetensors_dir(input_dir, output)
}

/// Parse a human-readable size string into bytes.
///
/// Supports `B`, `KB`, `MB`, `GB`, `TB` (decimal) and `KiB`, `MiB`, `GiB`, `TiB` (binary).
/// Numbers may be fractional (e.g. `1.5GB`). Unitless input is interpreted as bytes.
///
/// # Errors
///
/// Returns a descriptive string on malformed input.
pub fn parse_size(s: &str) -> Result<u64, String> {
    let s = s.trim();
    if s.is_empty() {
        return Err("empty size".to_string());
    }

    let split_at = s
        .find(|c: char| c.is_ascii_alphabetic())
        .unwrap_or(s.len());
    let (num_part, unit_part) = s.split_at(split_at);
    let num_part = num_part.trim();
    let unit_part = unit_part.trim();

    let num: f64 = num_part
        .parse()
        .map_err(|e| format!("invalid number '{num_part}': {e}"))?;
    if num < 0.0 || !num.is_finite() {
        return Err(format!("size must be a finite non-negative number: {num}"));
    }

    let multiplier: f64 = match unit_part.to_ascii_uppercase().as_str() {
        "" | "B" => 1.0,
        "K" | "KB" => 1_000.0,
        "M" | "MB" => 1_000_000.0,
        "G" | "GB" => 1_000_000_000.0,
        "T" | "TB" => 1_000_000_000_000.0,
        "KIB" => 1024.0,
        "MIB" => 1024.0 * 1024.0,
        "GIB" => 1024.0 * 1024.0 * 1024.0,
        "TIB" => 1024.0 * 1024.0 * 1024.0 * 1024.0,
        other => return Err(format!("unknown size unit '{other}'")),
    };

    let total = num * multiplier;
    if total > u64::MAX as f64 {
        return Err(format!("size overflow: {s}"));
    }
    Ok(total as u64)
}

#[cfg(test)]
mod size_tests {
    use super::parse_size;

    #[test]
    fn bytes_no_unit() {
        assert_eq!(parse_size("1024").unwrap(), 1024);
    }

    #[test]
    fn decimal_units() {
        assert_eq!(parse_size("1MB").unwrap(), 1_000_000);
        assert_eq!(parse_size("2GB").unwrap(), 2_000_000_000);
    }

    #[test]
    fn binary_units() {
        assert_eq!(parse_size("1MiB").unwrap(), 1_048_576);
        assert_eq!(parse_size("1GiB").unwrap(), 1_073_741_824);
    }

    #[test]
    fn case_insensitive() {
        assert_eq!(parse_size("5mb").unwrap(), 5_000_000);
        assert_eq!(parse_size("5Mb").unwrap(), 5_000_000);
    }

    #[test]
    fn fractional() {
        assert_eq!(parse_size("1.5GB").unwrap(), 1_500_000_000);
    }

    #[test]
    fn whitespace_tolerated() {
        assert_eq!(parse_size("  10 MB ").unwrap(), 10_000_000);
    }

    #[test]
    fn rejects_negative() {
        assert!(parse_size("-1MB").is_err());
    }

    #[test]
    fn rejects_unknown_unit() {
        assert!(parse_size("1XB").is_err());
    }

    #[test]
    fn rejects_empty() {
        assert!(parse_size("").is_err());
    }
}
