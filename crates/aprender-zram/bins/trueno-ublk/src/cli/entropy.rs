//! Entropy command - analyze file/directory entropy

use super::EntropyArgs;
use anyhow::Result;
use serde::Serialize;
use std::fs::File;
use std::io::Read;
use std::path::Path;

#[derive(Serialize)]
struct EntropyResult {
    path: String,
    entropy: f64,
    size: u64,
    compression_hint: String,
}

pub fn run(args: EntropyArgs) -> Result<()> {
    let mut results = Vec::new();

    for path in &args.paths {
        if path.is_dir() && args.recursive {
            analyze_directory(path, &mut results)?;
        } else if path.is_file() {
            if let Some(result) = analyze_file(path)? {
                results.push(result);
            }
        }
    }

    if args.json {
        println!("{}", serde_json::to_string_pretty(&results)?);
    } else {
        println!("{:>8} {:>12} {:>16}  PATH", "ENTROPY", "SIZE", "HINT");
        for r in &results {
            println!(
                "{:>8.2} {:>12} {:>16}  {}",
                r.entropy,
                super::format_size(r.size),
                r.compression_hint,
                r.path
            );
        }
    }

    Ok(())
}

fn analyze_directory(dir: &Path, results: &mut Vec<EntropyResult>) -> Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            analyze_directory(&path, results)?;
        } else if path.is_file() {
            if let Some(result) = analyze_file(&path)? {
                results.push(result);
            }
        }
    }
    Ok(())
}

fn analyze_file(path: &Path) -> Result<Option<EntropyResult>> {
    let metadata = std::fs::metadata(path)?;
    let size = metadata.len();

    // Skip empty files
    if size == 0 {
        return Ok(None);
    }

    // Sample up to 64KB for entropy calculation
    let sample_size = std::cmp::min(size, 65536) as usize;
    let mut file = File::open(path)?;
    let mut buffer = vec![0u8; sample_size];
    file.read_exact(&mut buffer)?;

    let entropy = calculate_entropy(&buffer);
    let hint = compression_hint(entropy);

    Ok(Some(EntropyResult {
        path: path.display().to_string(),
        entropy,
        size,
        compression_hint: hint.to_string(),
    }))
}

fn calculate_entropy(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    let mut counts = [0u64; 256];
    for &byte in data {
        counts[byte as usize] += 1;
    }

    let len = data.len() as f64;
    let mut entropy = 0.0;

    for &count in &counts {
        if count > 0 {
            let p = count as f64 / len;
            entropy -= p * p.log2();
        }
    }

    entropy
}

fn compression_hint(entropy: f64) -> &'static str {
    if entropy < 4.0 {
        "highly compressible"
    } else if entropy < 6.0 {
        "compressible"
    } else if entropy < 7.0 {
        "moderate"
    } else if entropy < 7.5 {
        "low benefit"
    } else {
        "skip (encrypted/compressed)"
    }
}
