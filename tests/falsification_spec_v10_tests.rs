#![allow(clippy::disallowed_methods)]
//! Popperian Falsification Tests -- Showcase Spec v10.4.0 (119 Gates)
//!
//! This file implements ALL 119 falsification gates from:
//!   docs/specifications/qwen2.5-coder-showcase-demo.md
//!
//! **GATED BY `model-tests` FEATURE** — these tests do NOT run with `cargo test`.
//! Many tests load GGUF/SafeTensors models, start servers, call ollama, and use GPU.
//! Running all at once WILL OOM the system.
//!
//! Run with: `cargo test --features model-tests --test falsification_spec_v10_tests <TEST_NAME>`
//! Never run the entire file at once without filtering.
//!
//! "We do not try to prove our theories are true, but to show that they
//!  are false." -- K. Popper (1963)
#![cfg(feature = "model-tests")]

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use aprender::format::layout_contract::{
    enforce_embedding_contract, enforce_import_contract, enforce_matmul_contract, LayoutContract,
};
use aprender::format::model_family::{
    build_default_registry, Activation, AttentionType, MlpType, NormType, PositionalEncoding,
    KNOWN_FAMILIES,
};
use aprender::format::rosetta::FormatType;
use aprender::format::validated_tensors::{RowMajor, ValidatedEmbedding, ValidatedWeight};
use tempfile::NamedTempFile;

// =============================================================================
// Helpers
// =============================================================================

fn collect_rs_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if !dir.exists() || !dir.is_dir() {
        return files;
    }
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                // Skip target/ and hidden directories
                let name = path.file_name().unwrap_or_default().to_string_lossy();
                if name.starts_with('.') || name == "target" {
                    continue;
                }
                files.extend(collect_rs_files(&path));
            } else if path.extension().map_or(false, |ext| ext == "rs") {
                files.push(path);
            }
        }
    }
    files
}

fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn search_dir_for_file(search_root: &Path, filename: &str) -> Option<PathBuf> {
    let mut dirs_to_visit = vec![search_root.to_path_buf()];
    while let Some(dir) = dirs_to_visit.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                dirs_to_visit.push(path);
            } else if path.file_name().map_or(false, |n| n == filename) {
                return Some(path);
            }
        }
    }
    None
}

fn find_generated_file(filename: &str) -> Option<PathBuf> {
    let target_dir = project_root().join("target");
    for profile in &["debug", "release"] {
        let search_root = target_dir.join(profile).join("build");
        if !search_root.exists() {
            continue;
        }
        if let Some(found) = search_dir_for_file(&search_root, filename) {
            return Some(found);
        }
    }
    None
}

// =============================================================================
// Model fixture helpers
// =============================================================================

/// Model directory: uses MODEL_DIR env var, or ./models/ relative to project root
fn model_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("MODEL_DIR") {
        PathBuf::from(dir)
    } else {
        project_root().join("models")
    }
}

/// Get path to the 0.5B GGUF model (fastest for testing)
fn gguf_model_path() -> Option<PathBuf> {
    let path = model_dir().join("qwen2.5-coder-0.5b-instruct-q4_k_m.gguf");
    if path.exists() {
        Some(path)
    } else {
        None
    }
}

/// Get path to the 0.5B APR model (validates it can be read by apr CLI)
fn apr_model_path() -> Option<PathBuf> {
    let path = model_dir().join("qwen2.5-coder-0.5b-instruct-q4_k_m.apr");
    if !path.exists() {
        return None;
    }
    // Validate APR file is usable (not corrupt)
    let bin = apr_binary();
    let output = Command::new(&bin)
        .args(["tensors", path.to_str().unwrap()])
        .output()
        .ok()?;
    if output.status.success() {
        Some(path)
    } else {
        eprintln!(
            "SKIP: APR model exists but is corrupt/incompatible: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        None
    }
}

/// Get path to SafeTensors model directory (0.5B)
fn safetensors_model_dir() -> Option<PathBuf> {
    let path = PathBuf::from("/home/noah/models/qwen2.5-coder-0.5b-instruct");
    if path.join("model.safetensors").exists() {
        Some(path)
    } else {
        None
    }
}

/// Find the apr CLI binary (release preferred, then debug)
fn apr_binary() -> PathBuf {
    let target_base = PathBuf::from("/mnt/nvme-raid0/targets/aprender");
    let release = target_base.join("release").join("apr");
    if release.exists() {
        return release;
    }
    let debug = target_base.join("debug").join("apr");
    if debug.exists() {
        return debug;
    }
    // Fallback to standard target dir
    let standard_release = project_root().join("target").join("release").join("apr");
    if standard_release.exists() {
        return standard_release;
    }
    project_root().join("target").join("debug").join("apr")
}

/// Find ollama binary if installed
fn which_ollama() -> Option<PathBuf> {
    let output = Command::new("which").arg("ollama").output().ok()?;
    if output.status.success() {
        let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if path.is_empty() {
            None
        } else {
            Some(PathBuf::from(path))
        }
    } else {
        None
    }
}

/// Run apr CLI command, return (success, stdout, stderr)
fn run_apr(args: &[&str]) -> (bool, String, String) {
    let bin = apr_binary();
    let output = Command::new(&bin)
        .args(args)
        .current_dir(project_root())
        .output()
        .unwrap_or_else(|e| panic!("Failed to run apr at {}: {}", bin.display(), e));
    (
        output.status.success(),
        String::from_utf8_lossy(&output.stdout).to_string(),
        String::from_utf8_lossy(&output.stderr).to_string(),
    )
}

/// Skip test if model not available (returns from calling function)
macro_rules! require_model {
    ($path_opt:expr, $name:expr) => {
        match $path_opt {
            Some(p) => p,
            None => {
                eprintln!(
                    "SKIP: {} not found. Set MODEL_DIR or download with `apr pull`",
                    $name
                );
                return;
            }
        }
    };
}

// =============================================================================
// Section 0: Ground Truth Testing (F-GT-*)
// These require model fixtures (SafeTensors BF16, 7B)
// =============================================================================

include!("includes/falsification_spec_v10_part_01.rs");
include!("includes/falsification_spec_v10_part_02.rs");
include!("includes/falsification_spec_v10_part_03.rs");
include!("includes/falsification_spec_v10_part_04.rs");
include!("includes/f_ollama_00.rs");
include!("includes/falsification_spec_v10_definition_of_done.rs");
include!("includes/falsification_spec_v10_part_07.rs");
include!("includes/f_trueno_00.rs");
include!("includes/f_realize_0.rs");
include!("includes/falsification_spec_v10_part_10.rs");
include!("includes/falsification_spec_v10_part_11.rs");
