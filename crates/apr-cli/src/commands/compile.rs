//! Compile command implementation (APR-SPEC §4.16)
//!
//! Builds standalone executables with embedded .apr models via `include_bytes!`.
//! Generates a temporary Cargo project, runs `cargo build`, and copies the output binary.

use crate::error::{CliError, Result};
use crate::output;
use aprender::format::v2::{AprV2Header, AprV2Metadata, HEADER_SIZE_V2, MAGIC_V2};
use std::fs;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Known compilation targets.
const TARGETS: &[(&str, &str)] = &[
    // Native
    ("x86_64-unknown-linux-gnu", "Linux x86_64 (glibc)"),
    ("x86_64-unknown-linux-musl", "Linux x86_64 (musl, fully static)"),
    ("aarch64-unknown-linux-gnu", "Linux ARM64"),
    ("x86_64-apple-darwin", "macOS x86_64"),
    ("aarch64-apple-darwin", "macOS ARM64 (Apple Silicon)"),
    ("x86_64-pc-windows-msvc", "Windows x86_64"),
    // WebAssembly
    ("wasm32-unknown-unknown", "Pure WASM (browser)"),
    ("wasm32-wasi", "WASM + WASI (server-side)"),
    ("wasm32-wasip1", "WASM + WASI Preview 1"),
    ("wasm32-wasip2", "WASM + WASI Preview 2 (component model)"),
];

/// Metadata extracted from .apr file for code generation.
struct ModelInfo {
    name: String,
    model_type: String,
    param_count: u64,
    tensor_count: u32,
    file_size: u64,
}

/// Run the compile command.
#[allow(clippy::fn_params_excessive_bools)]
pub(crate) fn run(
    file: Option<&Path>,
    output_path: Option<&Path>,
    target: Option<&str>,
    quantize: Option<&str>,
    release: bool,
    strip: bool,
    lto: bool,
    list_targets: bool,
    json_output: bool,
) -> Result<()> {
    if list_targets {
        return print_targets(json_output);
    }

    let file = file.ok_or_else(|| {
        CliError::ValidationFailed("Input .apr file is required (unless --list-targets)".into())
    })?;

    if quantize.is_some() {
        return Err(CliError::ValidationFailed(
            "Pre-embed quantization (--quantize) is not yet implemented. \
             Quantize with `apr quantize` first, then compile the quantized model."
                .into(),
        ));
    }

    if !file.exists() {
        return Err(CliError::FileNotFound(file.to_path_buf()));
    }

    let info = read_model_info(file)?;

    let bin_name = derive_binary_name(file);
    let output_path = output_path.map_or_else(|| PathBuf::from(&bin_name), Path::to_path_buf);

    if !json_output {
        output::header("APR Compile Pipeline");
        println!(
            "{}",
            output::kv_table(&[
                ("Model", file.display().to_string()),
                ("Name", info.name.clone()),
                ("Architecture", info.model_type.clone()),
                ("Parameters", format_param_count(info.param_count)),
                ("Tensors", info.tensor_count.to_string()),
                ("Output", output_path.display().to_string()),
            ])
        );
        println!();
    }

    // Generate ephemeral Cargo project for compilation
    let tmp_dir = tempfile::tempdir()
        .map_err(|e| CliError::Io(std::io::Error::other(e)))?;
    let project_dir = tmp_dir.path().join(&bin_name);

    generate_cargo_project(&project_dir, &bin_name, file, &info, release, strip, lto)?;

    if !json_output {
        output::pipeline_stage("Compiling", output::StageStatus::Running);
    }

    let built_binary = run_cargo_build(&project_dir, target, release, strip, lto, &bin_name)?;

    // Copy to output and make executable
    fs::copy(&built_binary, &output_path)?;
    make_executable(&output_path)?;

    let binary_size = fs::metadata(&output_path)?.len();

    if json_output {
        print_compile_result_json(file, &output_path, &info, binary_size, release, strip, lto, target);
    } else {
        print_compile_result_text(&output_path, &info, binary_size, release, strip, lto);
    }

    Ok(())
}

/// Run cargo build and return the path to the built binary.
fn run_cargo_build(
    project_dir: &Path,
    target: Option<&str>,
    release: bool,
    strip: bool,
    lto: bool,
    bin_name: &str,
) -> Result<PathBuf> {
    let mut cmd = Command::new("cargo");
    cmd.arg("build")
        .current_dir(project_dir)
        .env_remove("CARGO_TARGET_DIR");

    if release {
        cmd.arg("--release");
    }

    if let Some(t) = target {
        cmd.arg("--target").arg(t);
    }

    // Build RUSTFLAGS
    let mut rustflags = Vec::new();
    if strip {
        rustflags.push("-C strip=symbols".to_string());
    }
    if lto {
        rustflags.push("-C lto=fat".to_string());
    }
    if !rustflags.is_empty() {
        cmd.env("RUSTFLAGS", rustflags.join(" "));
    }

    let build_output = cmd.output().map_err(|e| {
        CliError::ValidationFailed(format!(
            "Failed to run cargo build. Is Rust installed?\n  {e}"
        ))
    })?;

    if !build_output.status.success() {
        let stderr = String::from_utf8_lossy(&build_output.stderr);
        return Err(CliError::ValidationFailed(format!(
            "Cargo build failed:\n{stderr}"
        )));
    }

    // Locate output binary
    let profile_dir = if release { "release" } else { "debug" };
    let built_binary = if let Some(t) = target {
        project_dir
            .join("target")
            .join(t)
            .join(profile_dir)
            .join(bin_name)
    } else {
        project_dir.join("target").join(profile_dir).join(bin_name)
    };

    if !built_binary.exists() {
        return Err(CliError::ValidationFailed(format!(
            "Build succeeded but binary not found at: {}",
            built_binary.display()
        )));
    }

    Ok(built_binary)
}

/// Make a file executable on Unix.
fn make_executable(_path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(_path)?.permissions();
        perms.set_mode(0o755);
        fs::set_permissions(_path, perms)?;
    }
    Ok(())
}

/// Print compile result as JSON.
#[allow(clippy::fn_params_excessive_bools)]
fn print_compile_result_json(
    input: &Path,
    output_path: &Path,
    info: &ModelInfo,
    binary_size: u64,
    release: bool,
    strip: bool,
    lto: bool,
    target: Option<&str>,
) {
    // serde_json::json!() macro uses infallible unwrap internally
    #[allow(clippy::disallowed_methods)]
    let result = serde_json::json!({
        "status": "success",
        "input": input.display().to_string(),
        "output": output_path.display().to_string(),
        "model_name": info.name,
        "architecture": info.model_type,
        "param_count": info.param_count,
        "model_size_bytes": info.file_size,
        "binary_size_bytes": binary_size,
        "release": release,
        "strip": strip,
        "lto": lto,
        "target": target,
    });
    println!("{}", serde_json::to_string_pretty(&result).unwrap_or_default());
}

/// Print compile result as text.
fn print_compile_result_text(
    output_path: &Path,
    info: &ModelInfo,
    binary_size: u64,
    release: bool,
    strip: bool,
    lto: bool,
) {
    println!();
    output::subheader("Build Report");
    println!(
        "{}",
        output::kv_table(&[
            ("Binary", output_path.display().to_string()),
            ("Binary size", output::format_size(binary_size)),
            ("Model size", output::format_size(info.file_size)),
            ("Mode", if release { "release" } else { "debug" }.into()),
            ("Strip", if strip { "yes" } else { "no" }.into()),
            ("LTO", if lto { "yes" } else { "no" }.into()),
        ])
    );
    println!();
    println!("  {}", output::badge_pass("Compile successful"));
    println!(
        "  Run with: {}",
        output_path.display().to_string().as_str()
    );
}

/// Print available compilation targets.
fn print_targets(json_output: bool) -> Result<()> {
    if json_output {
        // serde_json::json!() macro uses infallible unwrap internally
        #[allow(clippy::disallowed_methods)]
        let targets: Vec<_> = TARGETS
            .iter()
            .map(|(triple, desc)| {
                serde_json::json!({ "triple": triple, "description": desc })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&targets).unwrap_or_default());
    } else {
        output::header("Available Compilation Targets");
        println!();
        output::subheader("Native");
        for (triple, desc) in &TARGETS[..6] {
            output::kv(&format!("  {triple}"), desc);
        }
        println!();
        output::subheader("WebAssembly");
        for (triple, desc) in &TARGETS[6..] {
            output::kv(&format!("  {triple}"), desc);
        }
    }
    Ok(())
}

/// Read model metadata from .apr file header.
fn read_model_info(path: &Path) -> Result<ModelInfo> {
    let file = fs::File::open(path)?;
    let file_size = file.metadata()?.len();
    let mut reader = BufReader::new(file);

    // Read header
    let mut header_bytes = [0u8; HEADER_SIZE_V2];
    reader.read_exact(&mut header_bytes).map_err(|_| {
        CliError::InvalidFormat("File too small to contain valid APR header".into())
    })?;

    if header_bytes[0..4] != MAGIC_V2 {
        return Err(CliError::InvalidFormat(
            "Only APR v2 format (APR\\0) is supported for compilation. \
             Convert with `apr import` first."
                .into(),
        ));
    }

    let header = AprV2Header::from_bytes(&header_bytes)
        .map_err(|e| CliError::InvalidFormat(format!("Failed to parse header: {e}")))?;

    // Read metadata
    let (name, model_type, param_count) = if header.metadata_size > 0 {
        reader
            .seek(SeekFrom::Start(header.metadata_offset))
            .map_err(CliError::Io)?;
        let mut meta_bytes = vec![0u8; header.metadata_size as usize];
        reader.read_exact(&mut meta_bytes)?;

        match AprV2Metadata::from_json(&meta_bytes) {
            Ok(meta) => (
                meta.name.unwrap_or_else(|| "model".into()),
                meta.model_type.clone(),
                meta.param_count,
            ),
            Err(_) => ("model".into(), "unknown".into(), 0),
        }
    } else {
        ("model".into(), "unknown".into(), 0)
    };

    Ok(ModelInfo {
        name,
        model_type,
        param_count,
        tensor_count: header.tensor_count,
        file_size,
    })
}

/// Generate a temporary Cargo project that embeds the .apr model.
fn generate_cargo_project(
    project_dir: &Path,
    bin_name: &str,
    model_path: &Path,
    info: &ModelInfo,
    _release: bool,
    _strip: bool,
    _lto: bool,
) -> Result<()> {
    let src_dir = project_dir.join("src");
    fs::create_dir_all(&src_dir)?;

    // Copy model file into project
    let model_dest = project_dir.join("model.apr");
    fs::copy(model_path, &model_dest)?;

    // Generate Cargo.toml with realizar + server deps
    let cargo_toml = generate_cargo_toml(bin_name);
    fs::write(project_dir.join("Cargo.toml"), cargo_toml)?;

    // Generate main.rs
    let main_rs = generate_main_rs(bin_name, info);
    fs::write(src_dir.join("main.rs"), main_rs)?;

    Ok(())
}

include!("compile_codegen.rs");

/// Derive binary name from model file path.
fn derive_binary_name(path: &Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model")
        .to_lowercase()
        .replace(['.', ' ', '-'], "_")
}

/// Format parameter count in human-readable form.
fn format_param_count(count: u64) -> String {
    if count == 0 {
        return "unknown".into();
    }
    if count >= 1_000_000_000 {
        format!("{:.1}B", count as f64 / 1_000_000_000.0)
    } else if count >= 1_000_000 {
        format!("{:.1}M", count as f64 / 1_000_000.0)
    } else if count >= 1_000 {
        format!("{:.1}K", count as f64 / 1_000.0)
    } else {
        count.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derive_binary_name() {
        assert_eq!(derive_binary_name(Path::new("whisper-tiny.apr")), "whisper_tiny");
        assert_eq!(derive_binary_name(Path::new("/path/to/Qwen2.5-Coder.apr")), "qwen2_5_coder");
        assert_eq!(derive_binary_name(Path::new("model.apr")), "model");
    }

    #[test]
    fn test_format_param_count() {
        assert_eq!(format_param_count(0), "unknown");
        assert_eq!(format_param_count(500), "500");
        assert_eq!(format_param_count(1_500_000), "1.5M");
        assert_eq!(format_param_count(7_000_000_000), "7.0B");
        assert_eq!(format_param_count(39_000), "39.0K");
    }

    #[test]
    fn test_list_targets_json() {
        // Just verify it doesn't panic
        assert!(print_targets(true).is_ok());
    }

    #[test]
    fn test_run_missing_file() {
        let result = run(
            Some(Path::new("/nonexistent/model.apr")),
            None, None, None,
            false, false, false, false, false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_list_targets() {
        assert!(run(None, None, None, None, false, false, false, true, false).is_ok());
    }

    #[test]
    fn test_quantize_not_yet_supported() {
        let result = run(
            Some(Path::new("test.apr")),
            None, None, Some("int8"),
            false, false, false, false, false,
        );
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("not yet implemented"));
    }
}
