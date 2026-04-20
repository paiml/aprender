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
    (
        "x86_64-unknown-linux-musl",
        "Linux x86_64 (musl, fully static)",
    ),
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
#[provable_contracts_macros::contract(
    "apr-cli-operations-v1",
    equation = "mutating_output_contract"
)]
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

    let file = validate_compile_inputs(file, quantize)?;
    let info = read_model_info(file)?;
    let bin_name = derive_binary_name(file);
    let output_path = output_path.map_or_else(|| PathBuf::from(&bin_name), Path::to_path_buf);

    if !json_output {
        print_compile_banner(file, &info, &output_path);
    }

    let binary_size = compile_pipeline(
        file,
        &info,
        &bin_name,
        &output_path,
        target,
        release,
        strip,
        lto,
        json_output,
    )?;

    if json_output {
        print_compile_result_json(
            file,
            &output_path,
            &info,
            binary_size,
            release,
            strip,
            lto,
            target,
        );
    } else {
        print_compile_result_text(&output_path, &info, binary_size, release, strip, lto);
    }

    Ok(())
}

#[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
fn compile_pipeline(
    file: &Path,
    info: &ModelInfo,
    bin_name: &str,
    output_path: &Path,
    target: Option<&str>,
    release: bool,
    strip: bool,
    lto: bool,
    json_output: bool,
) -> Result<u64> {
    let tmp_dir = tempfile::tempdir().map_err(|e| CliError::Io(std::io::Error::other(e)))?;
    let project_dir = tmp_dir.path().join(bin_name);

    generate_cargo_project(&project_dir, bin_name, file, info, release, strip, lto)?;

    if !json_output {
        output::pipeline_stage("Compiling", output::StageStatus::Running);
    }

    let built_binary = run_cargo_build(&project_dir, target, release, strip, lto, bin_name)?;
    finalize_output(&built_binary, output_path)
}

fn validate_compile_inputs<'a>(file: Option<&'a Path>, quantize: Option<&str>) -> Result<&'a Path> {
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
    Ok(file)
}

fn print_compile_banner(file: &Path, info: &ModelInfo, output_path: &Path) {
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

fn finalize_output(built_binary: &Path, output_path: &Path) -> Result<u64> {
    fs::copy(built_binary, output_path)?;
    make_executable(output_path)?;
    Ok(fs::metadata(output_path)?.len())
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

    apply_rustflags(&mut cmd, strip, lto);

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

    locate_built_binary(project_dir, target, release, bin_name)
}

fn apply_rustflags(cmd: &mut Command, strip: bool, lto: bool) {
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
}

fn locate_built_binary(
    project_dir: &Path,
    target: Option<&str>,
    release: bool,
    bin_name: &str,
) -> Result<PathBuf> {
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
    println!(
        "{}",
        serde_json::to_string_pretty(&result).unwrap_or_default()
    );
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
    println!("  Run with: {}", output_path.display().to_string().as_str());
}

/// Print available compilation targets.
fn print_targets(json_output: bool) -> Result<()> {
    if json_output {
        // serde_json::json!() macro uses infallible unwrap internally
        #[allow(clippy::disallowed_methods)]
        let targets: Vec<_> = TARGETS
            .iter()
            .map(|(triple, desc)| serde_json::json!({ "triple": triple, "description": desc }))
            .collect();
        println!(
            "{}",
            serde_json::to_string_pretty(&targets).unwrap_or_default()
        );
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

    // ========================================================================
    // derive_binary_name Tests
    // ========================================================================

    #[test]
    fn test_derive_binary_name() {
        assert_eq!(
            derive_binary_name(Path::new("whisper-tiny.apr")),
            "whisper_tiny"
        );
        assert_eq!(
            derive_binary_name(Path::new("/path/to/Qwen2.5-Coder.apr")),
            "qwen2_5_coder"
        );
        assert_eq!(derive_binary_name(Path::new("model.apr")), "model");
    }

    #[test]
    fn derive_binary_name_no_extension() {
        assert_eq!(derive_binary_name(Path::new("modelfile")), "modelfile");
    }

    #[test]
    fn derive_binary_name_multiple_dots() {
        assert_eq!(
            derive_binary_name(Path::new("my.model.v2.apr")),
            "my_model_v2"
        );
    }

    #[test]
    fn derive_binary_name_spaces_replaced() {
        assert_eq!(derive_binary_name(Path::new("my model.apr")), "my_model");
    }

    #[test]
    fn derive_binary_name_uppercase_lowered() {
        assert_eq!(derive_binary_name(Path::new("MyModel.apr")), "mymodel");
    }

    #[test]
    fn derive_binary_name_all_special_chars() {
        assert_eq!(derive_binary_name(Path::new("a-b.c d.apr")), "a_b_c_d");
    }

    #[test]
    fn derive_binary_name_hidden_file() {
        assert_eq!(derive_binary_name(Path::new(".hidden.apr")), "_hidden");
    }

    // ========================================================================
    // format_param_count Tests
    // ========================================================================

    #[test]
    fn test_format_param_count() {
        assert_eq!(format_param_count(0), "unknown");
        assert_eq!(format_param_count(500), "500");
        assert_eq!(format_param_count(1_500_000), "1.5M");
        assert_eq!(format_param_count(7_000_000_000), "7.0B");
        assert_eq!(format_param_count(39_000), "39.0K");
    }

    #[test]
    fn format_param_count_boundary_999() {
        assert_eq!(format_param_count(999), "999");
    }

    #[test]
    fn format_param_count_boundary_1000() {
        assert_eq!(format_param_count(1_000), "1.0K");
    }

    #[test]
    fn format_param_count_boundary_999_999() {
        assert_eq!(format_param_count(999_999), "1000.0K");
    }

    #[test]
    fn format_param_count_boundary_1_million() {
        assert_eq!(format_param_count(1_000_000), "1.0M");
    }

    #[test]
    fn format_param_count_boundary_999_999_999() {
        assert_eq!(format_param_count(999_999_999), "1000.0M");
    }

    #[test]
    fn format_param_count_boundary_1_billion() {
        assert_eq!(format_param_count(1_000_000_000), "1.0B");
    }

    #[test]
    fn format_param_count_70b() {
        assert_eq!(format_param_count(70_000_000_000), "70.0B");
    }

    #[test]
    fn format_param_count_single() {
        assert_eq!(format_param_count(1), "1");
    }

    // ========================================================================
    // print_targets Tests
    // ========================================================================

    #[test]
    fn test_list_targets_json() {
        assert!(print_targets(true).is_ok());
    }

    #[test]
    fn test_list_targets_text() {
        assert!(print_targets(false).is_ok());
    }

    // ========================================================================
    // TARGETS constant Tests
    // ========================================================================

    #[test]
    fn targets_has_native_and_wasm() {
        assert!(TARGETS.len() >= 10, "Expected at least 10 targets");
        // Verify all targets have non-empty triple and description
        for (triple, desc) in TARGETS {
            assert!(!triple.is_empty(), "Target triple must not be empty");
            assert!(!desc.is_empty(), "Target description must not be empty");
        }
    }

    #[test]
    fn targets_includes_linux_x86_64() {
        assert!(
            TARGETS
                .iter()
                .any(|(t, _)| *t == "x86_64-unknown-linux-gnu"),
            "Missing Linux x86_64 target"
        );
    }

    #[test]
    fn targets_includes_macos_arm64() {
        assert!(
            TARGETS.iter().any(|(t, _)| *t == "aarch64-apple-darwin"),
            "Missing macOS ARM64 target"
        );
    }

    #[test]
    fn targets_includes_wasm() {
        let wasm_count = TARGETS
            .iter()
            .filter(|(t, _)| t.starts_with("wasm32"))
            .count();
        assert!(
            wasm_count >= 3,
            "Expected at least 3 WASM targets, found {wasm_count}"
        );
    }

    #[test]
    fn targets_native_before_wasm() {
        // Verify layout: first 6 are native, remaining are WASM
        for (triple, _) in &TARGETS[..6] {
            assert!(
                !triple.starts_with("wasm"),
                "First 6 should be native, found {triple}"
            );
        }
        for (triple, _) in &TARGETS[6..] {
            assert!(
                triple.starts_with("wasm"),
                "After 6 should be WASM, found {triple}"
            );
        }
    }

    // ========================================================================
    // run() Tests
    // ========================================================================

    #[test]
    fn test_run_missing_file() {
        let result = run(
            Some(Path::new("/nonexistent/model.apr")),
            None,
            None,
            None,
            false,
            false,
            false,
            false,
            false,
        );
        assert!(result.is_err());
        assert!(matches!(result, Err(CliError::FileNotFound(_))));
    }

    #[test]
    fn test_run_list_targets() {
        assert!(run(None, None, None, None, false, false, false, true, false).is_ok());
    }

    #[test]
    fn test_run_list_targets_json() {
        assert!(run(None, None, None, None, false, false, false, true, true).is_ok());
    }

    #[test]
    fn test_quantize_not_yet_supported() {
        let result = run(
            Some(Path::new("test.apr")),
            None,
            None,
            Some("int8"),
            false,
            false,
            false,
            false,
            false,
        );
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("not yet implemented"));
    }

    #[test]
    fn quantize_error_message_suggests_apr_quantize() {
        let result = run(
            Some(Path::new("test.apr")),
            None,
            None,
            Some("q4"),
            false,
            false,
            false,
            false,
            false,
        );
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("apr quantize"),
            "Error should suggest apr quantize: {err}"
        );
    }

    #[test]
    fn run_no_file_no_list_targets_errors() {
        let result = run(None, None, None, None, false, false, false, false, false);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Input .apr file is required"));
    }

    // ========================================================================
    // Codegen: generate_cargo_toml Tests
    // ========================================================================

    #[test]
    fn generate_cargo_toml_contains_package_name() {
        let toml = generate_cargo_toml("my_model");
        assert!(toml.contains("name = \"my_model\""));
    }

    #[test]
    fn generate_cargo_toml_has_realizar_dep() {
        let toml = generate_cargo_toml("test_bin");
        assert!(toml.contains("realizar"));
    }

    #[test]
    fn generate_cargo_toml_has_clap_dep() {
        let toml = generate_cargo_toml("test_bin");
        assert!(toml.contains("clap"));
    }

    #[test]
    fn generate_cargo_toml_has_tokio_dep() {
        let toml = generate_cargo_toml("test_bin");
        assert!(toml.contains("tokio"));
    }

    #[test]
    fn generate_cargo_toml_has_axum_dep() {
        let toml = generate_cargo_toml("test_bin");
        assert!(toml.contains("axum"));
    }

    #[test]
    fn generate_cargo_toml_has_release_profile() {
        let toml = generate_cargo_toml("test_bin");
        assert!(toml.contains("[profile.release]"));
        assert!(toml.contains("opt-level"));
    }

    #[test]
    fn generate_cargo_toml_valid_toml_syntax() {
        let toml = generate_cargo_toml("valid_name");
        // Basic structural checks
        assert!(toml.contains("[package]"));
        assert!(toml.contains("[dependencies]"));
        assert!(toml.contains("edition = \"2021\""));
    }

    // ========================================================================
    // Codegen: generate_header Tests
    // ========================================================================

    #[test]
    fn generate_header_contains_imports() {
        let header = generate_header("ML model", "7.0B");
        assert!(header.contains("use clap::Parser"));
        assert!(header.contains("use realizar"));
    }

    #[test]
    fn generate_header_embeds_arch_desc() {
        let header = generate_header("transformer model", "1.5B");
        assert!(header.contains("transformer model"));
    }

    #[test]
    fn generate_header_embeds_param_desc() {
        let header = generate_header("ML model", "3.5B");
        assert!(header.contains("3.5B"));
    }

    #[test]
    fn generate_header_has_include_bytes() {
        let header = generate_header("ML model", "unknown");
        assert!(header.contains("include_bytes!"));
        assert!(header.contains("MODEL_DATA"));
    }

    // ========================================================================
    // Codegen: generate_cli_struct Tests
    // ========================================================================

    #[test]
    fn generate_cli_struct_has_derive_parser() {
        let cli = generate_cli_struct("my_bin", "GPT model", "7B");
        assert!(cli.contains("#[derive(Parser)]"));
    }

    #[test]
    fn generate_cli_struct_has_command_name() {
        let cli = generate_cli_struct("whisper_tiny", "whisper model", "39M");
        assert!(cli.contains("name = \"whisper_tiny\""));
    }

    #[test]
    fn generate_cli_struct_has_prompt_field() {
        let cli = generate_cli_struct("bin", "model", "1B");
        assert!(cli.contains("prompt"));
    }

    #[test]
    fn generate_cli_struct_has_serve_field() {
        let cli = generate_cli_struct("bin", "model", "1B");
        assert!(cli.contains("serve"));
    }

    #[test]
    fn generate_cli_struct_has_max_tokens() {
        let cli = generate_cli_struct("bin", "model", "1B");
        assert!(cli.contains("max_tokens"));
    }

    #[test]
    fn generate_cli_struct_has_no_gpu() {
        let cli = generate_cli_struct("bin", "model", "1B");
        assert!(cli.contains("no_gpu"));
    }

    #[test]
    fn generate_cli_struct_has_port() {
        let cli = generate_cli_struct("bin", "model", "1B");
        assert!(cli.contains("port"));
        assert!(cli.contains("8080"));
    }

    // ========================================================================
    // Codegen: generate_main_fn Tests
    // ========================================================================

    #[test]
    fn generate_main_fn_has_main_function() {
        let main = generate_main_fn("bin", "test_model", "gpt", "7B");
        assert!(main.contains("fn main()"));
    }

    #[test]
    fn generate_main_fn_has_info_handler() {
        let main = generate_main_fn("bin", "test_model", "gpt", "7B");
        assert!(main.contains("print_info"));
        assert!(main.contains("cli.info"));
    }

    #[test]
    fn generate_main_fn_has_serve_handler() {
        let main = generate_main_fn("bin", "test_model", "gpt", "7B");
        assert!(main.contains("cli.serve"));
        assert!(main.contains("start_server"));
    }

    #[test]
    fn generate_main_fn_has_inference_call() {
        let main = generate_main_fn("bin", "test_model", "gpt", "7B");
        assert!(main.contains("run_inference"));
    }

    #[test]
    fn generate_main_fn_embeds_model_name() {
        let main = generate_main_fn("bin", "qwen2_coder", "transformer", "7.0B");
        assert!(main.contains("qwen2_coder"));
    }

    #[test]
    fn generate_main_fn_has_resolve_prompt() {
        let main = generate_main_fn("bin", "model", "gpt", "1B");
        assert!(main.contains("resolve_prompt"));
    }

    #[test]
    fn generate_main_fn_has_cleanup() {
        let main = generate_main_fn("bin", "model", "gpt", "1B");
        assert!(main.contains("remove_file"));
    }

    // ========================================================================
    // Codegen: generate_materialize_fn Tests
    // ========================================================================

    #[test]
    fn generate_materialize_fn_uses_bin_name_in_dir() {
        let mat = generate_materialize_fn("whisper_tiny");
        assert!(mat.contains("whisper_tiny-model"));
    }

    #[test]
    fn generate_materialize_fn_has_write_all() {
        let mat = generate_materialize_fn("bin");
        assert!(mat.contains("write_all(MODEL_DATA)"));
    }

    #[test]
    fn generate_materialize_fn_checks_existing() {
        let mat = generate_materialize_fn("bin");
        // Should check if existing file has correct size
        assert!(mat.contains("metadata"));
        assert!(mat.contains("MODEL_DATA.len()"));
    }

    // ========================================================================
    // Codegen: generate_server_fn Tests
    // ========================================================================

    #[test]
    fn generate_server_fn_has_health_endpoint() {
        let server = generate_server_fn("model", "7B");
        assert!(server.contains("/health"));
    }

    #[test]
    fn generate_server_fn_has_completions_endpoint() {
        let server = generate_server_fn("model", "7B");
        assert!(server.contains("/v1/chat/completions"));
    }

    #[test]
    fn generate_server_fn_uses_axum() {
        let server = generate_server_fn("model", "7B");
        assert!(server.contains("axum"));
        assert!(server.contains("Router"));
    }

    #[test]
    fn generate_server_fn_embeds_model_info() {
        let server = generate_server_fn("my_model", "3.5B");
        assert!(server.contains("my_model"));
        assert!(server.contains("3.5B"));
    }

    // ========================================================================
    // Codegen: generate_main_rs (full integration) Tests
    // ========================================================================

    #[test]
    fn generate_main_rs_unknown_model_type() {
        let info = ModelInfo {
            name: "model".to_string(),
            model_type: "unknown".to_string(),
            param_count: 0,
            tensor_count: 10,
            file_size: 1024,
        };
        let src = generate_main_rs("model", &info);
        assert!(src.contains("ML model")); // fallback for unknown model_type
    }

    #[test]
    fn generate_main_rs_empty_model_type() {
        let info = ModelInfo {
            name: "model".to_string(),
            model_type: String::new(),
            param_count: 100_000,
            tensor_count: 5,
            file_size: 2048,
        };
        let src = generate_main_rs("model", &info);
        assert!(src.contains("ML model")); // fallback for empty model_type
    }

    #[test]
    fn generate_main_rs_known_model_type() {
        let info = ModelInfo {
            name: "qwen".to_string(),
            model_type: "transformer".to_string(),
            param_count: 7_000_000_000,
            tensor_count: 290,
            file_size: 4_000_000_000,
        };
        let src = generate_main_rs("qwen", &info);
        assert!(src.contains("transformer model"));
        assert!(src.contains("7.0B"));
    }

    #[test]
    fn generate_main_rs_has_all_sections() {
        let info = ModelInfo {
            name: "test".to_string(),
            model_type: "linear".to_string(),
            param_count: 1000,
            tensor_count: 2,
            file_size: 100,
        };
        let src = generate_main_rs("test", &info);
        // Should contain: header, CLI struct, main fn, materialize fn, server fn
        assert!(src.contains("MODEL_DATA"));
        assert!(src.contains("#[derive(Parser)]"));
        assert!(src.contains("fn main()"));
        assert!(src.contains("fn materialize_model()"));
        assert!(src.contains("fn start_server"));
    }

    // ========================================================================
    // make_executable Tests
    // ========================================================================

    #[test]
    fn make_executable_succeeds_on_temp_file() {
        let file = tempfile::NamedTempFile::new().expect("create temp file");
        let result = make_executable(file.path());
        assert!(result.is_ok());
    }

    #[test]
    fn make_executable_nonexistent_file_errors() {
        let result = make_executable(Path::new("/nonexistent/file"));
        assert!(result.is_err());
    }
}
