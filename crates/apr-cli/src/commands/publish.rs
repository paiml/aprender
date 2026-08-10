//! HuggingFace Hub publish command (APR-PUB-001)
//!
//! Publishes models to HuggingFace Hub with auto-generated model cards.
//! Now uses native Rust HTTP upload instead of shelling out to Python CLI.
//!
//! # Toyota Way Principles
//!
//! - **Jidoka**: Auto-generate model cards to prevent incomplete documentation
//! - **Genchi Genbutsu**: Verify model before publishing
//! - **Muda**: Eliminate manual model card creation
//! - **Andon Cord**: Fail loudly on upload errors (APR-PUB-001)

use crate::error::CliError;
use aprender::format::model_card::ModelCard;
#[cfg(feature = "hf-hub")]
use aprender::hf_hub::{HfHubClient, PushOptions, UploadProgress};
use std::fs;
use std::path::Path;
#[cfg(feature = "hf-hub")]
use std::sync::Arc;

/// Validate publish inputs: repo ID format, directory exists, model files found.
fn validate_publish_inputs(
    directory: &Path,
    repo_id: &str,
) -> Result<Vec<std::path::PathBuf>, CliError> {
    if !repo_id.contains('/') || repo_id.split('/').count() != 2 {
        return Err(CliError::ValidationFailed(format!(
            "Invalid repo ID '{}'. Expected format: org/repo-name",
            repo_id
        )));
    }

    if !directory.exists() {
        return Err(CliError::FileNotFound(directory.to_path_buf()));
    }

    let files = find_model_files(directory)?;
    if files.is_empty() {
        return Err(CliError::ValidationFailed(format!(
            "No model files found in {}. Expected .apr, .safetensors, or .gguf files.",
            directory.display()
        )));
    }

    Ok(files)
}

/// Describe which HF Hub upload path a file of `size_bytes` will take.
///
/// Surfaces FALSIFY-PUB-LFS-001 (file-size dispatch) observably from the
/// CLI without needing an `HF_TOKEN`. The returned string is appended to
/// the dry-run file listing so a reviewer can verify routing at a glance.
///
/// Built with `--features xet`: `>5 GiB` reports `[→ Xet CAS]`.
/// Built without `--features xet`: `>5 GiB` reports a `FAIL` hint
/// pointing at the required rebuild.
#[cfg(feature = "hf-hub")]
fn format_upload_route(size_bytes: u64) -> &'static str {
    if aprender::hf_hub::xet::should_use_xet(size_bytes) {
        if cfg!(feature = "xet") {
            "[→ Xet CAS (>5 GiB)]"
        } else {
            "[✗ would FAIL: rebuild with --features xet]"
        }
    } else {
        "[→ HTTP LFS (≤5 GiB)]"
    }
}

/// Fallback when built without the `hf-hub` feature — dry-run only.
#[cfg(not(feature = "hf-hub"))]
fn format_upload_route(_size_bytes: u64) -> &'static str {
    "[? hf-hub feature off]"
}

/// One file the dry run would upload.
#[derive(Debug)]
pub(crate) struct PlannedFile {
    pub(crate) path: String,
    pub(crate) size_bytes: u64,
    /// `artifact`, `extra-file`, or `manifest`.
    pub(crate) kind: &'static str,
    pub(crate) route: &'static str,
}

/// What `apr publish --dry-run` would do, separated from how it is rendered.
///
/// Resolution and rendering are kept apart so that the exact bytes written to
/// stdout in JSON mode are [`DryRunPlan::to_json`] — a unit test over that
/// string tests what a consumer will parse.
#[derive(Debug)]
pub(crate) struct DryRunPlan {
    pub(crate) repo_id: String,
    pub(crate) files: Vec<PlannedFile>,
    /// `None` when a manifest supplies provenance and README generation is suppressed.
    pub(crate) readme: Option<String>,
}

impl DryRunPlan {
    /// The complete stdout of `apr publish --dry-run`, in whichever mode was
    /// asked for. Under `--json` that is exactly one JSON document — note that
    /// the generated model card lands in the `readme` string field, where it
    /// cannot corrupt the document the way the raw YAML front-matter did.
    pub(crate) fn stdout(&self, json: bool) -> String {
        if json {
            self.to_json()
        } else {
            self.to_human()
        }
    }

    // serde_json::json!() uses infallible unwrap internally
    #[allow(clippy::disallowed_methods)]
    fn to_json(&self) -> String {
        let files: Vec<serde_json::Value> = self
            .files
            .iter()
            .map(|f| {
                serde_json::json!({
                    "path": f.path,
                    "size_bytes": f.size_bytes,
                    "kind": f.kind,
                    "route": f.route,
                })
            })
            .collect();
        let doc = serde_json::json!({
            "repo_id": self.repo_id,
            "mode": "dry-run",
            "files": files,
            "readme": self.readme,
        });
        serde_json::to_string_pretty(&doc).unwrap_or_default()
    }

    fn to_human(&self) -> String {
        use std::fmt::Write as _;
        let mut out = format!("=== DRY RUN: Would publish to {} ===\n\n", self.repo_id);
        out.push_str("Files to upload:\n");
        for f in &self.files {
            // The manifest line keeps its own KB rendering and carries no upload
            // route, exactly as before — only the JSON mode is new here.
            let _ = match f.kind {
                "manifest" => writeln!(
                    out,
                    "  - {} ({:.1} KB) [manifest]",
                    f.path,
                    f.size_bytes as f64 / 1_000.0
                ),
                "artifact" => writeln!(
                    out,
                    "  - {} ({:.1} MB) {}",
                    f.path,
                    f.size_bytes as f64 / 1_000_000.0,
                    f.route
                ),
                other => writeln!(
                    out,
                    "  - {} ({:.1} MB) [{}] {}",
                    f.path,
                    f.size_bytes as f64 / 1_000_000.0,
                    other,
                    f.route
                ),
            };
        }
        match &self.readme {
            Some(readme) => {
                let _ = write!(out, "\nGenerated README.md:\n\n{readme}\n");
            }
            None => out.push_str(
                "\n(README.md auto-generation suppressed: manifest provides provenance)\n",
            ),
        }
        out.push_str("\n=== DRY RUN COMPLETE ===");
        out
    }
}

pub(crate) fn build_dry_run_plan(
    repo_id: &str,
    files: &[std::path::PathBuf],
    extra_files: &[std::path::PathBuf],
    manifest: Option<&Path>,
    readme_content: &str,
) -> DryRunPlan {
    let mut planned = Vec::new();
    let mut push = |path: &Path, kind: &'static str| {
        let size = fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        planned.push(PlannedFile {
            path: path.display().to_string(),
            size_bytes: size,
            kind,
            route: format_upload_route(size),
        });
    };
    for f in files {
        push(f, "artifact");
    }
    for ef in extra_files {
        push(ef, "extra-file");
    }
    if let Some(m) = manifest {
        push(m, "manifest");
    }
    DryRunPlan {
        repo_id: repo_id.to_string(),
        files: planned,
        readme: if manifest.is_some() {
            None
        } else {
            Some(readme_content.to_string())
        },
    }
}

/// Upload model files, sidecars, and either the manifest (when provided) or an
/// auto-generated README to HuggingFace Hub. Extends `upload_to_hub` for
/// F-PUBLISH-EXTRA-001: iterates extra_files and uploads manifest.yaml verbatim.
#[cfg(feature = "hf-hub")]
#[allow(clippy::too_many_arguments)]
fn upload_to_hub_extended(
    client: &HfHubClient,
    repo_id: &str,
    files: &[std::path::PathBuf],
    companion_files: &[std::path::PathBuf],
    readme_content: &str,
    manifest: Option<&Path>,
    extra_files: &[std::path::PathBuf],
    commit_msg: &str,
    verbose: bool,
) -> Result<(), CliError> {
    let progress_callback: Arc<dyn Fn(UploadProgress) + Send + Sync> = Arc::new(move |progress| {
        if verbose {
            println!(
                "  [{}/{}] {} ({:.1}%)",
                progress.files_completed + 1,
                progress.total_files,
                progress.current_file,
                progress.percentage()
            );
        }
    });

    let upload_one = |src: &Path, path_in_repo: &str| -> Result<(), CliError> {
        if verbose {
            let size = fs::metadata(src).map(|m| m.len()).unwrap_or(0);
            println!(
                "Uploading {} ({:.1} MB)...",
                path_in_repo,
                size as f64 / 1_000_000.0
            );
        }
        let file_data = fs::read(src)?;
        let options = PushOptions::new()
            .with_filename(path_in_repo.to_string())
            .with_commit_message(commit_msg)
            .with_progress_callback(progress_callback.clone())
            .with_create_repo(true);
        client
            .push_to_hub(repo_id, &file_data, options)
            .map_err(|e| CliError::NetworkError(format!("Upload failed: {e}")))?;
        Ok(())
    };

    for file in files {
        let filename = file
            .file_name()
            .ok_or_else(|| CliError::ValidationFailed("Invalid file path".into()))?
            .to_string_lossy()
            .to_string();
        upload_one(file, &filename)?;
    }

    // PMAT-690 defect 6 (2026-05-18): upload companion files (config.json,
    // tokenizer.json, LICENSE, etc.). Skip README.md — it's uploaded
    // separately below from `readme_content` which the caller may have
    // already populated with user-authored content.
    for cf in companion_files {
        let filename = cf
            .file_name()
            .ok_or_else(|| CliError::ValidationFailed("Invalid companion-file path".into()))?
            .to_string_lossy()
            .to_string();
        if filename == "README.md" {
            continue;
        }
        upload_one(cf, &filename)?;
    }

    for ef in extra_files {
        let filename = ef
            .file_name()
            .ok_or_else(|| CliError::ValidationFailed("Invalid extra-file path".into()))?
            .to_string_lossy()
            .to_string();
        upload_one(ef, &filename)?;
    }

    if let Some(manifest_path) = manifest {
        upload_one(manifest_path, "manifest.yaml")?;
    } else {
        if verbose {
            println!("Uploading README.md...");
        }
        let readme_options = PushOptions::new()
            .with_filename("README.md")
            .with_commit_message(commit_msg)
            .with_create_repo(false);
        client
            .push_to_hub(repo_id, readme_content.as_bytes(), readme_options)
            .map_err(|e| CliError::NetworkError(format!("README upload failed: {e}")))?;
    }

    // PMAT-690 defect 6 (2026-05-18): auto-emit `model.safetensors` LFS alias
    // for HF Transformers AutoModelForCausalLM auto-discovery. LFS dedup
    // makes the alias storage-free. See SPEC-HF-PUBLISH-001 §"Publishing
    // the `model.safetensors` alias".
    if let Some(src) = safetensors_needing_alias(files) {
        if verbose {
            println!(
                "Emitting model.safetensors LFS alias for {} (HF Transformers auto-load)",
                src.display()
            );
        }
        emit_safetensors_alias(client, repo_id, &src, commit_msg).map_err(|e| {
            CliError::NetworkError(format!("model.safetensors alias commit failed: {e}"))
        })?;
    }

    Ok(())
}

/// Upload model files and README to HuggingFace Hub.
#[cfg(feature = "hf-hub")]
#[allow(dead_code)]
fn upload_to_hub(
    client: &HfHubClient,
    repo_id: &str,
    files: &[std::path::PathBuf],
    readme_content: &str,
    commit_msg: &str,
    verbose: bool,
) -> Result<(), CliError> {
    let progress_callback: Arc<dyn Fn(UploadProgress) + Send + Sync> = Arc::new(move |progress| {
        if verbose {
            println!(
                "  [{}/{}] {} ({:.1}%)",
                progress.files_completed + 1,
                progress.total_files,
                progress.current_file,
                progress.percentage()
            );
        }
    });

    for file in files {
        let filename = file
            .file_name()
            .ok_or_else(|| CliError::ValidationFailed("Invalid file path".into()))?
            .to_string_lossy()
            .to_string();

        if verbose {
            let size = fs::metadata(file).map(|m| m.len()).unwrap_or(0);
            println!(
                "Uploading {} ({:.1} MB)...",
                filename,
                size as f64 / 1_000_000.0
            );
        }

        let file_data = fs::read(file)?;

        let options = PushOptions::new()
            .with_filename(filename)
            .with_commit_message(commit_msg)
            .with_progress_callback(progress_callback.clone())
            .with_create_repo(true);

        client
            .push_to_hub(repo_id, &file_data, options)
            .map_err(|e| CliError::NetworkError(format!("Upload failed: {e}")))?;
    }

    if verbose {
        println!("Uploading README.md...");
    }

    let readme_options = PushOptions::new()
        .with_filename("README.md")
        .with_commit_message(commit_msg)
        .with_create_repo(false);

    client
        .push_to_hub(repo_id, readme_content.as_bytes(), readme_options)
        .map_err(|e| CliError::NetworkError(format!("README upload failed: {e}")))?;

    Ok(())
}

/// Execute the publish command
///
/// F-PUBLISH-EXTRA-001 (contracts/apr-cli-publish-extra-v1.yaml):
/// When `manifest` is `Some`, validates the publish-manifest-v1.yaml, computes
/// sha256 of the declared local artifact, aborts before any network I/O on
/// mismatch, then uploads the manifest itself + sidecar files. Auto-README is
/// suppressed when the manifest carries provenance.
#[provable_contracts_macros::contract(
    "apr-cli-command-safety-v1",
    equation = "mutating_output_contract"
)]
#[allow(clippy::too_many_arguments)]
pub fn execute(
    directory: &Path,
    repo_id: &str,
    model_name: Option<&str>,
    license: &str,
    pipeline_tag: &str,
    library_name: Option<&str>,
    tags: &[String],
    commit_message: Option<&str>,
    dry_run: bool,
    verbose: bool,
    manifest: Option<&Path>,
    extra_files: &[std::path::PathBuf],
    json: bool,
) -> Result<(), CliError> {
    // When --manifest is provided, the manifest declares the single artifact
    // being shipped for this invocation. We restrict `files` to just that
    // artifact (F-PUBLISH-EXTRA-001::manifest_upload_roundtrip step 4) so
    // that a per-format manifest does not accidentally re-upload sibling
    // formats sitting next to it in the staging directory.
    let files = if let Some(manifest_path) = manifest {
        let artifact = preflight_manifest_guard(manifest_path, directory)?;
        vec![artifact]
    } else {
        validate_publish_inputs(directory, repo_id)?
    };

    // When a manifest is absent the guard above doesn't run; repo_id still
    // needs validation and directory existence must be checked.
    if manifest.is_some() {
        if !repo_id.contains('/') || repo_id.split('/').count() != 2 {
            return Err(CliError::ValidationFailed(format!(
                "Invalid repo ID '{}'. Expected format: org/repo-name",
                repo_id
            )));
        }
        if !directory.exists() {
            return Err(CliError::FileNotFound(directory.to_path_buf()));
        }
    }

    // PMAT-690 P3-C-prep defect 6 (2026-05-18): discover companion files
    // (config.json, vocab.json, merges.txt, tokenizer*.json, generation_config.json,
    // LICENSE, special_tokens_map.json, chat_template.jinja) so the publish includes
    // them automatically per SPEC-HF-PUBLISH-001. Manifest mode is unchanged
    // — manifest-driven publishes restrict to the declared artifact only.
    let companion_files: Vec<std::path::PathBuf> = if manifest.is_some() {
        Vec::new()
    } else {
        find_companion_files(directory)?
    };

    // If the user provided a README.md, use that instead of the auto-generated one
    // — empirical: the auto-generated stub is consistently weaker than what model
    // authors hand-craft (observed paiml/albor-370m-v1 publish 2026-05-17).
    let user_readme: Option<std::path::PathBuf> = companion_files
        .iter()
        .find(|p| p.file_name().and_then(|n| n.to_str()) == Some("README.md"))
        .cloned();

    if verbose {
        println!("Uploading {} primary artifact(s):", files.len());
        for f in &files {
            println!("  - {}", f.display());
        }
        if !companion_files.is_empty() {
            println!("Plus {} companion file(s):", companion_files.len());
            for f in &companion_files {
                println!("  - {}", f.display());
            }
        }
        if let Some(p) = &user_readme {
            println!(
                "User-provided README.md detected at {} — will replace auto-generated card",
                p.display()
            );
        }
    }

    for ef in extra_files {
        if !ef.exists() {
            return Err(CliError::FileNotFound(ef.clone()));
        }
    }

    let (model_card, file_names) = generate_model_card(
        repo_id,
        model_name,
        license,
        pipeline_tag,
        library_name,
        tags,
        &files,
    );
    let readme_content = if let Some(p) = &user_readme {
        fs::read_to_string(p).map_err(|e| {
            CliError::ValidationFailed(format!(
                "Failed to read user README at {}: {e}",
                p.display()
            ))
        })?
    } else {
        model_card.to_huggingface_extended(pipeline_tag, library_name, tags, &file_names)
    };

    if dry_run {
        let plan = build_dry_run_plan(repo_id, &files, extra_files, manifest, &readme_content);
        println!("{}", plan.stdout(json));
        return Ok(());
    }

    #[cfg(not(feature = "hf-hub"))]
    {
        let _ = (commit_message, verbose, manifest, extra_files);
        return Err(CliError::ValidationFailed(
            "Publishing requires the 'hf-hub' feature. Rebuild with: \
             cargo install --path crates/apr-cli --features hf-hub"
                .to_string(),
        ));
    }

    #[cfg(feature = "hf-hub")]
    {
        let client = HfHubClient::new().map_err(|e| {
            CliError::ValidationFailed(format!("Failed to create HF Hub client: {e}"))
        })?;

        if !client.is_authenticated() {
            return Err(CliError::ValidationFailed(
                "HF_TOKEN environment variable not set. Set it with: export HF_TOKEN=hf_...".into(),
            ));
        }

        let commit_msg = commit_message.unwrap_or("Upload via apr-cli publish");

        println!("Publishing to https://huggingface.co/{}", repo_id);
        let extras_size: u64 = extra_files
            .iter()
            .map(|f| fs::metadata(f).map(|m| m.len()).unwrap_or(0))
            .sum();
        let manifest_size: u64 = manifest
            .map(|m| fs::metadata(m).map(|meta| meta.len()).unwrap_or(0))
            .unwrap_or(0);
        let total_size: u64 = files
            .iter()
            .map(|f| fs::metadata(f).map(|m| m.len()).unwrap_or(0))
            .sum::<u64>()
            + extras_size
            + manifest_size
            + if manifest.is_some() {
                0
            } else {
                readme_content.len() as u64
            };
        println!(
            "Total upload size: {:.1} MB",
            total_size as f64 / 1_000_000.0
        );

        upload_to_hub_extended(
            &client,
            repo_id,
            &files,
            &companion_files,
            &readme_content,
            manifest,
            extra_files,
            commit_msg,
            verbose,
        )?;

        println!("\n✓ Published to https://huggingface.co/{}", repo_id);
        Ok(())
    }
}

/// Pre-flight manifest guard (F-PUBLISH-EXTRA-001::manifest_upload_roundtrip).
///
/// Parses the manifest, validates required top-level fields exist, locates the
/// declared artifact in `directory` (by basename of `artifact_url`), computes
/// its local sha256, and aborts on mismatch. Runs BEFORE any network I/O.
/// On success, returns the local path of the manifest-declared artifact so the
/// caller can restrict its upload set to exactly that one file.
fn preflight_manifest_guard(
    manifest_path: &Path,
    directory: &Path,
) -> Result<std::path::PathBuf, CliError> {
    let manifest_src = fs::read_to_string(manifest_path).map_err(|e| {
        CliError::ValidationFailed(format!(
            "Cannot read manifest {}: {e}",
            manifest_path.display()
        ))
    })?;

    let parsed: serde_yaml::Value = serde_yaml::from_str(&manifest_src)
        .map_err(|e| CliError::ValidationFailed(format!("Manifest YAML parse error: {e}")))?;

    let declared_sha = parsed
        .get("sha256")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            CliError::ValidationFailed("Manifest missing required field: sha256".into())
        })?;
    let artifact_url = parsed
        .get("artifact_url")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            CliError::ValidationFailed("Manifest missing required field: artifact_url".into())
        })?;
    let declared_size = parsed.get("size_bytes").and_then(serde_yaml::Value::as_u64);

    let artifact_basename = artifact_url
        .rsplit('/')
        .next()
        .ok_or_else(|| CliError::ValidationFailed("artifact_url has no basename".into()))?;
    let local_artifact = directory.join(artifact_basename);
    if !local_artifact.exists() {
        return Err(CliError::ValidationFailed(format!(
            "Manifest-declared artifact not found locally: {}",
            local_artifact.display()
        )));
    }

    let computed_sha = stream_sha256(&local_artifact)?;
    if computed_sha != declared_sha {
        return Err(CliError::ValidationFailed(format!(
            "sha256 mismatch — manifest-declared vs local artifact.\n  \
             manifest: {declared_sha}\n  \
             local:    {computed_sha}\n  \
             file:     {}",
            local_artifact.display()
        )));
    }

    if let Some(expected) = declared_size {
        let actual = fs::metadata(&local_artifact).map(|m| m.len()).unwrap_or(0);
        if expected != actual {
            return Err(CliError::ValidationFailed(format!(
                "size_bytes mismatch — manifest {expected}, local {actual}"
            )));
        }
    }

    Ok(local_artifact)
}

/// Streaming SHA-256 computation (64 KiB buffer) — same implementation used by
/// `apr validate-manifest` for the FALSIFY-PM-002 gate.
fn stream_sha256(path: &Path) -> Result<String, CliError> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut f = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 65_536];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Find model binary-artifact files in directory: `.apr`, `.safetensors`, `.gguf`.
fn find_model_files(directory: &Path) -> Result<Vec<std::path::PathBuf>, CliError> {
    let mut files = Vec::new();

    let entries = fs::read_dir(directory)?;

    for entry in entries.flatten() {
        let path = entry.path();

        if path.is_file() {
            if let Some(ext) = path.extension() {
                let ext_str = ext.to_string_lossy().to_lowercase();
                if ext_str == "apr" || ext_str == "safetensors" || ext_str == "gguf" {
                    files.push(path);
                }
            }
        }
    }

    // Sort for deterministic order
    files.sort();
    Ok(files)
}

/// Find companion files in directory: the standard HF / Transformers
/// integration set required by SPEC-HF-PUBLISH-001 (PMAT-690 defect 6).
///
/// Returns paths to files matching the allowlist below. The match is
/// case-sensitive exact filename (no extension globbing) because these
/// names are HF-standard and arbitrary `.json` / `.txt` files should NOT
/// be auto-published.
///
/// **Included filenames** (only when present in `directory`):
/// - `README.md` — user-authored model card (overrides auto-generated)
/// - `LICENSE` (and `LICENSE.md`, `LICENSE.txt`)
/// - `config.json` — HF Transformers `AutoConfig`
/// - `generation_config.json` — HF Transformers generation defaults
/// - `tokenizer.json` — HF fast tokenizer (single-file)
/// - `tokenizer_config.json` — chat_template + special tokens
/// - `vocab.json` — legacy BPE vocab
/// - `merges.txt` — legacy BPE merges
/// - `special_tokens_map.json` — special-tokens map (Llama/Mistral arches)
/// - `chat_template.jinja` — standalone chat template (newer HF convention)
///
/// First applied 2026-05-18 to remove the manual NDJSON companion-file
/// upload step the paiml/albor-370m-v1 publish needed.
fn find_companion_files(directory: &Path) -> Result<Vec<std::path::PathBuf>, CliError> {
    const COMPANION_NAMES: &[&str] = &[
        "README.md",
        "LICENSE",
        "LICENSE.md",
        "LICENSE.txt",
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "chat_template.jinja",
    ];

    let mut files = Vec::new();
    for name in COMPANION_NAMES {
        let path = directory.join(name);
        if path.is_file() {
            files.push(path);
        }
    }
    files.sort();
    Ok(files)
}

/// Check if any binary-artifact file is a SafeTensors export NOT named
/// `model.safetensors`. Returns the path of the first such file, which
/// will get an LFS-alias commit emitted after upload so that
/// `AutoModelForCausalLM.from_pretrained` can auto-discover the weights
/// without an explicit `weights_file` argument.
///
/// PMAT-690 defect 6 (2026-05-18): pin the alias-emission heuristic so a
/// regression test can falsify it.
fn safetensors_needing_alias(files: &[std::path::PathBuf]) -> Option<std::path::PathBuf> {
    files
        .iter()
        .find(|p| {
            p.extension().and_then(|e| e.to_str()) == Some("safetensors")
                && p.file_name().and_then(|n| n.to_str()) != Some("model.safetensors")
        })
        .cloned()
}

/// Emit an NDJSON `lfsFile` commit for `model.safetensors` pointing at the
/// same OID as `src` (which was already uploaded with its descriptive name).
///
/// LFS deduplicates by OID, so this is storage-free: both filenames resolve
/// to the same blob. Required for HF Transformers
/// `AutoModelForCausalLM.from_pretrained` to auto-discover the weights
/// without an explicit `weights_file` argument.
///
/// PMAT-690 defect 6 (2026-05-18). See SPEC-HF-PUBLISH-001 §"Publishing
/// the `model.safetensors` alias".
#[cfg(feature = "hf-hub")]
fn emit_safetensors_alias(
    client: &HfHubClient,
    repo_id: &str,
    src: &Path,
    commit_msg: &str,
) -> Result<(), aprender::hf_hub::HfHubError> {
    use sha2::{Digest, Sha256};

    let data = fs::read(src).map_err(|e| {
        aprender::hf_hub::HfHubError::NetworkError(format!(
            "Failed to read {} for alias hashing: {e}",
            src.display()
        ))
    })?;
    let size = data.len();
    let mut hasher = Sha256::new();
    hasher.update(&data);
    let sha256 = format!("{:x}", hasher.finalize());

    client.commit_lfs_alias(
        repo_id,
        "model.safetensors",
        &sha256,
        size,
        &format!("{commit_msg} (model.safetensors alias)"),
    )
}

/// Generate model card from parameters
fn generate_model_card(
    repo_id: &str,
    model_name: Option<&str>,
    license: &str,
    _pipeline_tag: &str,
    _library_name: Option<&str>,
    _tags: &[String],
    files: &[std::path::PathBuf],
) -> (ModelCard, Vec<String>) {
    let name = model_name.unwrap_or_else(|| repo_id.split('/').next_back().unwrap_or(repo_id));

    // GH-511: Collect actual file names for dynamic formats table
    let file_names: Vec<String> = files
        .iter()
        .filter_map(|f| f.file_name())
        .map(|f| f.to_string_lossy().to_string())
        .collect();

    let card = ModelCard::new(repo_id, "1.0.0")
        .with_name(name)
        .with_license(license)
        .with_description(format!("{} model published via aprender", name));

    (card, file_names)
}

// Note: upload_file function removed in APR-PUB-001
// Now using native aprender::hf_hub::HfHubClient instead of shelling out to huggingface-cli

/// Extended model card generation for HuggingFace format
trait ModelCardExt {
    fn to_huggingface_extended(
        &self,
        pipeline_tag: &str,
        library_name: Option<&str>,
        extra_tags: &[String],
        file_names: &[String],
    ) -> String;
}

impl ModelCardExt for ModelCard {
    fn to_huggingface_extended(
        &self,
        pipeline_tag: &str,
        library_name: Option<&str>,
        extra_tags: &[String],
        file_names: &[String],
    ) -> String {
        use std::fmt::Write;

        let mut output = String::from("---\n");

        // License
        if let Some(license) = &self.license {
            let _ = writeln!(output, "license: {}", license.to_lowercase());
        }

        // Language (default to multilingual for ASR)
        if pipeline_tag == "automatic-speech-recognition" {
            output.push_str("language:\n");
            output.push_str("  - en\n");
            output.push_str("  - multilingual\n");
        }

        // Pipeline tag
        let _ = writeln!(output, "pipeline_tag: {}", pipeline_tag);

        // Library name
        if let Some(lib) = library_name {
            let _ = writeln!(output, "library_name: {}", lib);
        }

        // Tags
        output.push_str("tags:\n");
        if let Some(arch) = &self.architecture {
            let _ = writeln!(output, "  - {}", arch.to_lowercase());
        }
        output.push_str("  - aprender\n");
        output.push_str("  - rust\n");

        // Extra tags (deduplicated)
        let mut seen_tags = std::collections::HashSet::new();
        seen_tags.insert("aprender");
        seen_tags.insert("rust");

        // Pipeline-specific tags
        if pipeline_tag == "automatic-speech-recognition" {
            if seen_tags.insert("speech-recognition") {
                output.push_str("  - speech-recognition\n");
            }
            if seen_tags.insert("audio") {
                output.push_str("  - audio\n");
            }
        }

        // Extra tags (skip duplicates)
        for tag in extra_tags {
            if seen_tags.insert(tag.as_str()) {
                let _ = writeln!(output, "  - {}", tag);
            }
        }

        // Model index (results, dataset, and metrics are all required by HuggingFace)
        output.push_str("model-index:\n");
        let _ = writeln!(output, "  - name: {}", self.model_id);
        output.push_str("    results:\n");
        output.push_str("      - task:\n");
        let _ = writeln!(output, "          type: {}", pipeline_tag);
        output.push_str("        dataset:\n");
        output.push_str("          name: custom\n");
        output.push_str("          type: custom\n");
        output.push_str("        metrics:\n");
        if self.metrics.is_empty() {
            // Add placeholder metric when none provided (required by HuggingFace)
            output.push_str("          - name: accuracy\n");
            output.push_str("            type: custom\n");
            output.push_str("            value: N/A\n");
        } else {
            for (key, value) in &self.metrics {
                let _ = writeln!(output, "          - name: {}", key);
                output.push_str("            type: custom\n");
                let _ = writeln!(output, "            value: {}", value);
            }
        }

        output.push_str("---\n\n");

        // Title
        let _ = writeln!(output, "# {}\n", self.name);

        // Description
        if let Some(desc) = &self.description {
            let _ = writeln!(output, "{}\n", desc);
        }

        // GH-511: Formats section generated from actual uploaded files
        output.push_str("## Available Formats\n\n");
        output.push_str("| Format | Description |\n");
        output.push_str("|--------|-------------|\n");
        if file_names.is_empty() {
            // Fallback if no files (shouldn't happen in practice)
            output.push_str("| `model.apr` | Native APR format (streaming, WASM-optimized) |\n");
        } else {
            for name in file_names {
                let desc = match std::path::Path::new(name)
                    .extension()
                    .and_then(|e| e.to_str())
                {
                    Some("apr") => "Native APR format (streaming, WASM-optimized)",
                    Some("safetensors") => "HuggingFace SafeTensors format",
                    Some("gguf") => "GGUF format (llama.cpp compatible)",
                    Some("bin") | Some("pt") | Some("pth") => "PyTorch binary format",
                    _ => "Model file",
                };
                let _ = writeln!(output, "| `{}` | {} |", name, desc);
            }
        }
        output.push('\n');

        // Usage section
        output.push_str("## Usage\n\n");
        output.push_str("```rust\n");
        output.push_str("use aprender::Model;\n");
        output.push('\n');
        output.push_str("let model = Model::load(\"model.apr\")?;\n");
        output.push_str("let result = model.run(&input)?;\n");
        output.push_str("```\n\n");

        // Framework
        output.push_str("## Framework\n\n");
        let _ = writeln!(output, "- **Version:** {}", self.framework_version);
        if let Some(rust) = &self.rust_version {
            let _ = writeln!(output, "- **Rust:** {}", rust);
        }
        output.push('\n');

        // Citation
        output.push_str("## Citation\n\n");
        output.push_str("```bibtex\n");
        output.push_str("@software{aprender,\n");
        output.push_str("  title = {aprender: Rust ML Library},\n");
        output.push_str("  author = {PAIML},\n");
        output.push_str("  year = {2025},\n");
        output.push_str("  url = {https://github.com/paiml/aprender}\n");
        output.push_str("}\n");
        output.push_str("```\n");

        output
    }
}

#[cfg(test)]
#[path = "publish_tests.rs"]
mod tests;
