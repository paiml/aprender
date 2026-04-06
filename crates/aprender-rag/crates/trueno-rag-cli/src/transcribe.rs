//! Batch transcription of media files to `.srt` sidecars.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

use crate::discover::{
    build_exclude_set, classify_media_sidecar_status, discover_and_report_media,
};
use crate::BackendType;

/// Transcription manifest for resume support.
#[derive(Serialize, Deserialize, Default)]
pub(crate) struct TranscribeManifest {
    /// Files that have been successfully transcribed.
    pub completed: Vec<String>,
    /// Files that failed transcription.
    pub failed: Vec<String>,
}

impl TranscribeManifest {
    pub(crate) fn load(root: &Path) -> Self {
        let manifest_path = root.join(".transcribe-manifest.json");
        fs::read_to_string(manifest_path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    #[allow(dead_code)]
    pub(crate) fn save(&self, root: &Path) -> Result<()> {
        let manifest_path = root.join(".transcribe-manifest.json");
        let json = serde_json::to_string_pretty(self)?;
        fs::write(manifest_path, json)?;
        Ok(())
    }
}

/// Filter media files by sidecar status and manifest, returning files to process.
pub(crate) fn filter_files_for_transcription(
    media_files: &[PathBuf],
    root: &Path,
    skip_existing: bool,
) -> Vec<PathBuf> {
    let (has_sidecar, needs_transcription) = classify_media_sidecar_status(media_files);
    let manifest = TranscribeManifest::load(root);
    let previously_completed = manifest.completed.len();

    let to_process: Vec<PathBuf> = if skip_existing {
        needs_transcription
            .into_iter()
            .filter(|f| !manifest.completed.contains(&f.to_string_lossy().to_string()))
            .collect()
    } else {
        media_files
            .iter()
            .filter(|f| !manifest.completed.contains(&f.to_string_lossy().to_string()))
            .cloned()
            .collect()
    };

    println!(
        "\nSidecar status: {} with .srt/.vtt, {} need transcription",
        has_sidecar.len(),
        to_process.len()
    );
    if previously_completed > 0 {
        println!("  {} previously completed (from manifest)", previously_completed);
    }
    to_process
}

/// Run the transcription feature gate (or print a message if not available).
#[allow(clippy::unnecessary_wraps)]
fn run_transcription_or_report(
    to_process: &[PathBuf],
    jobs: usize,
    model: Option<&str>,
    backend: BackendType,
    root: &Path,
    prompt: Option<&str>,
    hotwords: &[String],
) -> Result<()> {
    #[cfg(feature = "transcription")]
    {
        run_transcription_batch(to_process, jobs, model, backend, root, prompt, hotwords)?;
    }
    #[cfg(not(feature = "transcription"))]
    {
        let _ = (to_process, jobs, model, backend, root, prompt, hotwords);
        println!(
            "\nTranscription requires the 'transcription' feature.\n\
             Build with: cargo build --release --features transcription\n\n\
             {} files need transcription. Run with --dry-run to list them.",
            to_process.len()
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_transcribe(
    path: &str,
    recursive: bool,
    skip_existing: bool,
    jobs: usize,
    model: Option<&str>,
    backend: BackendType,
    dry_run: bool,
    prompt: Option<&str>,
    hotwords_file: Option<&str>,
    exclude_patterns: &[String],
) -> Result<()> {
    let root = Path::new(path);
    if !root.exists() {
        anyhow::bail!("Path not found: {}", root.display());
    }

    let exclude = build_exclude_set(exclude_patterns)?;
    let start_time = std::time::Instant::now();
    let media_files = discover_and_report_media(root, recursive, &exclude)?;
    if media_files.is_empty() {
        return Ok(());
    }

    let to_process = filter_files_for_transcription(&media_files, root, skip_existing);
    if to_process.is_empty() {
        println!("All files already have sidecars. Nothing to do.");
        return Ok(());
    }

    if dry_run {
        println!("\nDry run — files that would be transcribed:");
        for file in &to_process {
            println!("  {}", file.display());
        }
        println!("\nTotal: {} files", to_process.len());
        return Ok(());
    }

    // Load hotwords from file if provided (one word per line)
    let hotwords: Vec<String> = hotwords_file
        .map(|path| {
            fs::read_to_string(path)
                .unwrap_or_default()
                .lines()
                .map(|l| l.trim().to_string())
                .filter(|l| !l.is_empty() && !l.starts_with('#'))
                .collect()
        })
        .unwrap_or_default();

    if !hotwords.is_empty() {
        println!("Loaded {} hotwords for vocabulary biasing", hotwords.len());
    }
    if let Some(p) = prompt {
        println!("Using prompt: {:?}", p);
    }

    run_transcription_or_report(&to_process, jobs, model, backend, root, prompt, &hotwords)?;

    let elapsed = start_time.elapsed();
    println!(
        "\nTotal time: {:.1}s ({:.1} files/sec)",
        elapsed.as_secs_f64(),
        media_files.len() as f64 / elapsed.as_secs_f64().max(0.001)
    );
    Ok(())
}

/// Run transcription on a batch of media files (requires transcription feature).
#[cfg(feature = "transcription")]
fn run_transcription_batch(
    files: &[PathBuf],
    jobs: usize,
    model: Option<&str>,
    backend_type: BackendType,
    root: &Path,
    prompt: Option<&str>,
    hotwords: &[String],
) -> Result<()> {
    use rayon::prelude::*;
    use std::sync::Mutex;
    use trueno_rag::{
        DocumentLoader, TranscriptionBackend, TranscriptionConfig, TranscriptionLoader,
    };

    let backend = match backend_type {
        BackendType::Cpu => TranscriptionBackend::Cpu,
        BackendType::Gpu => TranscriptionBackend::Gpu,
        BackendType::Cuda => TranscriptionBackend::Cuda,
    };

    let config = TranscriptionConfig {
        model_path: model.map(PathBuf::from),
        backend,
        prompt: prompt.map(String::from),
        hotwords: hotwords.to_vec(),
        ..TranscriptionConfig::default()
    };
    let loader = TranscriptionLoader::new(config);

    if loader.has_model() {
        println!("\nWhisper model loaded. Transcribing {} files...", files.len());
    } else {
        println!(
            "\nNo model specified (use --model <path.apr>). \
             Only files with sidecars will be loaded."
        );
    }

    let batch_start = std::time::Instant::now();
    let manifest = Mutex::new(TranscribeManifest::load(root));
    let success = Mutex::new(0usize);
    let errors = Mutex::new(0usize);

    let process_file = |file: &PathBuf| {
        let filename = file.file_name().and_then(|n| n.to_str()).unwrap_or("?");

        match loader.load(file) {
            Ok(_doc) => {
                *success.lock().expect("success counter mutex poisoned") += 1;
                manifest
                    .lock()
                    .expect("manifest mutex poisoned")
                    .completed
                    .push(file.to_string_lossy().to_string());
                println!("  {} ... ok", filename);
            }
            Err(e) => {
                *errors.lock().expect("error counter mutex poisoned") += 1;
                manifest
                    .lock()
                    .expect("manifest mutex poisoned")
                    .failed
                    .push(file.to_string_lossy().to_string());
                println!("  {} ... FAILED: {e}", filename);
            }
        }
    };

    if jobs > 1 {
        println!("Using {} parallel transcription jobs", jobs);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(jobs)
            .build()
            .context("Failed to create thread pool")?;
        pool.install(|| {
            files.par_iter().for_each(process_file);
        });
    } else {
        files.iter().for_each(process_file);
    }

    // Final manifest save
    let manifest = manifest.into_inner().expect("manifest mutex poisoned");
    manifest.save(root)?;

    let success = success.into_inner().expect("success counter mutex poisoned");
    let errors = errors.into_inner().expect("error counter mutex poisoned");
    let elapsed = batch_start.elapsed();
    println!(
        "\nComplete: {} succeeded, {} failed out of {} total ({:.1}s, {:.1} files/sec)",
        success,
        errors,
        files.len(),
        elapsed.as_secs_f64(),
        files.len() as f64 / elapsed.as_secs_f64().max(0.001),
    );

    Ok(())
}
