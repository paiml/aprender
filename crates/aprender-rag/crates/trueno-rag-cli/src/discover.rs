//! File and directory discovery utilities.
//!
//! Walk directories, classify files by extension, filter by glob patterns,
//! and discover media files for transcription.

use anyhow::{Context, Result};
use globset::{Glob, GlobSet, GlobSetBuilder};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use trueno_rag::loader::LoaderRegistry;

/// Media file extensions that can be transcribed.
pub(crate) const MEDIA_EXTENSIONS: &[&str] = &[
    "mp4", "mkv", "avi", "mov", "webm", "flv", "wmv", // video
    "mp3", "wav", "flac", "ogg", "m4a", "aac", "wma", // audio
];

/// Build a GlobSet from exclude patterns. Returns None if no patterns.
pub(crate) fn build_exclude_set(patterns: &[String]) -> Result<Option<GlobSet>> {
    if patterns.is_empty() {
        return Ok(None);
    }
    let mut builder = GlobSetBuilder::new();
    for pattern in patterns {
        builder
            .add(Glob::new(pattern).with_context(|| format!("Invalid exclude glob: {pattern}"))?);
    }
    Ok(Some(builder.build().context("Failed to build exclude set")?))
}

/// Check if a path should be excluded by glob patterns.
pub(crate) fn is_excluded(path: &Path, exclude: &Option<GlobSet>) -> bool {
    match exclude {
        Some(set) => set.is_match(path),
        None => false,
    }
}

/// Process a single directory entry, returning it for collection or recursion.
pub(crate) enum DirEntryAction {
    AcceptFile(PathBuf),
    Recurse(PathBuf),
    Excluded,
    Skip,
}

/// Classify a directory entry for the walk.
pub(crate) fn classify_entry(
    path: PathBuf,
    recursive: bool,
    exclude: &Option<GlobSet>,
    accept: &impl Fn(&Path) -> bool,
) -> DirEntryAction {
    if is_excluded(&path, exclude) {
        return DirEntryAction::Excluded;
    }
    if path.is_dir() && recursive {
        return DirEntryAction::Recurse(path);
    }
    if path.is_file() && accept(&path) {
        return DirEntryAction::AcceptFile(path);
    }
    DirEntryAction::Skip
}

/// Walk a directory tree, collecting files that match `accept`.
pub(crate) fn walk_directory(
    root: &Path,
    recursive: bool,
    exclude: &Option<GlobSet>,
    accept: impl Fn(&Path) -> bool,
) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    let mut dirs_to_visit = vec![root.to_path_buf()];
    let mut excluded_count = 0usize;

    while let Some(dir) = dirs_to_visit.pop() {
        let entries = fs::read_dir(&dir)
            .with_context(|| format!("Failed to read directory: {}", dir.display()))?;

        for entry in entries {
            match classify_entry(entry?.path(), recursive, exclude, &accept) {
                DirEntryAction::AcceptFile(p) => files.push(p),
                DirEntryAction::Recurse(p) => dirs_to_visit.push(p),
                DirEntryAction::Excluded => excluded_count += 1,
                DirEntryAction::Skip => {}
            }
        }
    }

    if excluded_count > 0 {
        println!("Excluded {} paths by glob pattern", excluded_count);
    }

    files.sort();
    Ok(files)
}

/// Discover files from a path using the loader registry.
pub(crate) fn discover_files(
    root: &Path,
    recursive: bool,
    registry: &LoaderRegistry,
    exclude: &Option<GlobSet>,
) -> Result<Vec<PathBuf>> {
    if root.is_file() {
        if registry.loader_for(root).is_some() {
            return Ok(vec![root.to_path_buf()]);
        }
        anyhow::bail!(
            "Unsupported file format: {}",
            root.extension().and_then(|e| e.to_str()).unwrap_or("(none)")
        );
    }

    walk_directory(root, recursive, exclude, |p| registry.loader_for(p).is_some())
}

/// Check if a file has a media extension.
pub(crate) fn is_media_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| MEDIA_EXTENSIONS.contains(&ext.to_lowercase().as_str()))
}

/// Discover media files in a directory.
pub(crate) fn discover_media_files(
    root: &Path,
    recursive: bool,
    exclude: &Option<GlobSet>,
) -> Result<Vec<PathBuf>> {
    if root.is_file() {
        if is_media_file(root) {
            return Ok(vec![root.to_path_buf()]);
        }
        anyhow::bail!("Not a media file: {}", root.display());
    }

    walk_directory(root, recursive, exclude, |p| is_media_file(p))
}

/// Classify media files into those with/without existing sidecars.
pub(crate) fn classify_media_sidecar_status(files: &[PathBuf]) -> (Vec<PathBuf>, Vec<PathBuf>) {
    let mut has_sidecar = Vec::new();
    let mut needs_transcription = Vec::new();

    for file in files {
        if LoaderRegistry::find_sidecar(file).is_some() {
            has_sidecar.push(file.clone());
        } else {
            needs_transcription.push(file.clone());
        }
    }

    (has_sidecar, needs_transcription)
}

/// Classify discovered files by format for progress reporting.
pub(crate) fn classify_files(files: &[PathBuf]) -> HashMap<String, usize> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for file in files {
        let ext = file.extension().and_then(|e| e.to_str()).unwrap_or("other").to_lowercase();
        *counts.entry(ext).or_insert(0) += 1;
    }
    counts
}

/// Discover media files and print a summary of what was found.
pub(crate) fn discover_and_report_media(
    root: &Path,
    recursive: bool,
    exclude: &Option<GlobSet>,
) -> Result<Vec<PathBuf>> {
    println!("Discovering media files...");
    let media_files = discover_media_files(root, recursive, exclude)?;

    if media_files.is_empty() {
        println!("No media files found at: {}", root.display());
        return Ok(media_files);
    }

    let ext_counts = classify_files(&media_files);
    println!(
        "Found {} media files{}",
        media_files.len(),
        if recursive { " (recursive)" } else { "" }
    );
    for (ext, count) in &ext_counts {
        println!("  {} .{} files", count, ext);
    }
    Ok(media_files)
}
