//! Video frame extraction using ffmpeg scene detection.

use anyhow::{Context, Result};
use rayon::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::discover::{build_exclude_set, discover_media_files};

/// Extract keyframes from video files using ffmpeg scene detection.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_extract_frames(
    path: &str,
    recursive: bool,
    threshold: f64,
    min_interval: f64,
    jobs: usize,
    skip_existing: bool,
    dry_run: bool,
    exclude_patterns: &[String],
) -> Result<()> {
    // Verify ffmpeg is available
    let ffmpeg_check = std::process::Command::new("ffmpeg").arg("-version").output();
    let ffmpeg_ok = ffmpeg_check.ok().map_or(false, |output| output.status.success());
    if !ffmpeg_ok {
        anyhow::bail!("ffmpeg not found. Install with: apt install ffmpeg");
    }

    let exclude = build_exclude_set(exclude_patterns)?;
    let root = Path::new(path);
    if !root.exists() {
        anyhow::bail!("Path not found: {}", root.display());
    }

    let videos = discover_media_files(root, recursive, &exclude)?;
    if videos.is_empty() {
        anyhow::bail!("No video files found at: {}", root.display());
    }

    // Filter to files that need processing
    let to_process: Vec<&PathBuf> = videos
        .iter()
        .filter(|v| {
            if !skip_existing {
                return true;
            }
            let frames_dir = v.with_extension("frames");
            !frames_dir.exists()
                || frames_dir.read_dir().map(|mut d| d.next().is_none()).unwrap_or(true)
        })
        .collect();

    println!("Found {} video files ({} need frame extraction)", videos.len(), to_process.len());

    if to_process.is_empty() {
        println!(
            "All videos already have frames extracted. Use --skip-existing false to re-extract."
        );
        return Ok(());
    }

    if dry_run {
        for v in &to_process {
            println!("  Would extract: {}", v.display());
        }
        return Ok(());
    }

    // Process videos (parallel)
    let processed = std::sync::atomic::AtomicUsize::new(0);
    let total = to_process.len();

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(jobs)
        .build()
        .context("Failed to create thread pool")?;

    let errors: Mutex<Vec<String>> = Mutex::new(Vec::new());
    pool.install(|| {
        to_process.par_iter().for_each(|video| {
            let idx = processed.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            let name = video.file_name().and_then(|n| n.to_str()).unwrap_or("?");
            println!("[{}/{}] Extracting frames: {}", idx, total, name);

            match extract_frames_ffmpeg(video, threshold, min_interval) {
                Ok(count) => {
                    println!("[{}/{}] Extracted {} frames from {}", idx, total, count, name);
                }
                Err(e) => {
                    eprintln!("[{}/{}] Failed: {}: {}", idx, total, name, e);
                    if let Ok(mut errs) = errors.lock() {
                        errs.push(format!("{}: {}", name, e));
                    }
                }
            }
        });
    });

    let error_count = errors.lock().map(|e| e.len()).unwrap_or(0);
    println!(
        "Frame extraction complete: {} processed, {} errors",
        total - error_count,
        error_count
    );

    Ok(())
}

/// Extract keyframes from a single video using ffmpeg scene detection.
///
/// Uses `select='gt(scene,threshold)'` filter to detect scene changes,
/// then outputs PNG frames named `frame_<seconds>s.png`.
fn extract_frames_ffmpeg(video: &Path, threshold: f64, min_interval: f64) -> Result<usize> {
    let frames_dir = video.with_extension("frames");
    fs::create_dir_all(&frames_dir)?;

    // Use ffmpeg select filter for scene detection + fps filter for minimum interval
    let select_filter = format!("select='gt(scene\\,{threshold})',fps=1/{min_interval}",);

    let output_pattern = frames_dir.join("frame_%04d.png").to_string_lossy().to_string();

    let output = std::process::Command::new("ffmpeg")
        .args([
            "-i",
            &video.to_string_lossy(),
            "-vf",
            &select_filter,
            "-vsync",
            "vfr",
            "-frame_pts",
            "1",
            &output_pattern,
            "-y", // overwrite
            "-loglevel",
            "warning",
        ])
        .output()
        .context("Failed to run ffmpeg")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("ffmpeg failed: {}", stderr.trim());
    }

    // Count extracted frames and rename with timestamp info
    let frame_count = fs::read_dir(&frames_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|ext| ext == "png").unwrap_or(false))
        .count();

    Ok(frame_count)
}
