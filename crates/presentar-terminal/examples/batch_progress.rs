#![allow(clippy::unwrap_used, clippy::disallowed_methods)]
//! Batch Job Progress Monitor
//!
//! Demonstrates monitoring multiple batch jobs with progress bars
//! and completion estimates. Useful for data pipelines and ETL jobs.
//!
//! Run with: cargo run -p presentar-terminal --example `batch_progress`

use presentar_core::{Canvas, Color, Point, Rect, TextStyle};
use presentar_terminal::direct::{CellBuffer, DiffRenderer, DirectTerminalCanvas};
use presentar_terminal::ColorMode;

fn main() {
    println!("=== Batch Job Progress Monitor ===\n");

    // Simulate batch jobs
    let jobs = vec![
        Job::new(
            "data_ingest",
            JobStatus::Running,
            0.75,
            1_500_000,
            2_000_000,
        ),
        Job::new(
            "feature_extract",
            JobStatus::Running,
            0.42,
            840_000,
            2_000_000,
        ),
        Job::new("model_train", JobStatus::Running, 0.15, 15, 100),
        Job::new("validation", JobStatus::Pending, 0.0, 0, 50_000),
        Job::new("export_results", JobStatus::Pending, 0.0, 0, 100_000),
        Job::new("cleanup", JobStatus::Completed, 1.0, 10_000, 10_000),
        Job::new("archive_logs", JobStatus::Completed, 1.0, 5_000, 5_000),
        Job::new("notify_slack", JobStatus::Failed, 0.0, 0, 1),
    ];

    // Create buffer
    let mut buffer = CellBuffer::new(80, 24);
    let mut renderer = DiffRenderer::with_color_mode(ColorMode::TrueColor);

    {
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        // Background
        canvas.fill_rect(
            Rect::new(0.0, 0.0, 80.0, 24.0),
            Color::new(0.05, 0.05, 0.1, 1.0),
        );

        // Title
        let title_style = TextStyle {
            color: Color::new(0.9, 0.7, 0.3, 1.0),
            ..Default::default()
        };
        canvas.draw_text(
            "Batch Pipeline Monitor - jobs.yaml",
            Point::new(2.0, 1.0),
            &title_style,
        );

        // Job status summary
        draw_status_summary(&mut canvas, &jobs, 2.0, 3.0);

        // Job progress bars
        draw_job_progress(&mut canvas, &jobs, 2.0, 5.0);

        // Overall progress
        draw_overall_progress(&mut canvas, &jobs, 2.0, 17.0);

        // Resource usage
        draw_resource_usage(&mut canvas, 2.0, 20.0);
    }

    // Render
    let mut output = Vec::with_capacity(8192);
    let cells_written = renderer.flush(&mut buffer, &mut output).unwrap();

    println!("Buffer: {}x{}", buffer.width(), buffer.height());
    println!("Cells written: {cells_written}");
    println!("Output bytes: {}\n", output.len());

    println!("Rendered output:");
    println!("{}", "─".repeat(82));
    std::io::Write::write_all(&mut std::io::stdout(), &output).unwrap();
    println!();
    println!("{}", "─".repeat(82));
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum JobStatus {
    Pending,
    Running,
    Completed,
    Failed,
}

#[allow(clippy::trivially_copy_pass_by_ref)]
impl JobStatus {
    const fn symbol(&self) -> &'static str {
        match self {
            Self::Pending => "○",
            Self::Running => "◐",
            Self::Completed => "●",
            Self::Failed => "✗",
        }
    }

    fn color(&self) -> Color {
        match self {
            Self::Pending => Color::new(0.5, 0.5, 0.5, 1.0),
            Self::Running => Color::new(0.3, 0.7, 1.0, 1.0),
            Self::Completed => Color::new(0.3, 1.0, 0.5, 1.0),
            Self::Failed => Color::new(1.0, 0.3, 0.3, 1.0),
        }
    }
}

struct Job {
    name: String,
    status: JobStatus,
    progress: f64,
    processed: u64,
    total: u64,
}

impl Job {
    fn new(name: &str, status: JobStatus, progress: f64, processed: u64, total: u64) -> Self {
        Self {
            name: name.to_string(),
            status,
            progress,
            processed,
            total,
        }
    }
}

fn draw_status_summary(canvas: &mut DirectTerminalCanvas<'_>, jobs: &[Job], x: f32, y: f32) {
    let running = jobs
        .iter()
        .filter(|j| j.status == JobStatus::Running)
        .count();
    let completed = jobs
        .iter()
        .filter(|j| j.status == JobStatus::Completed)
        .count();
    let pending = jobs
        .iter()
        .filter(|j| j.status == JobStatus::Pending)
        .count();
    let failed = jobs
        .iter()
        .filter(|j| j.status == JobStatus::Failed)
        .count();

    let label_style = TextStyle {
        color: Color::new(0.6, 0.6, 0.6, 1.0),
        ..Default::default()
    };

    canvas.draw_text(
        &format!("Jobs: {} total", jobs.len()),
        Point::new(x, y),
        &label_style,
    );

    let status_items = [
        ("Running", running, Color::new(0.3, 0.7, 1.0, 1.0)),
        ("Completed", completed, Color::new(0.3, 1.0, 0.5, 1.0)),
        ("Pending", pending, Color::new(0.5, 0.5, 0.5, 1.0)),
        ("Failed", failed, Color::new(1.0, 0.3, 0.3, 1.0)),
    ];

    let mut offset = 18.0;
    for (name, count, color) in status_items {
        let style = TextStyle {
            color,
            ..Default::default()
        };
        canvas.draw_text(
            &format!("{name}:{count}"),
            Point::new(x + offset, y),
            &style,
        );
        offset += name.len() as f32 + 4.0;
    }
}

fn draw_job_progress(canvas: &mut DirectTerminalCanvas<'_>, jobs: &[Job], x: f32, y: f32) {
    let header_style = TextStyle {
        color: Color::new(0.5, 0.5, 0.5, 1.0),
        ..Default::default()
    };
    canvas.draw_text(
        "Job Name             Status      Progress                               Items",
        Point::new(x, y),
        &header_style,
    );
    canvas.draw_text(&"─".repeat(76), Point::new(x, y + 1.0), &header_style);

    for (i, job) in jobs.iter().enumerate() {
        let row_y = y + 2.0 + i as f32;
        draw_job_row(canvas, job, x, row_y);
    }
}

fn draw_job_row(canvas: &mut DirectTerminalCanvas<'_>, job: &Job, x: f32, y: f32) {
    // Job name
    let name_style = TextStyle {
        color: Color::new(0.9, 0.9, 0.9, 1.0),
        ..Default::default()
    };
    canvas.draw_text(&format!("{:<18}", job.name), Point::new(x, y), &name_style);

    // Status icon
    let status_style = TextStyle {
        color: job.status.color(),
        ..Default::default()
    };
    canvas.draw_text(
        &format!("{} {:>9}", job.status.symbol(), format!("{:?}", job.status)),
        Point::new(x + 19.0, y),
        &status_style,
    );

    // Progress bar
    let bar_width = 30;
    let filled = (job.progress * bar_width as f64).round() as usize;
    let mut bar = String::with_capacity(bar_width + 2);
    bar.push('[');
    for i in 0..bar_width {
        if i < filled {
            bar.push('█');
        } else {
            bar.push('░');
        }
    }
    bar.push(']');

    let bar_color = if job.status == JobStatus::Failed {
        Color::new(1.0, 0.3, 0.3, 1.0)
    } else {
        job.status.color()
    };
    let bar_style = TextStyle {
        color: bar_color,
        ..Default::default()
    };
    canvas.draw_text(&bar, Point::new(x + 32.0, y), &bar_style);

    // Item count
    let count_style = TextStyle {
        color: Color::new(0.7, 0.7, 0.7, 1.0),
        ..Default::default()
    };
    canvas.draw_text(
        &format_count(job.processed, job.total),
        Point::new(x + 65.0, y),
        &count_style,
    );
}

fn draw_overall_progress(canvas: &mut DirectTerminalCanvas<'_>, jobs: &[Job], x: f32, y: f32) {
    let total_jobs = jobs.len() as f64;
    let completed_jobs = jobs
        .iter()
        .map(|j| {
            if j.status == JobStatus::Completed {
                1.0
            } else {
                j.progress
            }
        })
        .sum::<f64>();
    let overall_pct = (completed_jobs / total_jobs) * 100.0;

    let label_style = TextStyle {
        color: Color::new(0.6, 0.6, 0.6, 1.0),
        ..Default::default()
    };
    canvas.draw_text("Overall Pipeline Progress:", Point::new(x, y), &label_style);

    // Large progress bar
    let bar_width = 50;
    let filled = ((overall_pct / 100.0) * bar_width as f64).round() as usize;
    let mut bar = String::with_capacity(bar_width + 2);
    bar.push('[');
    for i in 0..bar_width {
        if i < filled {
            bar.push('█');
        } else {
            bar.push('░');
        }
    }
    bar.push(']');

    let pct_color = if overall_pct > 90.0 {
        Color::new(0.3, 1.0, 0.5, 1.0)
    } else if overall_pct > 50.0 {
        Color::new(0.9, 0.7, 0.3, 1.0)
    } else {
        Color::new(0.3, 0.7, 1.0, 1.0)
    };
    let bar_style = TextStyle {
        color: pct_color,
        ..Default::default()
    };
    canvas.draw_text(&bar, Point::new(x, y + 1.0), &bar_style);
    canvas.draw_text(
        &format!("{overall_pct:5.1}%"),
        Point::new(x + 54.0, y + 1.0),
        &bar_style,
    );

    // ETA
    let eta_style = TextStyle {
        color: Color::new(0.8, 0.8, 0.8, 1.0),
        ..Default::default()
    };
    canvas.draw_text("ETA: 00:12:34", Point::new(x + 65.0, y + 1.0), &eta_style);
}

fn draw_resource_usage(canvas: &mut DirectTerminalCanvas<'_>, x: f32, y: f32) {
    let label_style = TextStyle {
        color: Color::new(0.5, 0.5, 0.5, 1.0),
        ..Default::default()
    };

    canvas.draw_text(
        "Resources: CPU: 85% | Memory: 12.4/16.0 GB | Disk I/O: 450 MB/s | Workers: 8/8",
        Point::new(x, y),
        &label_style,
    );
    canvas.draw_text(
        "[q] quit  [r] refresh  [p] pause  [c] cancel  [l] logs  [h] help",
        Point::new(x, y + 1.0),
        &label_style,
    );
}

fn format_count(processed: u64, total: u64) -> String {
    if total >= 1_000_000 {
        format!("{:.1}M/{:.1}M", processed as f64 / 1e6, total as f64 / 1e6)
    } else if total >= 1_000 {
        format!("{:.1}K/{:.1}K", processed as f64 / 1e3, total as f64 / 1e3)
    } else {
        format!("{processed}/{total}")
    }
}
