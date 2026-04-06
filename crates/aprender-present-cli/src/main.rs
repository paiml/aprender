//! Presentar CLI - serve and bundle WASM apps.

#![allow(
    clippy::needless_pass_by_value,
    clippy::uninlined_format_args,
    clippy::ptr_arg,
    clippy::unwrap_used,
    clippy::disallowed_methods,
    clippy::too_many_lines,
    clippy::too_many_arguments,
    clippy::module_name_repetitions,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::manual_let_else,
    clippy::collapsible_if,
    clippy::match_same_arms,
    clippy::if_not_else,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::single_match_else,
    clippy::items_after_statements,
    clippy::doc_markdown,
    clippy::needless_raw_string_hashes
)]

use clap::{Parser, Subcommand};
use std::fs;
use std::io::Read;
#[cfg(feature = "dev-server")]
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::Command;
#[cfg(feature = "dev-server")]
use std::sync::atomic::{AtomicU64, Ordering};
use tiny_http::{Response, Server};
#[cfg(feature = "dev-server")]
use tungstenite::accept;

#[derive(Parser)]
#[command(name = "presentar")]
#[command(about = "WASM-first visualization framework CLI")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start development server with hot reload
    Serve {
        /// Port to serve on
        #[arg(short, long, default_value = "8080")]
        port: u16,

        /// Directory to serve (default: www)
        #[arg(short, long, default_value = "www")]
        dir: PathBuf,

        /// Watch for changes and rebuild
        #[arg(short, long)]
        watch: bool,
    },

    /// Build optimized WASM bundle
    Bundle {
        /// Output directory
        #[arg(short, long, default_value = "dist")]
        output: PathBuf,

        /// Skip wasm-opt optimization
        #[arg(long)]
        no_optimize: bool,
    },

    /// Create new Presentar project
    New {
        /// Project name
        name: String,
    },

    /// Check YAML manifest validity
    Check {
        /// Path to manifest file
        #[arg(default_value = "app.yaml")]
        manifest: PathBuf,
    },

    /// Compute quality score for a manifest
    Score {
        /// Path to manifest file
        #[arg(default_value = "app.yaml")]
        manifest: PathBuf,

        /// Output format (text, json, badge)
        #[arg(short, long, default_value = "text")]
        format: String,

        /// Output file for badge (svg)
        #[arg(long)]
        badge: Option<PathBuf>,
    },

    /// Run quality gates validation
    Gate {
        /// Path to manifest file
        #[arg(default_value = "app.yaml")]
        manifest: PathBuf,

        /// Minimum passing grade (F, D, C, B, A)
        #[arg(short, long, default_value = "B")]
        min_grade: String,

        /// Minimum score (0-100)
        #[arg(short = 's', long)]
        min_score: Option<f64>,

        /// Strict mode - fail on any warning
        #[arg(long)]
        strict: bool,
    },

    /// Deploy application to cloud hosting
    Deploy {
        /// Source directory to deploy
        #[arg(short, long, default_value = "dist")]
        source: PathBuf,

        /// Deployment target (s3, cloudflare, vercel, netlify, local)
        #[arg(short, long, default_value = "s3")]
        target: String,

        /// S3 bucket name or deployment URL
        #[arg(short, long)]
        bucket: Option<String>,

        /// CloudFront distribution ID for cache invalidation
        #[arg(long)]
        distribution: Option<String>,

        /// AWS region for S3 deployment
        #[arg(long, default_value = "us-east-1")]
        region: String,

        /// Dry run - show what would be deployed without actually deploying
        #[arg(long)]
        dry_run: bool,

        /// Skip bundle step (deploy existing files)
        #[arg(long)]
        skip_build: bool,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Serve { port, dir, watch } => {
            serve(port, dir, watch);
        }
        Commands::Bundle {
            output,
            no_optimize,
        } => {
            bundle(output, no_optimize);
        }
        Commands::New { name } => {
            new_project(&name);
        }
        Commands::Check { manifest } => {
            check_manifest(&manifest);
        }
        Commands::Score {
            manifest,
            format,
            badge,
        } => {
            compute_score(&manifest, &format, badge.as_ref());
        }
        Commands::Gate {
            manifest,
            min_grade,
            min_score,
            strict,
        } => {
            run_gates(&manifest, &min_grade, min_score, strict);
        }
        Commands::Deploy {
            source,
            target,
            bucket,
            distribution,
            region,
            dry_run,
            skip_build,
        } => {
            deploy(
                &source,
                &target,
                bucket.as_deref(),
                distribution.as_deref(),
                &region,
                dry_run,
                skip_build,
            );
        }
    }
}

#[cfg(feature = "dev-server")]
/// Atomic counter for triggering hot reload
static RELOAD_COUNTER: AtomicU64 = AtomicU64::new(0);

#[cfg(feature = "dev-server")]
/// Hot reload WebSocket script to inject into HTML
const HOT_RELOAD_SCRIPT: &str = r#"
<script>
(function() {
    const ws = new WebSocket('ws://localhost:35729');
    let lastReload = 0;
    ws.onopen = () => console.log('[hot-reload] Connected');
    ws.onmessage = (e) => {
        const version = parseInt(e.data, 10);
        if (version > lastReload) {
            lastReload = version;
            console.log('[hot-reload] Reloading...');
            location.reload();
        }
    };
    ws.onclose = () => {
        console.log('[hot-reload] Disconnected, reconnecting in 2s...');
        setTimeout(() => location.reload(), 2000);
    };
})();
</script>
"#;

fn serve(port: u16, dir: PathBuf, watch: bool) {
    println!("Starting Presentar dev server...");
    println!("  Serving: {}", dir.display());
    println!("  URL: http://localhost:{}", port);
    if watch {
        println!("  Watch: enabled (rebuilds on file changes)");
        println!("  Hot reload: ws://localhost:35729");
    }
    println!();
    println!("Press Ctrl+C to stop");

    // Start file watcher in background thread (requires dev-server feature)
    #[cfg(feature = "dev-server")]
    if watch {
        let watch_dir = dir.clone();
        std::thread::spawn(move || {
            watch_and_rebuild(&watch_dir);
        });

        // Start WebSocket server for hot reload
        std::thread::spawn(|| {
            run_hot_reload_server();
        });
    }
    #[cfg(not(feature = "dev-server"))]
    if watch {
        eprintln!(
            "Warning: --watch requires the 'dev-server' feature. Serving without hot reload."
        );
    }

    let addr = format!("0.0.0.0:{}", port);
    let server = Server::http(&addr).expect("Failed to start server");

    for request in server.incoming_requests() {
        let url = request.url().to_string();
        let path = if url == "/" {
            dir.join("index.html")
        } else {
            dir.join(url.trim_start_matches('/'))
        };

        let response = if path.exists() && path.is_file() {
            let mut file = fs::File::open(&path).expect("open file");
            let mut content = Vec::new();
            file.read_to_end(&mut content).expect("read file");

            let content_type = match path.extension().and_then(|e| e.to_str()) {
                Some("html") => "text/html",
                Some("js") => "application/javascript",
                Some("wasm") => "application/wasm",
                Some("css") => "text/css",
                Some("json") => "application/json",
                Some("svg") => "image/svg+xml",
                Some("png") => "image/png",
                Some("yaml" | "yml") => "text/yaml",
                _ => "application/octet-stream",
            };

            // Inject hot reload script into HTML files when watching
            #[cfg(feature = "dev-server")]
            let content = if watch && content_type == "text/html" {
                inject_hot_reload_script(&content)
            } else {
                content
            };
            #[cfg(not(feature = "dev-server"))]
            let content = content;

            Response::from_data(content).with_header(
                tiny_http::Header::from_bytes(&b"Content-Type"[..], content_type.as_bytes())
                    .expect("header"),
            )
        } else {
            Response::from_string("404 Not Found").with_status_code(404)
        };

        let _ = request.respond(response);
    }
}

#[cfg(feature = "dev-server")]
/// Inject hot reload script before </body> or at end of HTML
fn inject_hot_reload_script(html: &[u8]) -> Vec<u8> {
    let html_str = String::from_utf8_lossy(html);
    if let Some(pos) = html_str.rfind("</body>") {
        let mut result = html_str[..pos].to_string();
        result.push_str(HOT_RELOAD_SCRIPT);
        result.push_str(&html_str[pos..]);
        result.into_bytes()
    } else {
        let mut result = html.to_vec();
        result.extend_from_slice(HOT_RELOAD_SCRIPT.as_bytes());
        result
    }
}

#[cfg(feature = "dev-server")]
/// Check if WebSocket client disconnected (non-blocking).
/// Returns true if client is still connected.
fn is_client_connected(websocket: &mut tungstenite::WebSocket<std::net::TcpStream>) -> bool {
    websocket.get_mut().set_nonblocking(true).ok();
    let connected = match websocket.read() {
        Ok(tungstenite::Message::Close(_)) => false,
        Err(tungstenite::Error::Io(ref e)) if e.kind() == std::io::ErrorKind::WouldBlock => true,
        Err(_) => false,
        _ => true,
    };
    websocket.get_mut().set_nonblocking(false).ok();
    connected
}

#[cfg(feature = "dev-server")]
/// Handle a single WebSocket client connection for hot reload.
fn handle_hot_reload_client(mut websocket: tungstenite::WebSocket<std::net::TcpStream>) {
    let mut last_sent = RELOAD_COUNTER.load(Ordering::Relaxed);

    // Send current version immediately
    let _ = websocket.send(tungstenite::Message::Text(last_sent.to_string().into()));

    // Poll for changes
    loop {
        std::thread::sleep(std::time::Duration::from_millis(100));
        let current = RELOAD_COUNTER.load(Ordering::Relaxed);
        if current > last_sent {
            last_sent = current;
            if websocket
                .send(tungstenite::Message::Text(current.to_string().into()))
                .is_err()
            {
                break;
            }
        }
        if !is_client_connected(&mut websocket) {
            break;
        }
    }
}

#[cfg(feature = "dev-server")]
/// Run WebSocket server that broadcasts reload events
fn run_hot_reload_server() {
    let server = match TcpListener::bind("127.0.0.1:35729") {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[hot-reload] Failed to bind WebSocket server: {}", e);
            return;
        }
    };

    for stream in server.incoming() {
        let Ok(stream) = stream else { continue };
        std::thread::spawn(move || {
            if let Ok(websocket) = accept(stream) {
                handle_hot_reload_client(websocket);
            }
        });
    }
}

#[cfg(feature = "dev-server")]
/// Trigger a hot reload for all connected clients
fn trigger_hot_reload() {
    RELOAD_COUNTER.fetch_add(1, Ordering::Relaxed);
}

#[cfg(feature = "dev-server")]
/// Handle YAML file change: validate and trigger reload if valid.
fn handle_yaml_change(path: &std::path::Path) {
    println!("[watch] YAML change: {}", path.display());
    if let Ok(content) = fs::read_to_string(path) {
        match presentar_yaml::Manifest::from_yaml(&content) {
            Ok(_) => {
                println!("[watch] YAML valid");
                trigger_hot_reload();
            }
            Err(e) => eprintln!("[watch] YAML error: {}", e),
        }
    }
}

#[cfg(feature = "dev-server")]
/// Handle file change based on extension.
fn handle_file_change(path: &std::path::Path) {
    let ext = path.extension().and_then(|e| e.to_str());
    match ext {
        Some("rs") => {
            println!("[watch] Rust change detected, rebuilding WASM...");
            rebuild_wasm();
            trigger_hot_reload();
        }
        Some("yaml" | "yml") => handle_yaml_change(path),
        Some("html" | "js" | "css") => {
            println!("[watch] Static file changed: {}", path.display());
            trigger_hot_reload();
        }
        _ => {}
    }
}

#[cfg(feature = "dev-server")]
fn watch_and_rebuild(dir: &PathBuf) {
    use notify::{Config, RecommendedWatcher, RecursiveMode, Watcher};
    use std::sync::mpsc::channel;
    use std::time::Duration;

    let (tx, rx) = channel();

    let mut watcher = RecommendedWatcher::new(
        move |res| {
            if let Ok(event) = res {
                let _ = tx.send(event);
            }
        },
        Config::default().with_poll_interval(Duration::from_secs(1)),
    )
    .expect("watcher");

    watcher.watch(dir, RecursiveMode::Recursive).expect("watch");

    // Also watch crates directory for Rust changes
    if PathBuf::from("crates").exists() {
        watcher
            .watch(&PathBuf::from("crates"), RecursiveMode::Recursive)
            .ok();
    }

    println!("[watch] Watching for changes...");

    let mut last_rebuild = std::time::Instant::now();
    let debounce = Duration::from_secs(1);

    loop {
        match rx.recv_timeout(Duration::from_millis(500)) {
            Ok(event) if last_rebuild.elapsed() > debounce => {
                if let Some(path) = event.paths.first() {
                    handle_file_change(path);
                    last_rebuild = std::time::Instant::now();
                }
            }
            Ok(_) => {} // Debounced
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(_) => break,
        }
    }
}

#[cfg(feature = "dev-server")]
fn rebuild_wasm() {
    let status = Command::new("wasm-pack")
        .args([
            "build",
            "crates/presentar",
            "--target",
            "web",
            "--out-dir",
            "../../www/pkg",
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();

    match status {
        Ok(s) if s.success() => println!("[watch] WASM rebuild complete"),
        _ => eprintln!("[watch] WASM rebuild failed"),
    }
}

fn bundle(output: PathBuf, no_optimize: bool) {
    println!("Building Presentar WASM bundle...");

    // Build with wasm-pack
    let status = Command::new("wasm-pack")
        .args([
            "build",
            "crates/presentar",
            "--target",
            "web",
            "--release",
            "--out-dir",
        ])
        .arg(format!("../../{}", output.join("pkg").display()))
        .status()
        .expect("wasm-pack failed");

    if !status.success() {
        eprintln!("wasm-pack build failed");
        std::process::exit(1);
    }

    // Run wasm-opt for size reduction.
    if !no_optimize {
        let wasm_file = output.join("pkg/presentar_bg.wasm");
        if wasm_file.exists() {
            println!("Optimizing with wasm-opt...");
            let _ = Command::new("wasm-opt")
                .args(["-Oz", "-o"])
                .arg(&wasm_file)
                .arg(&wasm_file)
                .status();
        }
    }

    // Copy index.html if exists
    if PathBuf::from("www/index.html").exists() {
        fs::create_dir_all(&output).ok();
        fs::copy("www/index.html", output.join("index.html")).ok();
        println!("Copied index.html");
    }

    // Print bundle size
    let wasm_file = output.join("pkg/presentar_bg.wasm");
    if wasm_file.exists() {
        let size = fs::metadata(&wasm_file).map(|m| m.len()).unwrap_or(0);
        println!();
        println!("Bundle built successfully!");
        println!("  Output: {}", output.display());
        println!("  WASM size: {} KB", size / 1024);
    }
}

fn new_project(name: &str) {
    println!("Creating new Presentar project: {}", name);

    let project_dir = PathBuf::from(name);
    fs::create_dir_all(&project_dir).expect("create project dir");
    fs::create_dir_all(project_dir.join("www")).expect("create www dir");

    // Create app.yaml
    let manifest = format!(
        r#"presentar: "0.1"
name: {}
version: "1.0.0"
description: A Presentar application

layout:
  type: dashboard
  columns: 12
  gap: 16
  sections:
    - id: header
      span: [1, 12]
      widgets:
        - type: text
          content: "Welcome to {}"
          style: heading
"#,
        name, name
    );
    fs::write(project_dir.join("app.yaml"), manifest).expect("write manifest");

    // Create index.html
    let html = r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Presentar App</title>
</head>
<body>
    <canvas id="canvas" width="800" height="600"></canvas>
    <script type="module">
        import init, { App } from './pkg/presentar.js';
        async function main() {
            await init();
            const app = new App('canvas');
            // Your app code here
        }
        main();
    </script>
</body>
</html>
"#;
    fs::write(project_dir.join("www/index.html"), html).expect("write html");

    println!();
    println!("Project created! Next steps:");
    println!("  cd {}", name);
    println!("  presentar serve");
}

fn check_manifest(path: &PathBuf) {
    println!("Checking manifest: {}", path.display());

    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to read manifest: {}", e);
            std::process::exit(1);
        }
    };

    match presentar_yaml::Manifest::from_yaml(&content) {
        Ok(manifest) => {
            println!("Manifest valid!");
            println!("  Name: {}", manifest.name);
            println!("  Version: {}", manifest.version);
            println!("  Data sources: {}", manifest.data.len());
            println!("  Sections: {}", manifest.layout.sections.len());
        }
        Err(e) => {
            eprintln!("Manifest invalid: {}", e);
            std::process::exit(1);
        }
    }
}

fn compute_score(path: &PathBuf, format: &str, badge_path: Option<&PathBuf>) {
    println!("Computing quality score for: {}", path.display());

    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to read manifest: {}", e);
            std::process::exit(1);
        }
    };

    let manifest = match presentar_yaml::Manifest::from_yaml(&content) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Manifest invalid: {}", e);
            std::process::exit(1);
        }
    };

    // Compute scores based on manifest analysis
    let score = analyze_manifest_quality(&manifest);

    match format {
        "json" => {
            println!(
                r#"{{"score": {:.1}, "grade": "{}", "structural": {:.1}, "performance": {:.1}, "accessibility": {:.1}, "data": {:.1}, "documentation": {:.1}, "consistency": {:.1}}}"#,
                score.overall,
                score.grade,
                score.structural,
                score.performance,
                score.accessibility,
                score.data,
                score.documentation,
                score.consistency
            );
        }
        "badge" => {
            let badge = generate_badge(&score);
            if let Some(p) = badge_path {
                fs::write(p, badge).expect("write badge");
                println!("Badge written to: {}", p.display());
            } else {
                println!("{badge}");
            }
        }
        _ => {
            println!();
            println!("Quality Score: {:.1}/100 ({})", score.overall, score.grade);
            println!();
            println!("Breakdown:");
            println!("  Structural:     {:.1}/25", score.structural);
            println!("  Performance:    {:.1}/20", score.performance);
            println!("  Accessibility:  {:.1}/20", score.accessibility);
            println!("  Data Quality:   {:.1}/15", score.data);
            println!("  Documentation:  {:.1}/10", score.documentation);
            println!("  Consistency:    {:.1}/10", score.consistency);
        }
    }

    // Generate badge if requested
    if let Some(p) = badge_path {
        if format != "badge" {
            let badge = generate_badge(&score);
            fs::write(p, badge).expect("write badge");
            println!("Badge written to: {}", p.display());
        }
    }
}

#[derive(Debug)]
struct QualityScore {
    overall: f64,
    grade: String,
    structural: f64,
    performance: f64,
    accessibility: f64,
    data: f64,
    documentation: f64,
    consistency: f64,
}

#[allow(clippy::too_many_lines)]
fn analyze_manifest_quality(manifest: &presentar_yaml::Manifest) -> QualityScore {
    // Structural (25 points max)
    let widget_count: usize = manifest
        .layout
        .sections
        .iter()
        .map(|s| s.widgets.len())
        .sum();
    let section_count = manifest.layout.sections.len();

    let structural_score = {
        let widget_score = (widget_count.min(20) as f64 / 20.0) * 10.0; // Up to 10 points for widgets
        let section_score = (section_count.min(5) as f64 / 5.0) * 8.0; // Up to 8 points for sections
        let layout_score = if manifest.layout.columns > 0 {
            7.0
        } else {
            0.0
        }; // 7 points for grid layout
        widget_score + section_score + layout_score
    };

    // Performance (20 points max) - estimate based on widget complexity
    let performance_score = {
        let complexity_penalty = (widget_count as f64 / 50.0).min(1.0) * 5.0;
        20.0 - complexity_penalty
    };

    // Accessibility (20 points max)
    let accessibility_score = {
        let has_description = !manifest.description.is_empty();
        let has_sections = !manifest.layout.sections.is_empty();
        let sections_have_ids = manifest.layout.sections.iter().all(|s| !s.id.is_empty());

        let mut score = 0.0;
        if has_description {
            score += 8.0;
        }
        if has_sections {
            score += 6.0;
        }
        if sections_have_ids {
            score += 6.0;
        }
        score
    };

    // Data Quality (15 points max)
    let data_score = {
        let has_data_sources = !manifest.data.is_empty();
        let data_count = manifest.data.len();
        let has_refresh = manifest.data.values().any(|d| d.refresh.is_some());

        let mut score = 0.0;
        if has_data_sources {
            score += 7.0;
        }
        score += (data_count.min(3) as f64 / 3.0) * 5.0;
        if has_refresh {
            score += 3.0;
        }
        score
    };

    // Documentation (10 points max)
    let documentation_score = {
        let has_name = !manifest.name.is_empty();
        let has_version = !manifest.version.is_empty();
        let has_description = !manifest.description.is_empty();

        let mut score = 0.0;
        if has_name {
            score += 3.0;
        }
        if has_version {
            score += 3.0;
        }
        if has_description {
            score += 4.0;
        }
        score
    };

    // Consistency (10 points max)
    let consistency_score = {
        // Check for consistent naming conventions
        let ids_consistent = manifest.layout.sections.iter().all(|s| {
            s.id.chars()
                .all(|c| c.is_ascii_lowercase() || c == '-' || c == '_')
        });

        let widgets_have_types = manifest
            .layout
            .sections
            .iter()
            .flat_map(|s| &s.widgets)
            .all(|w| !w.widget_type.is_empty());

        let mut score = 0.0;
        if ids_consistent {
            score += 5.0;
        }
        if widgets_have_types {
            score += 5.0;
        }
        score
    };

    let overall = structural_score
        + performance_score
        + accessibility_score
        + data_score
        + documentation_score
        + consistency_score;

    let grade = match overall as u32 {
        90..=100 => "A+",
        85..=89 => "A",
        80..=84 => "A-",
        77..=79 => "B+",
        73..=76 => "B",
        70..=72 => "B-",
        67..=69 => "C+",
        63..=66 => "C",
        60..=62 => "C-",
        50..=59 => "D",
        _ => "F",
    }
    .to_string();

    QualityScore {
        overall,
        grade,
        structural: structural_score,
        performance: performance_score,
        accessibility: accessibility_score,
        data: data_score,
        documentation: documentation_score,
        consistency: consistency_score,
    }
}

fn generate_badge(score: &QualityScore) -> String {
    let color = match score.overall as u32 {
        80..=100 => "#4c1",
        60..=79 => "#a3c51c",
        40..=59 => "#dfb317",
        _ => "#e05d44",
    };

    format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"100\" height=\"20\">\
<linearGradient id=\"b\" x2=\"0\" y2=\"100%\">\
<stop offset=\"0\" stop-color=\"#bbb\" stop-opacity=\".1\"/>\
<stop offset=\"1\" stop-opacity=\".1\"/>\
</linearGradient>\
<mask id=\"a\"><rect width=\"100\" height=\"20\" rx=\"3\" fill=\"#fff\"/></mask>\
<g mask=\"url(#a)\">\
<path fill=\"#555\" d=\"M0 0h55v20H0z\"/>\
<path fill=\"{color}\" d=\"M55 0h45v20H55z\"/>\
</g>\
<g fill=\"#fff\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"11\">\
<text x=\"27.5\" y=\"15\" fill=\"#010101\" fill-opacity=\".3\">quality</text>\
<text x=\"27.5\" y=\"14\">quality</text>\
<text x=\"77\" y=\"15\" fill=\"#010101\" fill-opacity=\".3\">{grade}</text>\
<text x=\"77\" y=\"14\">{grade}</text>\
</g>\
</svg>",
        color = color,
        grade = score.grade
    )
}

fn run_gates(path: &PathBuf, min_grade: &str, min_score: Option<f64>, strict: bool) {
    println!("Running quality gates for: {}", path.display());

    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to read manifest: {}", e);
            std::process::exit(1);
        }
    };

    let manifest = match presentar_yaml::Manifest::from_yaml(&content) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("GATE FAILED: Invalid manifest - {}", e);
            std::process::exit(1);
        }
    };

    let score = analyze_manifest_quality(&manifest);
    let mut failures = Vec::new();
    let mut warnings = Vec::new();

    // Check minimum grade
    let grade_value = grade_to_value(&score.grade);
    let min_grade_value = grade_to_value(min_grade);

    if grade_value < min_grade_value {
        failures.push(format!(
            "Grade {} is below minimum {}",
            score.grade, min_grade
        ));
    }

    // Check minimum score
    if let Some(min) = min_score {
        if score.overall < min {
            failures.push(format!(
                "Score {:.1} is below minimum {:.1}",
                score.overall, min
            ));
        }
    }

    // Additional gate checks
    if score.accessibility < 10.0 {
        warnings.push("Low accessibility score - consider adding descriptions and ARIA labels");
    }

    if score.documentation < 5.0 {
        warnings.push("Poor documentation - add name, version, and description");
    }

    if manifest.layout.sections.is_empty() {
        warnings.push("No sections defined in layout");
    }

    // Report results
    println!();
    println!("Score: {:.1}/100 ({})", score.overall, score.grade);
    println!();

    if !warnings.is_empty() {
        println!("Warnings:");
        for w in &warnings {
            println!("  - {}", w);
        }
        println!();
    }

    if !failures.is_empty() {
        println!("Failures:");
        for f in &failures {
            println!("  - {}", f);
        }
        println!();
        eprintln!("GATE FAILED");
        std::process::exit(1);
    }

    if strict && !warnings.is_empty() {
        eprintln!("GATE FAILED (strict mode)");
        std::process::exit(1);
    }

    println!("GATE PASSED");
}

fn grade_to_value(grade: &str) -> u32 {
    match grade.to_uppercase().as_str() {
        "A+" => 97,
        "A" => 93,
        "A-" => 90,
        "B+" => 87,
        "B" => 83,
        "B-" => 80,
        "C+" => 77,
        "C" => 73,
        "C-" => 70,
        "D+" => 67,
        "D" => 63,
        "D-" => 60,
        "F" => 50,
        _ => 0,
    }
}

/// Deploy application to cloud hosting.
fn deploy(
    source: &PathBuf,
    target: &str,
    bucket: Option<&str>,
    distribution: Option<&str>,
    region: &str,
    dry_run: bool,
    skip_build: bool,
) {
    println!("Deploying Presentar application...");
    println!("  Target: {}", target);
    println!("  Source: {}", source.display());
    if dry_run {
        println!("  Mode: DRY RUN (no actual changes)");
    }
    println!();

    // Build if needed
    if !skip_build {
        println!("Step 1: Building production bundle...");
        if dry_run {
            println!(
                "  [dry-run] Would run: presentar bundle --output {}",
                source.display()
            );
        } else {
            bundle(source.clone(), false);
        }
        println!();
    }

    // Verify source directory exists
    if !dry_run && !source.exists() {
        eprintln!(
            "Error: Source directory '{}' does not exist",
            source.display()
        );
        eprintln!("Run 'presentar bundle' first or use --skip-build with existing files");
        std::process::exit(1);
    }

    // Deploy based on target
    match target.to_lowercase().as_str() {
        "s3" => deploy_to_s3(source, bucket, distribution, region, dry_run),
        "cloudflare" => deploy_to_cloudflare(source, bucket, dry_run),
        "vercel" => deploy_to_vercel(source, dry_run),
        "netlify" => deploy_to_netlify(source, dry_run),
        "local" => deploy_to_local(source, bucket, dry_run),
        _ => {
            eprintln!("Unknown deployment target: {}", target);
            eprintln!("Supported targets: s3, cloudflare, vercel, netlify, local");
            std::process::exit(1);
        }
    }
}

/// Print files that would be uploaded (dry-run mode).
fn print_deploy_files(source: &PathBuf, files: &[(PathBuf, String)]) {
    println!();
    println!("Files that would be uploaded:");
    for (path, content_type) in files {
        let rel_path = path.strip_prefix(source).unwrap_or(path);
        println!(
            "  {} ({}, {} bytes)",
            rel_path.display(),
            content_type,
            fs::metadata(path).map(|m| m.len()).unwrap_or(0)
        );
    }
}

/// Upload a single file to S3.
fn upload_file_to_s3(path: &PathBuf, source: &PathBuf, bucket: &str, region: &str) {
    let rel_path = path.strip_prefix(source).unwrap_or(path);
    let s3_key = rel_path.to_string_lossy();
    let content_type = get_content_type(path);

    let mut cmd = Command::new("aws");
    cmd.args(["s3", "cp"])
        .arg(path)
        .arg(format!("s3://{}/{}", bucket, s3_key))
        .args(["--content-type", &content_type])
        .args(["--region", region]);

    let cache_control = get_cache_control(path);
    if !cache_control.is_empty() {
        cmd.args(["--cache-control", cache_control]);
    }

    match cmd.status() {
        Ok(s) if s.success() => println!("  Uploaded: {}", s3_key),
        Ok(_) => eprintln!("  Failed: {}", s3_key),
        Err(e) => {
            eprintln!("Error running aws cli: {}", e);
            eprintln!("Make sure AWS CLI is installed and configured");
            std::process::exit(1);
        }
    }
}

/// Invalidate CloudFront cache.
fn invalidate_cloudfront(dist_id: &str, dry_run: bool) {
    println!();
    println!("Step 3: Invalidating CloudFront cache...");
    println!("  Distribution: {}", dist_id);

    if dry_run {
        println!("  [dry-run] Would invalidate: /*");
        return;
    }

    let status = Command::new("aws")
        .args(["cloudfront", "create-invalidation"])
        .args(["--distribution-id", dist_id])
        .args(["--paths", "/*"])
        .status();

    match status {
        Ok(s) if s.success() => println!("  Cache invalidated successfully"),
        Ok(_) => eprintln!("  Cache invalidation failed"),
        Err(e) => eprintln!("  CloudFront invalidation error: {}", e),
    }
}

/// Deploy to AWS S3 with optional CloudFront invalidation.
fn deploy_to_s3(
    source: &PathBuf,
    bucket: Option<&str>,
    distribution: Option<&str>,
    region: &str,
    dry_run: bool,
) {
    let Some(bucket) = bucket else {
        eprintln!("Error: --bucket is required for S3 deployment");
        std::process::exit(1);
    };

    println!("Step 2: Deploying to S3...");
    println!("  Bucket: {}", bucket);
    println!("  Region: {}", region);

    let files = collect_deploy_files(source);
    println!("  Files: {} to upload", files.len());

    if dry_run {
        print_deploy_files(source, &files);
    } else {
        for (path, _) in &files {
            upload_file_to_s3(path, source, bucket, region);
        }
    }

    if let Some(dist_id) = distribution {
        invalidate_cloudfront(dist_id, dry_run);
    }

    println!();
    if dry_run {
        println!("Dry run complete. No files were uploaded.");
    } else {
        println!("Deployment complete!");
        println!(
            "  URL: https://{}.s3.{}.amazonaws.com/index.html",
            bucket, region
        );
        if distribution.is_some() {
            println!("  Note: CloudFront may take a few minutes to propagate");
        }
    }
}

/// Deploy to Cloudflare Pages.
fn deploy_to_cloudflare(source: &PathBuf, project: Option<&str>, dry_run: bool) {
    let project = project.unwrap_or("presentar-app");

    println!("Step 2: Deploying to Cloudflare Pages...");
    println!("  Project: {}", project);

    if dry_run {
        println!(
            "  [dry-run] Would run: npx wrangler pages deploy {} --project-name {}",
            source.display(),
            project
        );
    } else {
        let status = Command::new("npx")
            .args(["wrangler", "pages", "deploy"])
            .arg(source)
            .args(["--project-name", project])
            .status();

        match status {
            Ok(s) if s.success() => println!("Deployed to Cloudflare Pages successfully!"),
            Ok(_) => {
                eprintln!("Cloudflare Pages deployment failed");
                std::process::exit(1);
            }
            Err(e) => {
                eprintln!("Error running wrangler: {}", e);
                eprintln!("Make sure wrangler is installed: npm install -g wrangler");
                std::process::exit(1);
            }
        }
    }
}

/// Deploy to Vercel.
fn deploy_to_vercel(source: &PathBuf, dry_run: bool) {
    println!("Step 2: Deploying to Vercel...");

    if dry_run {
        println!(
            "  [dry-run] Would run: vercel deploy {} --prod",
            source.display()
        );
    } else {
        let status = Command::new("vercel")
            .args(["deploy", "--prod"])
            .arg(source)
            .status();

        match status {
            Ok(s) if s.success() => println!("Deployed to Vercel successfully!"),
            Ok(_) => {
                eprintln!("Vercel deployment failed");
                std::process::exit(1);
            }
            Err(e) => {
                eprintln!("Error running vercel: {}", e);
                eprintln!("Make sure Vercel CLI is installed: npm install -g vercel");
                std::process::exit(1);
            }
        }
    }
}

/// Deploy to Netlify.
fn deploy_to_netlify(source: &PathBuf, dry_run: bool) {
    println!("Step 2: Deploying to Netlify...");

    if dry_run {
        println!(
            "  [dry-run] Would run: netlify deploy --dir {} --prod",
            source.display()
        );
    } else {
        let status = Command::new("netlify")
            .args(["deploy", "--prod"])
            .args(["--dir", &source.to_string_lossy()])
            .status();

        match status {
            Ok(s) if s.success() => println!("Deployed to Netlify successfully!"),
            Ok(_) => {
                eprintln!("Netlify deployment failed");
                std::process::exit(1);
            }
            Err(e) => {
                eprintln!("Error running netlify: {}", e);
                eprintln!("Make sure Netlify CLI is installed: npm install -g netlify-cli");
                std::process::exit(1);
            }
        }
    }
}

/// Deploy to local directory (for testing or local server).
fn deploy_to_local(source: &PathBuf, dest: Option<&str>, dry_run: bool) {
    let dest = dest.unwrap_or("/var/www/html");
    let dest_path = PathBuf::from(dest);

    println!("Step 2: Copying to local directory...");
    println!("  Destination: {}", dest);

    if dry_run {
        let files = collect_deploy_files(source);
        println!("  [dry-run] Would copy {} files", files.len());
        for (path, _) in &files {
            let rel_path = path.strip_prefix(source).unwrap_or(path);
            println!(
                "    {} -> {}",
                rel_path.display(),
                dest_path.join(rel_path).display()
            );
        }
    } else {
        // Use cp -r for simplicity
        let status = Command::new("cp")
            .args(["-r", "-f"])
            .arg(format!("{}/.", source.display()))
            .arg(&dest_path)
            .status();

        match status {
            Ok(s) if s.success() => {
                println!("Copied to {} successfully!", dest);
            }
            Ok(_) => {
                eprintln!("Failed to copy files");
                std::process::exit(1);
            }
            Err(e) => {
                eprintln!("Error copying files: {}", e);
                std::process::exit(1);
            }
        }
    }
}

/// Collect files to deploy with their content types.
fn collect_deploy_files(source: &PathBuf) -> Vec<(PathBuf, String)> {
    let mut files = Vec::new();

    fn walk_dir(dir: &PathBuf, files: &mut Vec<(PathBuf, String)>) {
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    walk_dir(&path, files);
                } else if path.is_file() {
                    let content_type = get_content_type(&path);
                    files.push((path, content_type));
                }
            }
        }
    }

    walk_dir(source, &mut files);
    files.sort_by(|a, b| a.0.cmp(&b.0));
    files
}

/// Get content type for a file.
fn get_content_type(path: &PathBuf) -> String {
    match path.extension().and_then(|e| e.to_str()) {
        Some("html") => "text/html",
        Some("js") => "application/javascript",
        Some("wasm") => "application/wasm",
        Some("css") => "text/css",
        Some("json") => "application/json",
        Some("svg") => "image/svg+xml",
        Some("png") => "image/png",
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("gif") => "image/gif",
        Some("ico") => "image/x-icon",
        Some("woff") => "font/woff",
        Some("woff2") => "font/woff2",
        Some("ttf") => "font/ttf",
        Some("yaml" | "yml") => "text/yaml",
        Some("txt") => "text/plain",
        _ => "application/octet-stream",
    }
    .to_string()
}

/// Get cache control header for a file.
fn get_cache_control(path: &PathBuf) -> &'static str {
    match path.extension().and_then(|e| e.to_str()) {
        // WASM and JS with hash in filename can be cached forever
        Some("wasm") => "public, max-age=31536000, immutable",
        // CSS and fonts can be cached for a week
        Some("css" | "woff" | "woff2" | "ttf") => "public, max-age=604800",
        // Images can be cached for a day
        Some("png" | "jpg" | "jpeg" | "gif" | "svg" | "ico") => "public, max-age=86400",
        // HTML should be revalidated
        Some("html") => "no-cache, must-revalidate",
        // Default to short cache
        _ => "public, max-age=3600",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grade_to_value() {
        assert_eq!(grade_to_value("A+"), 97);
        assert_eq!(grade_to_value("A"), 93);
        assert_eq!(grade_to_value("A-"), 90);
        assert_eq!(grade_to_value("B+"), 87);
        assert_eq!(grade_to_value("B"), 83);
        assert_eq!(grade_to_value("B-"), 80);
        assert_eq!(grade_to_value("C+"), 77);
        assert_eq!(grade_to_value("C"), 73);
        assert_eq!(grade_to_value("C-"), 70);
        assert_eq!(grade_to_value("D+"), 67);
        assert_eq!(grade_to_value("D"), 63);
        assert_eq!(grade_to_value("D-"), 60);
        assert_eq!(grade_to_value("F"), 50);
        assert_eq!(grade_to_value("invalid"), 0);
    }

    #[test]
    fn test_grade_to_value_case_insensitive() {
        assert_eq!(grade_to_value("a+"), 97);
        assert_eq!(grade_to_value("b"), 83);
        assert_eq!(grade_to_value("c-"), 70);
    }

    #[test]
    fn test_generate_badge() {
        let score = QualityScore {
            overall: 85.0,
            grade: "A".to_string(),
            structural: 20.0,
            performance: 18.0,
            accessibility: 17.0,
            data: 12.0,
            documentation: 9.0,
            consistency: 9.0,
        };
        let badge = generate_badge(&score);
        assert!(badge.starts_with("<svg"));
        assert!(badge.ends_with("</svg>"));
        assert!(badge.contains("quality"));
        assert!(badge.contains("#4c1")); // green for 85
    }

    #[test]
    fn test_generate_badge_colors() {
        let make_score = |overall: f64| QualityScore {
            overall,
            grade: String::new(),
            structural: 0.0,
            performance: 0.0,
            accessibility: 0.0,
            data: 0.0,
            documentation: 0.0,
            consistency: 0.0,
        };

        assert!(generate_badge(&make_score(90.0)).contains("#4c1")); // green
        assert!(generate_badge(&make_score(70.0)).contains("#a3c51c")); // yellow-green
        assert!(generate_badge(&make_score(50.0)).contains("#dfb317")); // yellow
        assert!(generate_badge(&make_score(30.0)).contains("#e05d44")); // red
    }

    #[test]
    fn test_analyze_manifest_quality_minimal() {
        let manifest = presentar_yaml::Manifest::from_yaml(
            r#"
presentar: "1.0"
name: "Test"
version: "1.0.0"
layout:
  type: stack
  sections: []
"#,
        )
        .unwrap();

        let score = analyze_manifest_quality(&manifest);
        assert!(score.overall > 0.0);
        assert!(score.overall <= 100.0);
        assert!(!score.grade.is_empty());
    }

    #[test]
    fn test_analyze_manifest_quality_full() {
        let manifest = presentar_yaml::Manifest::from_yaml(
            r#"
presentar: "1.0"
name: "Dashboard"
version: "1.0.0"
description: "A test dashboard with comprehensive features"
data:
  metrics:
    source: "pacha://data/metrics"
    refresh: "1m"
layout:
  type: grid
  columns: 2
  sections:
    - id: header
      widgets:
        - type: text
          content: "Dashboard Header"
    - id: main-content
      widgets:
        - type: chart
          source: "{{ data.metrics }}"
        - type: table
          source: "{{ data.metrics | limit(10) }}"
"#,
        )
        .unwrap();

        let score = analyze_manifest_quality(&manifest);
        assert!(score.overall > 50.0); // Should have a reasonable score
        assert!(score.structural > 0.0);
        assert!(score.data > 0.0);
        assert!(score.documentation > 0.0);
    }

    #[test]
    fn test_quality_score_grade_mapping() {
        // Test grade computation is correct
        let manifest = presentar_yaml::Manifest::from_yaml(
            r#"
presentar: "1.0"
name: "Test"
version: "1.0.0"
description: "Testing grade calculation"
data:
  test:
    source: "pacha://test"
    refresh: "30s"
layout:
  type: stack
  sections:
    - id: section-one
      widgets:
        - type: text
          content: "Hello"
"#,
        )
        .unwrap();

        let score = analyze_manifest_quality(&manifest);

        // Verify grade matches overall score range
        let expected_grade = match score.overall as u32 {
            90..=100 => "A+",
            85..=89 => "A",
            80..=84 => "A-",
            77..=79 => "B+",
            73..=76 => "B",
            70..=72 => "B-",
            67..=69 => "C+",
            63..=66 => "C",
            60..=62 => "C-",
            50..=59 => "D",
            _ => "F",
        };
        assert_eq!(score.grade, expected_grade);
    }

    // ==========================================================================
    // Hot Reload Tests (require dev-server feature)
    // ==========================================================================

    #[cfg(feature = "dev-server")]
    #[test]
    fn test_inject_hot_reload_script_with_body() {
        let html = b"<!DOCTYPE html><html><body><p>Hello</p></body></html>";
        let result = inject_hot_reload_script(html);
        let result_str = String::from_utf8_lossy(&result);

        assert!(result_str.contains(HOT_RELOAD_SCRIPT));
        assert!(result_str.contains("<p>Hello</p>"));
        // Script should be before </body>
        let script_pos = result_str.find("new WebSocket").unwrap();
        let body_pos = result_str.rfind("</body>").unwrap();
        assert!(script_pos < body_pos);
    }

    #[cfg(feature = "dev-server")]
    #[test]
    fn test_inject_hot_reload_script_without_body() {
        let html = b"<html><p>No body tag</p></html>";
        let result = inject_hot_reload_script(html);
        let result_str = String::from_utf8_lossy(&result);

        // Script should be appended at end
        assert!(result_str.ends_with("</script>\n"));
        assert!(result_str.contains("new WebSocket"));
    }

    #[cfg(feature = "dev-server")]
    #[test]
    fn test_trigger_hot_reload_increments_counter() {
        let initial = RELOAD_COUNTER.load(std::sync::atomic::Ordering::Relaxed);
        trigger_hot_reload();
        let after = RELOAD_COUNTER.load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(after, initial + 1);
    }

    #[cfg(feature = "dev-server")]
    #[test]
    fn test_hot_reload_script_has_websocket_url() {
        assert!(HOT_RELOAD_SCRIPT.contains("ws://localhost:35729"));
    }

    #[cfg(feature = "dev-server")]
    #[test]
    fn test_hot_reload_script_handles_reconnect() {
        assert!(HOT_RELOAD_SCRIPT.contains("reconnecting"));
        assert!(HOT_RELOAD_SCRIPT.contains("setTimeout"));
    }

    // ==========================================================================
    // Deploy Tests
    // ==========================================================================

    #[test]
    fn test_get_content_type() {
        assert_eq!(get_content_type(&PathBuf::from("index.html")), "text/html");
        assert_eq!(
            get_content_type(&PathBuf::from("app.js")),
            "application/javascript"
        );
        assert_eq!(
            get_content_type(&PathBuf::from("app.wasm")),
            "application/wasm"
        );
        assert_eq!(get_content_type(&PathBuf::from("style.css")), "text/css");
        assert_eq!(
            get_content_type(&PathBuf::from("data.json")),
            "application/json"
        );
        assert_eq!(
            get_content_type(&PathBuf::from("logo.svg")),
            "image/svg+xml"
        );
        assert_eq!(get_content_type(&PathBuf::from("photo.png")), "image/png");
        assert_eq!(get_content_type(&PathBuf::from("photo.jpg")), "image/jpeg");
        assert_eq!(get_content_type(&PathBuf::from("photo.jpeg")), "image/jpeg");
        assert_eq!(get_content_type(&PathBuf::from("anim.gif")), "image/gif");
        assert_eq!(
            get_content_type(&PathBuf::from("favicon.ico")),
            "image/x-icon"
        );
        assert_eq!(get_content_type(&PathBuf::from("font.woff")), "font/woff");
        assert_eq!(get_content_type(&PathBuf::from("font.woff2")), "font/woff2");
        assert_eq!(get_content_type(&PathBuf::from("font.ttf")), "font/ttf");
        assert_eq!(get_content_type(&PathBuf::from("config.yaml")), "text/yaml");
        assert_eq!(get_content_type(&PathBuf::from("config.yml")), "text/yaml");
        assert_eq!(get_content_type(&PathBuf::from("readme.txt")), "text/plain");
        assert_eq!(
            get_content_type(&PathBuf::from("unknown.xyz")),
            "application/octet-stream"
        );
    }

    #[test]
    fn test_get_cache_control() {
        // WASM should have long cache
        assert!(get_cache_control(&PathBuf::from("app.wasm")).contains("31536000"));
        assert!(get_cache_control(&PathBuf::from("app.wasm")).contains("immutable"));

        // CSS and fonts should have week-long cache
        assert!(get_cache_control(&PathBuf::from("style.css")).contains("604800"));
        assert!(get_cache_control(&PathBuf::from("font.woff2")).contains("604800"));

        // Images should have day-long cache
        assert!(get_cache_control(&PathBuf::from("logo.png")).contains("86400"));
        assert!(get_cache_control(&PathBuf::from("photo.jpg")).contains("86400"));

        // HTML should be revalidated
        assert!(get_cache_control(&PathBuf::from("index.html")).contains("no-cache"));
        assert!(get_cache_control(&PathBuf::from("index.html")).contains("must-revalidate"));
    }

    #[test]
    fn test_collect_deploy_files() {
        // Create a temp directory with some files
        let temp_dir = std::env::temp_dir().join("presentar-test-deploy");
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir).unwrap();
        fs::create_dir_all(temp_dir.join("pkg")).unwrap();

        fs::write(temp_dir.join("index.html"), "<!DOCTYPE html>").unwrap();
        fs::write(temp_dir.join("pkg/app.js"), "// JS").unwrap();
        fs::write(temp_dir.join("pkg/app.wasm"), [0u8; 100]).unwrap();

        let files = collect_deploy_files(&temp_dir);

        assert_eq!(files.len(), 3);
        assert!(files
            .iter()
            .any(|(p, t)| p.ends_with("index.html") && t == "text/html"));
        assert!(files
            .iter()
            .any(|(p, t)| p.ends_with("app.js") && t == "application/javascript"));
        assert!(files
            .iter()
            .any(|(p, t)| p.ends_with("app.wasm") && t == "application/wasm"));

        // Cleanup
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_collect_deploy_files_empty() {
        let temp_dir = std::env::temp_dir().join("presentar-test-deploy-empty");
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir).unwrap();

        let files = collect_deploy_files(&temp_dir);
        assert!(files.is_empty());

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_collect_deploy_files_nested() {
        let temp_dir = std::env::temp_dir().join("presentar-test-deploy-nested");
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(temp_dir.join("a/b/c")).unwrap();

        fs::write(temp_dir.join("root.txt"), "root").unwrap();
        fs::write(temp_dir.join("a/level1.txt"), "level1").unwrap();
        fs::write(temp_dir.join("a/b/level2.txt"), "level2").unwrap();
        fs::write(temp_dir.join("a/b/c/level3.txt"), "level3").unwrap();

        let files = collect_deploy_files(&temp_dir);

        assert_eq!(files.len(), 4);
        assert!(files.iter().any(|(p, _)| p.ends_with("root.txt")));
        assert!(files.iter().any(|(p, _)| p.ends_with("level1.txt")));
        assert!(files.iter().any(|(p, _)| p.ends_with("level2.txt")));
        assert!(files.iter().any(|(p, _)| p.ends_with("level3.txt")));

        let _ = fs::remove_dir_all(&temp_dir);
    }
}
