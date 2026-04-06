//! Validate .prs (Presentar Scene Format) files.
//!
//! This example demonstrates how to parse and validate .prs files,
//! showing detailed information about the scene structure.
//!
//! # Usage
//!
//! ```bash
//! # Validate a single file
//! cargo run -p presentar-yaml --example validate_prs examples/prs/minimal.prs
//!
//! # Validate multiple files
//! cargo run -p presentar-yaml --example validate_prs examples/prs/*.prs
//!
//! # Validate with verbose output
//! cargo run -p presentar-yaml --example validate_prs -- -v examples/prs/sentiment-demo.prs
//! ```

use presentar_yaml::{Scene, SceneError};
use std::env;
use std::fs;
use std::path::Path;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: validate_prs [-v|--verbose] <file.prs> [file2.prs ...]");
        eprintln!();
        eprintln!("Options:");
        eprintln!("  -v, --verbose    Show detailed scene information");
        eprintln!();
        eprintln!("Examples:");
        eprintln!("  validate_prs examples/prs/minimal.prs");
        eprintln!("  validate_prs -v examples/prs/sentiment-demo.prs");
        eprintln!("  validate_prs examples/prs/*.prs");
        return ExitCode::from(1);
    }

    let mut verbose = false;
    let mut files: Vec<&str> = Vec::new();

    for arg in &args[1..] {
        match arg.as_str() {
            "-v" | "--verbose" => verbose = true,
            _ => files.push(arg),
        }
    }

    if files.is_empty() {
        eprintln!("Error: No .prs files specified");
        return ExitCode::from(1);
    }

    let mut passed = 0;
    let mut failed = 0;

    for file in &files {
        match validate_file(file, verbose) {
            Ok(()) => {
                passed += 1;
                println!("\x1b[32m✓\x1b[0m {file}");
            }
            Err(e) => {
                failed += 1;
                println!("\x1b[31m✗\x1b[0m {file}");
                eprintln!("  Error: {e}");
            }
        }
    }

    println!();
    println!(
        "Results: {} passed, {} failed, {} total",
        passed,
        failed,
        passed + failed
    );

    if failed > 0 {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    }
}

fn validate_file(path: &str, verbose: bool) -> Result<(), Box<dyn std::error::Error>> {
    let path = Path::new(path);

    if !path.exists() {
        return Err(format!("File not found: {}", path.display()).into());
    }

    if path.extension().map_or(true, |ext| ext != "prs") {
        return Err("File must have .prs extension".into());
    }

    let content = fs::read_to_string(path)?;
    let scene = Scene::from_yaml(&content).map_err(|e| format_scene_error(&e))?;

    if verbose {
        print_scene_info(&scene);
    }

    Ok(())
}

fn format_scene_error(e: &SceneError) -> String {
    match e {
        SceneError::Yaml(ye) => format!("YAML parse error: {ye}"),
        SceneError::InvalidVersion(v) => {
            format!("Invalid prs_version '{v}' (expected format: X.Y)")
        }
        SceneError::DuplicateWidgetId(id) => format!("Duplicate widget ID: '{id}'"),
        SceneError::InvalidBindingTarget { trigger, target } => {
            format!("Binding '{trigger}' references unknown target '{target}'")
        }
        SceneError::InvalidHashFormat { resource, hash } => {
            format!("Invalid hash for '{resource}': '{hash}' (expected blake3:<hex>)")
        }
        SceneError::MissingRemoteHash { resource } => {
            format!("Remote resource '{resource}' requires a hash for verification")
        }
        SceneError::InvalidExpression {
            context,
            expression,
            message,
        } => {
            format!("Invalid expression in {context}: '{expression}' - {message}")
        }
        SceneError::InvalidMetadataName(name) => {
            format!(
                "Invalid metadata name '{name}' (must be kebab-case: lowercase, numbers, hyphens)"
            )
        }
        SceneError::LayoutError(msg) => format!("Layout error: {msg}"),
    }
}

fn print_scene_metadata(scene: &Scene) {
    println!();
    println!("  \x1b[1mScene Information\x1b[0m");
    println!("  ─────────────────");
    println!("  Version:     {}", scene.prs_version);
    println!("  Name:        {}", scene.metadata.name);
    if let Some(title) = &scene.metadata.title {
        println!("  Title:       {title}");
    }
    if let Some(author) = &scene.metadata.author {
        println!("  Author:      {author}");
    }
    if let Some(license) = &scene.metadata.license {
        println!("  License:     {license}");
    }
    if !scene.metadata.tags.is_empty() {
        println!("  Tags:        {}", scene.metadata.tags.join(", "));
    }
}

fn print_scene_layout(scene: &Scene) {
    println!();
    println!("  \x1b[1mLayout\x1b[0m");
    println!("  ──────");
    println!("  Type:        {:?}", scene.layout.layout_type);
    if let Some(cols) = scene.layout.columns {
        println!("  Columns:     {cols}");
    }
    if let Some(rows) = scene.layout.rows {
        println!("  Rows:        {rows}");
    }
    println!("  Gap:         {}px", scene.layout.gap);
}

fn print_scene_widgets(scene: &Scene) {
    println!();
    println!("  \x1b[1mWidgets ({} total)\x1b[0m", scene.widgets.len());
    println!("  ─────────────────");
    for widget in &scene.widgets {
        let pos = widget
            .position
            .as_ref()
            .map(|p| format!(" @ ({}, {})", p.row, p.col))
            .unwrap_or_default();
        println!("  • {} ({:?}){}", widget.id, widget.widget_type, pos);
    }
}

fn print_scene_resources(scene: &Scene) {
    if !scene.resources.models.is_empty() {
        println!();
        println!(
            "  \x1b[1mModels ({} total)\x1b[0m",
            scene.resources.models.len()
        );
        println!("  ─────────────────");
        for (name, model) in &scene.resources.models {
            let hash_status = if model.hash.is_some() { "✓" } else { "○" };
            println!("  • {name} ({:?}) [{hash_status}]", model.resource_type);
            println!("    Source: {}", model.source.primary());
        }
    }
    if !scene.resources.datasets.is_empty() {
        println!();
        println!(
            "  \x1b[1mDatasets ({} total)\x1b[0m",
            scene.resources.datasets.len()
        );
        println!("  ───────────────────");
        for (name, dataset) in &scene.resources.datasets {
            let hash_status = if dataset.hash.is_some() { "✓" } else { "○" };
            println!("  • {name} ({:?}) [{hash_status}]", dataset.resource_type);
            println!("    Source: {}", dataset.source.primary());
        }
    }
}

fn print_scene_bindings(scene: &Scene) {
    if scene.bindings.is_empty() {
        return;
    }
    println!();
    println!("  \x1b[1mBindings ({} total)\x1b[0m", scene.bindings.len());
    println!("  ──────────────────");
    for binding in &scene.bindings {
        let debounce = binding
            .debounce_ms
            .map(|ms| format!(" (debounce: {ms}ms)"))
            .unwrap_or_default();
        println!("  • {}{debounce}", binding.trigger);
        for action in &binding.actions {
            let act = action.action.as_deref().unwrap_or("update");
            println!("    → {} [{}]", action.target, act);
        }
    }
}

fn print_scene_theme(scene: &Scene) {
    let Some(theme) = &scene.theme else {
        return;
    };
    println!();
    println!("  \x1b[1mTheme\x1b[0m");
    println!("  ─────");
    if let Some(preset) = &theme.preset {
        println!("  Preset:      {preset}");
    }
    if !theme.custom.is_empty() {
        println!("  Custom:      {} properties", theme.custom.len());
    }
}

fn print_scene_permissions(scene: &Scene) {
    let p = &scene.permissions;
    if p.network.is_empty() && p.filesystem.is_empty() && !p.clipboard && !p.camera {
        return;
    }
    println!();
    println!("  \x1b[1mPermissions\x1b[0m");
    println!("  ───────────");
    if !p.network.is_empty() {
        println!("  Network:     {} pattern(s)", p.network.len());
    }
    if !p.filesystem.is_empty() {
        println!("  Filesystem:  {} pattern(s)", p.filesystem.len());
    }
    if p.clipboard {
        println!("  Clipboard:   allowed");
    }
    if p.camera {
        println!("  Camera:      allowed");
    }
}

fn print_scene_info(scene: &Scene) {
    print_scene_metadata(scene);
    print_scene_layout(scene);
    print_scene_widgets(scene);
    print_scene_resources(scene);
    print_scene_bindings(scene);
    print_scene_theme(scene);
    print_scene_permissions(scene);
    println!();
}
