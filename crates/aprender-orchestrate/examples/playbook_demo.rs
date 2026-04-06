/// Playbook pipeline demonstration
///
/// Demonstrates YAML-based deterministic pipeline orchestration with
/// BLAKE3 content-addressable caching and cache miss explanations.
///
/// Run: cargo run --example playbook_demo --features native
use batuta::playbook::{self, RunConfig, RunResult};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("Batuta Playbook Demo");
    println!("====================\n");

    // Set up a temporary workspace
    let dir = tempfile::tempdir()?;
    let out1 = dir.path().join("greeting.txt");
    let out2 = dir.path().join("upper.txt");
    let out3 = dir.path().join("summary.txt");
    let yaml_path = dir.path().join("demo.yaml");

    // Write a 3-stage playbook YAML
    std::fs::write(
        &yaml_path,
        format!(
            r#"version: "1.0"
name: playbook-demo
params:
  greeting: "hello from batuta"
  chunk_size: 512
targets: {{}}
stages:
  write:
    cmd: "echo '{{{{params.greeting}}}}' > {out1}"
    deps: []
    outs:
      - path: {out1}
  transform:
    cmd: "tr a-z A-Z < {out1} > {out2}"
    deps:
      - path: {out1}
    outs:
      - path: {out2}
  summarize:
    cmd: "wc -c {out2} > {out3}"
    deps:
      - path: {out2}
    outs:
      - path: {out3}
    after:
      - transform
policy:
  failure: stop_on_first
  validation: checksum
  lock_file: true
"#,
            out1 = out1.display(),
            out2 = out2.display(),
            out3 = out3.display(),
        ),
    )?;

    // --- Step 1: Validate ---
    println!("Step 1: Validate playbook\n");
    let (pb, warnings) = playbook::validate_only(&yaml_path)?;
    println!("  Playbook '{}' is valid", pb.name);
    println!("  Stages: {}", pb.stages.len());
    println!("  Params: {}", pb.params.len());
    if !warnings.is_empty() {
        for w in &warnings {
            println!("  Warning: {}", w);
        }
    }
    println!();

    // --- Step 2: First run (all stages execute) ---
    println!("Step 2: First run (cold cache)\n");
    let config = RunConfig {
        playbook_path: yaml_path.clone(),
        stage_filter: None,
        force: false,
        dry_run: false,
        param_overrides: HashMap::new(),
    };
    let r1 = playbook::run_playbook(&config).await?;
    print_result(&r1);

    // Verify outputs
    println!("  Output: {}", std::fs::read_to_string(&out2)?.trim());
    println!();

    // --- Step 3: Second run (all cached) ---
    println!("Step 3: Second run (warm cache)\n");
    let r2 = playbook::run_playbook(&config).await?;
    print_result(&r2);
    println!();

    // --- Step 4: Parameter override triggers selective rerun ---
    println!("Step 4: Parameter change (selective cache invalidation)\n");
    let mut overrides = HashMap::new();
    overrides.insert(
        "greeting".to_string(),
        serde_yaml_ng::Value::String("sovereign ai stack".to_string()),
    );
    let config_override = RunConfig {
        playbook_path: yaml_path.clone(),
        stage_filter: None,
        force: false,
        dry_run: false,
        param_overrides: overrides,
    };
    let r3 = playbook::run_playbook(&config_override).await?;
    print_result(&r3);
    println!("  Output: {}", std::fs::read_to_string(&out2)?.trim());
    println!();

    // --- Step 5: Show status ---
    println!("Step 5: Pipeline status\n");
    playbook::show_status(&yaml_path)?;

    println!("\nPlaybook demo complete.");
    Ok(())
}

fn print_result(r: &RunResult) {
    println!(
        "  {} run, {} cached, {} failed ({:.1}s)",
        r.stages_run,
        r.stages_cached,
        r.stages_failed,
        r.total_duration.as_secs_f64()
    );
}
