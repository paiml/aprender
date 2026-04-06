//! Cluster Training Example (GPU-SHARE Phase 3, GH-210/211/212/218)
//!
//! Demonstrates multi-node adapter training with cluster config parsing,
//! job placement, checkpoint coordination, and SSH remote execution.
//!
//! ```bash
//! cargo run --example cluster_training
//! cargo run --example cluster_training -- --config cluster.yaml
//! ```

use entrenar::finetune::multi_adapter_pipeline::AdaptersConfigFile;
use entrenar::gpu::cluster::{ClusterConfig, GpuCostModel};
use entrenar::gpu::coordinator::{
    build_launch_command, check_cluster_health, exec_launch, CheckpointCoordinator,
    CheckpointMetadata,
};
use entrenar::gpu::mps::{validate_mps_config, MpsConfig};
use entrenar::gpu::placement::{place_adapters, AdapterJob, PlacementDecision};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_usage();
        return;
    }

    let config_path = parse_config_path(&args);
    let cluster = load_or_default_cluster(config_path.as_deref());

    println!("=== Cluster Training Demo ===");
    println!();
    print_cluster_info(&cluster);

    let jobs = create_demo_jobs();
    let placements = run_placement(&cluster, &jobs);
    print_launch_commands(&cluster, &placements);
    run_coordinator_demo(&cluster, &placements);
    run_cost_model_demo();
    run_mps_demo();
    show_exec_launch_api(&cluster, &placements);
    show_adapters_config_demo();
    show_health_check(&cluster);
}

fn print_usage() {
    println!("Cluster Training Example (GPU-SHARE Phase 3)");
    println!();
    println!("Usage:");
    println!("  cluster_training                        # use built-in demo cluster");
    println!("  cluster_training --config cluster.yaml  # use custom cluster config");
}

fn parse_config_path(args: &[String]) -> Option<PathBuf> {
    args.iter().position(|a| a == "--config").and_then(|i| args.get(i + 1)).map(PathBuf::from)
}

fn load_or_default_cluster(path: Option<&Path>) -> ClusterConfig {
    if let Some(p) = path {
        match ClusterConfig::from_file(p) {
            Ok(c) => return c,
            Err(e) => {
                eprintln!("Failed to load {}: {e}", p.display());
                eprintln!("Falling back to demo cluster.");
            }
        }
    }
    demo_cluster()
}

fn demo_cluster() -> ClusterConfig {
    ClusterConfig::from_yaml(
        r"
nodes:
  - name: desktop
    host: localhost
    gpus:
      - uuid: GPU-abcd-1234
        type: rtx-4090
        vram_mb: 24564
        memory_type: discrete
    max_adapters: 3
  - name: jetson
    host: jetson.local
    transport: ssh
    gpus:
      - uuid: GPU-efgh-5678
        type: jetson-orin
        vram_mb: 8192
        memory_type: unified
    max_adapters: 1
  - name: intel-box
    host: 10.0.0.5
    transport: ssh
    user: noah
    gpus: []
    cpu_cores: 16
    ram_mb: 65536
    max_adapters: 1
",
    )
    .expect("demo cluster should parse")
}

fn print_cluster_info(cluster: &ClusterConfig) {
    println!("{cluster}");
}

fn create_demo_jobs() -> Vec<AdapterJob> {
    vec![
        AdapterJob { adapter_idx: 0, budget_mb: 6000, label: "code-review".to_string() },
        AdapterJob { adapter_idx: 1, budget_mb: 6000, label: "bug-fixing".to_string() },
        AdapterJob { adapter_idx: 2, budget_mb: 3000, label: "docstring-gen".to_string() },
        AdapterJob { adapter_idx: 3, budget_mb: 3000, label: "test-gen".to_string() },
    ]
}

fn run_placement(cluster: &ClusterConfig, jobs: &[AdapterJob]) -> Vec<PlacementDecision> {
    println!("--- Job Placement ---");
    println!();
    let placements = place_adapters(cluster, jobs, &[]);

    for p in &placements {
        let job = &jobs[p.adapter_idx];
        println!(
            "  Adapter {} ({}): -> {} (score: {:.3})",
            p.adapter_idx, job.label, p.node_name, p.score
        );
    }

    let unplaced: Vec<_> = jobs
        .iter()
        .filter(|j| !placements.iter().any(|p| p.adapter_idx == j.adapter_idx))
        .collect();
    if !unplaced.is_empty() {
        println!();
        for j in &unplaced {
            println!("  Adapter {} ({}): UNPLACED (no eligible node)", j.adapter_idx, j.label);
        }
    }
    println!();
    placements
}

fn print_launch_commands(cluster: &ClusterConfig, placements: &[PlacementDecision]) {
    println!("--- Launch Commands ---");
    println!();
    for p in placements {
        if let Some(node) = cluster.find_node(&p.node_name) {
            let cmd = build_launch_command(
                node,
                Path::new("model.apr"),
                &PathBuf::from(format!("data/corpus-{}.jsonl", p.adapter_idx)),
                &PathBuf::from(format!("checkpoints/adapter-{}", p.adapter_idx)),
                16,
                3,
            );
            println!("  [{}] {cmd}", p.node_name);
        }
    }
    println!();
}

fn run_coordinator_demo(cluster: &ClusterConfig, placements: &[PlacementDecision]) {
    println!("--- Checkpoint Coordination ---");
    println!();

    let dirs = HashMap::new();
    let mut coord = CheckpointCoordinator::new(cluster.clone(), placements, &dirs, 300);

    // Simulate checkpoint data (would come from polling in real usage)
    simulate_checkpoints(&mut coord);

    println!("{}", coord.format_leaderboard());

    if let Some(best) = coord.best_adapter() {
        println!(
            "Best adapter: {} on node '{}' (loss: {:.4})",
            best.adapter_idx,
            best.node_name,
            best.latest.as_ref().map_or(0.0, |m| m.val_loss.unwrap_or(m.avg_loss))
        );
    }
}

fn run_cost_model_demo() {
    println!("--- GPU Cost Model (PW-01: 5x PCIe Rule) ---");
    println!();
    let model = GpuCostModel::default();

    let workloads = [
        ("Small matmul (1 MB, 100 MFLOPS)", 1.0, 100.0),
        ("Medium matmul (10 MB, 50k MFLOPS)", 10.0, 50_000.0),
        ("Large matmul (1 MB, 1M MFLOPS)", 1.0, 1_000_000.0),
    ];

    for (label, data_mb, mflops) in &workloads {
        let dispatch = model.should_dispatch_gpu(*data_mb, *mflops);
        let target = if dispatch { "GPU" } else { "CPU" };
        println!("  {label}: -> {target}");
    }
    println!();
}

fn run_mps_demo() {
    println!("--- Experimental MPS Validation (§1.5) ---");
    println!();

    let configs = [
        ("50% share + 8GB limit", MpsConfig::with_share(50).with_mem_limit(8000)),
        ("25% share (no limit)", MpsConfig::with_share(25)),
        ("5% share (error)", MpsConfig::with_share(5)),
    ];

    for (label, config) in &configs {
        let result = validate_mps_config(config);
        print!("  {label}: ");
        if result.has_errors() {
            println!("ERRORS: {}", result.errors.join("; "));
        } else if !result.warnings.is_empty() {
            println!("WARNINGS: {}", result.warnings.join("; "));
        } else {
            println!("OK");
        }
    }
    println!();
}

fn show_exec_launch_api(cluster: &ClusterConfig, placements: &[PlacementDecision]) {
    println!("--- Remote Execution API (GH-218) ---");
    println!();
    println!("  exec_launch() spawns training on local or SSH nodes.");
    println!("  SSH uses: ssh -o BatchMode=yes -o ConnectTimeout=5 host bash < script");
    println!();

    // Show that exec_launch works for local nodes
    if let Some(p) = placements.iter().find(|p| {
        cluster
            .find_node(&p.node_name)
            .is_some_and(|n| matches!(n.transport, entrenar::gpu::cluster::Transport::Local))
    }) {
        if let Some(node) = cluster.find_node(&p.node_name) {
            // Dry run: show what would happen (don't actually spawn apr)
            println!(
                "  Local exec_launch({}, adapter {}): would spawn bash -c 'apr finetune ...'",
                node.name, p.adapter_idx
            );
        }
    }

    // Show SSH exec_launch behavior
    if let Some(p) = placements.iter().find(|p| {
        cluster
            .find_node(&p.node_name)
            .is_some_and(|n| matches!(n.transport, entrenar::gpu::cluster::Transport::Ssh))
    }) {
        if let Some(node) = cluster.find_node(&p.node_name) {
            println!(
                "  SSH exec_launch({}, adapter {}): would run ssh {}@{} bash < script",
                node.name,
                p.adapter_idx,
                node.user.as_deref().unwrap_or("root"),
                node.host
            );
        }
    }
    // Reference exec_launch to suppress unused import warning
    _ = exec_launch;
    println!();
}

fn show_adapters_config_demo() {
    println!("--- Adapters Config TOML (§2.4) ---");
    println!();
    let toml = r#"[[adapter]]
data = "data/corpus-a.jsonl"
checkpoint = "checkpoints/adapter-a"
label = "code-review"
rank = 16
learning_rate = 0.0002

[[adapter]]
data = "data/corpus-b.jsonl"
checkpoint = "checkpoints/adapter-b"
label = "bug-fixing"
rank = 8
"#;
    match AdaptersConfigFile::from_toml(toml) {
        Ok(config) => {
            for (i, entry) in config.adapters.iter().enumerate() {
                println!(
                    "  Adapter {i}: {} -> {} (rank={}, lr={})",
                    entry.data.display(),
                    entry.checkpoint.display(),
                    entry.rank.map_or("default".to_string(), |r| r.to_string()),
                    entry.learning_rate.map_or("default".to_string(), |lr| format!("{lr}"))
                );
            }
        }
        Err(e) => eprintln!("  Parse error: {e}"),
    }
    println!();
}

fn show_health_check(cluster: &ClusterConfig) {
    println!("--- Cluster Health Check (§3.6) ---");
    println!();
    let results = check_cluster_health(cluster);
    for h in &results {
        let status = if h.reachable { "OK" } else { "UNREACHABLE" };
        let apr = h.apr_version.as_deref().unwrap_or("not found");
        let err = h.error.as_deref().unwrap_or("");
        if h.reachable {
            println!("  {}: {status} (apr: {apr})", h.node_name);
        } else {
            println!("  {}: {status} — {err}", h.node_name);
        }
    }
    println!();
}

fn simulate_checkpoints(coord: &mut CheckpointCoordinator) {
    let simulated = vec![
        (0, 0.42, Some(0.39)),
        (1, 0.55, Some(0.51)),
        (2, 0.38, Some(0.35)),
        (3, 0.65, Some(0.60)),
    ];

    for (idx, avg_loss, val_loss) in simulated {
        if let Some(status) = coord.adapters.get_mut(&idx) {
            status.latest = Some(CheckpointMetadata {
                adapter_idx: idx,
                epoch: 3,
                avg_loss,
                val_loss,
                node_name: Some(status.node_name.clone()),
                timestamp: None,
            });
        }
    }
}
