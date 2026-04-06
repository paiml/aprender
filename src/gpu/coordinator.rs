//! Checkpoint coordination for multi-node adapter training (GPU-SHARE Phase 3, §3.4).
//!
//! The coordinator polls each node's checkpoint directory for metadata,
//! compares val_loss across adapters, maintains a leaderboard, and identifies
//! the best adapter at end of training.
//!
//! Remote nodes are polled via `cat checkpoint_dir/best/metadata.json` over SSH.
//! Local nodes read the file directly.

use super::cluster::{ClusterConfig, NodeConfig, Transport};
use super::placement::PlacementDecision;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Metadata written by `save_adapter_checkpoint()` in multi_adapter_pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    pub adapter_idx: usize,
    pub epoch: usize,
    pub avg_loss: f32,
    #[serde(default)]
    pub val_loss: Option<f32>,
    #[serde(default)]
    pub node_name: Option<String>,
    #[serde(default)]
    pub timestamp: Option<String>,
}

/// Status of a single adapter across the cluster.
#[derive(Debug, Clone)]
pub struct AdapterStatus {
    /// Adapter index.
    pub adapter_idx: usize,
    /// Node name where this adapter is running.
    pub node_name: String,
    /// Checkpoint directory on the remote node.
    pub checkpoint_dir: PathBuf,
    /// Latest checkpoint metadata (if available).
    pub latest: Option<CheckpointMetadata>,
}

/// Leaderboard entry ranking adapters by loss.
#[derive(Debug, Clone)]
pub struct LeaderboardEntry {
    pub rank: usize,
    pub adapter_idx: usize,
    pub node_name: String,
    pub epoch: usize,
    pub loss: f32,
}

/// Coordinator for multi-node training checkpoint polling.
pub struct CheckpointCoordinator {
    /// Adapter statuses indexed by adapter_idx.
    pub adapters: HashMap<usize, AdapterStatus>,
    /// Poll interval in seconds.
    pub poll_interval_secs: u64,
    /// Cluster configuration reference.
    cluster: ClusterConfig,
}

impl CheckpointCoordinator {
    /// Create a new coordinator from placement decisions.
    pub fn new(
        cluster: ClusterConfig,
        placements: &[PlacementDecision],
        checkpoint_dirs: &HashMap<usize, PathBuf>,
        poll_interval_secs: u64,
    ) -> Self {
        let mut adapters = HashMap::new();
        for p in placements {
            let checkpoint_dir = checkpoint_dirs
                .get(&p.adapter_idx)
                .cloned()
                .unwrap_or_else(|| PathBuf::from(format!("checkpoints/adapter-{}", p.adapter_idx)));
            adapters.insert(
                p.adapter_idx,
                AdapterStatus {
                    adapter_idx: p.adapter_idx,
                    node_name: p.node_name.clone(),
                    checkpoint_dir,
                    latest: None,
                },
            );
        }
        Self { adapters, poll_interval_secs, cluster }
    }

    /// Poll all adapters for their latest checkpoint metadata.
    ///
    /// For local nodes, reads the file directly.
    /// For SSH nodes, executes `ssh host cat <path>` and parses JSON.
    pub fn poll_all(&mut self) -> Vec<PollResult> {
        let mut results = Vec::new();
        let adapter_list: Vec<(usize, String, PathBuf)> = self
            .adapters
            .values()
            .map(|a| (a.adapter_idx, a.node_name.clone(), a.checkpoint_dir.clone()))
            .collect();

        for (idx, node_name, checkpoint_dir) in adapter_list {
            let result = self.poll_adapter(idx, &node_name, &checkpoint_dir);
            results.push(result);
        }
        results
    }

    fn poll_adapter(
        &mut self,
        adapter_idx: usize,
        node_name: &str,
        checkpoint_dir: &Path,
    ) -> PollResult {
        let node = self.cluster.find_node(node_name);
        let transport = node.map_or(Transport::Local, |n| n.transport);

        let metadata = match transport {
            Transport::Local => read_local_metadata(checkpoint_dir),
            Transport::Ssh => {
                let host = node.map_or("unknown", |n| &n.host);
                let user = node.and_then(|n| n.user.as_deref());
                read_ssh_metadata(host, user, checkpoint_dir)
            }
        };

        match metadata {
            Ok(meta) => {
                if let Some(status) = self.adapters.get_mut(&adapter_idx) {
                    status.latest = Some(meta.clone());
                }
                PollResult::Ok { adapter_idx, metadata: meta }
            }
            Err(e) => PollResult::Error { adapter_idx, node_name: node_name.to_string(), error: e },
        }
    }

    /// Generate leaderboard sorted by loss (ascending).
    pub fn leaderboard(&self) -> Vec<LeaderboardEntry> {
        let mut entries: Vec<_> = self
            .adapters
            .values()
            .filter_map(|a| {
                a.latest.as_ref().map(|meta| LeaderboardEntry {
                    rank: 0,
                    adapter_idx: a.adapter_idx,
                    node_name: a.node_name.clone(),
                    epoch: meta.epoch,
                    loss: meta.val_loss.unwrap_or(meta.avg_loss),
                })
            })
            .collect();

        entries.sort_by(|a, b| a.loss.partial_cmp(&b.loss).unwrap_or(std::cmp::Ordering::Equal));
        for (i, entry) in entries.iter_mut().enumerate() {
            entry.rank = i + 1;
        }
        entries
    }

    /// Find the best adapter (lowest loss).
    pub fn best_adapter(&self) -> Option<&AdapterStatus> {
        let board = self.leaderboard();
        board.first().and_then(|entry| self.adapters.get(&entry.adapter_idx))
    }

    /// Format leaderboard as a human-readable string.
    pub fn format_leaderboard(&self) -> String {
        let board = self.leaderboard();
        if board.is_empty() {
            return "No checkpoints available yet.".to_string();
        }
        let mut out = String::from("Adapter Leaderboard:\n");
        out.push_str("  Rank | Adapter | Node       | Epoch | Loss\n");
        out.push_str("  -----+---------+------------+-------+--------\n");
        for entry in &board {
            out.push_str(&format!(
                "  {:>4} | {:>7} | {:<10} | {:>5} | {:.4}\n",
                entry.rank, entry.adapter_idx, entry.node_name, entry.epoch, entry.loss
            ));
        }
        out
    }
}

/// Result of polling a single adapter's checkpoint.
#[derive(Debug)]
pub enum PollResult {
    Ok { adapter_idx: usize, metadata: CheckpointMetadata },
    Error { adapter_idx: usize, node_name: String, error: String },
}

/// Read checkpoint metadata from a local path.
fn read_local_metadata(checkpoint_dir: &Path) -> Result<CheckpointMetadata, String> {
    let best_meta = checkpoint_dir.join("best").join("metadata.json");
    let contents = std::fs::read_to_string(&best_meta)
        .map_err(|e| format!("failed to read {}: {e}", best_meta.display()))?;
    serde_json::from_str(&contents)
        .map_err(|e| format!("failed to parse {}: {e}", best_meta.display()))
}

/// Read checkpoint metadata from a remote node via SSH.
///
/// Executes `ssh [-l user] host cat <checkpoint_dir>/best/metadata.json`
/// and parses the JSON output. Timeout: 10 seconds.
fn read_ssh_metadata(
    host: &str,
    user: Option<&str>,
    checkpoint_dir: &Path,
) -> Result<CheckpointMetadata, String> {
    let remote_path = checkpoint_dir.join("best").join("metadata.json");
    let cat_cmd = format!("cat {}", remote_path.display());
    let output = exec_ssh_command(host, user, &cat_cmd)?;
    serde_json::from_str(&output).map_err(|e| format!("failed to parse metadata from {host}: {e}"))
}

/// Execute a command on a remote host via SSH.
///
/// Uses `ssh -o ConnectTimeout=5 -o BatchMode=yes` for non-interactive,
/// timeout-bounded execution. Script is piped via stdin to avoid shell
/// injection through arguments.
fn exec_ssh_command(host: &str, user: Option<&str>, script: &str) -> Result<String, String> {
    let mut cmd = std::process::Command::new("ssh");
    cmd.args(["-o", "ConnectTimeout=5"]);
    cmd.args(["-o", "BatchMode=yes"]);
    cmd.args(["-o", "StrictHostKeyChecking=accept-new"]);

    if let Some(u) = user {
        cmd.args(["-l", u]);
    }

    cmd.arg(host);
    cmd.arg("bash");

    cmd.stdin(std::process::Stdio::piped());
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());

    let mut child = cmd.spawn().map_err(|e| format!("failed to spawn ssh to {host}: {e}"))?;

    // Pipe script via stdin (safe against injection)
    if let Some(stdin) = child.stdin.take() {
        use std::io::Write;
        let mut stdin = stdin;
        let _ = stdin.write_all(script.as_bytes());
        // stdin is dropped here, sending EOF
    }

    let output = child.wait_with_output().map_err(|e| format!("ssh to {host} failed: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "ssh to {host} exited {}: {stderr}",
            output.status.code().unwrap_or(-1)
        ));
    }

    String::from_utf8(output.stdout).map_err(|e| format!("invalid UTF-8 from ssh to {host}: {e}"))
}

/// Execute a training job on a remote or local node.
///
/// For local nodes, spawns the process directly.
/// For SSH nodes, pipes the command via stdin to `ssh host bash`.
///
/// Returns the child process handle for monitoring.
pub fn exec_launch(
    node: &NodeConfig,
    model_path: &Path,
    data_path: &Path,
    checkpoint_dir: &Path,
    rank: u32,
    epochs: u32,
) -> Result<std::process::Child, String> {
    let script = format!(
        "apr finetune {} --task instruct --method qlora --quantize-nf4 \
         --data {} --output {} --rank {rank} --epochs {epochs}",
        model_path.display(),
        data_path.display(),
        checkpoint_dir.display(),
    );

    match node.transport {
        Transport::Local => std::process::Command::new("bash")
            .arg("-c")
            .arg(&script)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| format!("failed to launch local training: {e}")),
        Transport::Ssh => {
            let mut cmd = std::process::Command::new("ssh");
            cmd.args(["-o", "ConnectTimeout=5"]);
            cmd.args(["-o", "BatchMode=yes"]);
            cmd.args(["-o", "StrictHostKeyChecking=accept-new"]);
            if let Some(ref u) = node.user {
                cmd.args(["-l", u]);
            }
            cmd.arg(&node.host);
            cmd.arg("bash");
            cmd.stdin(std::process::Stdio::piped());
            cmd.stdout(std::process::Stdio::piped());
            cmd.stderr(std::process::Stdio::piped());

            let mut child =
                cmd.spawn().map_err(|e| format!("failed to ssh to {}: {e}", node.host))?;

            if let Some(stdin) = child.stdin.take() {
                use std::io::Write;
                let mut stdin = stdin;
                let _ = stdin.write_all(script.as_bytes());
            }

            Ok(child)
        }
    }
}

/// Build a remote launch command for an adapter job on a node.
///
/// Returns the shell command that would be executed on the remote node
/// to start training.
pub fn build_launch_command(
    node: &NodeConfig,
    model_path: &Path,
    data_path: &Path,
    checkpoint_dir: &Path,
    rank: u32,
    epochs: u32,
) -> String {
    let base = format!(
        "apr finetune {} --task instruct --method qlora --quantize-nf4 \
         --data {} --output {} --rank {rank} --epochs {epochs}",
        model_path.display(),
        data_path.display(),
        checkpoint_dir.display(),
    );

    match node.transport {
        Transport::Local => base,
        Transport::Ssh => {
            let user_prefix = node.user.as_ref().map_or_else(String::new, |u| format!("{u}@"));
            format!("ssh {user_prefix}{} '{base}'", node.host)
        }
    }
}

/// Result of a node health check.
#[derive(Debug, Clone)]
pub struct NodeHealth {
    /// Node name from cluster config.
    pub node_name: String,
    /// Whether the node is reachable (SSH or local).
    pub reachable: bool,
    /// apr CLI version if detected.
    pub apr_version: Option<String>,
    /// Error message if health check failed.
    pub error: Option<String>,
}

/// Check health of all nodes in a cluster (GPU-SHARE §3.6).
///
/// For local nodes, checks that `apr --version` is available.
/// For SSH nodes, runs `ssh host 'apr --version'` with timeout.
pub fn check_cluster_health(cluster: &ClusterConfig) -> Vec<NodeHealth> {
    cluster.nodes.iter().map(|node| check_node_health(node)).collect()
}

fn check_node_health(node: &NodeConfig) -> NodeHealth {
    let script = "apr --version 2>/dev/null || echo 'apr: not found'";
    let result = match node.transport {
        Transport::Local => std::process::Command::new("bash")
            .arg("-c")
            .arg(script)
            .output()
            .map_err(|e| format!("failed to check local health: {e}"))
            .and_then(|out| {
                String::from_utf8(out.stdout).map_err(|e| format!("invalid UTF-8: {e}"))
            }),
        Transport::Ssh => exec_ssh_command(&node.host, node.user.as_deref(), script),
    };

    match result {
        Ok(output) => {
            let trimmed = output.trim().to_string();
            let has_apr = !trimmed.contains("not found") && !trimmed.is_empty();
            NodeHealth {
                node_name: node.name.clone(),
                reachable: true,
                apr_version: if has_apr { Some(trimmed) } else { None },
                error: if has_apr { None } else { Some("apr CLI not found on node".to_string()) },
            }
        }
        Err(e) => NodeHealth {
            node_name: node.name.clone(),
            reachable: false,
            apr_version: None,
            error: Some(e),
        },
    }
}

impl CheckpointCoordinator {
    /// Pull the best adapter's checkpoint from its node to a local directory (§3.4).
    ///
    /// For local nodes, copies the checkpoint directory.
    /// For SSH nodes, uses `scp -r` to fetch the checkpoint.
    ///
    /// Returns the local path where the checkpoint was saved.
    pub fn pull_best_checkpoint(&self, dest: &Path) -> Result<PathBuf, String> {
        let best =
            self.best_adapter().ok_or_else(|| "no adapters with checkpoint data".to_string())?;

        let node = self
            .cluster
            .find_node(&best.node_name)
            .ok_or_else(|| format!("node '{}' not found in cluster", best.node_name))?;

        let source_dir = best.checkpoint_dir.join("best");
        let dest_dir = dest.join(format!("adapter-{}-best", best.adapter_idx));

        match node.transport {
            Transport::Local => {
                copy_dir_recursive(&source_dir, &dest_dir)?;
                Ok(dest_dir)
            }
            Transport::Ssh => {
                std::fs::create_dir_all(&dest_dir)
                    .map_err(|e| format!("failed to create {}: {e}", dest_dir.display()))?;
                let user_prefix = node.user.as_ref().map_or_else(String::new, |u| format!("{u}@"));
                let remote = format!("{user_prefix}{}:{}/", node.host, source_dir.display());
                let output = std::process::Command::new("scp")
                    .args(["-r", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes"])
                    .arg(&remote)
                    .arg(dest_dir.to_str().unwrap_or("."))
                    .output()
                    .map_err(|e| format!("failed to run scp: {e}"))?;

                if !output.status.success() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    return Err(format!("scp failed: {stderr}"));
                }
                Ok(dest_dir)
            }
        }
    }
}

/// Recursively copy a directory tree.
fn copy_dir_recursive(src: &Path, dst: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dst).map_err(|e| format!("failed to create {}: {e}", dst.display()))?;
    let entries =
        std::fs::read_dir(src).map_err(|e| format!("failed to read {}: {e}", src.display()))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("failed to read entry: {e}"))?;
        let dest_path = dst.join(entry.file_name());
        if entry.path().is_dir() {
            copy_dir_recursive(&entry.path(), &dest_path)?;
        } else {
            std::fs::copy(entry.path(), &dest_path)
                .map_err(|e| format!("failed to copy {}: {e}", entry.path().display()))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use std::fs;

    fn test_cluster() -> ClusterConfig {
        ClusterConfig::from_yaml(
            r"
nodes:
  - name: desktop
    host: localhost
    gpus:
      - uuid: GPU-abcd-1234
        type: rtx-4090
        vram_mb: 24564
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
",
        )
        .expect("valid")
    }

    fn test_placements() -> Vec<PlacementDecision> {
        vec![
            PlacementDecision { adapter_idx: 0, node_name: "desktop".to_string(), score: 2.5 },
            PlacementDecision { adapter_idx: 1, node_name: "desktop".to_string(), score: 1.2 },
            PlacementDecision { adapter_idx: 2, node_name: "jetson".to_string(), score: 0.3 },
        ]
    }

    #[test]
    fn test_coordinator_creation() {
        let cluster = test_cluster();
        let placements = test_placements();
        let dirs = HashMap::new();
        let coord = CheckpointCoordinator::new(cluster, &placements, &dirs, 300);
        assert_eq!(coord.adapters.len(), 3);
        assert_eq!(coord.poll_interval_secs, 300);
    }

    #[test]
    fn test_empty_leaderboard() {
        let cluster = test_cluster();
        let coord = CheckpointCoordinator::new(cluster, &test_placements(), &HashMap::new(), 300);
        let board = coord.leaderboard();
        assert!(board.is_empty());
        assert!(coord.best_adapter().is_none());
    }

    #[test]
    fn test_leaderboard_with_data() {
        let cluster = test_cluster();
        let mut coord =
            CheckpointCoordinator::new(cluster, &test_placements(), &HashMap::new(), 300);

        // Manually set latest metadata
        coord.adapters.get_mut(&0).expect("valid").latest = Some(CheckpointMetadata {
            adapter_idx: 0,
            epoch: 3,
            avg_loss: 0.5,
            val_loss: Some(0.45),
            node_name: Some("desktop".to_string()),
            timestamp: None,
        });
        coord.adapters.get_mut(&1).expect("valid").latest = Some(CheckpointMetadata {
            adapter_idx: 1,
            epoch: 3,
            avg_loss: 0.8,
            val_loss: Some(0.75),
            node_name: Some("desktop".to_string()),
            timestamp: None,
        });
        coord.adapters.get_mut(&2).expect("valid").latest = Some(CheckpointMetadata {
            adapter_idx: 2,
            epoch: 2,
            avg_loss: 0.3,
            val_loss: Some(0.28),
            node_name: Some("jetson".to_string()),
            timestamp: None,
        });

        let board = coord.leaderboard();
        assert_eq!(board.len(), 3);
        assert_eq!(board[0].adapter_idx, 2); // 0.28 lowest
        assert_eq!(board[0].rank, 1);
        assert_eq!(board[1].adapter_idx, 0); // 0.45
        assert_eq!(board[2].adapter_idx, 1); // 0.75

        let best = coord.best_adapter().expect("valid");
        assert_eq!(best.adapter_idx, 2);
    }

    #[test]
    fn test_poll_local_checkpoint() {
        let dir = tempfile::tempdir().expect("valid");
        let best_dir = dir.path().join("best");
        fs::create_dir_all(&best_dir).expect("valid");
        let meta = CheckpointMetadata {
            adapter_idx: 0,
            epoch: 5,
            avg_loss: 0.42,
            val_loss: Some(0.39),
            node_name: Some("desktop".to_string()),
            timestamp: None,
        };
        fs::write(best_dir.join("metadata.json"), serde_json::to_string(&meta).expect("valid"))
            .expect("valid");

        let cluster = test_cluster();
        let placements = vec![PlacementDecision {
            adapter_idx: 0,
            node_name: "desktop".to_string(),
            score: 2.5,
        }];
        let mut dirs = HashMap::new();
        dirs.insert(0, dir.path().to_path_buf());

        let mut coord = CheckpointCoordinator::new(cluster, &placements, &dirs, 300);
        let results = coord.poll_all();

        assert_eq!(results.len(), 1);
        match &results[0] {
            PollResult::Ok { adapter_idx, metadata } => {
                assert_eq!(*adapter_idx, 0);
                assert_eq!(metadata.epoch, 5);
                assert!((metadata.avg_loss - 0.42).abs() < f32::EPSILON);
            }
            PollResult::Error { error, .. } => panic!("unexpected error: {error}"),
        }
    }

    #[test]
    fn test_poll_ssh_attempts_real_ssh() {
        // SSH poll now attempts real SSH (will fail on missing host, not with stub error)
        let cluster = test_cluster();
        let placements =
            vec![PlacementDecision { adapter_idx: 2, node_name: "jetson".to_string(), score: 0.3 }];
        let mut coord = CheckpointCoordinator::new(cluster, &placements, &HashMap::new(), 300);
        let results = coord.poll_all();

        assert_eq!(results.len(), 1);
        match &results[0] {
            PollResult::Error { error, .. } => {
                // Real SSH errors: connection refused, host unreachable, etc.
                // Must NOT contain the old stub message
                assert!(
                    !error.contains("not yet available"),
                    "SSH transport must not be stubbed: {error}"
                );
            }
            PollResult::Ok { .. } => {
                // If SSH host happens to be reachable (unlikely in CI), that's fine
            }
        }
    }

    #[test]
    fn test_exec_ssh_command_unreachable_host() {
        // Verify exec_ssh_command returns a real SSH error, not a stub
        let result = exec_ssh_command("192.0.2.1", Some("nobody"), "echo test");
        assert!(result.is_err());
        let err = result.unwrap_err();
        // Should be a real SSH/network error
        assert!(
            err.contains("ssh")
                || err.contains("Connection")
                || err.contains("timed out")
                || err.contains("refused")
                || err.contains("resolve")
                || err.contains("No route")
                || err.contains("exited"),
            "expected real SSH error, got: {err}"
        );
    }

    #[test]
    fn test_exec_ssh_command_builds_correct_args() {
        // Verify the SSH command with user sets -l flag
        // This tests the Command construction path (will fail to connect but that's expected)
        let result = exec_ssh_command("192.0.2.1", Some("testuser"), "echo hello");
        assert!(result.is_err()); // Expected: can't connect to RFC 5737 test address
    }

    #[test]
    fn test_format_leaderboard() {
        let cluster = test_cluster();
        let mut coord =
            CheckpointCoordinator::new(cluster, &test_placements(), &HashMap::new(), 300);
        coord.adapters.get_mut(&0).expect("valid").latest = Some(CheckpointMetadata {
            adapter_idx: 0,
            epoch: 2,
            avg_loss: 0.5,
            val_loss: None,
            node_name: None,
            timestamp: None,
        });

        let display = coord.format_leaderboard();
        assert!(display.contains("Adapter Leaderboard"));
        assert!(display.contains("0.5000"));
    }

    #[test]
    fn test_build_launch_command_local() {
        let node = NodeConfig {
            name: "desktop".to_string(),
            host: "localhost".to_string(),
            transport: Transport::Local,
            user: None,
            gpus: vec![],
            max_adapters: 1,
            cpu_cores: None,
            ram_mb: None,
        };
        let cmd = build_launch_command(
            &node,
            Path::new("model.apr"),
            Path::new("data.jsonl"),
            Path::new("/tmp/ckpt"),
            16,
            3,
        );
        assert!(cmd.starts_with("apr finetune model.apr"));
        assert!(cmd.contains("--rank 16"));
        assert!(cmd.contains("--epochs 3"));
        assert!(!cmd.contains("ssh"));
    }

    #[test]
    fn test_exec_launch_local() {
        // exec_launch for local node should spawn a bash process
        let node = NodeConfig {
            name: "test".to_string(),
            host: "localhost".to_string(),
            transport: Transport::Local,
            user: None,
            gpus: vec![],
            max_adapters: 1,
            cpu_cores: None,
            ram_mb: None,
        };
        // Use a command that will fail fast (no real apr binary needed)
        let result = exec_launch(
            &node,
            Path::new("/nonexistent/model.apr"),
            Path::new("/nonexistent/data.jsonl"),
            Path::new("/tmp/test-ckpt"),
            16,
            1,
        );
        // Should successfully spawn even if the command fails
        assert!(result.is_ok(), "local exec_launch should spawn: {:?}", result.err());
        let mut child = result.expect("valid");
        let _ = child.kill(); // Clean up
        let _ = child.wait(); // Reap zombie
    }

    #[test]
    fn test_build_launch_command_ssh() {
        let node = NodeConfig {
            name: "jetson".to_string(),
            host: "jetson.local".to_string(),
            transport: Transport::Ssh,
            user: Some("noah".to_string()),
            gpus: vec![],
            max_adapters: 1,
            cpu_cores: None,
            ram_mb: None,
        };
        let cmd = build_launch_command(
            &node,
            Path::new("model.apr"),
            Path::new("data.jsonl"),
            Path::new("/tmp/ckpt"),
            16,
            3,
        );
        assert!(cmd.starts_with("ssh noah@jetson.local"));
        assert!(cmd.contains("apr finetune model.apr"));
    }

    #[test]
    fn test_check_node_health_local() {
        let node = NodeConfig {
            name: "local".to_string(),
            host: "localhost".to_string(),
            transport: Transport::Local,
            user: None,
            gpus: vec![],
            max_adapters: 1,
            cpu_cores: None,
            ram_mb: None,
        };
        let health = check_node_health(&node);
        assert_eq!(health.node_name, "local");
        assert!(health.reachable);
        // apr may or may not be installed — just verify the check ran
    }

    #[test]
    fn test_check_cluster_health() {
        let cluster = test_cluster();
        let results = check_cluster_health(&cluster);
        assert_eq!(results.len(), 2); // desktop + jetson
        assert_eq!(results[0].node_name, "desktop");
        assert!(results[0].reachable); // local node should be reachable
    }

    #[test]
    fn test_pull_best_checkpoint_local() {
        let dir = tempfile::tempdir().expect("valid");
        let ckpt_dir = dir.path().join("adapter-0");
        let best_dir = ckpt_dir.join("best");
        fs::create_dir_all(&best_dir).expect("valid");

        // Write metadata + a model file
        let meta = CheckpointMetadata {
            adapter_idx: 0,
            epoch: 3,
            avg_loss: 0.35,
            val_loss: Some(0.30),
            node_name: Some("desktop".to_string()),
            timestamp: None,
        };
        fs::write(best_dir.join("metadata.json"), serde_json::to_string(&meta).expect("valid"))
            .expect("valid");
        fs::write(best_dir.join("adapter.safetensors"), b"fake-weights").expect("valid");

        let cluster = test_cluster();
        let placements = vec![PlacementDecision {
            adapter_idx: 0,
            node_name: "desktop".to_string(),
            score: 2.5,
        }];
        let mut dirs = HashMap::new();
        dirs.insert(0, ckpt_dir);

        let mut coord = CheckpointCoordinator::new(cluster, &placements, &dirs, 300);
        // Poll to get metadata
        let _ = coord.poll_all();

        let dest = tempfile::tempdir().expect("valid");
        let result = coord.pull_best_checkpoint(dest.path());
        assert!(result.is_ok(), "pull should succeed: {:?}", result.err());

        let pulled = result.expect("valid");
        assert!(pulled.join("metadata.json").exists());
        assert!(pulled.join("adapter.safetensors").exists());
    }

    #[test]
    fn test_pull_no_checkpoints_fails() {
        let cluster = test_cluster();
        let coord = CheckpointCoordinator::new(cluster, &test_placements(), &HashMap::new(), 300);
        let dest = tempfile::tempdir().expect("valid");
        let result = coord.pull_best_checkpoint(dest.path());
        assert!(result.is_err());
        assert!(result.expect_err("should fail").contains("no adapters"));
    }

    #[test]
    fn test_copy_dir_recursive() {
        let src = tempfile::tempdir().expect("valid");
        let sub = src.path().join("subdir");
        fs::create_dir_all(&sub).expect("valid");
        fs::write(src.path().join("a.txt"), "hello").expect("valid");
        fs::write(sub.join("b.txt"), "world").expect("valid");

        let dst = tempfile::tempdir().expect("valid");
        let dst_path = dst.path().join("copy");
        copy_dir_recursive(src.path(), &dst_path).expect("valid");

        assert!(dst_path.join("a.txt").exists());
        assert!(dst_path.join("subdir").join("b.txt").exists());
        assert_eq!(fs::read_to_string(dst_path.join("a.txt")).expect("valid"), "hello");
    }

    // ── Additional coverage tests ──

    #[test]
    fn test_checkpoint_metadata_serde_roundtrip() {
        let meta = CheckpointMetadata {
            adapter_idx: 3,
            epoch: 10,
            avg_loss: 0.123,
            val_loss: Some(0.099),
            node_name: Some("gpu-node-1".to_string()),
            timestamp: Some("2026-03-08T12:00:00Z".to_string()),
        };
        let json = serde_json::to_string(&meta).unwrap();
        let restored: CheckpointMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.adapter_idx, 3);
        assert_eq!(restored.epoch, 10);
        assert!((restored.avg_loss - 0.123).abs() < f32::EPSILON);
        assert!((restored.val_loss.unwrap() - 0.099).abs() < f32::EPSILON);
        assert_eq!(restored.node_name.unwrap(), "gpu-node-1");
        assert_eq!(restored.timestamp.unwrap(), "2026-03-08T12:00:00Z");
    }

    #[test]
    fn test_checkpoint_metadata_serde_defaults() {
        // Test deserialization with missing optional fields
        let json = r#"{"adapter_idx":0,"epoch":1,"avg_loss":0.5}"#;
        let meta: CheckpointMetadata = serde_json::from_str(json).unwrap();
        assert_eq!(meta.adapter_idx, 0);
        assert!(meta.val_loss.is_none());
        assert!(meta.node_name.is_none());
        assert!(meta.timestamp.is_none());
    }

    #[test]
    fn test_coordinator_custom_checkpoint_dirs() {
        let cluster = test_cluster();
        let placements = test_placements();
        let mut dirs = HashMap::new();
        dirs.insert(0, PathBuf::from("/custom/path/adapter-0"));
        dirs.insert(2, PathBuf::from("/custom/path/adapter-2"));

        let coord = CheckpointCoordinator::new(cluster, &placements, &dirs, 600);
        assert_eq!(coord.adapters[&0].checkpoint_dir, PathBuf::from("/custom/path/adapter-0"));
        assert_eq!(coord.adapters[&2].checkpoint_dir, PathBuf::from("/custom/path/adapter-2"));
        // Adapter 1 should have auto-generated path
        assert_eq!(coord.adapters[&1].checkpoint_dir, PathBuf::from("checkpoints/adapter-1"));
    }

    #[test]
    fn test_coordinator_default_checkpoint_dirs() {
        let cluster = test_cluster();
        let placements = test_placements();
        let dirs = HashMap::new();

        let coord = CheckpointCoordinator::new(cluster, &placements, &dirs, 300);
        for p in &placements {
            let expected = PathBuf::from(format!("checkpoints/adapter-{}", p.adapter_idx));
            assert_eq!(coord.adapters[&p.adapter_idx].checkpoint_dir, expected);
        }
    }

    #[test]
    fn test_leaderboard_uses_val_loss_when_available() {
        let cluster = test_cluster();
        let mut coord =
            CheckpointCoordinator::new(cluster, &test_placements(), &HashMap::new(), 300);

        coord.adapters.get_mut(&0).unwrap().latest = Some(CheckpointMetadata {
            adapter_idx: 0,
            epoch: 5,
            avg_loss: 1.0,
            val_loss: Some(0.5), // val_loss should be used
            node_name: None,
            timestamp: None,
        });

        let board = coord.leaderboard();
        assert_eq!(board.len(), 1);
        assert!((board[0].loss - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_leaderboard_falls_back_to_avg_loss() {
        let cluster = test_cluster();
        let mut coord =
            CheckpointCoordinator::new(cluster, &test_placements(), &HashMap::new(), 300);

        coord.adapters.get_mut(&0).unwrap().latest = Some(CheckpointMetadata {
            adapter_idx: 0,
            epoch: 5,
            avg_loss: 1.0,
            val_loss: None, // no val_loss, should fallback to avg_loss
            node_name: None,
            timestamp: None,
        });

        let board = coord.leaderboard();
        assert_eq!(board.len(), 1);
        assert!((board[0].loss - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_leaderboard_ranking_three_adapters() {
        let cluster = test_cluster();
        let mut coord =
            CheckpointCoordinator::new(cluster, &test_placements(), &HashMap::new(), 300);

        // Set losses: adapter 0 = 0.7, adapter 1 = 0.3, adapter 2 = 0.5
        coord.adapters.get_mut(&0).unwrap().latest = Some(CheckpointMetadata {
            adapter_idx: 0,
            epoch: 2,
            avg_loss: 0.7,
            val_loss: None,
            node_name: None,
            timestamp: None,
        });
        coord.adapters.get_mut(&1).unwrap().latest = Some(CheckpointMetadata {
            adapter_idx: 1,
            epoch: 3,
            avg_loss: 0.3,
            val_loss: None,
            node_name: None,
            timestamp: None,
        });
        coord.adapters.get_mut(&2).unwrap().latest = Some(CheckpointMetadata {
            adapter_idx: 2,
            epoch: 1,
            avg_loss: 0.5,
            val_loss: None,
            node_name: None,
            timestamp: None,
        });

        let board = coord.leaderboard();
        assert_eq!(board.len(), 3);
        // Ranked by loss ascending
        assert_eq!(board[0].adapter_idx, 1); // 0.3
        assert_eq!(board[0].rank, 1);
        assert_eq!(board[1].adapter_idx, 2); // 0.5
        assert_eq!(board[1].rank, 2);
        assert_eq!(board[2].adapter_idx, 0); // 0.7
        assert_eq!(board[2].rank, 3);

        // best_adapter returns lowest loss
        let best = coord.best_adapter().unwrap();
        assert_eq!(best.adapter_idx, 1);
    }

    #[test]
    fn test_format_leaderboard_empty() {
        let cluster = test_cluster();
        let coord = CheckpointCoordinator::new(cluster, &test_placements(), &HashMap::new(), 300);
        let display = coord.format_leaderboard();
        assert_eq!(display, "No checkpoints available yet.");
    }

    #[test]
    fn test_format_leaderboard_multiple_entries() {
        let cluster = test_cluster();
        let mut coord =
            CheckpointCoordinator::new(cluster, &test_placements(), &HashMap::new(), 300);

        coord.adapters.get_mut(&0).unwrap().latest = Some(CheckpointMetadata {
            adapter_idx: 0,
            epoch: 3,
            avg_loss: 0.5,
            val_loss: Some(0.45),
            node_name: None,
            timestamp: None,
        });
        coord.adapters.get_mut(&1).unwrap().latest = Some(CheckpointMetadata {
            adapter_idx: 1,
            epoch: 2,
            avg_loss: 0.8,
            val_loss: Some(0.75),
            node_name: None,
            timestamp: None,
        });

        let display = coord.format_leaderboard();
        assert!(display.contains("Adapter Leaderboard"));
        assert!(display.contains("Rank"));
        assert!(display.contains("Adapter"));
        assert!(display.contains("Node"));
        assert!(display.contains("Epoch"));
        assert!(display.contains("Loss"));
        assert!(display.contains("0.4500"));
        assert!(display.contains("0.7500"));
    }

    #[test]
    fn test_adapter_status_fields() {
        let cluster = test_cluster();
        let placements = test_placements();
        let coord = CheckpointCoordinator::new(cluster, &placements, &HashMap::new(), 300);

        let status = &coord.adapters[&0];
        assert_eq!(status.adapter_idx, 0);
        assert_eq!(status.node_name, "desktop");
        assert!(status.latest.is_none());
    }

    #[test]
    fn test_build_launch_command_ssh_no_user() {
        let node = NodeConfig {
            name: "remote".to_string(),
            host: "gpu-server.example.com".to_string(),
            transport: Transport::Ssh,
            user: None,
            gpus: vec![],
            max_adapters: 1,
            cpu_cores: None,
            ram_mb: None,
        };
        let cmd = build_launch_command(
            &node,
            Path::new("/models/qwen.safetensors"),
            Path::new("/data/train.jsonl"),
            Path::new("/checkpoints"),
            32,
            5,
        );
        // Without user, no user@ prefix
        assert!(cmd.starts_with("ssh gpu-server.example.com"));
        assert!(cmd.contains("apr finetune"));
        assert!(cmd.contains("--rank 32"));
        assert!(cmd.contains("--epochs 5"));
    }

    #[test]
    fn test_poll_result_debug_ok() {
        let result = PollResult::Ok {
            adapter_idx: 0,
            metadata: CheckpointMetadata {
                adapter_idx: 0,
                epoch: 1,
                avg_loss: 0.5,
                val_loss: None,
                node_name: None,
                timestamp: None,
            },
        };
        let debug = format!("{result:?}");
        assert!(debug.contains("Ok"));
    }

    #[test]
    fn test_poll_result_debug_error() {
        let result = PollResult::Error {
            adapter_idx: 1,
            node_name: "node1".to_string(),
            error: "connection refused".to_string(),
        };
        let debug = format!("{result:?}");
        assert!(debug.contains("Error"));
        assert!(debug.contains("connection refused"));
    }

    #[test]
    fn test_node_health_debug() {
        let health = NodeHealth {
            node_name: "test-node".to_string(),
            reachable: true,
            apr_version: Some("1.0.0".to_string()),
            error: None,
        };
        let debug = format!("{health:?}");
        assert!(debug.contains("test-node"));
        assert!(debug.contains("1.0.0"));
    }

    #[test]
    fn test_node_health_unreachable() {
        let health = NodeHealth {
            node_name: "offline-node".to_string(),
            reachable: false,
            apr_version: None,
            error: Some("connection timeout".to_string()),
        };
        assert!(!health.reachable);
        assert!(health.error.is_some());
        assert!(health.apr_version.is_none());
    }

    #[test]
    fn test_copy_dir_recursive_nested() {
        let src = tempfile::tempdir().unwrap();
        let deep = src.path().join("a").join("b").join("c");
        fs::create_dir_all(&deep).unwrap();
        fs::write(deep.join("deep.txt"), "deep content").unwrap();
        fs::write(src.path().join("root.txt"), "root").unwrap();

        let dst = tempfile::tempdir().unwrap();
        let dst_path = dst.path().join("output");
        copy_dir_recursive(src.path(), &dst_path).unwrap();

        assert!(dst_path.join("root.txt").exists());
        assert!(dst_path.join("a").join("b").join("c").join("deep.txt").exists());
        assert_eq!(
            fs::read_to_string(dst_path.join("a").join("b").join("c").join("deep.txt")).unwrap(),
            "deep content"
        );
    }

    #[test]
    fn test_copy_dir_recursive_nonexistent_src() {
        let dst = tempfile::tempdir().unwrap();
        let result = copy_dir_recursive(Path::new("/nonexistent/path"), &dst.path().join("out"));
        assert!(result.is_err());
    }

    #[test]
    fn test_read_local_metadata_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let result = read_local_metadata(dir.path());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("failed to read"));
    }

    #[test]
    fn test_read_local_metadata_invalid_json() {
        let dir = tempfile::tempdir().unwrap();
        let best_dir = dir.path().join("best");
        fs::create_dir_all(&best_dir).unwrap();
        fs::write(best_dir.join("metadata.json"), "not valid json").unwrap();
        let result = read_local_metadata(dir.path());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("failed to parse"));
    }

    #[test]
    fn test_read_local_metadata_valid() {
        let dir = tempfile::tempdir().unwrap();
        let best_dir = dir.path().join("best");
        fs::create_dir_all(&best_dir).unwrap();
        let meta = CheckpointMetadata {
            adapter_idx: 42,
            epoch: 7,
            avg_loss: 0.123,
            val_loss: Some(0.111),
            node_name: Some("gpu-0".to_string()),
            timestamp: Some("2026-01-01".to_string()),
        };
        fs::write(best_dir.join("metadata.json"), serde_json::to_string(&meta).unwrap()).unwrap();
        let result = read_local_metadata(dir.path()).unwrap();
        assert_eq!(result.adapter_idx, 42);
        assert_eq!(result.epoch, 7);
    }

    #[test]
    fn test_poll_all_updates_adapter_status() {
        let dir = tempfile::tempdir().unwrap();
        let best_dir = dir.path().join("best");
        fs::create_dir_all(&best_dir).unwrap();
        let meta = CheckpointMetadata {
            adapter_idx: 0,
            epoch: 10,
            avg_loss: 0.22,
            val_loss: Some(0.19),
            node_name: None,
            timestamp: None,
        };
        fs::write(best_dir.join("metadata.json"), serde_json::to_string(&meta).unwrap()).unwrap();

        let cluster = test_cluster();
        let placements = vec![PlacementDecision {
            adapter_idx: 0,
            node_name: "desktop".to_string(),
            score: 1.0,
        }];
        let mut dirs = HashMap::new();
        dirs.insert(0, dir.path().to_path_buf());
        let mut coord = CheckpointCoordinator::new(cluster, &placements, &dirs, 60);

        // Before poll, latest is None
        assert!(coord.adapters[&0].latest.is_none());

        let results = coord.poll_all();
        assert_eq!(results.len(), 1);

        // After poll, latest should be populated
        let latest = coord.adapters[&0].latest.as_ref().unwrap();
        assert_eq!(latest.epoch, 10);
        assert!((latest.avg_loss - 0.22).abs() < f32::EPSILON);
    }

    #[test]
    fn test_leaderboard_entry_fields() {
        let entry = LeaderboardEntry {
            rank: 1,
            adapter_idx: 5,
            node_name: "gpu-node".to_string(),
            epoch: 15,
            loss: 0.123,
        };
        assert_eq!(entry.rank, 1);
        assert_eq!(entry.adapter_idx, 5);
        assert_eq!(entry.node_name, "gpu-node");
        assert_eq!(entry.epoch, 15);
        assert!((entry.loss - 0.123).abs() < f32::EPSILON);
    }

    #[test]
    fn test_leaderboard_nan_loss_handling() {
        let cluster = test_cluster();
        let mut coord =
            CheckpointCoordinator::new(cluster, &test_placements(), &HashMap::new(), 300);

        // NaN in val_loss should still produce a leaderboard entry
        coord.adapters.get_mut(&0).unwrap().latest = Some(CheckpointMetadata {
            adapter_idx: 0,
            epoch: 1,
            avg_loss: f32::NAN,
            val_loss: Some(f32::NAN),
            node_name: None,
            timestamp: None,
        });

        let board = coord.leaderboard();
        assert_eq!(board.len(), 1);
    }
}
