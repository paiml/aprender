//! EXT-26 (aprender#4408, collapsed into #4401): the comparator harness (EXT-001 R-1a).
//!
//! A competitor arm (llama.cpp, Ollama, PEFT/TRL, Unsloth, MLflow) runs as a
//! pinned black box, and every run writes a comparator block
//! `{command, version, env_sha256, artifact_sha256, log_path}` into its
//! receipt. An arm missing any of those fields blocks (FALSIFY-EXT-020).
//!
//! The engine track's `AbRecord` (`aprender-test-lib` `perf_gate::ab`) is not
//! reused: PP-32 makes it structurally unable to hold a comparator.
//!
//! Determinism: the child sees only the env the arm declares (the host env is
//! cleared), and the log lands at `<log_dir>/<arm>.log`. So two runs of one
//! arm give identical blocks except `started_utc`/`finished_utc`.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// The fields every comparator block must carry (FALSIFY-EXT-020).
pub(crate) const REQUIRED: [&str; 5] = [
    "command",
    "version",
    "env_sha256",
    "artifact_sha256",
    "log_path",
];

/// What one comparator arm run recorded.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ComparatorBlock {
    /// The argv actually executed, container wrapper included.
    pub command: Vec<String>,
    /// First non-empty stdout line of the arm's version command.
    pub version: String,
    /// sha256 over the child's complete env, as sorted `K=V\n` lines.
    pub env_sha256: String,
    /// sha256 of the artifact the arm produced.
    pub artifact_sha256: String,
    /// Where the arm's stdout and stderr went.
    pub log_path: String,
    /// The digest-pinned image, when the arm ran in a container.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image: Option<String>,
    pub started_utc: String,
    pub finished_utc: String,
}

/// How to run one comparator arm.
#[derive(Debug, Clone)]
pub(crate) struct ArmSpec {
    /// `[a-z0-9._-]+`; names the log file.
    pub name: String,
    pub command: Vec<String>,
    pub version_command: Vec<String>,
    /// The complete env the child sees; nothing is inherited from the host.
    pub env: Vec<(String, String)>,
    /// The file the arm produces, hashed after the run.
    pub artifact: PathBuf,
    /// `name@sha256:<64 hex>`: run inside this image with the network denied.
    pub image: Option<String>,
    /// Mounted at the same path in the container and used as its workdir.
    pub workdir: PathBuf,
}

fn is_sha256(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

fn hex_lower(b: &[u8]) -> String {
    use std::fmt::Write;
    b.iter()
        .fold(String::with_capacity(b.len() * 2), |mut s, x| {
            let _ = write!(s, "{x:02x}");
            s
        })
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut f = File::open(path).map_err(|e| format!("artifact {}: {e}", path.display()))?;
    let mut h = Sha256::new();
    std::io::copy(&mut f, &mut h).map_err(|e| format!("artifact {}: {e}", path.display()))?;
    Ok(hex_lower(&h.finalize()))
}

/// sha256 over `env` as sorted `K=V\n` lines.
pub(crate) fn env_sha256(env: &[(String, String)]) -> String {
    let mut lines: Vec<String> = env.iter().map(|(k, v)| format!("{k}={v}\n")).collect();
    lines.sort();
    hex_lower(&Sha256::digest(lines.concat().as_bytes()))
}

/// R-1a: an image is pinned only by digest.
fn check_image(image: &str) -> Result<(), String> {
    match image.split_once("@sha256:") {
        Some((name, digest)) if !name.is_empty() && is_sha256(digest) => Ok(()),
        _ => Err(format!(
            "R-1a: comparator image `{image}` is not pinned by digest (want name@sha256:<64 hex>)"
        )),
    }
}

/// Wrap `argv` to run inside `image` with the network denied (R-1a).
pub(crate) fn container_argv(
    image: &str,
    argv: &[String],
    env: &[(String, String)],
    workdir: &Path,
) -> Result<Vec<String>, String> {
    check_image(image)?;
    let wd = workdir.display().to_string();
    let mut out: Vec<String> = ["docker", "run", "--rm", "--network=none"]
        .map(String::from)
        .to_vec();
    out.extend(["-v".into(), format!("{wd}:{wd}"), "-w".into(), wd]);
    for (k, v) in env {
        out.extend(["-e".into(), format!("{k}={v}")]);
    }
    out.push(image.to_string());
    out.extend(argv.iter().cloned());
    Ok(out)
}

fn argv_for(spec: &ArmSpec, argv: &[String]) -> Result<Vec<String>, String> {
    match &spec.image {
        Some(image) => container_argv(image, argv, &spec.env, &spec.workdir),
        None => Ok(argv.to_vec()),
    }
}

fn command(argv: &[String], env: &[(String, String)], workdir: &Path) -> Result<Command, String> {
    let (prog, args) = argv.split_first().ok_or("empty command")?;
    let mut c = Command::new(prog);
    c.args(args)
        .env_clear()
        .envs(env.iter().map(|(k, v)| (k, v)))
        .current_dir(workdir);
    Ok(c)
}

fn now_utc() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

fn valid_name(n: &str) -> bool {
    !n.is_empty()
        && n.bytes()
            .all(|b| matches!(b, b'a'..=b'z' | b'0'..=b'9' | b'.' | b'_' | b'-'))
}

/// Run one arm and return its comparator block. The declared artifact is
/// removed first, so it can only come from this run. A failed run, an empty
/// version or a missing artifact is an error, never a partial block.
pub(crate) fn run_arm(spec: &ArmSpec, log_dir: &Path) -> Result<ComparatorBlock, String> {
    if !valid_name(&spec.name) {
        return Err(format!("arm name `{}` is not [a-z0-9._-]+", spec.name));
    }
    let version_argv = argv_for(spec, &spec.version_command)?;
    let out = command(&version_argv, &spec.env, &spec.workdir)?
        .stdin(Stdio::null())
        .output()
        .map_err(|e| format!("{}: version command: {e}", spec.name))?;
    let version = String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .unwrap_or_default()
        .to_string();
    if !out.status.success() || version.is_empty() {
        return Err(format!("{}: version command gave no version", spec.name));
    }

    // A stale artifact from an earlier run must never be hashed as this one's.
    match std::fs::remove_file(&spec.artifact) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
            return Err(format!("{}: stale artifact: {e}", spec.artifact.display()));
        }
        _ => {}
    }
    let argv = argv_for(spec, &spec.command)?;
    let log_path = log_dir.join(format!("{}.log", spec.name));
    let log = File::create(&log_path).map_err(|e| format!("{}: {e}", log_path.display()))?;
    let err = log.try_clone().map_err(|e| e.to_string())?;
    let started_utc = now_utc();
    let status = command(&argv, &spec.env, &spec.workdir)?
        .stdin(Stdio::null())
        .stdout(log)
        .stderr(err)
        .status()
        .map_err(|e| format!("{}: {e}", spec.name))?;
    let finished_utc = now_utc();
    if !status.success() {
        return Err(format!(
            "{}: arm exited {status}; log {}",
            spec.name,
            log_path.display()
        ));
    }
    let block = ComparatorBlock {
        command: argv,
        version,
        env_sha256: env_sha256(&spec.env),
        artifact_sha256: sha256_file(&spec.artifact)?,
        log_path: log_path.display().to_string(),
        image: spec.image.clone(),
        started_utc,
        finished_utc,
    };
    check_block(&serde_json::to_value(&block).map_err(|e| e.to_string())?)
}

/// FALSIFY-EXT-020: a block missing or blanking any [`REQUIRED`] field, or
/// carrying a malformed hash or an unpinned image, is refused.
pub(crate) fn check_block(v: &Value) -> Result<ComparatorBlock, String> {
    for key in REQUIRED {
        let present = match v.get(key) {
            Some(Value::String(s)) => !s.trim().is_empty(),
            Some(Value::Array(a)) => !a.is_empty(),
            _ => false,
        };
        if !present {
            return Err(format!("FALSIFY-EXT-020: comparator block missing `{key}`"));
        }
    }
    let b: ComparatorBlock =
        serde_json::from_value(v.clone()).map_err(|e| format!("comparator block: {e}"))?;
    for (k, s) in [
        ("env_sha256", &b.env_sha256),
        ("artifact_sha256", &b.artifact_sha256),
    ] {
        if !is_sha256(s) {
            return Err(format!("comparator block: `{k}` is not 64 lowercase hex"));
        }
    }
    if let Some(image) = &b.image {
        check_image(image)?;
    }
    Ok(b)
}

/// Every arm in a receipt's `arms` list must carry a valid `comparator`
/// block. Returns the number checked.
pub(crate) fn check_arms(arms: &[Value]) -> Result<usize, String> {
    for (i, arm) in arms.iter().enumerate() {
        let name = arm.get("arm").and_then(Value::as_str).unwrap_or("?");
        let block = arm
            .get("comparator")
            .ok_or_else(|| format!("FALSIFY-EXT-020: arm {i} `{name}` has no comparator block"))?;
        check_block(block).map_err(|e| format!("arm {i} `{name}`: {e}"))?;
    }
    Ok(arms.len())
}

#[cfg(test)]
#[path = "comparator_tests.rs"]
mod tests;
