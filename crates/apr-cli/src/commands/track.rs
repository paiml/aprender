//! EXT-05 (aprender#4387): training verbs record to pacha by default.
//!
//! A tracked run writes, into the pacha registry (EXT-001 §3.1):
//! - one run row, identified by a ULID and carrying the engine identity (I-5);
//! - one `models` row for the produced artifact, whose `card.extra` holds its
//!   sha256 and the run id;
//! - three lineage edges: `base` (base model -> run), `dataset` (dataset ->
//!   run) and `produced` (run -> produced model).
//!
//! A base model already in `models` is referred to by its model id. An
//! unregistered base, and every dataset (until EXT-08's canonical manifest),
//! is referred to as `blake3:<hex>`, the content address pacha itself uses.
//! `--no-track` opts out.

use crate::error::CliError;
use entrenar::tracking::pacha::PachaBackend;
use entrenar::tracking::storage::TrackingBackend;
use entrenar::tracking::{ulid::new_ulid, Run, RunStatus};
use pacha::model::{ModelCard, ModelVersion};
use pacha::{Registry, RegistryConfig};
use sha2::Digest;
use std::path::{Path, PathBuf};

type Result<T> = std::result::Result<T, CliError>;

/// The producing engine (EXT-001 I-5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EngineIdentity {
    pub apr_version: String,
    pub git_sha: String,
    /// `"0"`, `"1"` or `"unknown"`, from build.rs.
    pub dirty: String,
}

impl EngineIdentity {
    /// The identity of this binary.
    pub(crate) fn current() -> Self {
        Self {
            apr_version: env!("CARGO_PKG_VERSION").to_string(),
            git_sha: env!("APR_GIT_SHA").to_string(),
            dirty: env!("APR_GIT_DIRTY").to_string(),
        }
    }

    /// I-5: refuse a dirty or unidentifiable engine, naming `--no-track`.
    pub(crate) fn check(&self) -> Result<()> {
        let why = if self.git_sha.is_empty() || self.git_sha.ends_with("+no-git") {
            Some(format!("unidentifiable build (git sha `{}`)", self.git_sha))
        } else if self.dirty != "0" {
            Some(format!(
                "dirty build ({} with uncommitted tracked changes, dirty={})",
                self.git_sha, self.dirty
            ))
        } else {
            None
        };
        match why {
            None => Ok(()),
            Some(why) => Err(CliError::ValidationFailed(format!(
                "cannot record this run to pacha: {why} (EXT-001 I-5). \
                 Build from a clean commit, or pass --no-track to train without recording"
            ))),
        }
    }
}

/// `~/.pacha`, the default pacha home.
fn default_pacha_home() -> Result<PathBuf> {
    dirs::home_dir()
        .map(|h| h.join(".pacha"))
        .ok_or_else(|| CliError::ValidationFailed("no home directory for ~/.pacha".into()))
}

fn pacha_err(e: impl std::fmt::Display) -> CliError {
    CliError::ValidationFailed(format!("pacha: {e}"))
}

/// Streamed BLAKE3 of one file: inputs need only pacha's content address.
fn blake3_file(path: &Path) -> Result<String> {
    let mut hasher = blake3::Hasher::new();
    hasher.update_reader(std::fs::File::open(path)?)?;
    Ok(hasher.finalize().to_hex().to_string())
}

/// BLAKE3 of a file, or of a directory as its sorted `(relative path, file
/// BLAKE3)` list, so a dataset directory is content-addressed too.
fn blake3_tree(path: &Path) -> Result<String> {
    if path.is_file() {
        return blake3_file(path);
    }
    let mut files = Vec::new();
    let mut stack = vec![path.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir)? {
            let p = entry?.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.is_file() {
                files.push(p);
            }
        }
    }
    files.sort();
    let mut tree = blake3::Hasher::new();
    for f in files {
        let rel = f
            .strip_prefix(path)
            .unwrap_or(&f)
            .to_string_lossy()
            .into_owned();
        tree.update(rel.as_bytes());
        tree.update(&[0]);
        tree.update(blake3_file(&f)?.as_bytes());
        tree.update(&[b'\n']);
    }
    Ok(tree.finalize().to_hex().to_string())
}

/// The artifact a verb produced at `output`: the file itself, or in an output
/// directory the first of the files entrenar writes. `None` when nothing
/// non-empty exists there, so an empty placeholder is never registered.
pub(crate) fn produced_artifact(output: &Path) -> Option<PathBuf> {
    let candidates: Vec<PathBuf> = if output.is_dir() {
        [
            "adapter_model.safetensors",
            "model.safetensors",
            "model.apr",
            "model-best.apr",
        ]
        .iter()
        .map(|f| output.join(f))
        .collect()
    } else {
        vec![output.to_path_buf()]
    };
    candidates
        .into_iter()
        .find(|p| p.is_file() && p.metadata().is_ok_and(|m| m.len() > 0))
}

/// What a finished tracked run recorded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Recorded {
    pub run_id: String,
    pub produced_model_id: Option<String>,
    pub produced_sha256: Option<String>,
    pub edges: usize,
}

/// A run in progress, recorded to pacha.
pub(crate) struct Recorder {
    registry: Registry,
    backend: PachaBackend,
    run: Run,
    inputs: Vec<(String, &'static str)>,
}

/// Start recording `verb` unless `no_track`. A refused identity refuses the
/// run, before any training starts.
pub(crate) fn start(
    verb: &str,
    base: Option<&Path>,
    dataset: Option<&Path>,
    no_track: bool,
) -> Result<Option<Recorder>> {
    if no_track {
        return Ok(None);
    }
    let home = default_pacha_home()?;
    Recorder::start_in(&home, &EngineIdentity::current(), verb, base, dataset).map(Some)
}

impl Recorder {
    /// Start recording into the pacha home `home` (tests use a tmp home).
    pub(crate) fn start_in(
        home: &Path,
        engine: &EngineIdentity,
        verb: &str,
        base: Option<&Path>,
        dataset: Option<&Path>,
    ) -> Result<Self> {
        engine.check()?;
        let registry = Registry::open(RegistryConfig::new(home)).map_err(pacha_err)?;
        let mut backend = PachaBackend::open(
            &RegistryConfig::new(home).db_path(),
            &home.join("tracking-metrics.db"),
        )
        .map_err(pacha_err)?;

        let mut run = Run::new(new_ulid(), Some(verb.to_string()), format!("apr-{verb}"));
        run.tags
            .insert("apr_version".into(), engine.apr_version.clone());
        run.tags.insert("git_sha".into(), engine.git_sha.clone());
        run.tags.insert("engine_dirty".into(), engine.dirty.clone());
        run.tags.insert("verb".into(), verb.to_string());

        let mut inputs = Vec::new();
        if let Some(base) = base.filter(|p| p.exists()) {
            let b3 = blake3_tree(base)?;
            let node = match registry
                .find_model_id_by_content_hash(&b3)
                .map_err(pacha_err)?
            {
                Some(id) => id,
                None => format!("blake3:{b3}"),
            };
            run.params.insert("base".into(), node.clone());
            run.params
                .insert("base_path".into(), base.display().to_string());
            inputs.push((node, "base"));
        }
        if let Some(dataset) = dataset.filter(|p| p.exists()) {
            let node = format!("blake3:{}", blake3_tree(dataset)?);
            run.params.insert("dataset".into(), node.clone());
            run.params
                .insert("dataset_path".into(), dataset.display().to_string());
            inputs.push((node, "dataset"));
        }
        backend.save_run(&run).map_err(pacha_err)?;
        Ok(Self {
            registry,
            backend,
            run,
            inputs,
        })
    }

    /// Close the run. On success, register what the verb produced at
    /// `output` and write the lineage edges.
    pub(crate) fn finish(mut self, succeeded: bool, output: Option<&Path>) -> Result<Recorded> {
        let run_id = self.run.run_id.clone();
        let mut recorded = Recorded {
            run_id: run_id.clone(),
            produced_model_id: None,
            produced_sha256: None,
            edges: 0,
        };
        for (node, edge) in &self.inputs {
            self.registry
                .add_lineage_edge(node, &run_id, edge, None)
                .map_err(pacha_err)?;
            recorded.edges += 1;
        }
        if succeeded {
            if let Some(artifact) = output.and_then(produced_artifact) {
                let (id, sha256) = self.register_produced(&artifact)?;
                let meta =
                    serde_json::json!({ "sha256": sha256, "path": artifact.display().to_string() });
                self.registry
                    .add_lineage_edge(&run_id, &id, "produced", Some(&meta))
                    .map_err(pacha_err)?;
                recorded.edges += 1;
                self.run
                    .params
                    .insert("output_sha256".into(), sha256.clone());
                self.run.params.insert("output_model".into(), id.clone());
                self.run.artifacts.push(artifact.display().to_string());
                recorded.produced_model_id = Some(id);
                recorded.produced_sha256 = Some(sha256);
            }
        }
        self.run.status = if succeeded {
            RunStatus::Completed
        } else {
            RunStatus::Failed
        };
        self.run.end_time_ms = Some(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX)),
        );
        self.backend.save_run(&self.run).map_err(pacha_err)?;
        Ok(recorded)
    }

    /// Register `artifact` as the next patch version of its file stem.
    fn register_produced(&self, artifact: &Path) -> Result<(String, String)> {
        let bytes = std::fs::read(artifact)?;
        let sha256 = format!("{:x}", sha2::Sha256::digest(&bytes));
        let name = artifact
            .file_stem()
            .map_or_else(|| "model".into(), |s| s.to_string_lossy().into_owned());
        let version = self
            .registry
            .list_model_versions(&name)
            .map_err(pacha_err)?
            .into_iter()
            .max()
            .map_or_else(|| ModelVersion::new(0, 1, 0), |v| v.bump_patch());
        let mut card = ModelCard::new(format!("produced by apr run {}", self.run.run_id));
        card.extra
            .insert("sha256".into(), serde_json::Value::String(sha256.clone()));
        card.extra.insert(
            "run_id".into(),
            serde_json::Value::String(self.run.run_id.clone()),
        );
        let id = self
            .registry
            .register_model(&name, &version, &bytes, card)
            .map_err(pacha_err)?;
        Ok((id.to_string(), sha256))
    }
}

/// Run `train` under a recorder: record the start (refusing an unidentified
/// engine before training), then close the run with the verb's outcome.
pub(crate) fn tracked(
    verb: &str,
    base: Option<&Path>,
    dataset: Option<&Path>,
    output: Option<&Path>,
    no_track: bool,
    train: impl FnOnce() -> Result<()>,
) -> Result<()> {
    let Some(recorder) = start(verb, base, dataset, no_track)? else {
        return train();
    };
    let outcome = train();
    let recorded = recorder.finish(outcome.is_ok(), output)?;
    eprintln!(
        "[apr] run {} recorded to pacha ({} lineage edges{})",
        recorded.run_id,
        recorded.edges,
        recorded
            .produced_sha256
            .as_deref()
            .map_or(String::new(), |s| format!(", output sha256 {s}"))
    );
    outcome
}

#[cfg(test)]
#[path = "track_tests.rs"]
mod tests;
