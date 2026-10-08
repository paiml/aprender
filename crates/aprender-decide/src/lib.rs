//! Method-neutral decision models over aprender-core (D-14).
//!
//! A *decision method* answers one fixed [`Task`] — "which of these K criteria does
//! this text belong to?" — with a calibrated probability per criterion, in the
//! task's criteria order. The order of `task.json`'s `criteria` object IS the label
//! index (D-05), so every probability vector here is an array in that order, never a
//! map.
//!
//! # The seam
//!
//! [`DecisionMethod`] is designed for exactly ONE implementation today —
//! [`laya::Laya`], a ModernBERT encoder with a typed decision head — with Kev (a
//! few-shot decoder classifier) as the known second. It carries only what every
//! method needs: the task, tokenization/row building ([`DecisionMethod::prepare`]),
//! and scoring of already-built rows ([`DecisionMethod::classify_prepared`]). It has
//! no Kev-speculative methods; Kev adds an implementation, not a new trait shape.
//!
//! `prepare` is separate from scoring so a server can price a request (its token
//! budget) from the rows it is about to score, tokenizing each text exactly once.
//!
//! # What this crate is not
//!
//! Serving is transport-only and lives elsewhere (the thin decide MCP servers call
//! this crate; they re-implement nothing — OPS-03). The encoder is not here either:
//! it is aprender-core's reusable `aprender::models::modernbert` (D-13), and Laya's
//! head and scorer reuse its `Linear`, `layer_norm`, `gelu_exact` and `attention`.
//!
//! # The served model
//!
//! A deployed decision model is ONE `decide-apr-v1` `.apr` (D-17; [`artifact`]), packed
//! from a Laya run dir by [`pack`]. The only way to a model a server may classify with
//! is [`Decider::load_bytes`] / [`Decider::load_path`], which run the whole load ladder;
//! [`Decider`] has private fields and no other constructor. Every [`Decider`] carries
//! its [`ModelIdentity`] (D-11): the sha256 of the whole file and the recipe_id.
//!
//! Whether a run (or the exact file packed from it) may be SERVED is [`verify`]'s decision
//! alone (decide-apr-v1 `deploy_eligibility`): it re-scores every eval row from the packed
//! bytes and the declared base, and decides the laya-finetune-gate-v1 gate on metrics it
//! recomputes itself. [`pack`] applies no policy.
//!
//! Contracts: `contracts/laya-parity-v1.yaml` (the torch -> .apr -> Rust parity
//! ladder) and `contracts/decide-apr-v1.yaml` (the task schema, marker rule, artifact
//! and load ladder).

pub mod artifact;
pub mod digest;
pub mod laya;
pub mod pack;
pub mod task;
pub mod verify;

#[cfg(test)]
pub(crate) mod test_support;

use std::fmt;

pub use artifact::{ArtifactError, BaseDecl, Manifest};
pub use laya::LayaError;
pub use pack::{pack_run_dir, PackError, PackInputs};
pub use task::{Criterion, Task, TaskError};

use aprender::models::modernbert::ModernBertError;

/// One scored text.
#[derive(Debug, Clone, PartialEq)]
pub struct Decision {
    /// Index of the most probable criterion, in task criteria order (first on ties).
    pub label_index: usize,
    /// One calibrated probability per criterion, in task criteria order.
    pub probabilities: Vec<f32>,
    /// Tokens in the row the method actually scored (after any truncation).
    pub tokens: usize,
    /// True when the caller's text was cut to fit the method's window (D-12).
    pub truncated: bool,
}

/// A text already tokenized and built into the row a method will score.
///
/// Only a method's own [`DecisionMethod::prepare`] constructs one, so a row can never
/// carry read positions the builder did not produce.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedRow {
    pub(crate) ids: Vec<u32>,
    pub(crate) markers: Vec<usize>,
    pub(crate) truncated: bool,
}

impl PreparedRow {
    /// The token ids of the built row.
    #[must_use]
    pub fn ids(&self) -> &[u32] {
        &self.ids
    }

    /// Method-specific read positions (Laya: the `[MASK]` option markers).
    #[must_use]
    pub fn markers(&self) -> &[usize] {
        &self.markers
    }

    /// Row length in tokens — what a server's token budget is priced on.
    #[must_use]
    pub fn tokens(&self) -> usize {
        self.ids.len()
    }

    /// True when the caller's text was cut to fit the window (D-12).
    #[must_use]
    pub fn truncated(&self) -> bool {
        self.truncated
    }
}

/// Why a decision method refused.
#[derive(Debug, Clone, PartialEq)]
pub enum DecideError {
    /// The task document was refused (D-05).
    Task(TaskError),
    /// The Laya method refused its inputs or its parts.
    Laya(LayaError),
    /// The ModernBERT encoder or one of its primitives refused at runtime.
    Encoder(ModernBertError),
}

impl fmt::Display for DecideError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Task(e) => write!(f, "decide: {e}"),
            Self::Laya(e) => write!(f, "decide: {e}"),
            Self::Encoder(e) => write!(f, "decide: {e}"),
        }
    }
}

impl std::error::Error for DecideError {}

impl From<TaskError> for DecideError {
    fn from(e: TaskError) -> Self {
        Self::Task(e)
    }
}

impl From<LayaError> for DecideError {
    fn from(e: LayaError) -> Self {
        Self::Laya(e)
    }
}

impl From<ModernBertError> for DecideError {
    fn from(e: ModernBertError) -> Self {
        Self::Encoder(e)
    }
}

/// A decision method bound to one task.
///
/// Designed for one implementation ([`laya::Laya`]) with Kev as the known second;
/// see the crate docs for why it has no other methods.
pub trait DecisionMethod: Send + Sync {
    /// The task every decision answers (ordered labels).
    fn task(&self) -> &Task;

    /// Tokenize and build the row for every text, in order.
    ///
    /// # Errors
    ///
    /// A typed refusal when a text cannot be tokenized or its row cannot be built.
    fn prepare(&self, texts: &[String]) -> Result<Vec<PreparedRow>, DecideError>;

    /// Score rows produced by [`DecisionMethod::prepare`], in order.
    ///
    /// # Errors
    ///
    /// A typed refusal from the method's forward pass.
    fn classify_prepared(&self, rows: &[PreparedRow]) -> Result<Vec<Decision>, DecideError>;

    /// [`DecisionMethod::prepare`] then [`DecisionMethod::classify_prepared`].
    ///
    /// # Errors
    ///
    /// Either step's refusal.
    fn classify(&self, texts: &[String]) -> Result<Vec<Decision>, DecideError> {
        let rows = self.prepare(texts)?;
        self.classify_prepared(&rows)
    }
}

/// The identity every classify response carries (D-11).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelIdentity {
    /// sha256 of the WHOLE `.apr` file bytes.
    pub artifact_sha256: String,
    /// sha256 of the `decide.recipe_json` blob (== the run dir's `recipe.json`).
    pub recipe_id: String,
    /// The decision method (`laya`).
    pub method: String,
    /// The declared base as a display string (`laya-en-root@55cf4c4e`).
    pub base: String,
    /// The declared base (D-04), as recorded in the manifest.
    pub base_decl: BaseDecl,
}

/// A decision model that passed every rung of the decide-apr-v1 load ladder.
///
/// Its fields are private and it has no public constructor: [`Decider::load_bytes`],
/// [`Decider::load_hashed`] and [`Decider::load_path`] are the only doors, and all run the
/// whole ladder
/// (decide-apr-v1 rung 8, proven by the trybuild case `tests/ui/decider_struct_literal.rs`).
pub struct Decider {
    method: Box<dyn DecisionMethod>,
    identity: ModelIdentity,
    manifest: Manifest,
}

impl fmt::Debug for Decider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Decider")
            .field("identity", &self.identity)
            .field("labels", &self.manifest.labels)
            .finish_non_exhaustive()
    }
}

impl Decider {
    /// Verify `bytes` through the whole decide-apr-v1 ladder (size cap first).
    ///
    /// # Errors
    ///
    /// An [`ArtifactError`] naming the rung that refused.
    pub fn load_bytes(bytes: &[u8]) -> Result<Self, ArtifactError> {
        artifact::load_verified(bytes)
    }

    /// [`Decider::load_bytes`] over bytes this crate already hashed: the whole ladder runs
    /// on [`artifact::HashedArtifact::bytes`], and rung 8 mints the identity from that
    /// digest instead of hashing the same bytes again. For a caller that checks a pinned
    /// sha256 before any parse (the Lambda's cold start), this is one pass, not two.
    ///
    /// # Errors
    ///
    /// An [`ArtifactError`] naming the rung that refused.
    pub fn load_hashed(hashed: &artifact::HashedArtifact<'_>) -> Result<Self, ArtifactError> {
        artifact::load_verified_hashed(hashed)
    }

    /// Open `path`, refuse it by its metadata length before reading, read it bounded
    /// (rung 1), then run the ladder.
    ///
    /// # Errors
    ///
    /// [`ArtifactError::Read`] for an I/O failure, otherwise the refusing rung.
    pub fn load_path(path: impl AsRef<std::path::Path>) -> Result<Self, ArtifactError> {
        let io = |e: std::io::Error| ArtifactError::Read {
            reason: e.to_string(),
        };
        let file = std::fs::File::open(path.as_ref()).map_err(io)?;
        let declared = file.metadata().map_err(io)?.len();
        let bytes = artifact::read_decide_apr_bytes_bounded(file, Some(declared))?;
        Self::load_bytes(&bytes)
    }

    /// The model identity (D-11).
    #[must_use]
    pub fn identity(&self) -> &ModelIdentity {
        &self.identity
    }

    /// The verified manifest.
    #[must_use]
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// The task every decision answers.
    #[must_use]
    pub fn task(&self) -> &Task {
        self.method.task()
    }

    /// Labels in task criteria order (the label index).
    #[must_use]
    pub fn labels(&self) -> Vec<&str> {
        self.method.task().labels()
    }

    /// Tokenize and build each text's row (price a request before scoring it).
    ///
    /// # Errors
    ///
    /// The method's refusal.
    pub fn prepare(&self, texts: &[String]) -> Result<Vec<PreparedRow>, DecideError> {
        self.method.prepare(texts)
    }

    /// Score rows from [`Decider::prepare`].
    ///
    /// # Errors
    ///
    /// The method's refusal.
    pub fn classify_prepared(&self, rows: &[PreparedRow]) -> Result<Vec<Decision>, DecideError> {
        self.method.classify_prepared(rows)
    }

    /// Prepare then score.
    ///
    /// # Errors
    ///
    /// The method's refusal.
    pub fn classify(&self, texts: &[String]) -> Result<Vec<Decision>, DecideError> {
        self.method.classify(texts)
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    /// Every `.rs` file under `dir`, recursively, sorted.
    fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
        let entries =
            std::fs::read_dir(dir).unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()));
        for entry in entries {
            let path = entry.expect("dir entry").path();
            if path.is_dir() {
                rust_sources(&path, out);
            } else if path.extension().is_some_and(|x| x == "rs") {
                out.push(path);
            }
        }
    }

    /// V14-d / CV5 (plan 08-31): production code reads the safetensors format only in
    /// `src/pack.rs`.
    ///
    /// Scans every `.rs` file under `src/` at test time for the two needles — the crate path
    /// and the type name — and allows them only in `src/pack.rs` and in the allowlist below.
    /// The needles are assembled with `concat!` so this file cannot match them literally,
    /// wherever the test lives. Two must-match controls keep the scan honest: `src/pack.rs`
    /// must hold a needle (else the scan is blind), and every allowlisted file must still hold
    /// one (else the allowlist entry has outlived its reason and FAILS).
    #[test]
    fn safetensors_is_read_only_by_pack() {
        const NEEDLES: [&str; 2] = [concat!("safe", "tensors::"), concat!("Safe", "Tensors")];
        const OWNER: &str = "src/pack.rs";
        const ALLOWLIST: [(&str, &str); 1] = [(
            "src/test_support.rs",
            "`#[cfg(test)]` fixture builder: parses the tiny checkpoint to build test \
             artifacts; never compiled into a server",
        )];

        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut files = Vec::new();
        rust_sources(&root.join("src"), &mut files);
        files.sort();
        assert!(
            files.len() > 5,
            "the scan found only {} files; it is not reading src/",
            files.len()
        );

        let mut owner_hits = 0usize;
        let mut allow_hits = [0usize; ALLOWLIST.len()];
        let mut offenders = Vec::new();
        for path in &files {
            let rel = path
                .strip_prefix(root)
                .expect("under the manifest dir")
                .to_string_lossy()
                .replace('\\', "/");
            let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {rel}: {e}"));
            let hits = NEEDLES.iter().filter(|n| text.contains(*n)).count();
            if hits == 0 {
                continue;
            }
            if rel == OWNER {
                owner_hits += hits;
            } else if let Some(i) = ALLOWLIST.iter().position(|(f, _)| *f == rel) {
                allow_hits[i] += hits;
            } else {
                offenders.push(rel);
            }
        }
        assert!(
            offenders.is_empty(),
            "production code outside {OWNER} names a safetensors needle: {offenders:?}"
        );
        assert!(
            owner_hits > 0,
            "must-match control: {OWNER} holds no needle, so the scan cannot see a reader"
        );
        for ((file, reason), hits) in ALLOWLIST.iter().zip(allow_hits) {
            assert!(
                hits > 0,
                "allowlist entry {file} ({reason}) no longer matches; remove it"
            );
        }
    }

    /// D-14 (owner decision `publish-false`, 2026-09-29, plan 08-31): the crate name becomes
    /// permanent at its first crates.io publish, so the crate stays `publish = false` until the
    /// name is confirmed. Reversing the decision is deliberate: delete the manifest line, add the
    /// crate to scripts/cascade-publish.sh TIERS after aprender-core, and delete this test.
    #[test]
    fn publish_is_false_until_the_crate_name_is_confirmed() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
        let manifest = std::fs::read_to_string(&path).expect("read the crate manifest");
        let package = manifest
            .split("\n[")
            .next()
            .expect("the manifest opens with [package]");
        assert!(
            package.lines().any(|l| l.trim() == "publish = false"),
            "{} [package] must carry `publish = false` (D-14): the crate name is not confirmed",
            path.display()
        );
    }
}
