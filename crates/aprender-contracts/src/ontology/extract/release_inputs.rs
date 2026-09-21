//! aprender#3715 — the inputs `extract:release-evidence` reads, each refused BY NAME when unreadable: the release
//! subject (the CLI's `--release-*` flags), the consumer-derived context rungs, the dogfood receipt R5 judged, and
//! the per-host kernel-diff receipts. The per-host MODEL receipts are `crate::ontology::receipts` (one reader for
//! the ladder join and the release matrix).
//!
//! Every reader here is pure and deterministic (R-15): files in byte order, no clock, no network, no git. What a
//! file does not say is carried as `None`, never defaulted to the passing value — an absent `fallback` is not
//! `false`, an absent `max_err` is not within any bound.

use std::path::{Path, PathBuf};

/// The context-rung declaration, relative to the repository root.
pub const CONTEXT_RUNGS_FILE: &str = "evidence/release/context-rungs.json";
pub const CONTEXT_SCHEMA: &str = "apr-release-context-rungs/v1";
/// The per-host kernel-diff receipts: `evidence/dogfood/kernels/<version>/<host>.json` (#3715 addendum).
pub const KERNEL_EVIDENCE_DIR: &str = "evidence/dogfood/kernels";
pub const KERNEL_SCHEMA: &str = "apr-kernel-diff-receipt/v1";
/// The tokenizer-parity receipts (aprender#3726): apr's token ids against the pinned llama.cpp comparator.
pub const TOKENIZER_EVIDENCE_DIR: &str = "evidence/dogfood/tokenizer";
pub const TOKENIZER_SCHEMA: &str = "apr-tokenizer-parity-receipt/v1";
/// The shrink-only ratchet baseline for what the v1 surface cannot declare (#3745 S2).
pub const SURFACE_RATCHET_FILE: &str = "evidence/release/surface-ratchet.json";
pub const SURFACE_RATCHET_SCHEMA: &str = "apr-release-surface-ratchet/v1";
/// The rung whose token count is DERIVED as the max over the consumer records, never written down.
pub const CONSUMER_MAX: &str = "consumer-max";
/// Reserved: the per-model rung whose size is the model's own GGUF `context_length` (added by the extractor).
pub const DECLARED: &str = "declared";

/// What a release-readiness run is asked about. Built from `--release-version` / `--release-commit` and friends.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Subject {
    pub version: String,
    /// The release commit (MC), 40 lowercase hex: the dogfood receipt's `commit` must equal it.
    pub commit: String,
    /// T-4 only: the sha the committed receipts were measured at, after R7 proved the tree equal modulo
    /// `evidence/`. `None` → the receipts must carry [`Subject::commit`].
    pub receipts_commit: Option<String>,
    /// `--receipts`: the model-receipt dir. `None` → `evidence/dogfood/models/<version>/`.
    pub receipts_dir: Option<PathBuf>,
    /// `--kernel-receipts`: `None` → `evidence/dogfood/kernels/<version>/`.
    pub kernel_receipts_dir: Option<PathBuf>,
    /// `--dogfood-receipt`: the exact file rule_r5 judged. `None` → the release has no dogfood receipt (a violation).
    pub dogfood_receipt: Option<PathBuf>,
    /// `--tokenizer-receipts`: `None` → `evidence/dogfood/tokenizer/<version>/`.
    pub tokenizer_receipts_dir: Option<PathBuf>,
    /// `--surface`: the release candidate's `apr surface --json` (#3745 S1). `None` → no cell can be derived,
    /// which the `.release` shape names.
    pub surface: Option<PathBuf>,
}

impl Subject {
    /// Refuse a subject no receipt could ever match: an empty version, or a commit that is not 40 hex. A short sha
    /// is refused rather than prefix-matched — two releases can share a prefix, and the gate would not know.
    pub fn new(version: &str, commit: &str) -> Result<Self, ReleaseError> {
        if version.trim().is_empty() {
            return Err(ReleaseError::Subject("--release-version is empty".into()));
        }
        Ok(Self {
            version: version.trim().to_string(),
            commit: full_sha("--release-commit", commit)?,
            receipts_commit: None,
            receipts_dir: None,
            kernel_receipts_dir: None,
            dogfood_receipt: None,
            tokenizer_receipts_dir: None,
            surface: None,
        })
    }

    /// Set `--receipts-commit` (T-4), under the same 40-hex rule.
    pub fn with_receipts_commit(mut self, sha: &str) -> Result<Self, ReleaseError> {
        self.receipts_commit = Some(full_sha("--receipts-commit", sha)?);
        Ok(self)
    }

    /// The sha every model and kernel receipt's `apr_sha` must equal.
    #[must_use]
    pub fn measured_commit(&self) -> &str {
        self.receipts_commit.as_deref().unwrap_or(&self.commit)
    }

    #[must_use]
    pub fn model_dir(&self, root: &Path) -> PathBuf {
        self.receipts_dir.clone().unwrap_or_else(|| {
            root.join(crate::ontology::receipts::EVIDENCE_DIR)
                .join(&self.version)
        })
    }

    #[must_use]
    pub fn tokenizer_dir(&self, root: &Path) -> PathBuf {
        self.tokenizer_receipts_dir
            .clone()
            .unwrap_or_else(|| root.join(TOKENIZER_EVIDENCE_DIR).join(&self.version))
    }

    #[must_use]
    pub fn kernel_dir(&self, root: &Path) -> PathBuf {
        self.kernel_receipts_dir
            .clone()
            .unwrap_or_else(|| root.join(KERNEL_EVIDENCE_DIR).join(&self.version))
    }
}

fn full_sha(flag: &str, s: &str) -> Result<String, ReleaseError> {
    let s = s.trim().to_ascii_lowercase();
    if s.len() == 40 && s.bytes().all(|b| b.is_ascii_hexdigit()) {
        Ok(s)
    } else {
        Err(ReleaseError::Subject(format!(
            "{flag} {s:?} is not a full 40-hex commit (a prefix is refused, never matched)"
        )))
    }
}

/// Why the release evidence could not be read. Every variant is the DECLARATION's fault (exit 3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReleaseError {
    Subject(String),
    Receipt(crate::ontology::receipts::ReceiptError),
    Input { file: String, what: String },
}

impl std::fmt::Display for ReleaseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Subject(w) => write!(f, "release subject: {w}"),
            Self::Receipt(e) => write!(f, "receipt {e}"),
            Self::Input { file, what } => write!(f, "{file}: {what}"),
        }
    }
}

impl std::error::Error for ReleaseError {}

fn input_err(file: &str, what: impl Into<String>) -> ReleaseError {
    ReleaseError::Input {
        file: file.to_string(),
        what: what.into(),
    }
}

/// Read `path` as JSON carrying `schema == want`; any other schema is refused by name.
fn read_json(path: &Path, name: &str, want: &str) -> Result<serde_json::Value, ReleaseError> {
    let text =
        std::fs::read_to_string(path).map_err(|e| input_err(name, format!("unreadable: {e}")))?;
    let v: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| input_err(name, format!("not JSON: {e}")))?;
    let schema = v
        .get("schema")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    if schema != want {
        return Err(input_err(
            name,
            format!("schema {schema:?} is not {want} — refused by name"),
        ));
    }
    Ok(v)
}

fn str_of(v: &serde_json::Value, key: &str) -> Option<String> {
    v.get(key)
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
}

fn list<'a>(v: &'a serde_json::Value, key: &str) -> impl Iterator<Item = &'a serde_json::Value> {
    v.get(key)
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
}

/// One consumer's measured (or planned) maximum prompt, quoted from its own record on #3710 / #3716.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Consumer {
    pub name: String,
    /// `None` = the consumer has not recorded a number yet — which makes `consumer-max` underived.
    pub max_prompt_tokens: Option<u64>,
    /// `measured` or `plan`, as the consumer's own record says.
    pub basis: String,
    pub source: String,
}

/// One context rung, as declared. `tokens` is `None` for the derived rung until [`derive_rungs`] fills it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextRung {
    pub id: String,
    pub tokens: Option<u64>,
    /// A LONG rung is owed only by the models `ladder.cells.long_rungs_for` names (#3712 amendment 2); a short
    /// rung is owed by every model it fits.
    pub long: bool,
    pub derived_from: Vec<String>,
    /// Consumers the derived rung could not include because they recorded no number.
    pub unmeasured_consumers: Vec<String>,
}

/// The context-rung declaration as read: the rungs (with `consumer-max` derived) and the consumer records.
pub type ContextDecl = (Vec<ContextRung>, Vec<Consumer>);

/// Read the context-rung declaration. `Ok(None)` when the file is absent — which the shape turns into a
/// violation (a release with no context rung proves nothing about context); a present-but-foreign file is exit 3.
pub fn read_context_rungs(root: &Path) -> Result<Option<ContextDecl>, ReleaseError> {
    let path = root.join(CONTEXT_RUNGS_FILE);
    if !path.exists() {
        return Ok(None);
    }
    let v = read_json(&path, CONTEXT_RUNGS_FILE, CONTEXT_SCHEMA)?;
    let consumers: Vec<Consumer> = list(&v, "consumers")
        .map(|c| Consumer {
            name: str_of(c, "consumer").unwrap_or_default(),
            max_prompt_tokens: c
                .get("max_prompt_tokens")
                .and_then(serde_json::Value::as_u64),
            basis: str_of(c, "basis").unwrap_or_default(),
            source: str_of(c, "source").unwrap_or_default(),
        })
        .collect();
    if let Some(r) = list(&v, "rungs").find(|r| str_of(r, "id").as_deref() == Some(DECLARED)) {
        return Err(input_err(
            CONTEXT_RUNGS_FILE,
            format!("rung {r} redeclares `{DECLARED}`, which is each model's own GGUF context_length and is never written down"),
        ));
    }
    let rungs = list(&v, "rungs")
        .map(|r| ContextRung {
            id: str_of(r, "id").unwrap_or_default(),
            tokens: r.get("tokens").and_then(serde_json::Value::as_u64),
            long: r
                .get("long")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
            derived_from: list(r, "derived_from")
                .filter_map(serde_json::Value::as_str)
                .map(str::to_string)
                .collect(),
            unmeasured_consumers: Vec::new(),
        })
        .collect();
    Ok(Some((derive_rungs(rungs, &consumers), consumers)))
}

/// `consumer-max` is COMPUTED here: the max over every consumer record that states a number, derived from every
/// consumer's source. A declared `tokens` on it is ignored, and a consumer with no number is carried by name so
/// the shape can refuse the rung (a max over a subset is a smaller number wearing the full name).
#[must_use]
pub fn derive_rungs(rungs: Vec<ContextRung>, consumers: &[Consumer]) -> Vec<ContextRung> {
    rungs
        .into_iter()
        .map(|mut r| {
            if r.id == CONSUMER_MAX {
                r.tokens = consumers.iter().filter_map(|c| c.max_prompt_tokens).max();
                r.derived_from = consumers
                    .iter()
                    .filter(|c| !c.source.is_empty())
                    .map(|c| format!("{}: {}", c.name, c.source))
                    .collect();
                r.unmeasured_consumers = consumers
                    .iter()
                    .filter(|c| c.max_prompt_tokens.is_none())
                    .map(|c| c.name.clone())
                    .collect();
            }
            r
        })
        .collect()
}

/// The dogfood receipt R5 judged (`scripts/dogfood.sh`: `{verdict, commit, version, phase, deferred, …}`).
/// Only verdict, commit and version are read: the pre-publish deferral whitelist is R5's, and a second copy of it
/// here would be a second declaration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dogfood {
    pub file: String,
    pub verdict: String,
    pub commit: String,
    pub version: String,
}

pub fn read_dogfood(path: &Path) -> Result<Dogfood, ReleaseError> {
    let name = path.display().to_string();
    let text =
        std::fs::read_to_string(path).map_err(|e| input_err(&name, format!("unreadable: {e}")))?;
    let v: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| input_err(&name, format!("not JSON: {e}")))?;
    Ok(Dogfood {
        file: name,
        verdict: str_of(&v, "verdict").unwrap_or_default(),
        commit: str_of(&v, "commit")
            .unwrap_or_default()
            .to_ascii_lowercase(),
        version: str_of(&v, "version").unwrap_or_default(),
    })
}

/// One inventory model's dispatch path on one host: the kernels `apr parity --per-op` reports for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dispatch {
    pub sha256: Option<String>,
    pub file: String,
    /// (kernel, quant). A GEMV is keyed on its quant — qwen3moe runs `ffn_down_exps` as Q4_K in 24 layers and
    /// Q6_K in 24 (#3712, aprender-eb) — so a kernel listed as a bare string carries quant `""`.
    pub kernels: Vec<(String, String)>,
}

/// One kernel-vs-reference measurement.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelRow {
    pub kernel: String,
    pub quant: String,
    pub reference: String,
    pub max_err: Option<f64>,
    /// A DISCRETE kernel (router top-k): token×layer slots whose selected sets differ, and the largest CPU logit
    /// gap between rank k and k+1 among them. `max_err` means nothing when the two sides chose different sets.
    pub index_mismatch: Option<u64>,
    pub tie_margin: Option<f64>,
    pub bound: Option<f64>,
    pub bound_source: String,
    pub verdict: String,
    pub reason: String,
}

/// One kernel-diff receipt file (`apr-kernel-diff-receipt/v1`).
#[derive(Debug, Clone, PartialEq)]
pub struct KernelReceipt {
    pub file: String,
    pub host: String,
    pub version: String,
    pub apr_sha: Option<String>,
    pub sm: String,
    pub dispatch: Vec<Dispatch>,
    pub rows: Vec<KernelRow>,
}

/// Every `*.json` under `dir`, in byte order (none when the dir is absent — the shape names the host).
pub fn read_kernel_receipts(dir: &Path, root: &Path) -> Result<Vec<KernelReceipt>, ReleaseError> {
    let mut files = Vec::new();
    walk_json(dir, &mut files);
    files.sort();
    files
        .iter()
        .map(|f| {
            let name = f
                .strip_prefix(root)
                .unwrap_or(f)
                .to_string_lossy()
                .replace('\\', "/");
            let v = read_json(f, &name, KERNEL_SCHEMA)?;
            Ok(parse_kernel_receipt(&name, &v))
        })
        .collect()
}

fn walk_json(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            walk_json(&p, out);
        } else if p.extension().and_then(|x| x.to_str()) == Some("json") {
            out.push(p);
        }
    }
}

fn parse_kernel_receipt(file: &str, v: &serde_json::Value) -> KernelReceipt {
    KernelReceipt {
        file: file.to_string(),
        host: str_of(v, "host").unwrap_or_default(),
        version: str_of(v, "version").unwrap_or_default(),
        apr_sha: str_of(v, "apr_sha").map(|s| s.to_ascii_lowercase()),
        sm: str_of(v, "sm").unwrap_or_default(),
        dispatch: list(v, "dispatch")
            .map(|d| Dispatch {
                sha256: str_of(d, "sha256").map(|s| s.to_ascii_lowercase()),
                file: str_of(d, "file").unwrap_or_default(),
                kernels: list(d, "kernels").filter_map(kernel_key).collect(),
            })
            .collect(),
        rows: list(v, "rows")
            .map(|r| KernelRow {
                kernel: str_of(r, "kernel").unwrap_or_default(),
                quant: str_of(r, "quant").unwrap_or_default().to_ascii_lowercase(),
                reference: str_of(r, "reference").unwrap_or_default(),
                max_err: r.get("max_err").and_then(serde_json::Value::as_f64),
                index_mismatch: r.get("index_mismatch").and_then(serde_json::Value::as_u64),
                tie_margin: r.get("tie_margin").and_then(serde_json::Value::as_f64),
                bound: r.get("bound").and_then(serde_json::Value::as_f64),
                bound_source: str_of(r, "bound_source").unwrap_or_default(),
                verdict: str_of(r, "verdict")
                    .unwrap_or_default()
                    .to_ascii_lowercase(),
                reason: str_of(r, "reason").unwrap_or_default(),
            })
            .collect(),
    }
}

/// A dispatch entry: `"kernel"` or `{"kernel": …, "quant": …}`.
fn kernel_key(k: &serde_json::Value) -> Option<(String, String)> {
    match k {
        serde_json::Value::String(s) => Some((s.clone(), String::new())),
        serde_json::Value::Object(_) => Some((
            str_of(k, "kernel")?,
            str_of(k, "quant").unwrap_or_default().to_ascii_lowercase(),
        )),
        _ => None,
    }
}

impl KernelRow {
    /// Within the measured bound. Continuous: `max_err ≤ bound`. Discrete (`index_mismatch` present): no mismatch,
    /// or every mismatch a near-tie (`tie_margin ≤ bound`). Anything unmeasured is outside.
    #[must_use]
    pub fn within_bound(&self) -> bool {
        match self.index_mismatch {
            Some(0) => true,
            Some(_) => matches!((self.tie_margin, self.bound), (Some(t), Some(b)) if t <= b),
            None => matches!((self.max_err, self.bound), (Some(e), Some(b)) if e <= b),
        }
    }
}

/// One model's tokenizer comparison (#3726): apr's ids vs the pinned llama.cpp on a named corpus.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokRow {
    pub sha256: Option<String>,
    pub file: String,
    pub family: String,
    pub corpus_sha256: String,
    pub comparator: String,
    pub comparator_sha: String,
    pub tokens_compared: Option<u64>,
    pub mismatches: Option<u64>,
    /// `decode(encode(x)) == x` over the corpus.
    pub roundtrip_ok: Option<bool>,
    pub verdict: String,
    pub reason: String,
}

impl TokRow {
    /// Identical ids over a corpus that is not empty: zero tokens compared is not parity, it is no measurement.
    #[must_use]
    pub fn exact(&self) -> bool {
        self.mismatches == Some(0) && self.tokens_compared.is_some_and(|n| n > 0)
    }
}

/// One tokenizer-parity receipt file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokReceipt {
    pub file: String,
    pub version: String,
    pub apr_sha: Option<String>,
    pub rows: Vec<TokRow>,
}

/// Every `*.json` under `dir` (none when absent — every model's tokenizer cell is then a missing row).
pub fn read_tokenizer_receipts(dir: &Path, root: &Path) -> Result<Vec<TokReceipt>, ReleaseError> {
    let mut files = Vec::new();
    walk_json(dir, &mut files);
    files.sort();
    files
        .iter()
        .map(|f| {
            let name = f
                .strip_prefix(root)
                .unwrap_or(f)
                .to_string_lossy()
                .replace('\\', "/");
            let v = read_json(f, &name, TOKENIZER_SCHEMA)?;
            Ok(TokReceipt {
                file: name,
                version: str_of(&v, "version").unwrap_or_default(),
                apr_sha: str_of(&v, "apr_sha").map(|s| s.to_ascii_lowercase()),
                rows: list(&v, "rows").map(parse_tok_row).collect(),
            })
        })
        .collect()
}

fn parse_tok_row(r: &serde_json::Value) -> TokRow {
    TokRow {
        sha256: str_of(r, "sha256").map(|s| s.to_ascii_lowercase()),
        file: str_of(r, "file").unwrap_or_default(),
        family: str_of(r, "family").unwrap_or_default(),
        corpus_sha256: str_of(r, "corpus_sha256").unwrap_or_default(),
        comparator: str_of(r, "comparator").unwrap_or_default(),
        comparator_sha: str_of(r, "comparator_sha").unwrap_or_default(),
        tokens_compared: r.get("tokens_compared").and_then(serde_json::Value::as_u64),
        mismatches: r.get("mismatches").and_then(serde_json::Value::as_u64),
        roundtrip_ok: r.get("roundtrip_ok").and_then(serde_json::Value::as_bool),
        verdict: str_of(r, "verdict")
            .unwrap_or_default()
            .to_ascii_lowercase(),
        reason: str_of(r, "reason").unwrap_or_default(),
    }
}

/// The committed ceilings for the two counts v1 cannot shrink by itself: args no role type claims, and
/// stdin-capable positionals whose stdin form v1 cannot declare. May only FALL (#3745, cop).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SurfaceRatchet {
    pub unknown_args: u64,
    pub stdin_undeclared: u64,
}

/// `Ok(None)` when the file is absent (the release then names the missing baseline); a foreign schema is exit 3.
pub fn read_surface_ratchet(root: &Path) -> Result<Option<SurfaceRatchet>, ReleaseError> {
    let path = root.join(SURFACE_RATCHET_FILE);
    if !path.exists() {
        return Ok(None);
    }
    let v = read_json(&path, SURFACE_RATCHET_FILE, SURFACE_RATCHET_SCHEMA)?;
    let n = |k: &str| {
        v.get(k)
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| input_err(SURFACE_RATCHET_FILE, format!("`{k}` is not a count")))
    };
    Ok(Some(SurfaceRatchet {
        unknown_args: n("unknown_args")?,
        stdin_undeclared: n("stdin_undeclared")?,
    }))
}
