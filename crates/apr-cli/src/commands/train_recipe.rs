//! Declarative training recipe: `apr-recipe-v1` (CRUX-001, PMAT-5316).
//!
//! Contract: `contracts/apr-recipe-v1.yaml`. One file names everything a
//! fine-tune or distillation run depends on; `apr finetune --recipe r.yaml`
//! refuses an invalid recipe before any model is opened, and every refusal
//! names the recipe field at fault. Relative paths resolve against the
//! recipe file's directory.
//!
//! This module is the schema and its validator only. The recipe is lowered
//! onto the existing `apr finetune` arguments by `finetune::run_from_recipe`.

use super::train_recipe_lora::{RawLoraFields, RecipeLora};
use crate::error::{CliError, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::{Path, PathBuf};

/// The value the recipe's `recipe:` key must carry.
pub(crate) const RECIPE_SCHEMA: &str = "apr-recipe-v1";
/// The `schema` value of a receipt built from a recipe.
pub(crate) const RECEIPT_SCHEMA: &str = "apr-recipe-receipt-v1";
/// Recipe fields this engine validates and hashes but does not yet feed into
/// the training loop. Kept in ONE place: when `apr finetune` threads the seed
/// or runs the eval, the name leaves this list and the receipt says so.
pub(crate) const FINETUNE_NOT_APPLIED: &[&str] = &["seed", "eval"];

/// The training method a recipe selects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum RecipeMethod {
    Lora,
    Qlora,
    Distill,
}

impl RecipeMethod {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "lora" => Some(Self::Lora),
            "qlora" => Some(Self::Qlora),
            "distill" => Some(Self::Distill),
            _ => None,
        }
    }

    /// The spelling used in the recipe and in `apr finetune --method`.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Lora => "lora",
            Self::Qlora => "qlora",
            Self::Distill => "distill",
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawRecipe {
    recipe: Option<String>,
    name: Option<String>,
    base: Option<String>,
    model_size: Option<String>,
    method: Option<String>,
    data: Option<RawData>,
    eval: Option<RawEval>,
    seed: Option<i64>,
    teacher: Option<RawTeacher>,
    #[serde(default)]
    training: RawTraining,
    lora: Option<RawLora>,
    output: Option<RawOutput>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawData {
    train: Option<String>,
    sha256: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawEval {
    heldout: Option<String>,
    benchmark: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTeacher {
    model: Option<String>,
    temperature: Option<f64>,
    alpha: Option<f64>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTraining {
    rank: Option<u32>,
    epochs: Option<u32>,
    learning_rate: Option<f64>,
    max_seq_len: Option<usize>,
    warmup: Option<f64>,
    weight_decay: Option<f64>,
}

/// The `lora:` block (issue #7240: `lora.alpha`, `lora.dropout`, `lora.target_modules`).
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawLora {
    alpha: Option<f64>,
    dropout: Option<f64>,
    target_modules: Option<Vec<String>>,
}

impl RawRecipe {
    /// The PMAT-7240 keys, handed to `train_recipe_lora` for validation.
    fn lora_fields(&self) -> RawLoraFields {
        let l = self.lora.clone().unwrap_or_default();
        RawLoraFields {
            alpha: l.alpha,
            dropout: l.dropout,
            target_modules: l.target_modules,
            warmup: self.training.warmup,
            weight_decay: self.training.weight_decay,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawOutput {
    dir: Option<String>,
}

/// The teacher half of a distillation recipe.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RecipeTeacher {
    pub(crate) model: PathBuf,
    pub(crate) temperature: f64,
    pub(crate) alpha: f64,
}

/// A recipe that passed every rule of `apr-recipe-v1`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TrainRecipe {
    pub(crate) name: String,
    pub(crate) base: PathBuf,
    /// Optional planning size ("0.8B"), so `--plan` works before the base is
    /// on disk. `--model-size` on the command line wins over it.
    pub(crate) model_size: Option<String>,
    pub(crate) method: RecipeMethod,
    pub(crate) train: PathBuf,
    /// Optional pin: the sha256 the training file must have.
    pub(crate) data_sha256_pin: Option<String>,
    pub(crate) heldout: PathBuf,
    pub(crate) benchmark: Option<String>,
    pub(crate) seed: u64,
    pub(crate) teacher: Option<RecipeTeacher>,
    pub(crate) rank: Option<u32>,
    pub(crate) epochs: Option<u32>,
    pub(crate) learning_rate: Option<f64>,
    pub(crate) max_seq_len: Option<usize>,
    /// LoRA alpha, dropout, target modules, warmup and weight decay (PMAT-7240).
    pub(crate) lora: RecipeLora,
    pub(crate) output_dir: Option<PathBuf>,
    /// sha256 of the recipe file's bytes. The eval block is part of it.
    pub(crate) recipe_sha256: String,
}

/// One engine that took part in a run.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct EngineId {
    pub(crate) name: String,
    pub(crate) version: String,
    pub(crate) sha: String,
}

/// The receipt fields `apr-recipe-v1` requires of every run.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct RecipeReceipt {
    pub(crate) schema: &'static str,
    pub(crate) recipe_name: String,
    pub(crate) recipe_sha256: String,
    /// `None` when the file was not there to hash (plan mode).
    pub(crate) data_sha256: Option<String>,
    pub(crate) eval_sha256: Option<String>,
    pub(crate) method: RecipeMethod,
    pub(crate) seed: u64,
    pub(crate) engines: Vec<EngineId>,
    pub(crate) overrides: Vec<String>,
    pub(crate) validated_not_applied: Vec<&'static str>,
}

/// Resolve `.` and `..` and repeated separators without touching the disk, so
/// two spellings of one path compare equal.
fn lexical_normalize(path: &Path) -> PathBuf {
    use std::path::Component;
    let mut out = PathBuf::new();
    for c in path.components() {
        match c {
            Component::CurDir => {}
            // `..` removes a name, stays at the root, and is kept when
            // there is no name left to remove (`../../a` stays `../../a`).
            Component::ParentDir => match out.components().next_back() {
                Some(Component::Normal(_)) => {
                    out.pop();
                }
                Some(Component::RootDir | Component::Prefix(_)) => {}
                _ => out.push(".."),
            },
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// True when both paths exist and are one file on disk (symlink, hard link).
fn same_file(a: &Path, b: &Path) -> bool {
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(x), Ok(y)) if x == y => true,
        _ => same_inode(a, b),
    }
}

#[cfg(unix)]
fn same_inode(a: &Path, b: &Path) -> bool {
    use std::os::unix::fs::MetadataExt;
    match (std::fs::metadata(a), std::fs::metadata(b)) {
        (Ok(x), Ok(y)) => x.dev() == y.dev() && x.ino() == y.ino(),
        _ => false,
    }
}

#[cfg(not(unix))]
fn same_inode(_: &Path, _: &Path) -> bool {
    false
}

fn refuse(origin: &str, violations: &[String]) -> CliError {
    CliError::ValidationFailed(format!(
        "recipe {origin} is not a valid {RECIPE_SCHEMA}: {}",
        violations.join("; ")
    ))
}

/// Lower-case hex of a sha256 digest.
fn hex(digest: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

pub(crate) fn sha256_bytes(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

/// sha256 of a file, streamed. `None` when the file cannot be read.
pub(crate) fn sha256_file(path: &Path) -> Option<String> {
    // Asked before the open: a device has no end, and opening a pipe waits
    // for a writer.
    if !std::fs::metadata(path).ok()?.is_file() {
        return None;
    }
    let mut file = std::fs::File::open(path).ok()?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0_u8; 1 << 16];
    loop {
        let n = file.read(&mut buf).ok()?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Some(hex(&hasher.finalize()))
}

fn is_sha256(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// A required, non-empty string; a miss is recorded against `field`.
fn required(value: Option<&String>, field: &str, why: &str, out: &mut Vec<String>) -> String {
    match value.map(|s| s.trim()) {
        Some(s) if !s.is_empty() => s.to_string(),
        Some(_) => {
            out.push(format!("field `{field}` is empty ({why})"));
            String::new()
        }
        None => {
            out.push(format!("field `{field}` is required ({why})"));
            String::new()
        }
    }
}

/// The `recipe` field must name the recipe schema.
fn check_schema_id(raw: &RawRecipe, bad: &mut Vec<String>) {
    match raw.recipe.as_deref() {
        Some(RECIPE_SCHEMA) => {}
        Some(other) => bad.push(format!(
            "field `recipe` must be `{RECIPE_SCHEMA}`, got `{other}`"
        )),
        None => bad.push(format!(
            "field `recipe` is required (the schema id, `{RECIPE_SCHEMA}`)"
        )),
    }
}

/// Resolve `method`, recording why it is missing or unknown.
fn check_method(raw: &RawRecipe, bad: &mut Vec<String>) -> Option<RecipeMethod> {
    let method_text = required(
        raw.method.as_ref(),
        "method",
        "one of lora, qlora, distill",
        bad,
    );
    let method = RecipeMethod::parse(&method_text);
    if method.is_none() && !method_text.is_empty() {
        bad.push(format!(
            "field `method` must be one of lora, qlora, distill, got `{method_text}`"
        ));
    }
    method
}

/// The training file and its optional pinned hash.
fn check_data(raw: &RawRecipe, bad: &mut Vec<String>) -> (String, Option<String>) {
    let (train, pin) = match raw.data.as_ref() {
        Some(d) => (
            required(
                d.train.as_ref(),
                "data.train",
                "the training data file",
                bad,
            ),
            d.sha256.clone(),
        ),
        None => {
            bad.push("field `data` is required (block with `train`)".to_string());
            (String::new(), None)
        }
    };
    if let Some(p) = pin.as_deref() {
        if !is_sha256(p) {
            bad.push("field `data.sha256` must be 64 lower-case hex characters".to_string());
        }
    }
    (train, pin)
}

/// The held-out eval file and optional benchmark name.
fn check_eval(
    raw: &RawRecipe,
    train: &str,
    base_dir: &Path,
    bad: &mut Vec<String>,
) -> (String, Option<String>) {
    let (heldout, benchmark) = match raw.eval.as_ref() {
        Some(e) => (
            required(
                e.heldout.as_ref(),
                "eval.heldout",
                "the held-out eval file",
                bad,
            ),
            e.benchmark.clone(),
        ),
        None => {
            bad.push(
                "field `eval` is required (block with `heldout`: a run without an eval \
                 cannot be compared)"
                    .to_string(),
            );
            (String::new(), None)
        }
    };
    if !heldout.is_empty()
        && !train.is_empty()
        && lexical_normalize(&base_dir.join(&heldout)) == lexical_normalize(&base_dir.join(train))
    {
        bad.push("field `eval.heldout` must not be the training file `data.train`".to_string());
    }
    if benchmark.as_deref().is_some_and(|b| b.trim().is_empty()) {
        bad.push("field `eval.benchmark` is empty (omit it or name a benchmark)".to_string());
    }
    (heldout, benchmark)
}

/// `seed` must be present and an unsigned integer.
fn check_seed(seed: Option<i64>, bad: &mut Vec<String>) -> Option<u64> {
    match seed {
        None => {
            bad.push("field `seed` is required (an unsigned integer)".to_string());
            None
        }
        Some(s) => u64::try_from(s).ok().or_else(|| {
            bad.push(format!("field `seed` must be an unsigned integer, got {s}"));
            None
        }),
    }
}

fn check_training(t: &RawTraining, out: &mut Vec<String>) {
    if t.rank == Some(0) {
        out.push("field `training.rank` must be greater than 0".to_string());
    }
    if t.epochs == Some(0) {
        out.push("field `training.epochs` must be greater than 0".to_string());
    }
    if let Some(lr) = t.learning_rate {
        if !(lr.is_finite() && lr > 0.0) {
            out.push(format!(
                "field `training.learning_rate` must be a positive number, got {lr}"
            ));
        }
    }
    if t.max_seq_len == Some(0) {
        out.push("field `training.max_seq_len` must be greater than 0".to_string());
    }
}

fn check_teacher(
    raw: Option<&RawTeacher>,
    method: Option<RecipeMethod>,
    base_dir: &Path,
    out: &mut Vec<String>,
) -> Option<RecipeTeacher> {
    let distill = method == Some(RecipeMethod::Distill);
    let Some(t) = raw else {
        if distill {
            out.push(
                "field `teacher` is required when `method` is distill (the model to distill from)"
                    .to_string(),
            );
        }
        return None;
    };
    if !distill && method.is_some() {
        out.push("field `teacher` is only valid when `method` is distill".to_string());
    }
    let model = required(
        t.model.as_ref(),
        "teacher.model",
        "the teacher model file",
        out,
    );
    let temperature = t.temperature.unwrap_or(3.0);
    if !(temperature.is_finite() && temperature > 0.0) {
        out.push(format!(
            "field `teacher.temperature` must be a positive number, got {temperature}"
        ));
    }
    let alpha = t.alpha.unwrap_or(0.7);
    if !(0.0..=1.0).contains(&alpha) {
        out.push(format!(
            "field `teacher.alpha` must be between 0 and 1, got {alpha}"
        ));
    }
    Some(RecipeTeacher {
        model: base_dir.join(model),
        temperature,
        alpha,
    })
}

impl TrainRecipe {
    /// Read and validate a recipe file. Nothing but the recipe is opened.
    pub(crate) fn load(path: &Path) -> Result<Self> {
        // Asked before the open: a device has no end, and opening a pipe waits
        // for a writer.
        let is_regular = std::fs::metadata(path).map_err(|e| {
            CliError::ValidationFailed(format!("cannot read recipe {}: {e}", path.display()))
        })?;
        if !is_regular.is_file() {
            return Err(CliError::ValidationFailed(format!(
                "recipe {} is not a regular file",
                path.display()
            )));
        }
        let bytes = std::fs::read(path).map_err(|e| {
            CliError::ValidationFailed(format!("cannot read recipe {}: {e}", path.display()))
        })?;
        let src = String::from_utf8(bytes).map_err(|e| {
            CliError::ValidationFailed(format!("recipe {} is not UTF-8: {e}", path.display()))
        })?;
        let base_dir = path.parent().unwrap_or_else(|| Path::new(""));
        Self::parse(&src, base_dir, &path.display().to_string())
    }

    /// Validate recipe text. `base_dir` anchors relative paths; `origin` is
    /// how the recipe is named in an error.
    pub(crate) fn parse(src: &str, base_dir: &Path, origin: &str) -> Result<Self> {
        let raw: RawRecipe = serde_saphyr::from_str_with_options(
            src,
            serde_saphyr::options! { strict_booleans: true, merge_keys: serde_saphyr::MergeKeyPolicy::AsOrdinary, emit_comments: false, reject_non_finite_typeless_float: false },
        )
        .map_err(|e| refuse(origin, &[e.to_string()]))?;

        let mut bad = Vec::new();
        check_schema_id(&raw, &mut bad);
        let name = required(raw.name.as_ref(), "name", "a label for receipts", &mut bad);
        let base = required(raw.base.as_ref(), "base", "the base model file", &mut bad);
        if raw
            .model_size
            .as_deref()
            .is_some_and(|m| m.trim().is_empty())
        {
            bad.push(
                "field `model_size` is empty (omit it or give a size such as 0.8B)".to_string(),
            );
        }
        let method = check_method(&raw, &mut bad);

        let (train, pin) = check_data(&raw, &mut bad);
        let (heldout, benchmark) = check_eval(&raw, &train, base_dir, &mut bad);
        let seed = check_seed(raw.seed, &mut bad);
        let teacher = check_teacher(raw.teacher.as_ref(), method, base_dir, &mut bad);
        check_training(&raw.training, &mut bad);
        let lora = RecipeLora::check(&raw.lora_fields(), &mut bad);
        let output_dir = raw.output.as_ref().map(|o| {
            required(
                o.dir.as_ref(),
                "output.dir",
                "where the run writes",
                &mut bad,
            )
        });

        let (Some(method), Some(seed), true) = (method, seed, bad.is_empty()) else {
            return Err(refuse(origin, &bad));
        };
        Ok(Self {
            name,
            base: base_dir.join(base),
            model_size: raw.model_size.map(|m| m.trim().to_string()),
            method,
            train: base_dir.join(train),
            data_sha256_pin: pin,
            heldout: base_dir.join(heldout),
            benchmark,
            seed,
            teacher,
            rank: raw.training.rank,
            epochs: raw.training.epochs,
            learning_rate: raw.training.learning_rate,
            max_seq_len: raw.training.max_seq_len,
            lora,
            output_dir: output_dir.map(|d| base_dir.join(d)),
            recipe_sha256: sha256_bytes(src.as_bytes()),
        })
    }

    /// Check the files a real run reads are there, and that the training data
    /// still has its pinned hash. Runs before any model is loaded.
    pub(crate) fn require_inputs(&self) -> Result<()> {
        let mut bad = Vec::new();
        let mut files = vec![
            ("base", &self.base),
            ("data.train", &self.train),
            ("eval.heldout", &self.heldout),
        ];
        if let Some(t) = &self.teacher {
            files.push(("teacher.model", &t.model));
        }
        for (field, path) in files {
            if !path.is_file() {
                bad.push(format!("field `{field}`: no such file {}", path.display()));
            }
        }
        if same_file(&self.train, &self.heldout) {
            bad.push(
                "field `eval.heldout` must not be the training file `data.train` \
                 (they are one file on disk)"
                    .to_string(),
            );
        }
        if let (Some(pin), Some(actual)) = (&self.data_sha256_pin, sha256_file(&self.train)) {
            if *pin != actual {
                bad.push(format!(
                    "field `data.sha256` pins {pin} but {} hashes to {actual}",
                    self.train.display()
                ));
            }
        }
        if bad.is_empty() {
            Ok(())
        } else {
            Err(CliError::ValidationFailed(format!(
                "recipe `{}` cannot run: {}",
                self.name,
                bad.join("; ")
            )))
        }
    }

    #[allow(clippy::disallowed_methods)]
    /// The recipe with every default filled in, as `--dry-run` / `--plan` print it (PMAT-7240).
    /// Every value is a number the run uses: `run` carries the planner's rank, alpha and learning
    /// rate (after the recipe's own values and `--rank`) and the sequence length of the selected
    /// training path. `epochs` is the count the run will use; `model_size` / `output` are after
    /// run flags.
    pub(crate) fn resolved(
        &self,
        model_size: Option<&str>,
        output: Option<&Path>,
        epochs: u32,
        run: &ResolvedRun,
    ) -> serde_json::Value {
        serde_json::json!({
            "recipe": RECIPE_SCHEMA,
            "name": self.name,
            "base": self.base.display().to_string(),
            "model_size": model_size,
            "method": self.method.as_str(),
            "data": {
                "train": self.train.display().to_string(),
                "sha256": self.data_sha256_pin,
            },
            "eval": {
                "heldout": self.heldout.display().to_string(),
                "benchmark": self.benchmark,
            },
            "seed": self.seed,
            "lora": {
                "alpha": run.alpha,
                "dropout": self.lora.dropout,
                "target_modules": self.lora.target_modules,
            },
            "training": {
                "rank": run.rank,
                "epochs": epochs,
                "learning_rate": run.learning_rate,
                "max_seq_len": run.max_seq_len,
                "warmup": self.lora.warmup_fraction,
                "weight_decay": self.lora.weight_decay,
            },
            "output": { "dir": output.map(|o| o.display().to_string()) },
        })
    }

    /// The receipt fields for a run of this recipe by `engines`.
    pub(crate) fn receipt(
        &self,
        engines: Vec<EngineId>,
        overrides: Vec<String>,
        validated_not_applied: &[&'static str],
    ) -> RecipeReceipt {
        RecipeReceipt {
            schema: RECEIPT_SCHEMA,
            recipe_name: self.name.clone(),
            recipe_sha256: self.recipe_sha256.clone(),
            data_sha256: sha256_file(&self.train),
            eval_sha256: sha256_file(&self.heldout),
            method: self.method,
            seed: self.seed,
            engines,
            overrides,
            validated_not_applied: validated_not_applied.to_vec(),
        }
    }
}

/// The numbers a run of a recipe settles on, which the recipe may leave to the planner.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ResolvedRun {
    pub(crate) rank: u32,
    pub(crate) alpha: f32,
    pub(crate) learning_rate: f64,
    pub(crate) max_seq_len: usize,
}

/// This binary, as a receipt engine.
pub(crate) fn apr_engine() -> EngineId {
    EngineId {
        name: "apr".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        sha: env!("APR_GIT_SHA").to_string(),
    }
}

#[cfg(test)]
#[path = "train_recipe_tests.rs"]
mod tests;
