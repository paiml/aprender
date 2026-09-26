//! `apr finetune --recipe FILE` (E8 #4002 row R-1b) and `apr distill --recipe
//! FILE` (row R-2c), both against `contracts/apr-recipe-v1.yaml`.
//!
//! The recipe is parsed, and the data and eval files are hashed against it,
//! before any model is opened. Every refusal names the recipe field, so a
//! recipe defect reads differently from a CLI flag error.

use std::io::Read;
use std::path::{Path, PathBuf};

use entrenar::recipe::{EvalMetric, MethodKind, Recipe, RecipeError};
use sha2::{Digest, Sha256};

use crate::error::{CliError, Result};

/// The finetune arguments a recipe supplies.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RecipeArgs {
    pub model: PathBuf,
    pub method: String,
    pub rank: Option<u32>,
    pub data: PathBuf,
    /// The validation set (`eval.held_out`), hashed against `eval.sha256`.
    pub held_out: PathBuf,
    pub epochs: u32,
    pub learning_rate: f64,
    pub seed: u64,
    pub hash: String,
}

fn refuse(e: RecipeError) -> CliError {
    CliError::ValidationFailed(e.to_string())
}

fn refuse_field(field: &str, reason: String) -> CliError {
    refuse(RecipeError {
        field: field.to_string(),
        reason,
    })
}

fn sha256_file(path: &Path) -> std::io::Result<String> {
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// Resolve a path from the recipe relative to the recipe file's directory.
fn resolve(recipe_dir: &Path, p: &str) -> PathBuf {
    let p = Path::new(p);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        recipe_dir.join(p)
    }
}

fn check_hash(field: &str, path: &Path, want: &str) -> Result<()> {
    let got = sha256_file(path)
        .map_err(|e| refuse_field(field, format!("cannot hash {}: {e}", path.display())))?;
    if got != want {
        return Err(refuse_field(
            field,
            format!("{} hashes to {got}, recipe says {want}", path.display()),
        ));
    }
    Ok(())
}

/// Parse and check a recipe file. Opens no model.
pub(crate) fn load(path: &Path) -> Result<RecipeArgs> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| refuse_field("<file>", format!("cannot read {}: {e}", path.display())))?;
    let recipe = Recipe::parse(&text).map_err(refuse)?;
    let dir = path.parent().unwrap_or_else(|| Path::new("."));

    let method = match recipe.method.kind {
        MethodKind::Lora => "lora",
        MethodKind::Qlora => "qlora",
        MethodKind::Full => "full",
        MethodKind::Distill => {
            return Err(refuse_field(
                "method.kind",
                "distill recipes run through `apr distill`, not `apr finetune`".to_string(),
            ))
        }
    };

    // The finetune trainer steps one sample at a time; a recipe that asks for
    // a larger batch would get a run that differs from what it declares.
    if recipe.training.batch_size != 1 {
        return Err(refuse_field(
            "training.batch_size",
            format!(
                "apr finetune trains one sample per step; batch_size {} is not honored yet (use 1)",
                recipe.training.batch_size
            ),
        ));
    }

    // The finetune trainer reports validation loss only.
    if recipe.eval.metric != EvalMetric::Loss {
        return Err(refuse_field(
            "eval.metric",
            "apr finetune reports validation loss; metric accuracy is not honored yet (use loss)"
                .to_string(),
        ));
    }

    let data = resolve(dir, &recipe.data.train);
    check_hash("data.sha256", &data, &recipe.data.sha256)?;
    let held_out = resolve(dir, &recipe.eval.held_out);
    check_hash("eval.sha256", &held_out, &recipe.eval.sha256)?;

    Ok(RecipeArgs {
        model: PathBuf::from(&recipe.base.model),
        method: method.to_string(),
        rank: recipe.method.rank,
        data,
        held_out,
        epochs: recipe.training.epochs,
        learning_rate: recipe.training.learning_rate,
        seed: recipe.training.seed,
        hash: recipe.hash(),
    })
}

/// Distill temperature when the recipe names none. Equal to the
/// `apr distill --temperature` default.
pub(crate) const DISTILL_DEFAULT_TEMPERATURE: f64 = 3.0;

/// The `apr distill` (cuda backend) arguments a distill recipe supplies.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DistillRecipeArgs {
    pub student: PathBuf,
    pub teacher: PathBuf,
    /// One `.bin` token shard, hashed against `data.sha256`.
    pub data: PathBuf,
    /// The held-out `.bin` shard (`eval.held_out`), hashed against
    /// `eval.sha256`; its KD loss is measured after training.
    pub held_out: PathBuf,
    pub temperature: f64,
    pub epochs: u32,
    pub batch_size: u32,
    pub learning_rate: f64,
    pub seed: u64,
    pub hash: String,
}

/// Parse and check a distill recipe file. Opens no model.
pub(crate) fn load_distill(path: &Path) -> Result<DistillRecipeArgs> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| refuse_field("<file>", format!("cannot read {}: {e}", path.display())))?;
    let recipe = Recipe::parse(&text).map_err(refuse)?;
    let dir = path.parent().unwrap_or_else(|| Path::new("."));

    if recipe.method.kind != MethodKind::Distill {
        return Err(refuse_field(
            "method.kind",
            "only distill recipes run through `apr distill`; use `apr finetune --recipe`"
                .to_string(),
        ));
    }
    // The distill student trains in full; an adapter rank/alpha would not be
    // applied, so a recipe naming one is refused rather than ignored.
    if recipe.method.rank.is_some() {
        return Err(refuse_field(
            "method.rank",
            "apr distill trains the full student; an adapter rank is not honored".to_string(),
        ));
    }
    if recipe.method.alpha.is_some() {
        return Err(refuse_field(
            "method.alpha",
            "apr distill trains the full student; an adapter alpha is not honored".to_string(),
        ));
    }

    let data = resolve(dir, &recipe.data.train);
    // The shard reader reads u32 LE `.bin` token shards; any other file would
    // hash fine and then fail (or be misread) at the first batch.
    if data.extension().is_none_or(|e| e != "bin") {
        return Err(refuse_field(
            "data.train",
            format!(
                "{} is not a .bin token shard (apr tokenize encode-corpus writes one)",
                data.display()
            ),
        ));
    }
    check_hash("data.sha256", &data, &recipe.data.sha256)?;
    // The held-out set is evaluated through the same shard reader.
    let held_out = resolve(dir, &recipe.eval.held_out);
    if held_out.extension().is_none_or(|e| e != "bin") {
        return Err(refuse_field(
            "eval.held_out",
            format!(
                "{} is not a .bin token shard (apr distill evaluates the held-out set as one)",
                held_out.display()
            ),
        ));
    }
    check_hash("eval.sha256", &held_out, &recipe.eval.sha256)?;
    // The distill pipeline measures held-out KD loss only.
    if recipe.eval.metric != EvalMetric::Loss {
        return Err(refuse_field(
            "eval.metric",
            "apr distill reports held-out KD loss; metric accuracy is not honored yet (use loss)"
                .to_string(),
        ));
    }

    Ok(DistillRecipeArgs {
        student: PathBuf::from(&recipe.base.model),
        // parse() guarantees a distill recipe names a teacher.
        teacher: PathBuf::from(recipe.method.teacher.clone().unwrap_or_default()),
        data,
        held_out,
        temperature: recipe
            .method
            .temperature
            .unwrap_or(DISTILL_DEFAULT_TEMPERATURE),
        epochs: recipe.training.epochs,
        batch_size: recipe.training.batch_size,
        learning_rate: recipe.training.learning_rate,
        seed: recipe.training.seed,
        hash: recipe.hash(),
    })
}

/// Distill window length (tokens per sample, before the +1 label shift) when
/// no override is set. A recipe run always uses this value.
pub(crate) const DISTILL_SEQ_LEN: usize = 256;

/// The environment override for the distill window length. Outside a recipe
/// it is honored; under `--recipe` it is refused, because the recipe hash is
/// the run's identity and an env var would change the run without changing it.
pub(crate) const DISTILL_SEQ_LEN_ENV: &str = "APR_DISTILL_SMOKE_SEQ_LEN";

/// Refuse a window-length override under `apr distill --recipe`. `env` is the
/// value of [`DISTILL_SEQ_LEN_ENV`], if set.
pub(crate) fn refuse_seq_len_override(env: Option<&std::ffi::OsStr>) -> Result<()> {
    match env {
        None => Ok(()),
        Some(v) => Err(refuse_field(
            "<env>",
            format!(
                "{DISTILL_SEQ_LEN_ENV}={} would change the run without changing the recipe hash; \
                 a recipe run uses seq_len {DISTILL_SEQ_LEN} (unset it)",
                v.to_string_lossy()
            ),
        )),
    }
}

/// The distill window length outside a recipe: [`DISTILL_SEQ_LEN`] when
/// [`DISTILL_SEQ_LEN_ENV`] is unset, else its value. A value that is not a
/// positive integer is refused, never read as the default.
#[cfg_attr(not(all(feature = "training", feature = "cuda")), allow(dead_code))]
pub(crate) fn distill_seq_len(env: Option<&std::ffi::OsStr>) -> Result<usize> {
    let Some(v) = env else {
        return Ok(DISTILL_SEQ_LEN);
    };
    let s = v.to_string_lossy();
    match s.trim().parse::<usize>() {
        Ok(n) if n > 0 => Ok(n),
        _ => Err(CliError::ValidationFailed(format!(
            "{DISTILL_SEQ_LEN_ENV}={s} is not a positive integer (unset it for {DISTILL_SEQ_LEN})"
        ))),
    }
}

/// Full batches in one pass over a `.bin` shard of `shard_bytes` bytes: the
/// reader cuts u32 tokens into `seq_len + 1` windows and groups `batch_size`
/// windows per batch. Evaluating exactly this many never wraps around.
#[cfg_attr(not(all(feature = "training", feature = "cuda")), allow(dead_code))]
pub(crate) fn held_out_batches(shard_bytes: u64, batch_size: usize, seq_len: usize) -> usize {
    let tokens = usize::try_from(shard_bytes / 4).unwrap_or(usize::MAX);
    tokens / (seq_len + 1) / batch_size.max(1)
}

#[cfg(test)]
#[path = "finetune_recipe_tests.rs"]
mod tests;
