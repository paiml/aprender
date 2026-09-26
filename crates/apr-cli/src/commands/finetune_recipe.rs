//! `apr finetune --recipe FILE` (E8 #4002 row R-1b, `contracts/apr-recipe-v1.yaml`).
//!
//! The recipe is parsed, and the data and eval files are hashed against it,
//! before any model is opened. Every refusal names the recipe field, so a
//! recipe defect reads differently from a CLI flag error.

use std::io::Read;
use std::path::{Path, PathBuf};

use entrenar::recipe::{MethodKind, Recipe, RecipeError};
use sha2::{Digest, Sha256};

use crate::error::{CliError, Result};

/// The finetune arguments a recipe supplies.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RecipeArgs {
    pub model: PathBuf,
    pub method: String,
    pub rank: Option<u32>,
    pub data: PathBuf,
    pub epochs: u32,
    pub learning_rate: f64,
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

    let data = resolve(dir, &recipe.data.train);
    check_hash("data.sha256", &data, &recipe.data.sha256)?;
    let held_out = resolve(dir, &recipe.eval.held_out);
    check_hash("eval.sha256", &held_out, &recipe.eval.sha256)?;

    Ok(RecipeArgs {
        model: PathBuf::from(&recipe.base.model),
        method: method.to_string(),
        rank: recipe.method.rank,
        data,
        epochs: recipe.training.epochs,
        learning_rate: recipe.training.learning_rate,
        hash: recipe.hash(),
    })
}

#[cfg(test)]
#[path = "finetune_recipe_tests.rs"]
mod tests;
