//! `apr registry` subcommand — operations over the local model registry.
//!
//! Contract: `contracts/crux-A-01-v1.yaml` — closes FALSIFY-CRUX-A-01-004
//! (`apr registry aliases --json` emits the full str→str alias map).
//! `apr registry lineage` walks a model's recorded ancestry (EXT-001 EXT-07,
//! `contracts/dogfood-model-lifecycle-v1.yaml` FALSIFY-EXT-006).

use crate::error::{CliError, Result};
use batuta_common::cli_roles::{DirPath, FreeText};
use clap::Subcommand;
use pacha::{Ancestry, Registry, RegistryConfig};
use std::path::Path;

#[derive(Subcommand, Clone, Debug)]
pub enum RegistryCommands {
    /// List short-name → canonical-URL aliases from configs/aliases.yaml.
    Aliases {
        /// Emit the alias map as a single JSON object on stdout.
        #[arg(long)]
        json: bool,
    },
    /// Every recorded ancestor of a model: runs, parents, bases, datasets.
    Lineage {
        /// A model id, a BLAKE3 content hash, a produced model's sha256, a
        /// lineage node id (`blake3:<hex>`, a run id), or a model file.
        target: FreeText,
        /// Emit the ancestry (root, nodes, edges) as one JSON object.
        #[arg(long)]
        json: bool,
        /// Pacha home to read (default `~/.pacha`).
        #[arg(long, value_name = "DIR")]
        pacha_home: Option<DirPath>,
    },
}

pub fn run(command: RegistryCommands) -> Result<()> {
    match command {
        RegistryCommands::Aliases { json } => aliases(json),
        RegistryCommands::Lineage {
            target,
            json,
            pacha_home,
        } => {
            let home = match pacha_home {
                Some(h) => h.into_path_buf(),
                None => dirs::home_dir().map(|h| h.join(".pacha")).ok_or_else(|| {
                    CliError::ValidationFailed("no home directory for ~/.pacha".into())
                })?,
            };
            print_ancestry(&lineage_in(&home, &target)?, json)
        }
    }
}

fn pacha_err(e: impl std::fmt::Display) -> CliError {
    CliError::ValidationFailed(format!("pacha: {e}"))
}

fn is_hex64(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

/// The ancestry of `target` in the pacha home `home`. A cycle is refused.
pub(crate) fn lineage_in(home: &Path, target: &str) -> Result<Ancestry> {
    if !RegistryConfig::new(home).db_path().exists() {
        return Err(CliError::ValidationFailed(format!(
            "no pacha registry at {}",
            home.display()
        )));
    }
    let registry = Registry::open(RegistryConfig::new(home)).map_err(pacha_err)?;
    let (node, known) = resolve(&registry, target)?;
    let ancestry = registry.ancestry(&node).map_err(pacha_err)?;
    if !known && ancestry.edges.is_empty() {
        return Err(CliError::ValidationFailed(format!(
            "{target}: not a registered model and no recorded lineage"
        )));
    }
    Ok(ancestry)
}

/// The registered model `target` names, if any (EXT-11 `apr model pack`).
pub(crate) fn resolve_model(registry: &Registry, target: &str) -> Result<Option<String>> {
    Ok(match resolve(registry, target)? {
        (id, true) => Some(id),
        (_, false) => None,
    })
}

/// The lineage node `target` names, and whether it is a registered model.
fn resolve(registry: &Registry, target: &str) -> Result<(String, bool)> {
    if let Ok(id) = target.parse::<pacha::model::ModelId>() {
        if registry.get_model_by_id(&id).is_ok() {
            return Ok((id.to_string(), true));
        }
    }
    let path = Path::new(target);
    let Some(hash) = target_hash(target, path)? else {
        return Ok((target.to_string(), false));
    };
    let bare_hex = !path.is_file() && !target.starts_with("blake3:");
    match model_by_hash(registry, &hash, bare_hex)? {
        Some(id) => Ok((id, true)),
        None => Ok((format!("blake3:{hash}"), false)),
    }
}

/// The BLAKE3 hash `target` names: a model file's content hash, or a 64-hex
/// string with or without its `blake3:` prefix, lowercased.
fn target_hash(target: &str, path: &Path) -> Result<Option<String>> {
    if path.is_file() {
        let mut hasher = blake3::Hasher::new();
        hasher.update_reader(std::fs::File::open(path)?)?;
        return Ok(Some(hasher.finalize().to_hex().to_string()));
    }
    Ok(target
        .strip_prefix("blake3:")
        .or(Some(target))
        .filter(|h| is_hex64(h))
        .map(str::to_ascii_lowercase))
}

/// The registered model `hash` names: its content hash first, then, for a
/// `bare_hex` target, a recorded sha256.
fn model_by_hash(registry: &Registry, hash: &str, bare_hex: bool) -> Result<Option<String>> {
    if let Some(id) = registry
        .find_model_id_by_content_hash(hash)
        .map_err(pacha_err)?
    {
        return Ok(Some(id));
    }
    // A bare 64-hex that is not a content hash may be a recorded sha256.
    if bare_hex {
        return registry
            .find_model_id_by_card_sha256(hash)
            .map_err(pacha_err);
    }
    Ok(None)
}

/// Which run, datasets and bases produced a model — the direct producer only
/// (EXT-31, FALSIFY-CRUX-P-04-001). `produced_by: None` is an orphan.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Provenance {
    pub produced_by: Option<String>,
    pub datasets: Vec<String>,
    pub bases: Vec<String>,
}

impl Provenance {
    pub(crate) fn is_orphan(&self) -> bool {
        self.produced_by.is_none()
    }

    pub(crate) fn summary(&self, root: &str) -> String {
        match &self.produced_by {
            None => format!("{root}: orphan — no run recorded as producing it (I-1)"),
            Some(run) => format!(
                "{root}: produced by {run}; dataset(s) [{}]; base(s) [{}]",
                self.datasets.join(", "),
                self.bases.join(", ")
            ),
        }
    }
}

pub(crate) fn provenance(ancestry: &Ancestry) -> Provenance {
    let Some(run) = ancestry
        .edges
        .iter()
        .find(|e| e.to_id == ancestry.root && e.edge_type == "produced")
        .map(|e| e.from_id.clone())
    else {
        return Provenance::default();
    };
    let into_run = |kind: &str| -> Vec<String> {
        ancestry
            .edges
            .iter()
            .filter(|e| e.to_id == run && e.edge_type == kind)
            .map(|e| e.from_id.clone())
            .collect()
    };
    Provenance {
        datasets: into_run("dataset"),
        bases: into_run("base"),
        produced_by: Some(run),
    }
}

fn print_ancestry(ancestry: &Ancestry, json: bool) -> Result<()> {
    let p = provenance(ancestry);
    if json {
        let mut value = serde_json::to_value(ancestry)
            .map_err(|e| CliError::ValidationFailed(format!("ancestry json: {e}")))?;
        value["provenance"] = serde_json::json!({
            "produced_by": p.produced_by,
            "datasets": p.datasets,
            "bases": p.bases,
            "orphan": p.is_orphan(),
        });
        println!("{value}");
    } else {
        println!("{}", p.summary(&ancestry.root));
        println!(
            "{}: {} ancestor(s), {} edge(s)",
            ancestry.root,
            ancestry.nodes.len() - 1,
            ancestry.edges.len()
        );
        for e in &ancestry.edges {
            println!("  {} --{}--> {}", e.from_id, e.edge_type, e.to_id);
        }
    }
    Ok(())
}

fn aliases(json: bool) -> Result<()> {
    use super::aliases;
    let map = aliases::alias_map();
    if json {
        let value = serde_json::to_string(map)
            .expect("CRUX-A-01 FALSIFY-004: BTreeMap<String, String> always serializes");
        println!("{value}");
    } else {
        for (name, url) in map {
            println!("{name}\t{url}");
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "registry_tests.rs"]
mod tests;
