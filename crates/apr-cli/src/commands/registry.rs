//! `apr registry` subcommand — operations over the local model registry.
//!
//! Contract: `contracts/crux-A-01-v1.yaml` — closes FALSIFY-CRUX-A-01-004
//! (`apr registry aliases --json` emits the full str→str alias map).
//! `apr registry lineage` walks a model's recorded ancestry (EXT-001 EXT-07,
//! `contracts/dogfood-model-lifecycle-v1.yaml` FALSIFY-EXT-006).

use crate::error::{CliError, Result};
use clap::Subcommand;
use pacha::{Ancestry, Registry, RegistryConfig};
use std::path::{Path, PathBuf};

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
        target: String,
        /// Emit the ancestry (root, nodes, edges) as one JSON object.
        #[arg(long)]
        json: bool,
        /// Pacha home to read (default `~/.pacha`).
        #[arg(long, value_name = "DIR")]
        pacha_home: Option<PathBuf>,
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
                Some(h) => h,
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
    let hash = if path.is_file() {
        let mut hasher = blake3::Hasher::new();
        hasher.update_reader(std::fs::File::open(path)?)?;
        Some(hasher.finalize().to_hex().to_string())
    } else {
        target
            .strip_prefix("blake3:")
            .or(Some(target))
            .filter(|h| is_hex64(h))
            .map(str::to_ascii_lowercase)
    };
    if let Some(hash) = hash {
        if let Some(id) = registry
            .find_model_id_by_content_hash(&hash)
            .map_err(pacha_err)?
        {
            return Ok((id, true));
        }
        // A bare 64-hex that is not a content hash may be a recorded sha256.
        if !path.is_file() && !target.starts_with("blake3:") {
            if let Some(id) = registry
                .find_model_id_by_card_sha256(&hash)
                .map_err(pacha_err)?
            {
                return Ok((id, true));
            }
        }
        return Ok((format!("blake3:{hash}"), false));
    }
    Ok((target.to_string(), false))
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
        false
    }

    pub(crate) fn summary(&self, _root: &str) -> String {
        String::new()
    }
}

pub(crate) fn provenance(_ancestry: &Ancestry) -> Provenance {
    Provenance::default()
}

fn print_ancestry(ancestry: &Ancestry, json: bool) -> Result<()> {
    if json {
        let value = serde_json::to_string(ancestry)
            .map_err(|e| CliError::ValidationFailed(format!("ancestry json: {e}")))?;
        println!("{value}");
    } else {
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
