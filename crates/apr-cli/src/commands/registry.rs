//! `apr registry` subcommand — operations over the local model registry.
//!
//! Contract: `contracts/crux-A-01-v1.yaml` — closes FALSIFY-CRUX-A-01-004
//! (`apr registry aliases --json` emits the full str→str alias map).

use crate::error::Result;
use clap::Subcommand;

#[derive(Subcommand, Clone, Debug)]
pub enum RegistryCommands {
    /// List short-name → canonical-URL aliases from configs/aliases.yaml.
    Aliases {
        /// Emit the alias map as a single JSON object on stdout.
        #[arg(long)]
        json: bool,
    },
}

pub fn run(command: RegistryCommands) -> Result<()> {
    match command {
        RegistryCommands::Aliases { json } => aliases(json),
    }
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
