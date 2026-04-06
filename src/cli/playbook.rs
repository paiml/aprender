//! CLI subcommands for `batuta playbook`

use anyhow::Result;
use clap::Subcommand;
use std::path::PathBuf;

use crate::playbook;

#[derive(Subcommand, Debug)]
pub enum PlaybookCommand {
    /// Run a playbook pipeline
    Run {
        /// Path to the playbook YAML file
        playbook_path: PathBuf,

        /// Only run specific stages
        #[arg(long, value_delimiter = ',')]
        stages: Option<Vec<String>>,

        /// Force re-run (ignore cache)
        #[arg(long)]
        force: bool,

        /// Parameter overrides (key=value)
        #[arg(short = 'p', long = "param", value_parser = parse_param)]
        params: Vec<(String, String)>,
    },

    /// Validate a playbook (parse, check refs, detect cycles)
    Validate {
        /// Path to the playbook YAML file
        playbook_path: PathBuf,
    },

    /// Show pipeline execution status from lock file
    Status {
        /// Path to the playbook YAML file
        playbook_path: PathBuf,
    },

    /// Show lock file contents
    Lock {
        /// Path to the playbook YAML file
        playbook_path: PathBuf,
    },
}

fn parse_param(s: &str) -> Result<(String, String), String> {
    let pos = s.find('=').ok_or_else(|| format!("invalid param '{}': expected key=value", s))?;
    Ok((s[..pos].to_string(), s[pos + 1..].to_string()))
}

pub fn cmd_playbook(command: PlaybookCommand) -> Result<()> {
    match command {
        PlaybookCommand::Run { playbook_path, stages, force, params } => {
            let param_overrides: std::collections::HashMap<String, serde_yaml_ng::Value> =
                params.into_iter().map(|(k, v)| (k, serde_yaml_ng::Value::String(v))).collect();

            let config = playbook::RunConfig {
                playbook_path: playbook_path.clone(),
                stage_filter: stages,
                force,
                dry_run: false,
                param_overrides,
            };

            println!("Running playbook: {}", playbook_path.display());

            let rt = tokio::runtime::Runtime::new()?;
            let result = rt.block_on(playbook::run_playbook(&config))?;

            println!(
                "\nDone: {} run, {} cached, {} failed ({:.1}s)",
                result.stages_run,
                result.stages_cached,
                result.stages_failed,
                result.total_duration.as_secs_f64()
            );

            Ok(())
        }
        PlaybookCommand::Validate { playbook_path } => {
            println!("Validating: {}", playbook_path.display());

            let (pb, warnings) = playbook::validate_only(&playbook_path)?;

            println!("Playbook '{}' is valid", pb.name);
            println!("  Stages: {}", pb.stages.len());
            println!("  Params: {}", pb.params.len());

            if !warnings.is_empty() {
                println!("\nWarnings:");
                for w in &warnings {
                    println!("  - {}", w);
                }
            }

            Ok(())
        }
        PlaybookCommand::Status { playbook_path } => playbook::show_status(&playbook_path),
        PlaybookCommand::Lock { playbook_path } => {
            let lock = playbook::cache::load_lock_file(&playbook_path)?;
            match lock {
                Some(l) => {
                    let yaml = serde_yaml_ng::to_string(&l)?;
                    println!("{}", yaml);
                }
                None => {
                    println!("No lock file found for {}", playbook_path.display());
                }
            }
            Ok(())
        }
    }
}
