//! Init command types

use clap::Parser;
use std::path::PathBuf;

/// Arguments for the init command
#[derive(Parser, Debug, Clone, PartialEq)]
pub struct InitArgs {
    /// Template to use for initialization
    #[arg(short, long, default_value = "minimal")]
    pub template: InitTemplate,

    /// Output path (stdout if not specified)
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Experiment name
    #[arg(long, default_value = "my-experiment")]
    pub name: String,

    /// Model source path or URI
    #[arg(long)]
    pub model: Option<String>,

    /// Base model HF repo ID (e.g., Qwen/Qwen2.5-Coder-0.5B)
    #[arg(long)]
    pub base: Option<String>,

    /// Training method (overrides --template)
    #[arg(long)]
    pub method: Option<TrainingMethod>,

    /// Data source path or URI
    #[arg(long)]
    pub data: Option<String>,
}

/// Training method for --method flag
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TrainingMethod {
    /// Full fine-tuning
    Full,
    /// LoRA fine-tuning
    Lora,
    /// QLoRA (quantized LoRA) fine-tuning
    Qlora,
}

impl std::str::FromStr for TrainingMethod {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "full" => Ok(TrainingMethod::Full),
            "lora" => Ok(TrainingMethod::Lora),
            "qlora" => Ok(TrainingMethod::Qlora),
            _ => Err(format!("Unknown method: {s}. Valid methods: full, lora, qlora")),
        }
    }
}

impl std::fmt::Display for TrainingMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrainingMethod::Full => write!(f, "full"),
            TrainingMethod::Lora => write!(f, "lora"),
            TrainingMethod::Qlora => write!(f, "qlora"),
        }
    }
}

/// Init template type
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum InitTemplate {
    /// Minimal manifest with required fields only
    #[default]
    Minimal,
    /// LoRA fine-tuning template
    Lora,
    /// QLoRA fine-tuning template
    Qlora,
    /// Full template with all sections
    Full,
}

impl std::str::FromStr for InitTemplate {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "minimal" | "min" => Ok(InitTemplate::Minimal),
            "lora" => Ok(InitTemplate::Lora),
            "qlora" => Ok(InitTemplate::Qlora),
            "full" | "complete" => Ok(InitTemplate::Full),
            _ => Err(format!("Unknown template: {s}. Valid templates: minimal, lora, qlora, full")),
        }
    }
}

impl std::fmt::Display for InitTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InitTemplate::Minimal => write!(f, "minimal"),
            InitTemplate::Lora => write!(f, "lora"),
            InitTemplate::Qlora => write!(f, "qlora"),
            InitTemplate::Full => write!(f, "full"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_template_from_str() {
        assert_eq!(
            "minimal".parse::<InitTemplate>().expect("parsing should succeed"),
            InitTemplate::Minimal
        );
        assert_eq!(
            "min".parse::<InitTemplate>().expect("parsing should succeed"),
            InitTemplate::Minimal
        );
        assert_eq!(
            "lora".parse::<InitTemplate>().expect("parsing should succeed"),
            InitTemplate::Lora
        );
        assert_eq!(
            "qlora".parse::<InitTemplate>().expect("parsing should succeed"),
            InitTemplate::Qlora
        );
        assert_eq!(
            "full".parse::<InitTemplate>().expect("parsing should succeed"),
            InitTemplate::Full
        );
        assert_eq!(
            "complete".parse::<InitTemplate>().expect("parsing should succeed"),
            InitTemplate::Full
        );
        assert!("invalid".parse::<InitTemplate>().is_err());
    }

    #[test]
    fn test_init_template_display() {
        assert_eq!(format!("{}", InitTemplate::Minimal), "minimal");
        assert_eq!(format!("{}", InitTemplate::Lora), "lora");
        assert_eq!(format!("{}", InitTemplate::Qlora), "qlora");
        assert_eq!(format!("{}", InitTemplate::Full), "full");
    }

    #[test]
    fn test_init_template_default() {
        assert_eq!(InitTemplate::default(), InitTemplate::Minimal);
    }

    #[test]
    fn test_training_method_from_str() {
        assert_eq!(
            "full".parse::<TrainingMethod>().expect("parsing should succeed"),
            TrainingMethod::Full
        );
        assert_eq!(
            "lora".parse::<TrainingMethod>().expect("parsing should succeed"),
            TrainingMethod::Lora
        );
        assert_eq!(
            "qlora".parse::<TrainingMethod>().expect("parsing should succeed"),
            TrainingMethod::Qlora
        );
        assert_eq!(
            "LORA".parse::<TrainingMethod>().expect("parsing should succeed"),
            TrainingMethod::Lora
        );
        assert!("invalid".parse::<TrainingMethod>().is_err());
    }

    #[test]
    fn test_training_method_display() {
        assert_eq!(format!("{}", TrainingMethod::Full), "full");
        assert_eq!(format!("{}", TrainingMethod::Lora), "lora");
        assert_eq!(format!("{}", TrainingMethod::Qlora), "qlora");
    }
}
