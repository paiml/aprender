//! Quantization and merge command types

use clap::Parser;
use std::path::PathBuf;

/// Arguments for the quantize command
#[derive(Parser, Debug, Clone, PartialEq)]
pub struct QuantizeArgs {
    /// Path to model file
    #[arg(value_name = "MODEL")]
    pub model: PathBuf,

    /// Output path for quantized model
    #[arg(short, long)]
    pub output: PathBuf,

    /// Quantization bits (4 or 8)
    #[arg(short, long, default_value = "4")]
    pub bits: u8,

    /// Quantization method (symmetric or asymmetric)
    #[arg(short, long, default_value = "symmetric")]
    pub method: QuantMethod,

    /// Use per-channel quantization
    #[arg(long)]
    pub per_channel: bool,

    /// Path to calibration data
    #[arg(long)]
    pub calibration_data: Option<PathBuf>,

    /// Output as SafeTensors with I8 dtype + scale tensors (for WASM deployment)
    #[arg(long)]
    pub safetensors: bool,
}

/// Arguments for the merge command
#[derive(Parser, Debug, Clone, PartialEq)]
pub struct MergeArgs {
    /// Paths to models to merge (for ties/dare/slerp/average)
    #[arg(value_name = "MODELS", num_args = 0..)]
    pub models: Vec<PathBuf>,

    /// Output path for merged model
    #[arg(short, long)]
    pub output: PathBuf,

    /// Merge method (ties, dare, slerp, average, lora-adapter)
    #[arg(short, long, default_value = "ties")]
    pub method: MergeMethod,

    /// Interpolation weight (for slerp)
    #[arg(short, long)]
    pub weight: Option<f32>,

    /// Density threshold (for ties/dare)
    #[arg(short, long)]
    pub density: Option<f32>,

    /// Model weights (comma-separated, for weighted average)
    #[arg(long)]
    pub weights: Option<String>,

    /// Base model path (for lora-adapter merge, ENT-LoRA-017)
    #[arg(long)]
    pub base: Option<PathBuf>,

    /// LoRA adapter directory (for lora-adapter merge, ENT-LoRA-017)
    #[arg(long)]
    pub adapter: Option<PathBuf>,
}

/// Quantization method
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum QuantMethod {
    #[default]
    Symmetric,
    Asymmetric,
}

impl std::str::FromStr for QuantMethod {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "symmetric" | "sym" => Ok(QuantMethod::Symmetric),
            "asymmetric" | "asym" => Ok(QuantMethod::Asymmetric),
            _ => Err(format!(
                "Unknown quantization method: {s}. Valid methods: symmetric, asymmetric"
            )),
        }
    }
}

/// Merge method
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum MergeMethod {
    #[default]
    Ties,
    Dare,
    Slerp,
    Average,
    /// Merge LoRA adapter into base model (ENT-LoRA-017)
    LoraAdapter,
}

impl std::str::FromStr for MergeMethod {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "ties" => Ok(MergeMethod::Ties),
            "dare" => Ok(MergeMethod::Dare),
            "slerp" => Ok(MergeMethod::Slerp),
            "average" | "avg" => Ok(MergeMethod::Average),
            "lora-adapter" | "lora" => Ok(MergeMethod::LoraAdapter),
            _ => Err(format!(
                "Unknown merge method: {s}. Valid: ties, dare, slerp, average, lora-adapter"
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quant_method_from_str() {
        assert_eq!(
            "symmetric".parse::<QuantMethod>().expect("parsing should succeed"),
            QuantMethod::Symmetric
        );
        assert_eq!(
            "sym".parse::<QuantMethod>().expect("parsing should succeed"),
            QuantMethod::Symmetric
        );
        assert_eq!(
            "asymmetric".parse::<QuantMethod>().expect("parsing should succeed"),
            QuantMethod::Asymmetric
        );
        assert_eq!(
            "asym".parse::<QuantMethod>().expect("parsing should succeed"),
            QuantMethod::Asymmetric
        );
        assert!("invalid".parse::<QuantMethod>().is_err());
    }

    #[test]
    fn test_merge_method_from_str() {
        assert_eq!(
            "ties".parse::<MergeMethod>().expect("parsing should succeed"),
            MergeMethod::Ties
        );
        assert_eq!(
            "dare".parse::<MergeMethod>().expect("parsing should succeed"),
            MergeMethod::Dare
        );
        assert_eq!(
            "slerp".parse::<MergeMethod>().expect("parsing should succeed"),
            MergeMethod::Slerp
        );
        assert_eq!(
            "average".parse::<MergeMethod>().expect("parsing should succeed"),
            MergeMethod::Average
        );
        assert_eq!(
            "avg".parse::<MergeMethod>().expect("parsing should succeed"),
            MergeMethod::Average
        );
        assert_eq!(
            "lora-adapter".parse::<MergeMethod>().expect("parsing should succeed"),
            MergeMethod::LoraAdapter
        );
        assert_eq!(
            "lora".parse::<MergeMethod>().expect("parsing should succeed"),
            MergeMethod::LoraAdapter
        );
        assert!("invalid".parse::<MergeMethod>().is_err());
    }

    #[test]
    fn test_quant_method_default() {
        assert_eq!(QuantMethod::default(), QuantMethod::Symmetric);
    }

    #[test]
    fn test_merge_method_default() {
        assert_eq!(MergeMethod::default(), MergeMethod::Ties);
    }
}
