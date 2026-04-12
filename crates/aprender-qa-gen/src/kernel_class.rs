//! Kernel Equivalence Classes
//!
//! Maps model families to kernel equivalence classes. Models sharing the same
//! kernel class use identical compute kernels (attention, normalization, activation,
//! positional encoding). Once kernels are proven via full MVP on one representative
//! model, other models in the class only need dimensional smoke verification.

use serde::{Deserialize, Serialize};
use std::str::FromStr;

/// Kernel equivalence class identifier.
///
/// Each class groups model families that share identical kernel pipelines.
/// Class A covers ~70% of all models (GQA+RMSNorm+SiLU+SwiGLU+RoPE).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KernelClass {
    /// GQA + RMSNorm + SiLU + SwiGLU + RoPE (LLaMA 3, Qwen2, Mistral, Yi, DeepSeek, InternLM2, Falcon-H1, Phi-3/4, StableLM)
    A,
    /// MHA + LayerNorm + GELU (GPT-2, GPT-Neo, GPT-NeoX, GPT-J, OPT, GPT-BigCode, CodeGen, XGLM, Falcon 7B, Phi-2)
    B,
    /// MQA + LayerNorm + GELU + ALiBi (BLOOM, Falcon-40B)
    C,
    /// Mixed LayerNorm: gegelu/SiLU + LayerNorm + GQA (Phi-3-small, Moonshine)
    D,
    /// MoE + GQA + RMSNorm + SwiGLU (Mixtral, Qwen-MoE)
    E,
    /// RMSNorm + GELU + GatedMlp + RoPE (Gemma, Gemma2, CodeGemma)
    F,
    /// State Space Model: selective scan, no attention (Mamba)
    Ssm,
    /// Linear Attention: WKV recurrence, no softmax (RWKV7)
    Linear,
}

impl KernelClass {
    /// Map a model family string to its kernel equivalence class.
    ///
    /// Returns `None` for unknown families.
    #[must_use]
    pub fn from_family(family: &str) -> Option<Self> {
        match family.to_lowercase().as_str() {
            // Class A: GQA + RMSNorm + SiLU + SwiGLU + RoPE
            // Phi-3/4 and StableLM moved here from D — they use SiLU+RMSNorm+GQA+RoPE (identical kernel pipeline to LLaMA)
            "llama" | "llama3" | "llama-3" | "llama3.2" | "codellama" | "tinyllama" | "qwen"
            | "qwen2" | "qwen2.5" | "qwen3" | "qwen-coder" | "mistral" | "yi" | "deepseek"
            | "deepseek-v2" | "deepseek-coder" | "deepseek-r1" | "internlm2" | "internlm"
            | "smollm" | "smollm2" | "olmo" | "olmo2" | "granite" | "granite-code"
            | "starcoder2" | "nemotron" | "falcon-h1" | "falcon_h1" | "deepseek2"
            | "deepseek-coder-v2" | "openelm" | "qwen3_5" | "qwen3.5" | "phi-3" | "phi3"
            | "phi4" | "stablelm" | "stable-lm" | "vicuna" | "solar" | "baichuan" => Some(Self::A),

            // Class B: MHA + LayerNorm + GELU
            // phi/phi-2 (Phi-2) stays here — MHA+LayerNorm+GELU, not SiLU
            "gpt2" | "gpt-2" | "gpt-neo" | "gpt_neo" | "gpt-neox" | "gptneox" | "gpt-j"
            | "gptj" | "opt" | "galactica" | "gpt-bigcode" | "gpt_bigcode" | "falcon-7b"
            | "falcon7b" | "codegen" | "xglm" | "starcoder" | "starcoder1" | "bert" | "roberta"
            | "deberta" | "electra" | "distilbert" | "whisper" | "phi" | "phi2" | "phi-2" => {
                Some(Self::B)
            }

            // Class C: MQA + LayerNorm + GELU + ALiBi
            // NOTE: bare "falcon" removed — ambiguous (7B=Class B, 40B=Class C)
            "bloom" | "bloomz" | "falcon-40b" | "falcon40b" => Some(Self::C),

            // Class D: mixed LayerNorm variants (gegelu, SiLU+LayerNorm)
            "phi3small" | "phi-3-small" | "moonshine" => Some(Self::D),

            // Class E: MoE + GQA + RMSNorm + SwiGLU
            "mixtral" | "qwen-moe" | "qwenmoe" | "qwen3_moe" => Some(Self::E),

            // Class F: RMSNorm + GELU + GatedMlp + RoPE
            "gemma" | "gemma2" | "gemma3" | "codegemma" => Some(Self::F),

            // SSM: State Space Models (selective scan, no attention)
            "mamba" | "mamba2" => Some(Self::Ssm),

            // Linear: Linear attention (WKV recurrence, no softmax)
            "rwkv" | "rwkv7" | "rwkv6" | "rwkv5" => Some(Self::Linear),

            _ => None,
        }
    }

    /// Get the representative model for this kernel class.
    ///
    /// This is the model that should have full MVP certification to prove
    /// kernel correctness for the entire class.
    #[must_use]
    pub const fn representative_model(&self) -> &'static str {
        match self {
            Self::A => "Qwen/Qwen2.5-Coder-0.5B-Instruct",
            Self::B => "EleutherAI/gpt-neox-20b",
            Self::C => "tiiuae/falcon-40b",
            Self::D => "microsoft/Phi-3-small-8k-instruct",
            Self::E => "mistralai/Mixtral-8x7B-Instruct-v0.1",
            Self::F => "google/gemma-2-2b-it",
            Self::Ssm => "state-spaces/mamba-130m",
            Self::Linear => "RWKV/rwkv-7-world-0.1b",
        }
    }

    /// Human-readable label for this kernel class.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::A => "GQA+RMSNorm+SiLU+SwiGLU+RoPE",
            Self::B => "MHA+LayerNorm+GELU",
            Self::C => "MQA+LayerNorm+GELU+ALiBi",
            Self::D => "Mixed+LayerNorm+GELU/SiLU",
            Self::E => "MoE+GQA+RMSNorm+SwiGLU",
            Self::F => "RMSNorm+GELU+GatedMlp+RoPE",
            Self::Ssm => "SSM+RMSNorm+SiLU",
            Self::Linear => "WKV+LayerNorm+GELU",
        }
    }

    /// Return all kernel class variants.
    #[must_use]
    pub const fn all() -> &'static [Self] {
        &[
            Self::A,
            Self::B,
            Self::C,
            Self::D,
            Self::E,
            Self::F,
            Self::Ssm,
            Self::Linear,
        ]
    }
}

impl FromStr for KernelClass {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "A" => Ok(Self::A),
            "B" => Ok(Self::B),
            "C" => Ok(Self::C),
            "D" => Ok(Self::D),
            "E" => Ok(Self::E),
            "F" => Ok(Self::F),
            "SSM" => Ok(Self::Ssm),
            "LINEAR" => Ok(Self::Linear),
            _ => Err(format!(
                "Unknown kernel class: {s}. Use: A, B, C, D, E, F, SSM, Linear"
            )),
        }
    }
}

impl std::fmt::Display for KernelClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::A => write!(f, "A"),
            Self::B => write!(f, "B"),
            Self::C => write!(f, "C"),
            Self::D => write!(f, "D"),
            Self::E => write!(f, "E"),
            Self::F => write!(f, "F"),
            Self::Ssm => write!(f, "SSM"),
            Self::Linear => write!(f, "Linear"),
        }
    }
}

/// Filter certification records by kernel class.
///
/// Returns model IDs from the certifications list whose family maps to the given class.
#[must_use]
pub fn models_in_class(class: KernelClass, families_and_ids: &[(String, String)]) -> Vec<String> {
    families_and_ids
        .iter()
        .filter(|(family, _)| KernelClass::from_family(family) == Some(class))
        .map(|(_, model_id)| model_id.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_family_class_a() {
        assert_eq!(KernelClass::from_family("llama"), Some(KernelClass::A));
        assert_eq!(KernelClass::from_family("qwen"), Some(KernelClass::A));
        assert_eq!(KernelClass::from_family("qwen-coder"), Some(KernelClass::A));
        assert_eq!(KernelClass::from_family("qwen3"), Some(KernelClass::A));
        assert_eq!(KernelClass::from_family("mistral"), Some(KernelClass::A));
        assert_eq!(KernelClass::from_family("yi"), Some(KernelClass::A));
        assert_eq!(KernelClass::from_family("deepseek"), Some(KernelClass::A));
        assert_eq!(KernelClass::from_family("internlm2"), Some(KernelClass::A));
        assert_eq!(KernelClass::from_family("codellama"), Some(KernelClass::A));
        assert_eq!(KernelClass::from_family("tinyllama"), Some(KernelClass::A));
        assert_eq!(
            KernelClass::from_family("deepseek-coder"),
            Some(KernelClass::A)
        );
        assert_eq!(
            KernelClass::from_family("deepseek-r1"),
            Some(KernelClass::A)
        );
        assert_eq!(KernelClass::from_family("smollm"), Some(KernelClass::A));
        assert_eq!(KernelClass::from_family("smollm2"), Some(KernelClass::A));
        assert_eq!(KernelClass::from_family("olmo"), Some(KernelClass::A));
        assert_eq!(KernelClass::from_family("olmo2"), Some(KernelClass::A));
        assert_eq!(KernelClass::from_family("internlm"), Some(KernelClass::A));
        assert_eq!(KernelClass::from_family("granite"), Some(KernelClass::A));
        assert_eq!(
            KernelClass::from_family("granite-code"),
            Some(KernelClass::A)
        );
        assert_eq!(KernelClass::from_family("starcoder2"), Some(KernelClass::A));
        assert_eq!(KernelClass::from_family("nemotron"), Some(KernelClass::A));
        assert_eq!(KernelClass::from_family("falcon-h1"), Some(KernelClass::A));
        assert_eq!(KernelClass::from_family("falcon_h1"), Some(KernelClass::A));
        // Phi-3/4 and StableLM: SiLU+RMSNorm+GQA+RoPE (identical to LLaMA)
        assert_eq!(KernelClass::from_family("phi-3"), Some(KernelClass::A));
        assert_eq!(KernelClass::from_family("phi3"), Some(KernelClass::A));
        assert_eq!(KernelClass::from_family("phi4"), Some(KernelClass::A));
        assert_eq!(KernelClass::from_family("stablelm"), Some(KernelClass::A));
        assert_eq!(KernelClass::from_family("stable-lm"), Some(KernelClass::A));
        assert_eq!(KernelClass::from_family("vicuna"), Some(KernelClass::A));
        assert_eq!(KernelClass::from_family("solar"), Some(KernelClass::A));
        assert_eq!(KernelClass::from_family("baichuan"), Some(KernelClass::A));
    }

    #[test]
    fn test_from_family_class_b() {
        assert_eq!(KernelClass::from_family("gpt2"), Some(KernelClass::B));
        assert_eq!(KernelClass::from_family("gpt-2"), Some(KernelClass::B));
        assert_eq!(KernelClass::from_family("gpt-neo"), Some(KernelClass::B));
        assert_eq!(KernelClass::from_family("gpt_neo"), Some(KernelClass::B));
        assert_eq!(KernelClass::from_family("gpt-neox"), Some(KernelClass::B));
        assert_eq!(KernelClass::from_family("gpt-j"), Some(KernelClass::B));
        assert_eq!(KernelClass::from_family("opt"), Some(KernelClass::B));
        assert_eq!(KernelClass::from_family("galactica"), Some(KernelClass::B));
        assert_eq!(
            KernelClass::from_family("gpt-bigcode"),
            Some(KernelClass::B)
        );
        assert_eq!(
            KernelClass::from_family("gpt_bigcode"),
            Some(KernelClass::B)
        );
        assert_eq!(KernelClass::from_family("falcon-7b"), Some(KernelClass::B));
        assert_eq!(KernelClass::from_family("codegen"), Some(KernelClass::B));
        assert_eq!(KernelClass::from_family("xglm"), Some(KernelClass::B));
        // Phi-2: MHA+LayerNorm+GELU (NOT SiLU like Phi-3/4)
        assert_eq!(KernelClass::from_family("phi"), Some(KernelClass::B));
        assert_eq!(KernelClass::from_family("phi2"), Some(KernelClass::B));
        assert_eq!(KernelClass::from_family("phi-2"), Some(KernelClass::B));
    }

    #[test]
    fn test_from_family_class_c() {
        assert_eq!(KernelClass::from_family("bloom"), Some(KernelClass::C));
        assert_eq!(KernelClass::from_family("bloomz"), Some(KernelClass::C));
        assert_eq!(KernelClass::from_family("falcon-40b"), Some(KernelClass::C));
        // Bare "falcon" is ambiguous (7B=B, 40B=C) — returns None
        assert_eq!(KernelClass::from_family("falcon"), None);
    }

    #[test]
    fn test_from_family_class_d() {
        // Class D now only contains mixed LayerNorm variants
        assert_eq!(KernelClass::from_family("phi3small"), Some(KernelClass::D));
        assert_eq!(
            KernelClass::from_family("phi-3-small"),
            Some(KernelClass::D)
        );
        assert_eq!(KernelClass::from_family("moonshine"), Some(KernelClass::D));
    }

    #[test]
    fn test_from_family_class_ssm() {
        assert_eq!(KernelClass::from_family("mamba"), Some(KernelClass::Ssm));
        assert_eq!(KernelClass::from_family("mamba2"), Some(KernelClass::Ssm));
    }

    #[test]
    fn test_from_family_class_linear() {
        assert_eq!(KernelClass::from_family("rwkv7"), Some(KernelClass::Linear));
        assert_eq!(KernelClass::from_family("rwkv6"), Some(KernelClass::Linear));
        assert_eq!(KernelClass::from_family("rwkv"), Some(KernelClass::Linear));
    }

    #[test]
    fn test_from_family_class_e() {
        assert_eq!(KernelClass::from_family("mixtral"), Some(KernelClass::E));
        assert_eq!(KernelClass::from_family("qwen-moe"), Some(KernelClass::E));
    }

    #[test]
    fn test_from_family_class_f() {
        assert_eq!(KernelClass::from_family("gemma"), Some(KernelClass::F));
        assert_eq!(KernelClass::from_family("gemma2"), Some(KernelClass::F));
        assert_eq!(KernelClass::from_family("gemma3"), Some(KernelClass::F));
        assert_eq!(KernelClass::from_family("codegemma"), Some(KernelClass::F));
    }

    /// Bare "falcon" must NOT resolve to any class — it's ambiguous.
    /// Falcon-7B = Class B (MHA), Falcon-40B = Class C (MQA+ALiBi).
    #[test]
    fn test_bare_falcon_is_ambiguous() {
        assert_eq!(KernelClass::from_family("falcon"), None);
        assert_eq!(KernelClass::from_family("falcon-7b"), Some(KernelClass::B));
        assert_eq!(KernelClass::from_family("falcon-40b"), Some(KernelClass::C));
    }

    #[test]
    fn test_from_family_unknown() {
        assert_eq!(KernelClass::from_family("unknown-model"), None);
        assert_eq!(KernelClass::from_family(""), None);
    }

    #[test]
    fn test_from_family_case_insensitive() {
        assert_eq!(KernelClass::from_family("LLAMA"), Some(KernelClass::A));
        assert_eq!(KernelClass::from_family("Qwen"), Some(KernelClass::A));
        assert_eq!(KernelClass::from_family("Mixtral"), Some(KernelClass::E));
    }

    #[test]
    fn test_representative_models() {
        assert!(KernelClass::A.representative_model().contains("Qwen"));
        assert!(KernelClass::B.representative_model().contains("neox"));
        assert!(KernelClass::C.representative_model().contains("falcon"));
        assert!(KernelClass::D.representative_model().contains("Phi"));
        assert!(KernelClass::E.representative_model().contains("Mixtral"));
        assert!(KernelClass::Ssm.representative_model().contains("mamba"));
        assert!(KernelClass::Linear.representative_model().contains("rwkv"));
    }

    #[test]
    fn test_labels() {
        assert!(KernelClass::A.label().contains("GQA"));
        assert!(KernelClass::B.label().contains("MHA"));
        assert!(KernelClass::C.label().contains("MQA"));
        assert!(KernelClass::D.label().contains("LayerNorm"));
        assert!(KernelClass::E.label().contains("MoE"));
        assert!(KernelClass::F.label().contains("GatedMlp"));
        assert!(KernelClass::Ssm.label().contains("SSM"));
        assert!(KernelClass::Linear.label().contains("WKV"));
    }

    #[test]
    fn test_from_str() {
        assert_eq!("A".parse::<KernelClass>().unwrap(), KernelClass::A);
        assert_eq!("b".parse::<KernelClass>().unwrap(), KernelClass::B);
        assert_eq!("C".parse::<KernelClass>().unwrap(), KernelClass::C);
        assert_eq!("d".parse::<KernelClass>().unwrap(), KernelClass::D);
        assert_eq!("E".parse::<KernelClass>().unwrap(), KernelClass::E);
        assert_eq!("F".parse::<KernelClass>().unwrap(), KernelClass::F);
        assert_eq!("SSM".parse::<KernelClass>().unwrap(), KernelClass::Ssm);
        assert_eq!(
            "LINEAR".parse::<KernelClass>().unwrap(),
            KernelClass::Linear
        );
        assert!("X".parse::<KernelClass>().is_err());
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", KernelClass::A), "A");
        assert_eq!(format!("{}", KernelClass::E), "E");
        assert_eq!(format!("{}", KernelClass::Ssm), "SSM");
        assert_eq!(format!("{}", KernelClass::Linear), "Linear");
    }

    #[test]
    fn test_models_in_class() {
        let data = vec![
            ("qwen".to_string(), "Qwen/Qwen2.5-0.5B".to_string()),
            (
                "qwen-coder".to_string(),
                "Qwen/Qwen2.5-Coder-1.5B".to_string(),
            ),
            ("llama".to_string(), "meta-llama/Llama-3-8B".to_string()),
            ("phi".to_string(), "microsoft/phi-2".to_string()),
            ("mixtral".to_string(), "mistralai/Mixtral-8x7B".to_string()),
            ("mamba".to_string(), "state-spaces/mamba-130m".to_string()),
        ];

        let class_a = models_in_class(KernelClass::A, &data);
        assert_eq!(class_a.len(), 3);
        assert!(class_a.contains(&"Qwen/Qwen2.5-0.5B".to_string()));
        assert!(class_a.contains(&"Qwen/Qwen2.5-Coder-1.5B".to_string()));
        assert!(class_a.contains(&"meta-llama/Llama-3-8B".to_string()));

        // phi (Phi-2) is now Class B
        let class_b = models_in_class(KernelClass::B, &data);
        assert_eq!(class_b.len(), 1);
        assert!(class_b.contains(&"microsoft/phi-2".to_string()));

        let class_e = models_in_class(KernelClass::E, &data);
        assert_eq!(class_e.len(), 1);

        let class_ssm = models_in_class(KernelClass::Ssm, &data);
        assert_eq!(class_ssm.len(), 1);
        assert!(class_ssm.contains(&"state-spaces/mamba-130m".to_string()));
    }

    #[test]
    fn test_all_classes() {
        let all = KernelClass::all();
        assert_eq!(all.len(), 8);
        assert_eq!(all[0], KernelClass::A);
        assert_eq!(all[5], KernelClass::F);
        assert_eq!(all[6], KernelClass::Ssm);
        assert_eq!(all[7], KernelClass::Linear);
    }

    #[test]
    fn test_serialize_deserialize() {
        let class = KernelClass::A;
        let json = serde_json::to_string(&class).unwrap();
        assert_eq!(json, "\"A\"");
        let deserialized: KernelClass = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, KernelClass::A);

        // SSM and Linear round-trip
        let ssm = serde_json::to_string(&KernelClass::Ssm).unwrap();
        assert_eq!(ssm, "\"Ssm\"");
        let linear = serde_json::to_string(&KernelClass::Linear).unwrap();
        assert_eq!(linear, "\"Linear\"");
    }
}
