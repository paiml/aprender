use super::*;

fn sample_provenance() -> Provenance {
    Provenance {
        source: SourceProvenance {
            format: "safetensors".to_string(),
            path: "model.safetensors".to_string(),
            sha256: "a1b2c3d4e5f6".to_string(),
            hf_repo: "Qwen/Qwen2.5-Coder-0.5B-Instruct".to_string(),
            downloaded_at: "2026-02-01T12:00:00Z".to_string(),
        },
        derived: vec![
            DerivedProvenance {
                format: "gguf".to_string(),
                path: "model.gguf".to_string(),
                sha256: "f6e5d4c3b2a1".to_string(),
                converter: "apr-cli".to_string(),
                converter_version: "0.2.12".to_string(),
                quantization: None,
                created_at: "2026-02-01T12:05:00Z".to_string(),
            },
            DerivedProvenance {
                format: "apr".to_string(),
                path: "model.apr".to_string(),
                sha256: "1a2b3c4d5e6f".to_string(),
                converter: "apr-cli".to_string(),
                converter_version: "0.2.12".to_string(),
                quantization: None,
                created_at: "2026-02-01T12:06:00Z".to_string(),
            },
        ],
    }
}

// PMAT-PROV-001: Reject certification with mismatched sources

include!("f_prov_1.rs");
include!("f_prov_2.rs");
