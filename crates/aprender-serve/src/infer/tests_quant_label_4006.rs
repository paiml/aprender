//! #4006: `quant=` names the transformer body, not the tied lm_head.

use super::{
    body_qtypes, body_quant_label, model_body_qtypes, qtype_to_dtype_str,
    safetensors_dtype_ggml_id, safetensors_quant_label,
};

const Q5_K: u32 = 13;
const Q4_K: u32 = 12;
const Q6_K: u32 = 14;
const IQ2_XXS: u32 = 16;
const IQ3_XXS: u32 = 18;

/// A model under `$HOME/models` (host-local; these tests SKIP when it is absent).
fn home_model(name: &str) -> String {
    format!(
        "{}/models/{name}",
        std::env::var("HOME").unwrap_or_default()
    )
}

/// The ticket's census shape: a Q5_K tied head over an IQ2_XXS-dominated,
/// mixed body. The old label was the head's type.
#[test]
fn a_mixed_ud_body_is_not_labelled_by_its_head() {
    let mut body = vec![IQ2_XXS; 95];
    body.extend(vec![IQ3_XXS; 24]);
    body.extend(vec![Q4_K; 18]);
    let label = body_quant_label(&body, Q5_K);
    assert_ne!(label, "Q5_K", "labelled by the head again");
    assert!(
        label.starts_with("mixed(IQ2_XXS×95,"),
        "dominant first: {label}"
    );
    assert!(
        label.contains("IQ3_XXS×24") && label.contains("Q4_K×18"),
        "{label}"
    );
    assert!(
        label.ends_with("lm_head=Q5_K"),
        "the head is still reported: {label}"
    );
    // The old label, for the record: this is what receipts quoted.
    assert_eq!(qtype_to_dtype_str(Q5_K), "Q5_K");
}

/// A uniform body prints its one type, with the head only when it differs —
/// so an ordinary Q4_K_M file keeps a short label.
#[test]
fn a_uniform_body_prints_one_type() {
    assert_eq!(body_quant_label(&[Q4_K; 7], Q4_K), "Q4_K");
    assert_eq!(body_quant_label(&[Q4_K; 7], Q6_K), "Q4_K lm_head=Q6_K");
    // No body tensors at all: fall back to the head rather than print nothing.
    assert_eq!(body_quant_label(&[], Q6_K), "Q6_K");
    // An id the table does not know is shown as a number, never guessed.
    assert_eq!(body_quant_label(&[9999], 9999), "ggml type 9999");
}

/// The real file the ticket measured (`apr run` printed `quant=Q5_K`). Host-local:
/// it SKIPs loudly when the file is absent, so it is a receipt, not a CI gate —
/// the census test above is the CI gate.
#[test]
fn the_real_qwen35_ud_iq2_xxs_header_is_labelled_iq2_xxs() {
    let path = home_model("Qwen3.5-0.8B-UD-IQ2_XXS.gguf");
    let Ok(mapped) = crate::gguf::MappedGGUFModel::from_path(&path) else {
        eprintln!("SKIP (not a pass): {path} is absent");
        return;
    };
    // Header only: this qwen35 hybrid does not load through the dense model. The
    // head is the tied token_embd (there is no output.weight).
    let head = |n: &str| {
        mapped
            .model
            .tensors
            .iter()
            .find(|t| t.name == n)
            .map(|t| t.qtype)
    };
    let lm_head = head("output.weight")
        .or_else(|| head("token_embd.weight"))
        .expect("a head tensor");
    let old = qtype_to_dtype_str(lm_head);
    let new = body_quant_label(&body_qtypes(&mapped.model), lm_head);
    eprintln!("RECEIPT #4006: old quant={old}  new quant={new}");
    assert_eq!(
        old, "Q5_K",
        "precondition: the head is the tied Q5_K token_embd"
    );
    assert!(new.starts_with("mixed(IQ2_XXS×"), "{new}");
    assert!(new.contains("lm_head=Q5_K"), "{new}");
}

/// SafeTensors dtypes name the same way GGUF qtypes do; integers are not weights.
#[test]
fn safetensors_dtypes_map_onto_ggml_names() {
    use crate::safetensors::SafetensorsDtype as D;
    let label = |d: D| body_quant_label(&[safetensors_dtype_ggml_id(&d).expect("float")], 0);
    assert_eq!(label(D::BF16), "BF16 lm_head=F32");
    assert_eq!(label(D::F16), "F16 lm_head=F32");
    assert_eq!(label(D::F32), "F32");
    assert_eq!(safetensors_dtype_ggml_id(&D::I32), None);
}

/// The #4006 follow-up must-RED: `apr run x.apr --no-gpu -v` printed the
/// hardcoded `quant=Q4_K (OwnedQuantizedModel CPU)` for a BF16 `.apr`. The label is
/// now the loaded layers' census. Host-local (SKIPs loudly when absent); the Q4_K
/// file is the positive control, so "never Q4_K" cannot pass by printing nothing.
#[test]
fn a_bf16_apr_is_not_labelled_q4k_and_a_q4k_apr_is() {
    for (path, want, must_not) in [
        (
            home_model("qwen2.5-coder-0.5b-instruct.apr"),
            "BF16",
            Some("Q4_K"),
        ),
        (
            "/mnt/nvme-raid0/models/qwen2.5-coder-0.5b-instruct-q4k.apr".to_string(),
            "Q4_K",
            None,
        ),
    ] {
        let Ok(mapped) = crate::apr::MappedAprModel::from_path(&path) else {
            eprintln!("SKIP (not a pass): {path} is absent");
            continue;
        };
        let model = crate::gguf::OwnedQuantizedModel::from_apr(&mapped).expect("load apr");
        let label = body_quant_label(&model_body_qtypes(&model), model.lm_head_weight.qtype);
        eprintln!("RECEIPT #4006 apr: {path} -> quant={label}");
        assert!(label.starts_with(want), "{path}: {label}");
        if let Some(bad) = must_not {
            assert!(!label.contains(bad), "{path}: {label}");
        }
    }
}

/// The SafeTensors CUDA path printed `quant=F16/BF16` — an either/or guess. Host-local.
#[test]
fn a_safetensors_label_is_read_from_its_header() {
    let path = home_model("qwen2.5-coder-1.5b-instruct.safetensors");
    let path = std::path::Path::new(&path);
    if !path.exists() {
        eprintln!("SKIP (not a pass): {} is absent", path.display());
        return;
    }
    let label = safetensors_quant_label(path);
    eprintln!("RECEIPT #4006 safetensors: quant={label}");
    assert!(
        !label.contains('/'),
        "an either/or is not a measurement: {label}"
    );
    assert!(!label.starts_with("unknown"), "{label}");
}
