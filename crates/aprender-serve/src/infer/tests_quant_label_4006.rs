//! #4006: `quant=` names the transformer body, not the tied lm_head.

use super::{body_qtypes, body_quant_label, qtype_to_dtype_str};

const Q5_K: u32 = 13;
const Q4_K: u32 = 12;
const Q6_K: u32 = 14;
const IQ2_XXS: u32 = 16;
const IQ3_XXS: u32 = 18;

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
    let path = "/home/noah/models/Qwen3.5-0.8B-UD-IQ2_XXS.gguf";
    let Ok(mapped) = crate::gguf::MappedGGUFModel::from_path(path) else {
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
