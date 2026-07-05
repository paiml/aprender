//! Example: Rosetta Testing (PMAT-ROSETTA-002/003)
//!
//! This example demonstrates the metamorphic relation and multi-hop
//! conversion testing types added by the Rosetta-Testing spec.
//!
//! Run with: `cargo run --example rosetta_testing -p apr-qa-runner`

#![allow(clippy::missing_panics_doc, clippy::expect_used)]

use aprender_qa_gen::{Backend, Format, ModelId};
use aprender_qa_runner::{
    CommutativityTest, ConversionConfig, IdempotencyTest, InspectResult, RoundTripTest,
};

/// Demonstrate Rosetta metamorphic and multi-hop conversion testing types
fn main() {
    let model = ModelId::new("Qwen", "Qwen2.5-Coder-0.5B-Instruct");

    // =========================================================================
    // InspectResult: parse `apr rosetta inspect --json` output
    // =========================================================================
    println!("=== InspectResult (T-GH192-01) ===\n");

    let json = r#"{
        "tensor_count": 338,
        "tensor_names": ["model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight"],
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "hidden_size": 896,
        "architecture": "Qwen2ForCausalLM"
    }"#;
    let inspect: InspectResult = serde_json::from_str(json).expect("valid JSON test data");

    println!("Tensor count:          {}", inspect.tensor_count);
    println!("Tensor names (first):  {:?}", &inspect.tensor_names[..2]);
    println!("Attention heads:       {:?}", inspect.num_attention_heads);
    println!("KV heads:              {:?}", inspect.num_key_value_heads);
    println!("Hidden size:           {:?}", inspect.hidden_size);
    println!("Architecture:          {:?}", inspect.architecture);
    println!();

    // =========================================================================
    // Multi-hop round-trip chains (T-QKV-03, T-QKV-04)
    // =========================================================================
    println!("=== Multi-Hop Round-Trip Chains ===\n");

    // T-QKV-03: ST → APR → GGUF → ST (F-CONV-RT-002)
    let rt_002 = RoundTripTest::new(
        vec![
            Format::SafeTensors,
            Format::Apr,
            Format::Gguf,
            Format::SafeTensors,
        ],
        Backend::Cpu,
        model.clone(),
    );
    println!(
        "F-CONV-RT-002: {:?} ({} hops)",
        rt_002.formats,
        rt_002.formats.len() - 1
    );

    // T-QKV-04: ST → APR → GGUF → APR → ST (F-CONV-RT-003)
    let rt_003 = RoundTripTest::new(
        vec![
            Format::SafeTensors,
            Format::Apr,
            Format::Gguf,
            Format::Apr,
            Format::SafeTensors,
        ],
        Backend::Cpu,
        model.clone(),
    );
    println!(
        "F-CONV-RT-003: {:?} ({} hops)",
        rt_003.formats,
        rt_003.formats.len() - 1
    );
    println!();

    // =========================================================================
    // Idempotency test (MR-IDEM)
    // =========================================================================
    println!("=== Idempotency Test (MR-IDEM) ===\n");

    let idem = IdempotencyTest::new(Format::Gguf, Format::Apr, Backend::Cpu, model.clone());
    println!(
        "F-CONV-IDEM-001: convert {:?} → {:?} twice, compare outputs",
        idem.format_a, idem.format_b
    );
    println!();

    // =========================================================================
    // Commutativity test (MR-COM)
    // =========================================================================
    println!("=== Commutativity Test (MR-COM) ===\n");

    let com = CommutativityTest::new(Backend::Cpu, model);
    println!(
        "F-CONV-COM-001: GGUF→APR vs GGUF→ST→APR (backend: {:?})",
        com.backend
    );
    println!();

    // =========================================================================
    // ConversionConfig with new fields
    // =========================================================================
    println!("=== ConversionConfig (new fields) ===\n");

    let config = ConversionConfig::default();
    println!("test_multi_hop:     {}", config.test_multi_hop);
    println!("test_cardinality:   {}", config.test_cardinality);
    println!("test_tensor_names:  {}", config.test_tensor_names);
    println!("test_idempotency:   {}", config.test_idempotency);
    println!("test_commutativity: {}", config.test_commutativity);
    println!();

    // =========================================================================
    // Gate ID summary
    // =========================================================================
    println!("=== New Gate IDs (PMAT-ROSETTA-002/003) ===\n");
    println!("| Gate ID            | Source   | Description                          |");
    println!("|--------------------+----------+--------------------------------------|");
    println!("| F-CONV-CARD-001    | MR-CARD  | Silent tensor loss (QKV fusion)      |");
    println!("| F-CONV-NAME-001    | T-QKV-02 | Unexpected tensor renaming           |");
    println!("| F-CONV-RT-002      | T-QKV-03 | ST→APR→GGUF→ST round-trip failure    |");
    println!("| F-CONV-RT-003      | T-QKV-04 | Multi-hop chain failure              |");
    println!("| F-CONV-IDEM-001    | MR-IDEM  | Double-convert instability           |");
    println!("| F-CONV-COM-001     | MR-COM   | Path-dependent conversion bugs       |");
    println!("| F-INSPECT-META-001 | T-GH192  | Missing/wrong model metadata         |");
}
