//! Example: Generating RAG-Optimized Markdown Reports
//!
//! This example demonstrates how to generate markdown reports optimized for
//! batuta's RAG (Retrieval-Augmented Generation) oracle indexing.
//!
//! Run with: `cargo run --example generate_rag_markdown -p apr-qa-report`

#![allow(clippy::missing_panics_doc)]

use aprender_qa_gen::{Backend, Format, Modality, ModelId, QaScenario};
use aprender_qa_report::mqs::{CategoryScores, GatewayResult, MqsScore, Penalty};
use aprender_qa_report::popperian::{FalsificationDetail, PopperianScore};
use aprender_qa_report::{generate_evidence_detail, generate_index_entry, generate_rag_markdown};
use aprender_qa_runner::{Evidence, EvidenceCollector};

fn main() {
    // Create sample MQS score
    let mqs = create_sample_mqs();

    // Create sample Popperian score
    let popperian = create_sample_popperian();

    // Create sample evidence
    let collector = create_sample_evidence();

    // Generate full RAG-optimized markdown report
    println!("=== Full RAG Markdown Report ===\n");
    let full_report = generate_rag_markdown(&mqs, &popperian, &collector);
    println!("{full_report}");

    // Generate compact index entry
    println!("\n=== Index Entry (for summary tables) ===\n");
    println!("| Model | Score | Grade | Status | Prod Ready |");
    println!("|-------|-------|-------|--------|------------|");
    print!("{}", generate_index_entry(&mqs));

    // Generate individual evidence detail
    println!("\n=== Individual Evidence Detail ===\n");
    for evidence in collector.all().iter().take(3) {
        println!("{}", generate_evidence_detail(evidence));
    }
}

fn create_sample_mqs() -> MqsScore {
    MqsScore {
        model_id: "qwen/Qwen2.5-Coder-7B-Instruct".to_string(),
        raw_score: 892,
        normalized_score: 89.2,
        grade: "A-".to_string(),
        gateways: vec![
            GatewayResult::passed("G1", "Model loads successfully"),
            GatewayResult::passed("G2", "Basic inference works"),
            GatewayResult::passed("G3", "No crashes during testing"),
            GatewayResult::passed("G4", "Output is coherent text"),
        ],
        gateways_passed: true,
        categories: CategoryScores {
            qual: 175,
            perf: 140,
            stab: 180,
            comp: 147,
            edge: 130,
            regr: 120,
        },
        total_tests: 150,
        tests_passed: 142,
        tests_failed: 8,
        penalties: vec![Penalty {
            code: "TIMEOUT".to_string(),
            description: "2 tests exceeded time limit".to_string(),
            points: 15,
        }],
        total_penalty: 15,
        proof_bonus: None,
    }
}

fn create_sample_popperian() -> PopperianScore {
    PopperianScore {
        model_id: "qwen/Qwen2.5-Coder-7B-Instruct".to_string(),
        hypotheses_tested: 150,
        corroborated: 142,
        falsified: 8,
        inconclusive: 0,
        corroboration_ratio: 0.947,
        severity_weighted_score: 0.92,
        confidence_level: 0.98,
        reproducibility_index: 0.99,
        black_swan_count: 0,
        falsifications: vec![
            FalsificationDetail {
                gate_id: "F-PERF-023".to_string(),
                hypothesis: "Inference completes under 500ms for short prompts".to_string(),
                evidence: "Measured: 623ms (exceeded by 24.6%)".to_string(),
                severity: 2,
                is_black_swan: false,
                occurrence_count: 1,
            },
            FalsificationDetail {
                gate_id: "F-EDGE-017".to_string(),
                hypothesis: "Model handles empty input gracefully".to_string(),
                evidence: "Returned error instead of empty response".to_string(),
                severity: 3,
                is_black_swan: false,
                occurrence_count: 2,
            },
        ],
    }
}

fn create_sample_evidence() -> EvidenceCollector {
    let mut collector = EvidenceCollector::new();
    let model = ModelId::new("qwen", "Qwen2.5-Coder-7B-Instruct");

    // Add gateway evidence
    collector.add(Evidence::corroborated(
        "G1-LOAD",
        create_scenario(&model, "Loading model..."),
        "Model loaded in 3.2s",
        3200,
    ));

    collector.add(Evidence::corroborated(
        "G2-INFER",
        create_scenario(&model, "Hello, world!"),
        "Hello! I'm ready to help.",
        120,
    ));

    // Add functional evidence
    collector.add(Evidence::corroborated(
        "F-QUAL-001",
        create_scenario(&model, "Write a hello world in Rust"),
        "fn main() { println!(\"Hello, world!\"); }",
        250,
    ));

    collector.add(Evidence::falsified(
        "F-PERF-023",
        create_scenario(&model, "Complex code generation"),
        "Response time exceeded threshold",
        "Generated code but took 623ms",
        623,
    ));

    collector.add(Evidence::falsified(
        "F-EDGE-017",
        create_scenario(&model, ""),
        "Empty input not handled gracefully",
        "Error: empty input",
        50,
    ));

    collector
}

fn create_scenario(model: &ModelId, prompt: &str) -> QaScenario {
    QaScenario::new(
        model.clone(),
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        prompt.to_string(),
        42,
    )
}
