//! Example: Generating Model Certification Certificate
//!
//! This example demonstrates how to generate CERTIFICATE.md files
//! using the Popperian falsification methodology.
//!
//! Run with: `cargo run --example generate_certificate -p apr-qa-report`

#![allow(clippy::missing_panics_doc)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::unwrap_used)]

use aprender_qa_gen::{Backend, Format, Modality, ModelId, QaScenario};
use aprender_qa_report::popperian::PopperianCalculator;
use aprender_qa_report::{CertificateGenerator, MqsCalculator};
use aprender_qa_runner::{Evidence, EvidenceCollector};

/// Generate a sample model certification certificate and print it
fn main() {
    let model_id = "Qwen/Qwen2.5-Coder-0.5B-Instruct";
    let version = "1.0.0";
    let model = ModelId::new("Qwen", "Qwen2.5-Coder-0.5B-Instruct");

    // Create sample evidence from test runs
    let mut collector = EvidenceCollector::new();

    // Gateway tests (G1-G4) - all must pass for certification
    add_gateway_evidence(&mut collector, &model);

    // Verification Matrix tests (F-INT, F-API, F-NUM, F-SEC)
    add_verification_evidence(&mut collector, &model);

    // Calculate MQS score
    let mqs_calc = MqsCalculator::new();
    let mqs = mqs_calc.calculate(model_id, &collector).unwrap();

    // Calculate Popperian score
    let popperian_calc = PopperianCalculator::new();
    let popperian = popperian_calc.calculate(model_id, &collector);

    // Generate certificate
    let generator = CertificateGenerator::new("APR QA Framework v0.1.0");
    let evidence_hash = "sha256:abc123def456..."; // In reality, hash of evidence.json

    let certificate = generator.generate(model_id, version, &mqs, &popperian, evidence_hash);

    // Print certificate details
    println!("Certificate Generated");
    println!("=====================");
    println!();
    println!("Model:    {}", certificate.model_id);
    println!("Version:  {}", certificate.version);
    println!("Status:   {}", certificate.status);
    println!("Grade:    {}", certificate.grade);
    println!("MQS:      {}/1000", certificate.mqs_score);
    println!(
        "Score:    {}/{} ({:.1}%)",
        certificate.verification_score,
        certificate.max_score,
        (certificate.verification_score as f64 / certificate.max_score as f64) * 100.0
    );
    println!("Black Swans: {}", certificate.black_swans_caught);
    println!();

    // Generate markdown
    let markdown = generator.to_markdown(&certificate);
    println!("CERTIFICATE.md Preview:");
    println!("========================");
    println!();

    // Print first 40 lines of markdown
    for (i, line) in markdown.lines().enumerate() {
        if i >= 40 {
            println!("... (truncated)");
            break;
        }
        println!("{line}");
    }
}

/// Add gateway evidence (G1-G4) to the evidence collector
fn add_gateway_evidence(collector: &mut EvidenceCollector, model: &ModelId) {
    // G1: Model loads
    collector.add(Evidence::corroborated(
        "G1-LOAD",
        create_scenario(model, "Loading model..."),
        "Model loaded successfully",
        1200,
    ));

    // G2: Basic inference
    collector.add(Evidence::corroborated(
        "G2-INFER",
        create_scenario(model, "def hello():"),
        "Generated valid Python code",
        150,
    ));

    // G3: No crashes
    collector.add(Evidence::corroborated(
        "G3-STABLE",
        create_scenario(model, "Test stability"),
        "Process completed cleanly",
        100,
    ));

    // G4: Output not garbage
    collector.add(Evidence::corroborated(
        "G4-VALID",
        create_scenario(model, "# Calculate factorial"),
        "Output is syntactically valid code",
        80,
    ));
}

/// Add verification matrix evidence (F-INT, F-API, F-NUM, F-SEC, F-PAR, F-PERF)
fn add_verification_evidence(collector: &mut EvidenceCollector, model: &ModelId) {
    // F-INT-001..005: Fundamental Integrity
    for i in 1..=5 {
        collector.add(Evidence::corroborated(
            format!("F-INT-00{i}"),
            create_scenario(model, &format!("Integrity test {i}")),
            "Integrity check passed",
            50 + i as u64 * 10,
        ));
    }

    // F-API-001..005: Interface Compliance
    for i in 1..=5 {
        collector.add(Evidence::corroborated(
            format!("F-API-00{i}"),
            create_scenario(model, &format!("API test {i}")),
            "API compliance verified",
            30 + i as u64 * 5,
        ));
    }

    // F-NUM-001..004: Numerical Stability
    for i in 1..=4 {
        collector.add(Evidence::corroborated(
            format!("F-NUM-00{i}"),
            create_scenario(model, &format!("Numerical test {i}")),
            "Numerical stability verified",
            40 + i as u64 * 5,
        ));
    }

    // F-SEC-001..003: Security & Safety
    for i in 1..=3 {
        collector.add(Evidence::corroborated(
            format!("F-SEC-00{i}"),
            create_scenario(model, &format!("Security test {i}")),
            "Security check passed",
            60 + i as u64 * 10,
        ));
    }

    // F-PAR-001..003: Cross-Platform Parity
    for i in 1..=3 {
        collector.add(Evidence::corroborated(
            format!("F-PAR-00{i}"),
            create_scenario(model, &format!("Parity test {i}")),
            "Platform parity verified",
            80 + i as u64 * 5,
        ));
    }

    // F-PERF-001..004: Performance Boundaries
    for i in 1..=4 {
        collector.add(Evidence::corroborated(
            format!("F-PERF-00{i}"),
            create_scenario(model, &format!("Performance test {i}")),
            "Performance threshold met",
            100 + i as u64 * 10,
        ));
    }
}

/// Create a QA scenario with default CPU/GGUF/Run settings
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
