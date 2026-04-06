//! Tests for experiment runs, storage, research artifacts, and sovereign distribution.

use crate::experiment::*;

// -------------------------------------------------------------------------
// SovereignDistribution Tests
// -------------------------------------------------------------------------

#[test]
fn test_sovereign_distribution_creation() {
    let dist = SovereignDistribution::new("my-model", "1.0.0");
    assert_eq!(dist.name, "my-model");
    assert_eq!(dist.version, "1.0.0");
    assert!(dist.platforms.is_empty());
    assert!(dist.artifacts.is_empty());
}

#[test]
fn test_sovereign_distribution_add_platform() {
    let mut dist = SovereignDistribution::new("my-model", "1.0.0");
    dist.add_platform("linux-x86_64");
    dist.add_platform("darwin-aarch64");

    assert_eq!(dist.platforms.len(), 2);
    assert!(dist.platforms.contains(&"linux-x86_64".to_string()));
}

#[test]
fn test_sovereign_distribution_add_artifact() {
    let mut dist = SovereignDistribution::new("my-model", "1.0.0");
    dist.add_artifact(SovereignArtifact {
        name: "model.onnx".to_string(),
        artifact_type: ArtifactType::Model,
        sha256: "abc123".to_string(),
        size_bytes: 1_000_000,
        source_url: None,
    });

    assert_eq!(dist.artifacts.len(), 1);
    assert_eq!(dist.total_size_bytes(), 1_000_000);
}

#[test]
fn test_sovereign_distribution_validate_signatures_missing() {
    let mut dist = SovereignDistribution::new("my-model", "1.0.0");
    dist.add_artifact(SovereignArtifact {
        name: "model.onnx".to_string(),
        artifact_type: ArtifactType::Model,
        sha256: "abc123".to_string(),
        size_bytes: 1_000_000,
        source_url: None,
    });

    let result = dist.validate_signatures();
    assert!(result.is_err());
    match result {
        Err(ExperimentError::SovereignValidationFailed(msg)) => {
            assert!(msg.contains("model.onnx"));
        }
        _ => panic!("Expected SovereignValidationFailed error"),
    }
}

#[test]
fn test_sovereign_distribution_validate_signatures_present() {
    let mut dist = SovereignDistribution::new("my-model", "1.0.0");
    dist.add_artifact(SovereignArtifact {
        name: "model.onnx".to_string(),
        artifact_type: ArtifactType::Model,
        sha256: "abc123".to_string(),
        size_bytes: 1_000_000,
        source_url: None,
    });
    dist.signatures.push(ArtifactSignature {
        artifact_name: "model.onnx".to_string(),
        algorithm: SignatureAlgorithm::Ed25519,
        signature: "sig123".to_string(),
        key_id: "key1".to_string(),
    });

    let result = dist.validate_signatures();
    assert!(result.is_ok());
}

// -------------------------------------------------------------------------
// ORCID Tests
// -------------------------------------------------------------------------

#[test]
fn test_orcid_valid() {
    let orcid = Orcid::new("0000-0002-1825-0097");
    assert!(orcid.is_ok());
    assert_eq!(orcid.unwrap().as_str(), "0000-0002-1825-0097");
}

#[test]
fn test_orcid_valid_with_x() {
    let orcid = Orcid::new("0000-0002-1825-009X");
    assert!(orcid.is_ok());
}

#[test]
fn test_orcid_invalid_format() {
    let orcid = Orcid::new("invalid-orcid");
    assert!(orcid.is_err());
    match orcid {
        Err(ExperimentError::InvalidOrcid(s)) => {
            assert_eq!(s, "invalid-orcid");
        }
        _ => panic!("Expected InvalidOrcid error"),
    }
}

#[test]
fn test_orcid_invalid_too_short() {
    let orcid = Orcid::new("0000-0002-1825");
    assert!(orcid.is_err());
}

// -------------------------------------------------------------------------
// CitationMetadata Tests
// -------------------------------------------------------------------------

#[test]
fn test_citation_metadata_to_bibtex() {
    let citation = CitationMetadata {
        citation_type: CitationType::Article,
        title: "Test Paper".to_string(),
        authors: vec!["Alice Smith".to_string(), "Bob Jones".to_string()],
        year: 2024,
        month: Some(6),
        doi: Some("10.1234/test".to_string()),
        url: None,
        venue: Some("Journal of Testing".to_string()),
        volume: Some("42".to_string()),
        pages: Some("1-10".to_string()),
        publisher: None,
        version: None,
    };

    let bibtex = citation.to_bibtex("smith2024test");
    assert!(bibtex.contains("@article{smith2024test,"));
    assert!(bibtex.contains("title = {Test Paper}"));
    assert!(bibtex.contains("author = {Alice Smith and Bob Jones}"));
    assert!(bibtex.contains("year = {2024}"));
    assert!(bibtex.contains("doi = {10.1234/test}"));
    assert!(bibtex.contains("journal = {Journal of Testing}"));
}

#[test]
fn test_citation_metadata_to_cff() {
    let citation = CitationMetadata {
        citation_type: CitationType::Software,
        title: "My Tool".to_string(),
        authors: vec!["Developer".to_string()],
        year: 2024,
        month: Some(11),
        doi: Some("10.5281/zenodo.123".to_string()),
        url: Some("https://github.com/example/tool".to_string()),
        venue: None,
        volume: None,
        pages: None,
        publisher: None,
        version: Some("1.0.0".to_string()),
    };

    let cff = citation.to_cff();
    assert!(cff.contains("cff-version: 1.2.0"));
    assert!(cff.contains("title: \"My Tool\""));
    assert!(cff.contains("name: \"Developer\""));
    assert!(cff.contains("version: \"1.0.0\""));
    assert!(cff.contains("doi: \"10.5281/zenodo.123\""));
}

// -------------------------------------------------------------------------
// ExperimentRun Tests
// -------------------------------------------------------------------------

#[test]
fn test_experiment_run_creation() {
    let device =
        ComputeDevice::Cpu { cores: 8, threads_per_core: 2, architecture: CpuArchitecture::X86_64 };

    let run = ExperimentRun::new("run-001", "my-experiment", ModelParadigm::DeepLearning, device);

    assert_eq!(run.run_id, "run-001");
    assert_eq!(run.experiment_name, "my-experiment");
    assert_eq!(run.paradigm, ModelParadigm::DeepLearning);
    assert_eq!(run.status, RunStatus::Running);
    assert!(run.ended_at.is_none());
}

#[test]
fn test_experiment_run_log_metric() {
    let device =
        ComputeDevice::Cpu { cores: 8, threads_per_core: 2, architecture: CpuArchitecture::X86_64 };

    let mut run =
        ExperimentRun::new("run-001", "my-experiment", ModelParadigm::DeepLearning, device);
    run.log_metric("accuracy", 0.95);
    run.log_metric("loss", 0.05);

    assert_eq!(run.metrics.get("accuracy"), Some(&0.95));
    assert_eq!(run.metrics.get("loss"), Some(&0.05));
}

#[test]
fn test_experiment_run_log_param() {
    let device =
        ComputeDevice::Cpu { cores: 8, threads_per_core: 2, architecture: CpuArchitecture::X86_64 };

    let mut run =
        ExperimentRun::new("run-001", "my-experiment", ModelParadigm::DeepLearning, device);
    run.log_param("learning_rate", serde_json::json!(0.001));
    run.log_param("batch_size", serde_json::json!(32));

    assert_eq!(run.hyperparameters.get("learning_rate"), Some(&serde_json::json!(0.001)));
}

#[test]
fn test_experiment_run_complete() {
    let device =
        ComputeDevice::Cpu { cores: 8, threads_per_core: 2, architecture: CpuArchitecture::X86_64 };

    let mut run =
        ExperimentRun::new("run-001", "my-experiment", ModelParadigm::DeepLearning, device);
    run.complete();

    assert_eq!(run.status, RunStatus::Completed);
    assert!(run.ended_at.is_some());
}

#[test]
fn test_experiment_run_fail() {
    let device =
        ComputeDevice::Cpu { cores: 8, threads_per_core: 2, architecture: CpuArchitecture::X86_64 };

    let mut run =
        ExperimentRun::new("run-001", "my-experiment", ModelParadigm::DeepLearning, device);
    run.fail();

    assert_eq!(run.status, RunStatus::Failed);
    assert!(run.ended_at.is_some());
}

#[test]
fn test_experiment_run_serialization() {
    let device =
        ComputeDevice::Cpu { cores: 8, threads_per_core: 2, architecture: CpuArchitecture::X86_64 };

    let mut run =
        ExperimentRun::new("run-001", "my-experiment", ModelParadigm::DeepLearning, device);
    run.log_metric("accuracy", 0.95);

    let json = serde_json::to_string(&run).unwrap();
    let deserialized: ExperimentRun = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.run_id, "run-001");
    assert_eq!(deserialized.metrics.get("accuracy"), Some(&0.95));
}

// -------------------------------------------------------------------------
// ExperimentStorage Tests
// -------------------------------------------------------------------------

#[test]
fn test_in_memory_storage_store_and_get() {
    let storage = InMemoryExperimentStorage::new();
    let device =
        ComputeDevice::Cpu { cores: 8, threads_per_core: 2, architecture: CpuArchitecture::X86_64 };

    let run = ExperimentRun::new("run-001", "my-experiment", ModelParadigm::DeepLearning, device);
    storage.store_run(&run).unwrap();

    let retrieved = storage.get_run("run-001").unwrap();
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().run_id, "run-001");
}

#[test]
fn test_in_memory_storage_get_nonexistent() {
    let storage = InMemoryExperimentStorage::new();
    let retrieved = storage.get_run("nonexistent").unwrap();
    assert!(retrieved.is_none());
}

#[test]
fn test_in_memory_storage_list_runs() {
    let storage = InMemoryExperimentStorage::new();
    let device =
        ComputeDevice::Cpu { cores: 8, threads_per_core: 2, architecture: CpuArchitecture::X86_64 };

    let run1 = ExperimentRun::new("run-001", "exp-a", ModelParadigm::DeepLearning, device.clone());
    let run2 = ExperimentRun::new("run-002", "exp-a", ModelParadigm::FineTuning, device.clone());
    let run3 = ExperimentRun::new("run-003", "exp-b", ModelParadigm::TraditionalML, device);

    storage.store_run(&run1).unwrap();
    storage.store_run(&run2).unwrap();
    storage.store_run(&run3).unwrap();

    let runs = storage.list_runs("exp-a").unwrap();
    assert_eq!(runs.len(), 2);
}

#[test]
fn test_in_memory_storage_delete_run() {
    let storage = InMemoryExperimentStorage::new();
    let device =
        ComputeDevice::Cpu { cores: 8, threads_per_core: 2, architecture: CpuArchitecture::X86_64 };

    let run = ExperimentRun::new("run-001", "my-experiment", ModelParadigm::DeepLearning, device);
    storage.store_run(&run).unwrap();

    storage.delete_run("run-001").unwrap();
    let retrieved = storage.get_run("run-001").unwrap();
    assert!(retrieved.is_none());
}

// -------------------------------------------------------------------------
// CRediT Role Tests
// -------------------------------------------------------------------------

#[test]
fn test_credit_role_variants() {
    let roles = vec![
        CreditRole::Conceptualization,
        CreditRole::DataCuration,
        CreditRole::FormalAnalysis,
        CreditRole::FundingAcquisition,
        CreditRole::Investigation,
        CreditRole::Methodology,
        CreditRole::ProjectAdministration,
        CreditRole::Resources,
        CreditRole::Software,
        CreditRole::Supervision,
        CreditRole::Validation,
        CreditRole::Visualization,
        CreditRole::WritingOriginalDraft,
        CreditRole::WritingReviewEditing,
    ];
    assert_eq!(roles.len(), 14);
}

#[test]
fn test_research_contributor_creation() {
    let contributor = ResearchContributor {
        name: "Alice Researcher".to_string(),
        orcid: Orcid::new("0000-0002-1825-0097").ok(),
        affiliation: "MIT".to_string(),
        roles: vec![CreditRole::Conceptualization, CreditRole::Software],
        email: Some("alice@mit.edu".to_string()),
    };

    assert_eq!(contributor.name, "Alice Researcher");
    assert!(contributor.orcid.is_some());
    assert_eq!(contributor.roles.len(), 2);
}

// -------------------------------------------------------------------------
// BibTeX/CFF Coverage Gap Tests
// -------------------------------------------------------------------------

#[test]
fn test_bibtex_inproceedings_venue_is_booktitle() {
    let citation = CitationMetadata {
        citation_type: CitationType::InProceedings,
        title: "Conference Paper".to_string(),
        authors: vec!["Author One".to_string()],
        year: 2025,
        month: None,
        doi: None,
        url: Some("https://example.com/paper".to_string()),
        venue: Some("ICML 2025".to_string()),
        volume: None,
        pages: None,
        publisher: Some("ACM".to_string()),
        version: None,
    };

    let bibtex = citation.to_bibtex("one2025conf");
    assert!(bibtex.contains("@inproceedings{one2025conf,"));
    assert!(bibtex.contains("booktitle = {ICML 2025}"));
    assert!(bibtex.contains("url = {https://example.com/paper}"));
    assert!(bibtex.contains("publisher = {ACM}"));
    // month should NOT appear
    assert!(!bibtex.contains("month"));
}

#[test]
fn test_bibtex_book_venue_is_howpublished() {
    let citation = CitationMetadata {
        citation_type: CitationType::Book,
        title: "A Textbook".to_string(),
        authors: vec!["Writer A".to_string()],
        year: 2023,
        month: Some(3),
        doi: None,
        url: None,
        venue: Some("O'Reilly".to_string()),
        volume: None,
        pages: None,
        publisher: None,
        version: Some("2nd".to_string()),
    };

    let bibtex = citation.to_bibtex("writer2023book");
    assert!(bibtex.contains("@book{writer2023book,"));
    assert!(bibtex.contains("howpublished = {O'Reilly}"));
    assert!(bibtex.contains("version = {2nd}"));
    assert!(bibtex.contains("month = {3}"));
    // url, publisher, volume, pages should NOT appear
    assert!(!bibtex.contains("url ="));
    assert!(!bibtex.contains("publisher ="));
}

#[test]
fn test_bibtex_software_type() {
    let citation = CitationMetadata {
        citation_type: CitationType::Software,
        title: "MySoftware".to_string(),
        authors: vec!["Dev A".to_string(), "Dev B".to_string()],
        year: 2024,
        month: None,
        doi: Some("10.5281/zenodo.999".to_string()),
        url: None,
        venue: None,
        volume: None,
        pages: None,
        publisher: None,
        version: Some("3.1.0".to_string()),
    };

    let bibtex = citation.to_bibtex("soft2024");
    assert!(bibtex.contains("@software{soft2024,"));
    assert!(bibtex.contains("version = {3.1.0}"));
    assert!(bibtex.contains("doi = {10.5281/zenodo.999}"));
    assert!(bibtex.contains("author = {Dev A and Dev B}"));
}

#[test]
fn test_bibtex_dataset_type() {
    let citation = CitationMetadata {
        citation_type: CitationType::Dataset,
        title: "MyDataset".to_string(),
        authors: vec!["Data Curator".to_string()],
        year: 2024,
        month: None,
        doi: None,
        url: None,
        venue: None,
        volume: None,
        pages: None,
        publisher: None,
        version: None,
    };

    let bibtex = citation.to_bibtex("ds2024");
    assert!(bibtex.contains("@dataset{ds2024,"));
}

#[test]
fn test_bibtex_misc_type() {
    let citation = CitationMetadata {
        citation_type: CitationType::Misc,
        title: "A Note".to_string(),
        authors: vec!["Nobody".to_string()],
        year: 2020,
        month: None,
        doi: None,
        url: None,
        venue: Some("Blog Post".to_string()),
        volume: None,
        pages: None,
        publisher: None,
        version: None,
    };

    let bibtex = citation.to_bibtex("misc2020");
    assert!(bibtex.contains("@misc{misc2020,"));
    assert!(bibtex.contains("howpublished = {Blog Post}"));
}

#[test]
fn test_cff_no_month_defaults_to_january() {
    let citation = CitationMetadata {
        citation_type: CitationType::Software,
        title: "Tool".to_string(),
        authors: vec!["Dev".to_string()],
        year: 2025,
        month: None,
        doi: None,
        url: None,
        venue: None,
        volume: None,
        pages: None,
        publisher: None,
        version: None,
    };

    let cff = citation.to_cff();
    assert!(cff.contains("date-released: \"2025-01-01\""));
    // Optional version/doi/url fields should NOT appear (note: "cff-version:" is always present)
    assert!(!cff.contains("version: \""));
    assert!(!cff.contains("doi: \""));
    assert!(!cff.contains("url: \""));
}

#[test]
fn test_cff_multiple_authors() {
    let citation = CitationMetadata {
        citation_type: CitationType::Article,
        title: "Multi Author".to_string(),
        authors: vec!["Alice".to_string(), "Bob".to_string(), "Charlie".to_string()],
        year: 2024,
        month: Some(12),
        doi: None,
        url: None,
        venue: None,
        volume: None,
        pages: None,
        publisher: None,
        version: None,
    };

    let cff = citation.to_cff();
    assert!(cff.contains("- name: \"Alice\""));
    assert!(cff.contains("- name: \"Bob\""));
    assert!(cff.contains("- name: \"Charlie\""));
    assert!(cff.contains("date-released: \"2024-12-01\""));
}

#[test]
fn test_bibtex_all_optional_fields() {
    let citation = CitationMetadata {
        citation_type: CitationType::Article,
        title: "Complete".to_string(),
        authors: vec!["Author".to_string()],
        year: 2024,
        month: Some(6),
        doi: Some("10.1234/test".to_string()),
        url: Some("https://example.com".to_string()),
        venue: Some("Nature".to_string()),
        volume: Some("100".to_string()),
        pages: Some("10-20".to_string()),
        publisher: Some("Springer".to_string()),
        version: Some("1.0".to_string()),
    };

    let bibtex = citation.to_bibtex("complete2024");
    assert!(bibtex.contains("month = {6}"));
    assert!(bibtex.contains("doi = {10.1234/test}"));
    assert!(bibtex.contains("url = {https://example.com}"));
    assert!(bibtex.contains("journal = {Nature}"));
    assert!(bibtex.contains("volume = {100}"));
    assert!(bibtex.contains("pages = {10-20}"));
    assert!(bibtex.contains("publisher = {Springer}"));
    assert!(bibtex.contains("version = {1.0}"));
}

// -------------------------------------------------------------------------
// ExperimentRun Coverage Gap Tests
// -------------------------------------------------------------------------

#[test]
fn test_run_status_cancelled_variant() {
    let device =
        ComputeDevice::Cpu { cores: 4, threads_per_core: 1, architecture: CpuArchitecture::X86_64 };

    let mut run =
        ExperimentRun::new("run-cancel", "cancel-experiment", ModelParadigm::TraditionalML, device);
    // Manually set to cancelled since there's no cancel() method
    run.status = RunStatus::Cancelled;
    run.ended_at = Some(chrono::Utc::now().to_rfc3339());

    assert_eq!(run.status, RunStatus::Cancelled);
    assert!(run.ended_at.is_some());
}

#[test]
fn test_run_status_serialization_all_variants() {
    // Test each RunStatus variant serializes and deserializes correctly
    let statuses =
        vec![RunStatus::Running, RunStatus::Completed, RunStatus::Failed, RunStatus::Cancelled];

    for status in statuses {
        let json = serde_json::to_string(&status).unwrap();
        let deserialized: RunStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(status, deserialized);
    }
}

#[test]
fn test_run_status_equality_and_copy() {
    let s1 = RunStatus::Running;
    let s2 = s1; // Copy
    assert_eq!(s1, s2); // s1 still usable

    assert_ne!(RunStatus::Running, RunStatus::Completed);
    assert_ne!(RunStatus::Completed, RunStatus::Failed);
    assert_ne!(RunStatus::Failed, RunStatus::Cancelled);
}

#[test]
fn test_run_status_debug_format() {
    assert_eq!(format!("{:?}", RunStatus::Running), "Running");
    assert_eq!(format!("{:?}", RunStatus::Completed), "Completed");
    assert_eq!(format!("{:?}", RunStatus::Failed), "Failed");
    assert_eq!(format!("{:?}", RunStatus::Cancelled), "Cancelled");
}

#[test]
fn test_experiment_run_with_tags() {
    let device =
        ComputeDevice::Cpu { cores: 8, threads_per_core: 2, architecture: CpuArchitecture::X86_64 };

    let mut run =
        ExperimentRun::new("run-tags", "tag-experiment", ModelParadigm::DeepLearning, device);
    run.tags.push("baseline".to_string());
    run.tags.push("production".to_string());
    run.tags.push("v2".to_string());

    assert_eq!(run.tags.len(), 3);
    assert!(run.tags.contains(&"baseline".to_string()));
    assert!(run.tags.contains(&"production".to_string()));
}

#[test]
fn test_experiment_run_with_energy_metrics() {
    let device = ComputeDevice::Cpu {
        cores: 4,
        threads_per_core: 2,
        architecture: CpuArchitecture::Aarch64,
    };

    let mut run =
        ExperimentRun::new("run-energy", "energy-experiment", ModelParadigm::FineTuning, device);
    run.energy = Some(EnergyMetrics::new(1800.0, 100.0, 250.0, 3600.0).with_pue(1.2));

    assert!(run.energy.is_some());
    let energy = run.energy.as_ref().unwrap();
    assert_eq!(energy.total_joules, 1800.0);
    assert_eq!(energy.pue, 1.2);
}

#[test]
fn test_experiment_run_with_cost_metrics() {
    let device = ComputeDevice::Gpu {
        name: "A100".to_string(),
        memory_gb: 80.0,
        compute_capability: Some("8.0".to_string()),
        vendor: GpuVendor::Nvidia,
    };

    let mut run =
        ExperimentRun::new("run-cost", "cost-experiment", ModelParadigm::DeepLearning, device);
    run.cost = Some(CostMetrics::new(40.0, 5.0, 5.0));

    assert!(run.cost.is_some());
    let cost = run.cost.as_ref().unwrap();
    assert_eq!(cost.total_cost_usd, 50.0);
    assert_eq!(cost.compute_cost_usd, 40.0);
}

#[test]
fn test_experiment_run_with_platform() {
    let device =
        ComputeDevice::Cpu { cores: 8, threads_per_core: 2, architecture: CpuArchitecture::X86_64 };

    let mut run =
        ExperimentRun::new("run-plat", "platform-experiment", ModelParadigm::TraditionalML, device);
    run.platform = PlatformEfficiency::Edge;

    assert_eq!(run.platform, PlatformEfficiency::Edge);
}

#[test]
fn test_experiment_run_full_lifecycle_serialization() {
    let device = ComputeDevice::Gpu {
        name: "RTX 4090".to_string(),
        memory_gb: 24.0,
        compute_capability: Some("8.9".to_string()),
        vendor: GpuVendor::Nvidia,
    };

    let mut run =
        ExperimentRun::new("run-full", "full-lifecycle", ModelParadigm::DeepLearning, device);
    run.log_metric("accuracy", 0.99);
    run.log_metric("f1_score", 0.98);
    run.log_param("epochs", serde_json::json!(100));
    run.log_param("optimizer", serde_json::json!("adam"));
    run.tags.push("best-model".to_string());
    run.platform = PlatformEfficiency::Server;
    run.complete();

    let json = serde_json::to_string(&run).unwrap();
    let deserialized: ExperimentRun = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.run_id, "run-full");
    assert_eq!(deserialized.status, RunStatus::Completed);
    assert!(deserialized.ended_at.is_some());
    assert_eq!(deserialized.metrics.len(), 2);
    assert_eq!(deserialized.hyperparameters.len(), 2);
    assert_eq!(deserialized.tags.len(), 1);
}

#[test]
fn test_in_memory_storage_delete_nonexistent() {
    let storage = InMemoryExperimentStorage::new();
    // Deleting a non-existent run should succeed (no-op)
    let result = storage.delete_run("nonexistent");
    assert!(result.is_ok());
}

#[test]
fn test_in_memory_storage_list_empty() {
    let storage = InMemoryExperimentStorage::new();
    let runs = storage.list_runs("nonexistent-experiment").unwrap();
    assert!(runs.is_empty());
}

#[test]
fn test_in_memory_storage_overwrite_run() {
    let storage = InMemoryExperimentStorage::new();
    let device =
        ComputeDevice::Cpu { cores: 4, threads_per_core: 1, architecture: CpuArchitecture::X86_64 };

    let mut run = ExperimentRun::new(
        "run-001",
        "overwrite-test",
        ModelParadigm::TraditionalML,
        device.clone(),
    );
    run.log_metric("accuracy", 0.5);
    storage.store_run(&run).unwrap();

    // Update and overwrite
    run.log_metric("accuracy", 0.9);
    run.complete();
    storage.store_run(&run).unwrap();

    let retrieved = storage.get_run("run-001").unwrap().unwrap();
    assert_eq!(retrieved.metrics.get("accuracy"), Some(&0.9));
    assert_eq!(retrieved.status, RunStatus::Completed);
}

#[test]
fn test_in_memory_storage_default() {
    let storage = InMemoryExperimentStorage::default();
    let runs = storage.list_runs("any").unwrap();
    assert!(runs.is_empty());
}

#[test]
fn test_experiment_run_multiple_paradigms() {
    let device =
        ComputeDevice::Cpu { cores: 4, threads_per_core: 1, architecture: CpuArchitecture::X86_64 };

    let paradigms = vec![
        ModelParadigm::TraditionalML,
        ModelParadigm::DeepLearning,
        ModelParadigm::FineTuning,
        ModelParadigm::Distillation,
        ModelParadigm::MoE,
    ];

    for paradigm in paradigms {
        let run = ExperimentRun::new("run", "exp", paradigm, device.clone());
        assert_eq!(run.paradigm, paradigm);
    }
}
