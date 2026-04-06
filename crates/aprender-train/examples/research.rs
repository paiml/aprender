//! Academic Research Artifacts Example
//!
//! Demonstrates research workflow capabilities:
//! - Research artifact metadata with CRediT roles
//! - Citation generation (BibTeX, CFF)
//! - Pre-registration with cryptographic commitment
//! - Anonymization for double-blind review
//! - Jupyter notebook export
//! - RO-Crate packaging
//!
//! Run with: cargo run --example research

use ed25519_dalek::SigningKey;
use entrenar::research::{
    Affiliation, AnonymizationConfig, ArchiveDeposit, ArchiveProvider, ArtifactType, Author,
    CitationGraph, CitationMetadata, CitationNode, ContributorRole, KernelSpec, License,
    LiterateDocument, NotebookExporter, PreRegistration, ResearchArtifact, RoCrate,
    SignedPreRegistration, TimestampProof,
};
use std::path::PathBuf;

/// Section 1: Create Research Artifact with Authors
fn demo_artifact() -> (ResearchArtifact, Author) {
    println!("--- Research Artifact (ENT-019) ---");

    let author1 = Author::new("Alice Smith")
        .with_orcid("0000-0002-1825-0097")
        .expect("operation should succeed")
        .with_affiliation(
            Affiliation::new("MIT")
                .with_ror_id("https://ror.org/03yrm5c26")
                .expect("operation should succeed")
                .with_country("US"),
        )
        .with_roles([
            ContributorRole::Conceptualization,
            ContributorRole::Software,
            ContributorRole::WritingOriginal,
        ]);

    let author2 = Author::new("Bob Jones")
        .with_affiliation(Affiliation::new("Stanford University").with_country("US"))
        .with_roles([ContributorRole::DataCuration, ContributorRole::Validation]);

    let artifact = ResearchArtifact::new(
        "dataset-2024-ml",
        "Deep Learning Dataset for Computer Vision",
        ArtifactType::Dataset,
        License::CcBy4,
    )
    .with_authors([author1.clone(), author2])
    .with_doi("10.5281/zenodo.1234567")
    .with_version("2.0.0")
    .with_description("A curated dataset for training computer vision models")
    .with_keywords(["machine learning", "computer vision", "deep learning"]);

    println!("Artifact: {}", artifact.title);
    println!("Type: {}", artifact.artifact_type);
    println!("License: {}", artifact.license);
    println!("Authors: {} total", artifact.authors.len());
    for author in &artifact.authors {
        println!("  - {} (ORCID: {:?})", author.name, author.orcid);
        for role in &author.roles {
            println!("    Role: {role}");
        }
    }
    println!();

    (artifact, author1)
}

/// Section 2: Citation Generation
fn demo_citation(artifact: &ResearchArtifact) -> CitationMetadata {
    println!("--- Citation Generation (ENT-020) ---");

    let citation = CitationMetadata::new(artifact.clone(), 2024)
        .with_journal("Nature Machine Intelligence")
        .with_volume("6")
        .with_pages("123-145")
        .with_url("https://example.com/dataset")
        .with_keywords(["benchmark", "evaluation"]);

    println!("Citation key: {}", citation.generate_citation_key());
    println!();

    println!("BibTeX output:");
    println!("{}", citation.to_bibtex());
    println!();

    println!("CITATION.cff output (first 20 lines):");
    for line in citation.to_cff().lines().take(20) {
        println!("{line}");
    }
    println!("...\n");

    citation
}

/// Section 3: Literate Document
fn demo_literate_document() -> LiterateDocument {
    println!("--- Literate Document (ENT-021) ---");

    let literate_content = r#"
# Analysis Report

This document contains our analysis code.

```python
import numpy as np
import matplotlib.pyplot as plt

data = np.load("results.npy")
plt.plot(data)
```

## Results

The analysis shows significant improvement.

```rust
fn main() {
    println!("Hello from Rust!");
}
```
"#;

    let doc = LiterateDocument::parse_markdown(literate_content);
    let blocks = doc.extract_code_blocks();

    println!("Extracted {} code blocks:", blocks.len());
    for block in &blocks {
        println!(
            "  - {:?} at line {} ({} chars)",
            block.language,
            block.line_number,
            block.content.len()
        );
    }
    println!();

    doc
}

/// Section 4: Pre-Registration
fn demo_preregistration() {
    println!("--- Pre-Registration (ENT-022) ---");

    let prereg = PreRegistration::new(
        "Effect of Data Augmentation on Model Accuracy",
        "Data augmentation will improve model accuracy by at least 15%",
        "Train ResNet-50 on ImageNet with/without augmentation, n=5 runs each",
        "Two-sample t-test comparing mean accuracies, alpha=0.05",
    )
    .with_notes("Secondary analysis: effect on different image categories");

    let commitment = prereg.commit();
    println!("Pre-registration commitment:");
    println!("  Hash: {}...", &commitment.hash[..32]);
    println!("  Created: {}", commitment.created_at);

    // Sign with Ed25519
    let signing_key = SigningKey::from_bytes(&[42u8; 32]);
    let signed = SignedPreRegistration::sign(&prereg, &signing_key)
        .with_timestamp_proof(TimestampProof::git("abc123def456"));

    println!("  Signed: Yes (Ed25519)");
    println!("  Signature valid: {}", signed.verify().expect("operation should succeed"));
    println!();
}

/// Section 5: Anonymization
fn demo_anonymization(artifact: &ResearchArtifact) {
    println!("--- Anonymization (ENT-023) ---");

    let anon_config = AnonymizationConfig::new("review-salt-2024")
        .with_author_replacement("Anonymous Author")
        .with_affiliation_replacement("Anonymous Institution");

    let anon_artifact = anon_config.anonymize(artifact);

    println!("Original artifact:");
    println!("  ID: {}", artifact.id);
    println!("  Authors: {:?}", artifact.authors.iter().map(|a| &a.name).collect::<Vec<_>>());

    println!("Anonymized artifact:");
    println!("  Anonymous ID: {}", anon_artifact.anonymous_id);
    println!(
        "  Authors: {:?}",
        anon_artifact.authors.iter().map(|a| &a.placeholder).collect::<Vec<_>>()
    );

    // Verify original ID
    println!(
        "  Can verify original: {}",
        anon_config.verify_original_id(&anon_artifact, "dataset-2024-ml")
    );
    println!();
}

/// Section 6: Jupyter Notebook Export
fn demo_notebook(doc: &LiterateDocument) {
    println!("--- Notebook Export (ENT-024) ---");

    let notebook = NotebookExporter::from_literate(doc);

    println!("Generated notebook:");
    println!("  Kernel: {} ({})", notebook.kernel.display_name, notebook.kernel.language);
    println!("  Cells: {} total", notebook.cell_count());
    println!("    Code cells: {}", notebook.code_cells().len());
    println!("    Markdown cells: {}", notebook.markdown_cells().len());

    // Also demonstrate manual notebook creation
    let mut rust_notebook = NotebookExporter::with_kernel(KernelSpec::evcxr());
    rust_notebook.add_markdown("# Rust ML Analysis\n\nUsing entrenar for machine learning.");
    rust_notebook.add_code("let x = vec![1.0, 2.0, 3.0];\nprintln!(\"{:?}\", x);");

    println!("  Manual Rust notebook: {} cells", rust_notebook.cell_count());
    println!();
}

/// Section 7: Citation Graph
fn demo_citation_graph(citation: &CitationMetadata) {
    println!("--- Citation Graph (ENT-025) ---");

    let mut graph = CitationGraph::new();

    // Create citation chain: our paper -> foundational papers
    let citation_b = CitationMetadata::new(
        ResearchArtifact::new(
            "resnet",
            "Deep Residual Learning",
            ArtifactType::Paper,
            License::Mit,
        )
        .with_author(Author::new("Kaiming He")),
        2016,
    );
    let citation_c = CitationMetadata::new(
        ResearchArtifact::new(
            "imagenet",
            "ImageNet Dataset",
            ArtifactType::Dataset,
            License::CcBy4,
        )
        .with_author(Author::new("Jia Deng")),
        2009,
    );

    graph.add_node("our-paper", CitationNode::new(citation.clone(), false));
    graph.add_node("resnet", CitationNode::new(citation_b, true));
    graph.add_node("imagenet", CitationNode::new(citation_c, true));

    graph.add_citation("our-paper", "resnet");
    graph.add_citation("our-paper", "imagenet");
    graph.add_citation("resnet", "imagenet");

    println!("Citation graph:");
    println!("  Nodes: {}", graph.node_count());
    println!("  Edges: {}", graph.edge_count());
    println!("  Upstream from 'our-paper': {}", graph.cite_upstream("our-paper").len());
    println!("  All citations (transitive): {}", graph.aggregate_all_citations("our-paper").len());
    println!();
}

/// Section 8: RO-Crate Packaging
fn demo_rocrate(artifact: &ResearchArtifact) {
    println!("--- RO-Crate Packaging (ENT-026) ---");

    let crate_path = PathBuf::from("/tmp/entrenar-research-demo");
    let mut ro_crate = RoCrate::from_artifact(artifact, &crate_path);

    ro_crate.add_text_file("README.md", "# Dataset\n\nThis is a demo dataset.");
    ro_crate.add_text_file("data/train.csv", "x,y,label\n1,2,0\n3,4,1");
    ro_crate.add_text_file("data/test.csv", "x,y,label\n5,6,0\n7,8,1");

    println!("RO-Crate package:");
    println!("  Root: {}", crate_path.display());
    println!("  Entities: {}", ro_crate.entity_count());
    println!("  Data files: {}", ro_crate.file_count());

    // Create ZIP
    let zip_data = ro_crate.to_zip().expect("operation should succeed");
    println!("  ZIP size: {} bytes", zip_data.len());
    println!();
}

/// Section 9: Archive Deposit
fn demo_archive_deposit(artifact: &ResearchArtifact) {
    println!("--- Archive Deposit (ENT-027) ---");

    let deposit = ArchiveDeposit::new(ArchiveProvider::Zenodo, artifact.clone())
        .with_text_file("README.md", "# Dataset Documentation");

    println!("Deposit prepared for: {}", deposit.provider);
    println!("  Files: {}", deposit.files.len());
    println!("  Metadata title: {}", deposit.metadata.title);
    println!("  Resource type: {:?}", deposit.metadata.resource_type);

    // Supported providers
    println!("\nSupported archive providers:");
    for provider in [
        ArchiveProvider::Zenodo,
        ArchiveProvider::Figshare,
        ArchiveProvider::Dryad,
        ArchiveProvider::Dataverse,
    ] {
        println!("  - {}: {}", provider, provider.base_url());
    }
    println!();
}

fn main() -> entrenar::Result<()> {
    println!("=== Entrenar Academic Research Demo ===\n");

    let (artifact, _author1) = demo_artifact();
    let citation = demo_citation(&artifact);
    let doc = demo_literate_document();
    demo_preregistration();
    demo_anonymization(&artifact);
    demo_notebook(&doc);
    demo_citation_graph(&citation);
    demo_rocrate(&artifact);
    demo_archive_deposit(&artifact);

    // Summary
    println!("=== Demo Complete ===");
    println!();
    println!("The research module provides:");
    println!("  - CRediT taxonomy for contributor roles");
    println!("  - ORCID/ROR validation for identifiers");
    println!("  - BibTeX and CFF citation export");
    println!("  - Cryptographic pre-registration");
    println!("  - Double-blind anonymization");
    println!("  - Jupyter notebook generation");
    println!("  - RO-Crate 1.1 compliant packaging");
    println!("  - Archive deposit preparation");

    Ok(())
}
