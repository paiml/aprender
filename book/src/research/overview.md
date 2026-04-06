# Academic Research Artifacts

The research module provides tools for academic research workflows, supporting FAIR data principles, proper attribution,
and reproducible science.

## Why Research Artifacts?

Academic research requires:
- **Proper Attribution**: CRediT taxonomy for contributor roles
- **Persistent Identifiers**: ORCID for authors, ROR for institutions, DOI for artifacts
- **Reproducibility**: Pre-registration with cryptographic commitments
- **FAIR Data**: RO-Crate packaging for findable, accessible, interoperable, reusable data
- **Double-Blind Review**: Anonymization support for peer review

## Components

### 1. Research Artifacts (ENT-019)

Core types for research metadata:

```rust
use entrenar::research::{
    ResearchArtifact, Author, Affiliation, ArtifactType,
    License, ContributorRole
};

// Create an author with ORCID and CRediT roles
let author = Author::new("Alice Smith")
    .with_orcid("0000-0002-1825-0097")?
    .with_affiliation(
        Affiliation::new("MIT")
            .with_ror_id("https://ror.org/03yrm5c26")?
            .with_country("US")
    )
    .with_roles([
        ContributorRole::Conceptualization,
        ContributorRole::Software,
        ContributorRole::WritingOriginal,
    ]);

// Create a research artifact
let artifact = ResearchArtifact::new(
    "dataset-2024-001",
    "Novel Deep Learning Dataset",
    ArtifactType::Dataset,
    License::CcBy4,
)
.with_author(author)
.with_doi("10.5281/zenodo.1234567")
.with_description("A curated dataset for ML research")
.with_keywords(["machine learning", "computer vision"]);
```

### 2. Citation Generation (ENT-020)

Export citations in BibTeX and CITATION.cff formats:

```rust
use entrenar::research::CitationMetadata;

let citation = CitationMetadata::new(artifact, 2024)
    .with_journal("Nature Machine Intelligence")
    .with_volume("6")
    .with_pages("123-145")
    .with_url("https://example.com/paper");

// Generate BibTeX
let bibtex = citation.to_bibtex();
// @article{smith_2024_novel,
//   author = {Smith, Alice},
//   title = {{Novel Deep Learning Dataset}},
//   year = {2024},
//   ...
// }

// Generate CITATION.cff
let cff = citation.to_cff();
```

### 3. Literate Documents (ENT-021)

Parse and extract code from literate programming documents:

```rust
use entrenar::research::LiterateDocument;

let doc = LiterateDocument::parse_markdown(r#"
# Analysis

```python
import numpy as np
data = np.load("dataset.npy")
```

Results show significant improvement.
"#);

// Extract code blocks
let blocks = doc.extract_code_blocks();
for block in blocks {
    println!("Language: {:?}", block.language);
    println!("Line: {}", block.line_number);
    println!("Code: {}", block.content);
}

// Convert to HTML
let html = doc.to_html();
```

### 4. Pre-Registration (ENT-022)

Cryptographic commitment for research protocols:

```rust
use entrenar::research::{PreRegistration, SignedPreRegistration, TimestampProof};
use ed25519_dalek::SigningKey;

// Create pre-registration
let prereg = PreRegistration::new(
    "Effect of Treatment A on Outcome B",
    "Treatment A improves Outcome B by 20%",
    "Randomized controlled trial, n=100",
    "Two-sample t-test, alpha=0.05",
);

// Create cryptographic commitment
let commitment = prereg.commit();
println!("Commitment hash: {}", commitment.hash);

// Sign with Ed25519
let signing_key = SigningKey::generate(&mut rand::rngs::OsRng);
let signed = SignedPreRegistration::sign(&prereg, &signing_key)
    .with_timestamp_proof(TimestampProof::git("abc123"));

// Later: verify the commitment
let reveal = prereg.reveal(&commitment)?;
assert!(signed.verify()?);
```

### 5. Anonymization (ENT-023)

Double-blind review support:

```rust
use entrenar::research::{AnonymizationConfig, anonymize_text};

let config = AnonymizationConfig::new("review-salt-2024")
    .with_author_replacement("Anonymous Author")
    .with_affiliation_replacement("Anonymous Institution")
    .with_strip_doi(true);

// Anonymize artifact
let anon_artifact = config.anonymize(&artifact);

// Export for double-blind review
let json = anon_artifact.to_double_blind_json();
// No author names, affiliations, or identifying info
```

### 6. Jupyter Export (ENT-024)

Export to Jupyter notebook format:

```rust
use entrenar::research::{NotebookExporter, KernelSpec, Cell};

// Create notebook with Rust kernel
let mut notebook = NotebookExporter::with_kernel(KernelSpec::evcxr());

notebook.add_markdown("# Rust Analysis\n\nUsing entrenar for ML.");
notebook.add_code(r#"
use entrenar::autograd::Tensor;
let x = Tensor::from_vec(vec![1.0, 2.0, 3.0]);
"#);

// Or convert from literate document
let notebook = NotebookExporter::from_literate(&doc);

// Export to .ipynb
let ipynb = notebook.to_ipynb();
std::fs::write("analysis.ipynb", ipynb)?;
```

### 7. Citation Graph (ENT-025)

Track and aggregate citations:

```rust
use entrenar::research::{CitationGraph, CitationNode, EdgeType};

let mut graph = CitationGraph::new();

// Add citation nodes
graph.add_node("paper-a", CitationNode::new(citation_a, false));
graph.add_node("paper-b", CitationNode::new(citation_b, true));
graph.add_node("paper-c", CitationNode::new(citation_c, true));

// Build citation relationships
graph.add_citation("paper-a", "paper-b");
graph.add_citation("paper-b", "paper-c");

// Get all upstream citations (including transitive)
let all_citations = graph.aggregate_all_citations("paper-a");

// Export all to BibTeX
let bibtex_all = graph.to_bibtex_all();
```

### 8. RO-Crate Packaging (ENT-026)

Create FAIR-compliant research object packages:

```rust
use entrenar::research::RoCrate;

// Create RO-Crate from artifact
let mut crate_pkg = RoCrate::from_artifact(&artifact, "./my-dataset");

// Add data files
crate_pkg.add_text_file("data/train.csv", "x,y\n1,2\n3,4");
crate_pkg.add_text_file("README.md", "# Dataset\n\nDescription...");
crate_pkg.add_file("model.safetensors", model_bytes);

// Write to directory (creates ro-crate-metadata.json)
crate_pkg.to_directory()?;

// Or create ZIP archive
let zip_bytes = crate_pkg.to_zip()?;
std::fs::write("dataset.zip", zip_bytes)?;
```

### 9. Archive Deposits (ENT-027)

Deposit to academic archives:

```rust
use entrenar::research::{ArchiveDeposit, ArchiveProvider, ZenodoConfig};

// Configure Zenodo
let config = ZenodoConfig::new("your-api-token")
    .with_sandbox(true)  // Use sandbox for testing
    .with_community("ml-research");

// Create deposit
let deposit = ArchiveDeposit::new(ArchiveProvider::Zenodo, artifact)
    .with_text_file("README.md", readme_content)
    .with_file("data.zip", data_bytes);

// Submit (returns DOI)
let result = deposit.deposit()?;
println!("DOI: {}", result.doi);
println!("URL: {}", result.url);
```

## CRediT Contributor Roles

The module supports all 14 CRediT taxonomy roles:

| Role | Description |
|------|-------------|
| Conceptualization | Ideas and research goals |
| DataCuration | Data management and annotation |
| FormalAnalysis | Statistical/computational analysis |
| FundingAcquisition | Financial support |
| Investigation | Research execution |
| Methodology | Method development |
| ProjectAdministration | Project management |
| Resources | Materials and infrastructure |
| Software | Programming and development |
| Supervision | Oversight and leadership |
| Validation | Verification of results |
| Visualization | Data presentation |
| WritingOriginal | Initial draft |
| WritingReview | Critical review and editing |

## Supported Formats

| Format | Use Case |
|--------|----------|
| BibTeX | LaTeX citations |
| CFF | GitHub CITATION.cff |
| RO-Crate | FAIR data packages |
| Jupyter | Interactive notebooks |
| JSON-LD | Linked data |

## Archive Providers

| Provider | URL |
|----------|-----|
| Zenodo | https://zenodo.org |
| Figshare | https://figshare.com |
| Dryad | https://datadryad.org |
| Dataverse | https://dataverse.harvard.edu |
