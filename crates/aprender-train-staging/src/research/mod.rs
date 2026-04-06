//! Academic Research Artifacts (Phase 7)
//!
//! This module provides tools for academic research workflows:
//! - Research artifact metadata with CRediT taxonomy
//! - Citation generation (BibTeX, CFF)
//! - Literate programming document support
//! - Pre-registration with cryptographic commitments
//! - Double-blind anonymization
//! - Jupyter notebook export
//! - Citation graph management
//! - RO-Crate packaging
//! - Archive deposits (Zenodo, Figshare)

pub mod anonymization;
pub mod archive;
pub mod artifact;
pub mod citation;
pub mod citation_graph;
pub mod literate;
pub mod notebook;
pub mod preregistration;
pub mod ro_crate;

// Re-export commonly used types
pub use anonymization::{AnonymizationConfig, AnonymizedArtifact, AnonymousAuthor};
pub use archive::{
    ArchiveDeposit, ArchiveProvider, DepositError, DepositMetadata, DepositResult, FigshareConfig,
    ZenodoConfig,
};
pub use artifact::{
    Affiliation, ArtifactType, Author, ContributorRole, License, ResearchArtifact, ValidationError,
};
pub use citation::CitationMetadata;
pub use citation_graph::{CitationEdge, CitationGraph, CitationNode, EdgeType};
pub use literate::{CodeBlock, LiterateDocument};
pub use notebook::{Cell, CellOutput, CellType, KernelSpec, NotebookExporter};
pub use preregistration::{
    PreRegistration, PreRegistrationCommitment, PreRegistrationError, PreRegistrationReveal,
    SignedPreRegistration, TimestampProof,
};
pub use ro_crate::{EntityType, RoCrate, RoCrateDescriptor, RoCrateEntity};
