//! Data Platforms Integration
//!
//! Unified interface for enterprise data platforms:
//! - Databricks (Delta Lake, Unity Catalog, MLflow)
//! - Snowflake (Iceberg, Snowpark, Data Sharing)
//! - AWS (S3, SageMaker, Bedrock)
//! - HuggingFace (Hub, Datasets, Transformers)
//!
//! ## Toyota Way Principles
//!
//! - Genchi Genbutsu: Direct platform API queries
//! - Poka-Yoke: OS-level egress filtering for sovereignty
//! - Heijunka: Adaptive throttling for shared resources
//! - Jidoka: Schema drift detection stops the line
//! - Muda: Federation over migration (zero-copy)
//! - Andon: Cost estimation before execution

pub mod tree;

// Re-export types for library users (used by lib.rs, not by main.rs binary)
#[allow(unused_imports)]
pub use tree::{
    DataPlatformTree, IntegrationMapping, IntegrationType, Platform, PlatformCategory,
    PlatformComponent,
};
