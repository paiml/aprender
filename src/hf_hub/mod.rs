//! HuggingFace Hub dataset importer.
//!
//! Provides functionality to import datasets from the HuggingFace Hub.
//! Supports downloading parquet files directly from HF datasets.
//!
//! # Example
//!
//! ```no_run
//! use alimentar::{hf_hub::HfDataset, Dataset};
//!
//! // Import a dataset from HuggingFace Hub
//! let hf = HfDataset::builder("squad")
//!     .revision("main")
//!     .split("train")
//!     .build()
//!     .unwrap();
//!
//! let dataset = hf.download().unwrap();
//! println!("Loaded {} rows", dataset.len());
//! ```

mod download;
mod upload;
pub mod validation;

#[cfg(test)]
mod tests;

// Re-export download types
// Internal re-exports for tests
#[cfg(test)]
pub(crate) use download::default_cache_dir;
pub use download::{list_dataset_files, DatasetInfo, HfDataset, HfDatasetBuilder};
#[cfg(test)]
pub(crate) use upload::HF_API_URL;
// Re-export upload types
#[cfg(feature = "hf-hub")]
pub use upload::{
    build_lfs_batch_request, build_lfs_preupload_request, build_ndjson_lfs_commit,
    build_ndjson_upload_payload, compute_sha256, is_binary_file,
};
pub use upload::{HfPublisher, HfPublisherBuilder};
// Re-export validation types
pub use validation::{
    DatasetCardValidator, ValidationError, VALID_LICENSES, VALID_SIZE_CATEGORIES,
    VALID_TASK_CATEGORIES,
};
