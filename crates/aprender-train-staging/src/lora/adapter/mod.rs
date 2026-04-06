//! LoRA adapter serialization and deserialization
//!
//! Enables saving and loading LoRA adapters independently of base models.
//! This allows:
//! - Training adapters and saving them separately
//! - Loading adapters into frozen base models for inference
//! - Sharing adapters without sharing full model weights
//! - Switching between multiple adapters for the same base model
//! - PEFT-compatible export for HuggingFace ecosystem interop
//! - Merging adapters into base weights for deployment

mod error;
mod io;
mod lora_adapter;
pub(crate) mod merge_export;
#[cfg(feature = "hub-publish")]
pub(crate) mod merge_pipeline;
mod metadata;
pub(crate) mod peft_config;
pub(crate) mod peft_export;

pub use error::AdapterError;
pub use io::{load_adapter, load_adapter_peft, save_adapter, save_adapter_peft, AdapterFormat};
pub use lora_adapter::LoRAAdapter;
pub use merge_export::{merge_and_collect, merge_qlora_and_collect, MergedModel};
#[cfg(feature = "hub-publish")]
pub use merge_pipeline::{
    merge_export_publish, merge_qlora_export_publish, MergePublishError, MergePublishResult,
};
pub use metadata::AdapterMetadata;
pub use peft_config::PeftAdapterConfig;
pub use peft_export::PeftAdapterBundle;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod falsify_tests;
