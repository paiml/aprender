//! LoRA (Low-Rank Adaptation) implementation
//!
//! LoRA enables parameter-efficient fine-tuning of large pretrained models
//! by adding trainable low-rank decomposition matrices to frozen weights.

mod adapter;
mod config;
mod dora;
mod layer;
mod multi_adapter;
mod paged_optim;
mod pissa;
mod qlora;

#[cfg(test)]
mod benchmarks;
#[cfg(test)]
mod gradient_tests;

pub use adapter::{
    load_adapter, load_adapter_peft, merge_and_collect, merge_qlora_and_collect, save_adapter,
    save_adapter_peft, AdapterError, AdapterFormat, AdapterMetadata, LoRAAdapter, MergedModel,
    PeftAdapterBundle, PeftAdapterConfig,
};
#[cfg(feature = "hub-publish")]
pub use adapter::{
    merge_export_publish, merge_qlora_export_publish, MergePublishError, MergePublishResult,
};
pub use config::LoRAConfig;
pub use dora::DoRALayer;
pub use layer::{LoRALayer, LoRAScaling};
pub use multi_adapter::{MultiAdapterManager, NamedAdapter};
pub use paged_optim::{PagedOptimStates, PagedState, PagingStats, PagingStrategy, VramBudget};
pub use pissa::pissa_init;
pub use qlora::{MemoryStats, QLoRALayer};

#[cfg(test)]
pub use benchmarks::{
    benchmark_model, run_transformer_benchmarks, BenchmarkResults, LayerMemoryStats,
};
