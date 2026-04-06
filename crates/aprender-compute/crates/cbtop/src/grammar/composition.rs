//! Composition mode (Facets equivalent in Grammar of Graphics).

/// Composition mode (analogous to Facets)
#[derive(Debug, Clone, PartialEq, Default)]
pub enum CompositionMode {
    /// Single execution
    #[default]
    None,
    /// Data parallelism (same op, different data)
    DataParallel { shards: usize },
    /// Model parallelism (different ops, same data)
    ModelParallel { stages: usize },
    /// Pipeline parallelism
    Pipeline { depth: usize, overlap: bool },
    /// Batch processing
    Batch { batch_size: usize, prefetch: usize },
}

impl CompositionMode {
    /// Create data parallel mode
    pub fn data_parallel(shards: usize) -> Self {
        CompositionMode::DataParallel { shards }
    }

    /// Create batch mode
    pub fn batch(size: usize) -> Self {
        CompositionMode::Batch { batch_size: size, prefetch: 2 }
    }

    /// Create pipeline mode
    pub fn pipeline(depth: usize) -> Self {
        CompositionMode::Pipeline { depth, overlap: true }
    }
}
