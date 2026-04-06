//! Resource mapping (Aesthetics equivalent in Grammar of Graphics).

/// Scale binding for resource mapping
#[derive(Debug, Clone, PartialEq)]
pub enum ScaleBinding {
    /// Bind to problem size
    ProblemSize,
    /// Bind to data volume
    DataVolume,
    /// Bind to throughput requirement
    Throughput,
    /// Custom binding expression
    Custom(String),
}

/// Byte size helper
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ByteSize(pub usize);

impl ByteSize {
    /// Create from megabytes
    pub fn mb(mb: usize) -> Self {
        ByteSize(mb * 1024 * 1024)
    }

    /// Create from gigabytes
    pub fn gb(gb: usize) -> Self {
        ByteSize(gb * 1024 * 1024 * 1024)
    }

    /// Get raw bytes
    pub fn bytes(&self) -> usize {
        self.0
    }
}

/// Resource mapping (analogous to Aesthetics)
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ResourceMapping {
    /// Map problem size to cores
    pub cores: Option<ScaleBinding>,
    /// Map data volume to memory
    pub memory: Option<ScaleBinding>,
    /// Map throughput to bandwidth
    pub bandwidth: Option<ScaleBinding>,
    /// Map latency constraints
    pub latency: Option<ScaleBinding>,
    /// Fixed core count override
    pub cores_value: Option<usize>,
    /// Fixed memory limit override
    pub memory_value: Option<ByteSize>,
}

impl ResourceMapping {
    /// Create empty resource mapping
    pub fn new() -> Self {
        Self::default()
    }

    /// Set core binding
    pub fn cores(mut self, binding: ScaleBinding) -> Self {
        self.cores = Some(binding);
        self
    }

    /// Set fixed core count
    pub fn cores_value(mut self, count: usize) -> Self {
        self.cores_value = Some(count);
        self
    }

    /// Set memory binding
    pub fn memory(mut self, binding: ScaleBinding) -> Self {
        self.memory = Some(binding);
        self
    }

    /// Set fixed memory limit
    pub fn memory_value(mut self, size: ByteSize) -> Self {
        self.memory_value = Some(size);
        self
    }
}
