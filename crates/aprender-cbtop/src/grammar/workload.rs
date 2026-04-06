//! Workload specification (Data equivalent in Grammar of Graphics).

/// Compute operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Operation {
    /// Element-wise operations
    Elementwise,
    /// Dot product
    Dot,
    /// Matrix multiplication
    Matmul,
    /// 2D convolution
    Conv2d,
    /// Multi-head attention
    Attention,
    /// Softmax
    Softmax,
    /// Layer normalization
    LayerNorm,
    /// Feed-forward network
    Ffn,
    /// Reduction (sum, mean, max, etc.)
    Reduce,
    /// Custom operation
    Custom(u32),
}

/// Data type for compute operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    /// 32-bit floating point
    F32,
    /// 16-bit floating point
    F16,
    /// Brain float 16
    Bf16,
    /// 8-bit integer
    I8,
    /// Unsigned 8-bit integer
    U8,
    /// 4-bit quantized (packed)
    Q4,
}

impl DataType {
    /// Get byte size of data type
    pub fn byte_size(&self) -> usize {
        match self {
            DataType::F32 => 4,
            DataType::F16 | DataType::Bf16 => 2,
            DataType::I8 | DataType::U8 => 1,
            DataType::Q4 => 1, // 2 values per byte
        }
    }
}

/// Problem dimensions
#[derive(Debug, Clone, PartialEq)]
pub struct Dimensions {
    /// Batch size
    pub batch: usize,
    /// Sequence length (for attention)
    pub seq_len: usize,
    /// Number of heads (for attention)
    pub num_heads: usize,
    /// Head dimension (for attention)
    pub head_dim: usize,
    /// Hidden dimension (for FFN)
    pub hidden_dim: usize,
    /// M dimension (for matmul)
    pub m: usize,
    /// N dimension (for matmul)
    pub n: usize,
    /// K dimension (for matmul)
    pub k: usize,
}

impl Default for Dimensions {
    fn default() -> Self {
        Self { batch: 1, seq_len: 1, num_heads: 1, head_dim: 64, hidden_dim: 1, m: 1, n: 1, k: 1 }
    }
}

impl Dimensions {
    /// Create dimensions for vector operation
    pub fn vector(size: usize) -> Self {
        Self { n: size, ..Default::default() }
    }

    /// Create dimensions for matrix multiplication
    pub fn matmul(m: usize, n: usize, k: usize) -> Self {
        Self { m, n, k, ..Default::default() }
    }

    /// Create dimensions for attention
    pub fn attention(batch: usize, seq_len: usize, num_heads: usize, head_dim: usize) -> Self {
        Self { batch, seq_len, num_heads, head_dim, ..Default::default() }
    }
}

/// Tensor specification
#[derive(Debug, Clone, PartialEq)]
pub struct TensorSpec {
    /// Tensor name/identifier
    pub name: String,
    /// Shape dimensions
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: DataType,
    /// Stride (optional, for non-contiguous tensors)
    pub stride: Option<Vec<usize>>,
}

impl TensorSpec {
    /// Create a new tensor spec
    pub fn new(name: impl Into<String>, shape: Vec<usize>, dtype: DataType) -> Self {
        Self { name: name.into(), shape, dtype, stride: None }
    }

    /// Total number of elements
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Total byte size
    pub fn byte_size(&self) -> usize {
        self.numel() * self.dtype.byte_size()
    }
}

/// Workload specification (analogous to DataFrame)
#[derive(Debug, Clone, PartialEq)]
pub struct WorkloadSpec {
    /// Operation type
    pub operation: Operation,
    /// Problem dimensions
    pub dimensions: Dimensions,
    /// Primary data type
    pub dtype: DataType,
    /// Input tensor specifications
    pub inputs: Vec<TensorSpec>,
    /// Output tensor specifications
    pub outputs: Vec<TensorSpec>,
}

impl WorkloadSpec {
    /// Create a dot product workload
    pub fn dot(size: usize) -> Self {
        Self {
            operation: Operation::Dot,
            dimensions: Dimensions::vector(size),
            dtype: DataType::F32,
            inputs: vec![
                TensorSpec::new("a", vec![size], DataType::F32),
                TensorSpec::new("b", vec![size], DataType::F32),
            ],
            outputs: vec![TensorSpec::new("result", vec![1], DataType::F32)],
        }
    }

    /// Create a matrix multiplication workload
    pub fn matmul(m: usize, n: usize, k: usize) -> Self {
        Self {
            operation: Operation::Matmul,
            dimensions: Dimensions::matmul(m, n, k),
            dtype: DataType::F32,
            inputs: vec![
                TensorSpec::new("a", vec![m, k], DataType::F32),
                TensorSpec::new("b", vec![k, n], DataType::F32),
            ],
            outputs: vec![TensorSpec::new("c", vec![m, n], DataType::F32)],
        }
    }

    /// Create an attention workload
    pub fn attention(batch: usize, seq_len: usize, num_heads: usize, head_dim: usize) -> Self {
        let embed_dim = num_heads * head_dim;
        Self {
            operation: Operation::Attention,
            dimensions: Dimensions::attention(batch, seq_len, num_heads, head_dim),
            dtype: DataType::F32,
            inputs: vec![
                TensorSpec::new("q", vec![batch, seq_len, embed_dim], DataType::F32),
                TensorSpec::new("k", vec![batch, seq_len, embed_dim], DataType::F32),
                TensorSpec::new("v", vec![batch, seq_len, embed_dim], DataType::F32),
            ],
            outputs: vec![TensorSpec::new("out", vec![batch, seq_len, embed_dim], DataType::F32)],
        }
    }

    /// Create an elementwise workload
    pub fn elementwise(size: usize) -> Self {
        Self {
            operation: Operation::Elementwise,
            dimensions: Dimensions::vector(size),
            dtype: DataType::F32,
            inputs: vec![TensorSpec::new("input", vec![size], DataType::F32)],
            outputs: vec![TensorSpec::new("output", vec![size], DataType::F32)],
        }
    }

    /// Total FLOP count estimate
    pub fn flop_count(&self) -> usize {
        match self.operation {
            Operation::Dot => self.dimensions.n * 2, // mul + add
            Operation::Matmul => self.dimensions.m * self.dimensions.n * self.dimensions.k * 2,
            Operation::Attention => {
                let b = self.dimensions.batch;
                let s = self.dimensions.seq_len;
                let h = self.dimensions.num_heads;
                let d = self.dimensions.head_dim;
                // QK^T + softmax + AV
                b * h * (s * s * d * 2 + s * s + s * s * d * 2)
            }
            // Default estimate for remaining operations
            Operation::Elementwise
            | Operation::Conv2d
            | Operation::Softmax
            | Operation::LayerNorm
            | Operation::Ffn
            | Operation::Reduce
            | Operation::Custom(_) => self.dimensions.n,
        }
    }
}
