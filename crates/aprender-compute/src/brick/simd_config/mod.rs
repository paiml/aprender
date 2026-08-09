//! SIMD Configuration and Lazy Initialization
//!
//! LCP-07: Lazy AMX/SIMD tile configuration for expensive state setup.
//! LCP-13: Unroll-and-tail vectorization patterns.

use super::ComputeBackend;

// ----------------------------------------------------------------------------
// LCP-07: Lazy AMX Tile Config
// ----------------------------------------------------------------------------

/// SIMD backend state for lazy initialization.
///
/// AMX (Advanced Matrix Extensions) and AVX-512 require tile configuration
/// that's expensive to set up. This tracks whether initialization has occurred.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SimdBackendState {
    /// Not initialized - will configure on first use
    #[default]
    Uninitialized,
    /// Configuration in progress
    Configuring,
    /// Ready to use
    Ready,
    /// Failed to initialize (fallback to scalar)
    Failed,
}

/// Lazy SIMD tile configuration manager.
///
/// Defers expensive SIMD state setup until actually needed.
#[derive(Debug)]
pub struct LazySimdConfig {
    /// Current state
    state: SimdBackendState,
    /// Best available backend
    best_backend: ComputeBackend,
    /// Whether AMX is supported
    amx_supported: bool,
    /// Tile configuration (for AMX)
    tile_config: Option<AmxTileConfig>,
}

/// AMX tile configuration (8x8 tile palette).
#[derive(Debug, Clone, Copy, Default)]
pub struct AmxTileConfig {
    /// Palette ID (0-1)
    pub palette: u8,
    /// Start row
    pub start_row: u8,
    /// Number of rows per tile
    pub rows: u8,
    /// Bytes per row
    pub bytes_per_row: u16,
}

impl LazySimdConfig {
    /// Create new lazy config, detecting best backend.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: SimdBackendState::Uninitialized,
            best_backend: Self::detect_best_backend(),
            amx_supported: Self::detect_amx(),
            tile_config: None,
        }
    }

    /// Detect best available SIMD backend.
    fn detect_best_backend() -> ComputeBackend {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                return ComputeBackend::Avx512;
            }
            if is_x86_feature_detected!("avx2") {
                return ComputeBackend::Avx2;
            }
            if is_x86_feature_detected!("sse2") {
                return ComputeBackend::Sse2;
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            // NEON is always available on aarch64
            return ComputeBackend::Neon;
        }
        #[allow(unreachable_code)]
        ComputeBackend::Scalar
    }

    /// Detect AMX support (Intel Sapphire Rapids+).
    fn detect_amx() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            // AMX requires specific CPUID checks
            // For now, return false as AMX is rare
            false
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }

    /// Ensure SIMD is configured, initializing lazily if needed.
    pub fn ensure_ready(&mut self) -> Result<ComputeBackend, SimdBackendState> {
        match self.state {
            SimdBackendState::Ready => Ok(self.best_backend),
            SimdBackendState::Failed => Err(SimdBackendState::Failed),
            SimdBackendState::Configuring => Err(SimdBackendState::Configuring),
            SimdBackendState::Uninitialized => {
                self.state = SimdBackendState::Configuring;

                // Configure AMX tiles if supported
                if self.amx_supported {
                    self.tile_config = Some(AmxTileConfig {
                        palette: 1,
                        start_row: 0,
                        rows: 16,
                        bytes_per_row: 64,
                    });
                    // In real implementation, would call LDTILECFG here
                }

                self.state = SimdBackendState::Ready;
                Ok(self.best_backend)
            }
        }
    }

    /// Get current state.
    #[must_use]
    pub fn state(&self) -> SimdBackendState {
        self.state
    }

    /// Get best backend without initializing.
    #[must_use]
    pub fn best_backend(&self) -> ComputeBackend {
        self.best_backend
    }

    /// Check if AMX is supported.
    #[must_use]
    pub fn has_amx(&self) -> bool {
        self.amx_supported
    }

    /// Reset to uninitialized state.
    pub fn reset(&mut self) {
        self.state = SimdBackendState::Uninitialized;
        self.tile_config = None;
    }
}

impl Default for LazySimdConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ----------------------------------------------------------------------------
// LCP-13: Unroll-and-Tail Vectorization
// ----------------------------------------------------------------------------

/// Unroll factor for SIMD loops.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnrollFactor {
    /// No unrolling (1x)
    None,
    /// 2x unroll
    X2,
    /// 4x unroll
    X4,
    /// 8x unroll (AVX-512)
    X8,
}

impl UnrollFactor {
    /// Get numeric factor.
    #[must_use]
    pub fn value(&self) -> usize {
        match self {
            UnrollFactor::None => 1,
            UnrollFactor::X2 => 2,
            UnrollFactor::X4 => 4,
            UnrollFactor::X8 => 8,
        }
    }

    /// Get optimal factor for backend.
    #[must_use]
    pub fn for_backend(backend: ComputeBackend) -> Self {
        match backend {
            ComputeBackend::Avx512 => UnrollFactor::X8,
            ComputeBackend::Avx2 => UnrollFactor::X4,
            ComputeBackend::Sse2 | ComputeBackend::Neon => UnrollFactor::X2,
            _ => UnrollFactor::None,
        }
    }
}

/// Helper for unroll-and-tail loop pattern.
///
/// Processes data in unrolled chunks, then handles the tail.
#[derive(Debug)]
pub struct UnrollTailIterator {
    /// Total elements
    total: usize,
    /// Current position
    position: usize,
    /// Elements per unrolled iteration
    chunk_size: usize,
}

impl UnrollTailIterator {
    /// Create iterator for given size and unroll factor.
    pub fn new(total: usize, factor: UnrollFactor) -> Self {
        Self { total, position: 0, chunk_size: factor.value() }
    }

    /// Get number of full unrolled iterations.
    #[must_use]
    pub fn full_iterations(&self) -> usize {
        self.total / self.chunk_size
    }

    /// Get tail size (remainder).
    #[must_use]
    pub fn tail_size(&self) -> usize {
        self.total % self.chunk_size
    }

    /// Check if there's a tail to process.
    #[must_use]
    pub fn has_tail(&self) -> bool {
        self.tail_size() > 0
    }

    /// Get next chunk range for unrolled iteration.
    pub fn next_chunk(&mut self) -> Option<(usize, usize)> {
        if self.position + self.chunk_size <= self.total {
            let start = self.position;
            self.position += self.chunk_size;
            Some((start, start + self.chunk_size))
        } else {
            None
        }
    }

    /// Get tail range (call after all chunks consumed).
    pub fn tail_range(&self) -> Option<(usize, usize)> {
        let tail_start = self.full_iterations() * self.chunk_size;
        if tail_start < self.total {
            Some((tail_start, self.total))
        } else {
            None
        }
    }
}

/// Process a slice with unroll-and-tail pattern.
///
/// # Example
/// ```ignore
/// let result = unroll_tail_process(
///     &data,
///     UnrollFactor::X4,
///     |chunk| chunk.iter().sum::<f32>(), // Unrolled body
///     |elem| *elem,                       // Tail body
/// );
/// ```
pub fn unroll_tail_process<T, U, F, G>(
    data: &[T],
    factor: UnrollFactor,
    mut process_chunk: F,
    mut process_elem: G,
) -> Vec<U>
where
    F: FnMut(&[T]) -> U,
    G: FnMut(&T) -> U,
{
    let mut iter = UnrollTailIterator::new(data.len(), factor);
    let mut results =
        Vec::with_capacity(iter.full_iterations() + if iter.has_tail() { 1 } else { 0 });

    // Process full chunks
    while let Some((start, end)) = iter.next_chunk() {
        results.push(process_chunk(&data[start..end]));
    }

    // Process tail
    if let Some((start, end)) = iter.tail_range() {
        for elem in &data[start..end] {
            results.push(process_elem(elem));
        }
    }

    results
}

#[cfg(test)]
mod tests;
