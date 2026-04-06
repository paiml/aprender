//! WASM Pixel GUI Demo - GPU-Accelerated Random Fill
//!
//! Demonstrates probar's pixel coverage capabilities with a TUI that
//! randomly fills a 1080p grid using GPU-accelerated computation.
//!
//! Implements PROBAR-SPEC-009 with:
//! - PCG-XSH-RR deterministic RNG
//! - Wilson score confidence intervals
//! - FalsifiabilityGate integration
//! - **Actual GPU acceleration via trueno** (when `gpu` feature enabled)
//!
//! ## References
//!
//! - Nickolls et al. (2008): GPU parallel computing model
//! - O'Neill (2014): PCG random number generation
//! - Wilson (1927): Wilson score interval
//! - W3C (2021): WebGPU specification
//! - Mahajan et al. (2021): Pixel-based visual testing

use super::ConfidenceInterval;
use std::time::Duration;

#[cfg(feature = "gpu")]
use trueno::backends::gpu::GpuDevice;

/// PCG-XSH-RR constants from O'Neill (2014)
const PCG_MULTIPLIER: u32 = 747796405;
const PCG_INCREMENT: u32 = 2891336453;
const PCG_OUTPUT_MUL: u32 = 277803737;

/// Configuration for the WASM pixel demo
#[derive(Debug, Clone)]
pub struct WasmDemoConfig {
    /// Screen width in pixels (default: 1920)
    pub width: u32,
    /// Screen height in pixels (default: 1080)
    pub height: u32,
    /// Probability of filling a pixel per frame (default: 0.01)
    pub fill_probability: f32,
    /// Target coverage percentage (default: 0.99)
    pub target_coverage: f32,
    /// RNG seed for determinism
    pub seed: u64,
    /// Color palette for rendering
    pub palette: DemoPalette,
}

impl Default for WasmDemoConfig {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            fill_probability: 0.01,
            target_coverage: 0.99,
            seed: 42,
            palette: DemoPalette::Viridis,
        }
    }
}

impl WasmDemoConfig {
    /// Create 1080p configuration
    #[must_use]
    pub fn hd_1080p() -> Self {
        Self::default()
    }

    /// Create 720p configuration
    #[must_use]
    pub fn hd_720p() -> Self {
        Self {
            width: 1280,
            height: 720,
            ..Self::default()
        }
    }

    /// Create small test configuration
    #[must_use]
    pub fn test_small() -> Self {
        Self {
            width: 100,
            height: 100,
            fill_probability: 0.1,
            seed: 42,
            ..Self::default()
        }
    }

    /// Set custom seed
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set fill probability
    #[must_use]
    pub fn with_fill_probability(mut self, prob: f32) -> Self {
        self.fill_probability = prob.clamp(0.0, 1.0);
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.width == 0 || self.height == 0 {
            return Err(ConfigError::InvalidDimensions {
                width: self.width,
                height: self.height,
            });
        }
        if !(0.0..=1.0).contains(&self.fill_probability) {
            return Err(ConfigError::InvalidProbability(self.fill_probability));
        }
        if !(0.0..=1.0).contains(&self.target_coverage) {
            return Err(ConfigError::InvalidTargetCoverage(self.target_coverage));
        }
        Ok(())
    }
}

/// Color palette options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DemoPalette {
    /// Viridis - colorblind-safe (default)
    #[default]
    Viridis,
    /// Magma - dark to bright
    Magma,
    /// Heat - traditional heat map
    Heat,
    /// Grayscale
    Grayscale,
}

/// Severity level for gap regions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GapSeverity {
    /// Informational (small gap, <25 pixels)
    Info,
    /// Warning (medium gap, 25-100 pixels)
    Warning,
    /// Critical (large gap, >100 pixels)
    Critical,
}

/// A gap region in the pixel buffer
#[derive(Debug, Clone)]
pub struct DemoGapRegion {
    /// X position
    pub x: usize,
    /// Y position
    pub y: usize,
    /// Width
    pub width: usize,
    /// Height
    pub height: usize,
    /// Total pixel count
    pub size: usize,
    /// Severity level
    pub severity: GapSeverity,
}

/// Configuration errors
#[derive(Debug, Clone, PartialEq)]
pub enum ConfigError {
    /// Invalid screen dimensions (width and height must be > 0)
    InvalidDimensions {
        /// Width value
        width: u32,
        /// Height value
        height: u32,
    },
    /// Invalid fill probability
    InvalidProbability(f32),
    /// Invalid target coverage
    InvalidTargetCoverage(f32),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidDimensions { width, height } => {
                write!(f, "Invalid dimensions: {}x{} (must be > 0)", width, height)
            }
            Self::InvalidProbability(p) => {
                write!(f, "Invalid probability: {} (must be 0.0..=1.0)", p)
            }
            Self::InvalidTargetCoverage(c) => {
                write!(f, "Invalid target coverage: {} (must be 0.0..=1.0)", c)
            }
        }
    }
}

impl std::error::Error for ConfigError {}

/// PCG-XSH-RR random number generator (O'Neill, 2014)
///
/// Provides deterministic, high-quality random numbers with minimal state.
#[derive(Debug, Clone)]
pub struct PcgRng {
    state: u64,
    increment: u64,
}

impl PcgRng {
    /// Create new RNG with seed
    #[must_use]
    pub fn new(seed: u64) -> Self {
        let mut rng = Self {
            state: 0,
            increment: (seed << 1) | 1, // Must be odd
        };
        // Warm up state
        let _ = rng.next_u32();
        rng.state = rng.state.wrapping_add(seed);
        let _ = rng.next_u32();
        rng
    }

    /// Generate next 32-bit random value
    #[must_use]
    pub fn next_u32(&mut self) -> u32 {
        let old_state = self.state;

        // Advance state using LCG
        self.state = old_state
            .wrapping_mul(u64::from(PCG_MULTIPLIER))
            .wrapping_add(self.increment);

        // XSH-RR output function
        let xorshifted = ((old_state >> 18) ^ old_state) >> 27;
        let rot = (old_state >> 59) as u32;

        #[allow(clippy::cast_possible_truncation)]
        let result = xorshifted as u32;
        result.rotate_right(rot)
    }

    /// Generate random float in [0, 1)
    #[must_use]
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f64 / u32::MAX as f64) as f32
    }

    /// Generate hash for pixel index (pure function for GPU)
    #[must_use]
    pub fn hash_pixel(seed: u32, index: u32, frame: u32) -> u32 {
        let input = seed ^ index ^ frame.wrapping_mul(12345);
        let state = input
            .wrapping_mul(PCG_MULTIPLIER)
            .wrapping_add(PCG_INCREMENT);
        let word =
            ((state >> ((state >> 28).wrapping_add(4))) ^ state).wrapping_mul(PCG_OUTPUT_MUL);
        (word >> 22) ^ word
    }

    /// Check if pixel should be filled based on probability
    #[must_use]
    pub fn should_fill(seed: u32, index: u32, frame: u32, probability: f32) -> bool {
        let hash = Self::hash_pixel(seed, index, frame);
        let random_value = hash as f32 / u32::MAX as f32;
        random_value < probability
    }
}

/// GPU pixel buffer for demo rendering
///
/// When `gpu` feature is enabled, this uses actual GPU compute via trueno/wgpu.
/// Otherwise falls back to CPU implementation.
#[derive(Debug, Clone)]
pub struct GpuPixelBuffer {
    /// Pixel data (0.0 = uncovered, 1.0 = fully covered)
    pub pixels: Vec<f32>,
    /// Buffer width
    pub width: u32,
    /// Buffer height
    pub height: u32,
    /// Current frame number
    pub frame: u32,
    /// RNG seed
    pub seed: u32,
    /// Whether GPU is being used
    pub using_gpu: bool,
}

/// GPU backend for actual hardware acceleration
#[cfg(feature = "gpu")]
pub struct GpuAccelerator {
    device: GpuDevice,
}

#[cfg(feature = "gpu")]
impl GpuAccelerator {
    /// Create new GPU accelerator
    pub fn new() -> Result<Self, String> {
        let device = GpuDevice::new()?;
        Ok(Self { device })
    }

    /// Check if GPU is available
    #[allow(dead_code)]
    pub fn is_available() -> bool {
        GpuDevice::is_available()
    }

    /// Execute parallel fill on GPU using wgpu compute
    ///
    /// Uses trueno's GPU backend (wgpu/Vulkan) for actual hardware acceleration.
    /// The RNG is computed on CPU but the comparison and assignment can be
    /// parallelized via GPU vector operations.
    pub fn parallel_fill(
        &self,
        pixels: &mut [f32],
        width: u32,
        height: u32,
        frame: u32,
        seed: u32,
        probability: f32,
    ) -> Result<(), String> {
        let total = pixels.len();

        // Generate random values (this computation is embarrassingly parallel)
        let random_values: Vec<f32> = (0..total)
            .map(|idx| {
                let hash = PcgRng::hash_pixel(seed, idx as u32, frame);
                hash as f32 / u32::MAX as f32
            })
            .collect();

        // Generate gradient values for positions
        let gradient_values: Vec<f32> = (0..total)
            .map(|idx| {
                let x = idx as u32 % width;
                let y = idx as u32 / width;
                ((x + y) as f32 / (width + height) as f32).max(0.001)
            })
            .collect();

        // Use GPU for vector operations when buffer is large enough
        // trueno's GPU backend works best for operations on >100K elements
        if total > 100_000 {
            // GPU-accelerated threshold comparison using trueno's sigmoid
            // (sigmoid of large negative = ~0, sigmoid of large positive = ~1)
            // This demonstrates actual GPU compute dispatch
            let scaled: Vec<f32> = random_values
                .iter()
                .map(|&r| (probability - r) * 100.0) // Scale difference for sigmoid
                .collect();

            let mut mask = vec![0.0f32; total];
            // Use GPU sigmoid as a soft threshold function
            self.device.sigmoid(&scaled, &mut mask)?;

            // Apply mask to update pixels
            for (idx, pixel) in pixels.iter_mut().enumerate() {
                if *pixel == 0.0 && mask[idx] > 0.5 {
                    *pixel = gradient_values[idx];
                }
            }
        } else {
            // For smaller buffers, CPU is faster due to transfer overhead
            for (idx, pixel) in pixels.iter_mut().enumerate() {
                if *pixel == 0.0 && random_values[idx] < probability {
                    *pixel = gradient_values[idx];
                }
            }
        }

        Ok(())
    }
}

impl GpuPixelBuffer {
    /// Create new pixel buffer (tries GPU first, falls back to CPU)
    #[must_use]
    pub fn new(width: u32, height: u32, seed: u64) -> Self {
        let total_pixels = (width as usize) * (height as usize);
        let using_gpu = Self::gpu_available();

        Self {
            pixels: vec![0.0; total_pixels],
            width,
            height,
            frame: 0,
            seed: (seed & 0xFFFF_FFFF) as u32,
            using_gpu,
        }
    }

    /// Check if GPU acceleration is available
    #[must_use]
    pub fn gpu_available() -> bool {
        #[cfg(feature = "gpu")]
        {
            GpuDevice::is_available()
        }
        #[cfg(not(feature = "gpu"))]
        {
            false
        }
    }

    /// Get GPU device name if available
    #[must_use]
    pub fn gpu_device_name() -> Option<String> {
        #[cfg(feature = "gpu")]
        {
            if GpuDevice::is_available() {
                // trueno uses wgpu which auto-detects best GPU
                // Return generic name since adapter_info is internal
                Some("wgpu (Vulkan/Metal/DX12)".to_string())
            } else {
                None
            }
        }
        #[cfg(not(feature = "gpu"))]
        {
            None
        }
    }

    /// Create 1080p buffer
    #[must_use]
    pub fn new_1080p() -> Self {
        Self::new(1920, 1080, 42)
    }

    /// Create 720p buffer
    #[must_use]
    pub fn new_720p() -> Self {
        Self::new(1280, 720, 42)
    }

    /// Total number of pixels
    #[must_use]
    pub fn total_pixels(&self) -> usize {
        self.pixels.len()
    }

    /// Execute random fill pass (GPU-accelerated when available)
    pub fn random_fill_pass(&mut self, probability: f32) {
        self.frame += 1;

        #[cfg(feature = "gpu")]
        {
            if self.using_gpu {
                if let Ok(accelerator) = GpuAccelerator::new() {
                    if accelerator
                        .parallel_fill(
                            &mut self.pixels,
                            self.width,
                            self.height,
                            self.frame,
                            self.seed,
                            probability,
                        )
                        .is_ok()
                    {
                        return;
                    }
                }
            }
        }

        // CPU fallback
        self.random_fill_pass_cpu(probability);
    }

    /// CPU-only random fill pass (fallback)
    fn random_fill_pass_cpu(&mut self, probability: f32) {
        let frame = self.frame;
        let seed = self.seed;

        for (idx, pixel) in self.pixels.iter_mut().enumerate() {
            if *pixel == 0.0 && PcgRng::should_fill(seed, idx as u32, frame, probability) {
                // Calculate color based on position (viridis-like gradient)
                let x = idx as u32 % self.width;
                let y = idx as u32 / self.width;
                let normalized = (x + y) as f32 / (self.width + self.height) as f32;
                *pixel = normalized.max(0.001); // Minimum non-zero for "covered"
            }
        }
    }

    /// Run multiple fill passes until target coverage
    pub fn fill_to_coverage(&mut self, target: f32, probability: f32, max_frames: u32) {
        for _ in 0..max_frames {
            self.random_fill_pass(probability);
            if self.coverage_percentage() >= target {
                break;
            }
        }
    }

    /// Calculate coverage statistics
    #[must_use]
    pub fn coverage_stats(&self) -> CoverageStats {
        let covered = self.pixels.iter().filter(|&&v| v > 0.0).count();
        let total = self.pixels.len();
        let percentage = covered as f32 / total as f32;

        CoverageStats {
            covered,
            total,
            percentage,
            wilson_ci: wilson_confidence_interval(covered, total, 0.95),
            gaps: self.find_gaps(),
        }
    }

    /// Get coverage percentage
    #[must_use]
    pub fn coverage_percentage(&self) -> f32 {
        let covered = self.pixels.iter().filter(|&&v| v > 0.0).count();
        covered as f32 / self.pixels.len() as f32
    }

    /// Find gap regions (contiguous uncovered areas)
    #[must_use]
    pub fn find_gaps(&self) -> Vec<DemoGapRegion> {
        let mut gaps = Vec::new();
        let mut visited = vec![false; self.pixels.len()];

        for y in 0..self.height {
            for x in 0..self.width {
                let idx = (y * self.width + x) as usize;
                if self.pixels[idx] == 0.0 && !visited[idx] {
                    // Found start of gap - flood fill to find extent
                    let gap = self.flood_fill_gap(x, y, &mut visited);
                    if gap.size > 0 {
                        gaps.push(gap);
                    }
                }
            }
        }

        gaps
    }

    /// Flood fill to find connected gap region
    fn flood_fill_gap(&self, start_x: u32, start_y: u32, visited: &mut [bool]) -> DemoGapRegion {
        let mut stack = vec![(start_x, start_y)];
        let mut min_x = start_x;
        let mut max_x = start_x;
        let mut min_y = start_y;
        let mut max_y = start_y;
        let mut size = 0;

        while let Some((x, y)) = stack.pop() {
            if x >= self.width || y >= self.height {
                continue;
            }
            let idx = (y * self.width + x) as usize;
            if visited[idx] || self.pixels[idx] > 0.0 {
                continue;
            }

            visited[idx] = true;
            size += 1;
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);

            // Add neighbors (4-connected)
            if x > 0 {
                stack.push((x - 1, y));
            }
            if x < self.width - 1 {
                stack.push((x + 1, y));
            }
            if y > 0 {
                stack.push((x, y - 1));
            }
            if y < self.height - 1 {
                stack.push((x, y + 1));
            }
        }

        DemoGapRegion {
            x: min_x as usize,
            y: min_y as usize,
            width: (max_x - min_x + 1) as usize,
            height: (max_y - min_y + 1) as usize,
            size,
            severity: if size > 100 {
                GapSeverity::Critical
            } else if size > 25 {
                GapSeverity::Warning
            } else {
                GapSeverity::Info
            },
        }
    }

    /// Downsample to terminal resolution
    #[must_use]
    pub fn downsample(&self, term_width: usize, term_height: usize) -> Vec<f32> {
        let scale_x = self.width as usize / term_width.max(1);
        let scale_y = self.height as usize / term_height.max(1);
        let mut result = vec![0.0; term_width * term_height];

        for ty in 0..term_height {
            for tx in 0..term_width {
                // Average pixels in this cell
                let mut sum = 0.0;
                let mut count = 0;

                for py in 0..scale_y {
                    for px in 0..scale_x {
                        let src_x = tx * scale_x + px;
                        let src_y = ty * scale_y + py;
                        if src_x < self.width as usize && src_y < self.height as usize {
                            let idx = src_y * self.width as usize + src_x;
                            sum += self.pixels[idx];
                            count += 1;
                        }
                    }
                }

                if count > 0 {
                    result[ty * term_width + tx] = sum / count as f32;
                }
            }
        }

        result
    }

    /// Reset buffer
    pub fn reset(&mut self) {
        self.pixels.fill(0.0);
        self.frame = 0;
    }

    /// Check if this buffer is using GPU acceleration
    #[must_use]
    pub fn is_using_gpu(&self) -> bool {
        self.using_gpu
    }
}

/// Coverage statistics with Wilson CI
#[derive(Debug, Clone)]
pub struct CoverageStats {
    /// Number of covered pixels
    pub covered: usize,
    /// Total number of pixels
    pub total: usize,
    /// Coverage percentage (0.0 - 1.0)
    pub percentage: f32,
    /// Wilson confidence interval
    pub wilson_ci: ConfidenceInterval,
    /// Gap regions
    pub gaps: Vec<DemoGapRegion>,
}

impl CoverageStats {
    /// Check if coverage meets threshold
    #[must_use]
    pub fn meets_threshold(&self, threshold: f32) -> bool {
        self.percentage >= threshold
    }

    /// Get largest gap size
    #[must_use]
    pub fn max_gap_size(&self) -> usize {
        self.gaps.iter().map(|g| g.size).max().unwrap_or(0)
    }
}

/// Calculate Wilson score confidence interval (Wilson, 1927)
///
/// Provides better coverage for small samples than normal approximation.
#[must_use]
pub fn wilson_confidence_interval(
    successes: usize,
    total: usize,
    confidence: f32,
) -> ConfidenceInterval {
    if total == 0 {
        return ConfidenceInterval {
            lower: 0.0,
            upper: 0.0,
            level: confidence,
        };
    }

    let n = total as f32;
    let p = successes as f32 / n;

    // Z-score for confidence level (95% = 1.96)
    let z: f32 = if (confidence - 0.90).abs() < 0.01 {
        1.645
    } else if (confidence - 0.95).abs() < 0.01 {
        1.96
    } else if (confidence - 0.99).abs() < 0.01 {
        2.576
    } else {
        1.96
    };

    let z2 = z * z;
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let margin = (z / denominator) * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt());

    ConfidenceInterval {
        lower: (center - margin).max(0.0),
        upper: (center + margin).min(1.0),
        level: confidence,
    }
}

/// Demo state for TUI rendering
#[derive(Debug)]
pub struct WasmPixelDemo {
    /// GPU pixel buffer
    pub buffer: GpuPixelBuffer,
    /// Configuration
    pub config: WasmDemoConfig,
    /// Start time for measuring convergence
    pub start_time: std::time::Instant,
    /// Whether demo is complete
    pub complete: bool,
}

impl WasmPixelDemo {
    /// Create new demo with configuration
    #[must_use]
    pub fn new(config: WasmDemoConfig) -> Self {
        Self {
            buffer: GpuPixelBuffer::new(config.width, config.height, config.seed),
            config,
            start_time: std::time::Instant::now(),
            complete: false,
        }
    }

    /// Create 1080p demo
    #[must_use]
    pub fn hd_1080p() -> Self {
        Self::new(WasmDemoConfig::hd_1080p())
    }

    /// Execute one frame
    pub fn tick(&mut self) {
        if self.complete {
            return;
        }

        self.buffer.random_fill_pass(self.config.fill_probability);

        if self.buffer.coverage_percentage() >= self.config.target_coverage {
            self.complete = true;
        }
    }

    /// Get current stats
    #[must_use]
    pub fn stats(&self) -> CoverageStats {
        self.buffer.coverage_stats()
    }

    /// Get elapsed time
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Check if demo is complete
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.complete
    }

    /// Get frame count
    #[must_use]
    pub fn frame_count(&self) -> u32 {
        self.buffer.frame
    }

    /// Reset demo
    pub fn reset(&mut self) {
        self.buffer.reset();
        self.start_time = std::time::Instant::now();
        self.complete = false;
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
mod tests {
    use super::*;

    // =========================================================================
    // Section 1: Configuration Tests (QA 6-10)
    // =========================================================================

    #[test]
    fn h0_demo_01_config_default() {
        let config = WasmDemoConfig::default();
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
        assert!((config.fill_probability - 0.01).abs() < f32::EPSILON);
    }

    #[test]
    fn h0_demo_02_config_validation_valid() {
        let config = WasmDemoConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn h0_demo_03_config_validation_zero_width() {
        let config = WasmDemoConfig {
            width: 0,
            ..Default::default()
        };
        assert!(matches!(
            config.validate(),
            Err(ConfigError::InvalidDimensions { .. })
        ));
    }

    #[test]
    fn h0_demo_04_config_validation_invalid_probability() {
        let config = WasmDemoConfig {
            fill_probability: -0.5,
            ..Default::default()
        };
        assert!(matches!(
            config.validate(),
            Err(ConfigError::InvalidProbability(_))
        ));
    }

    #[test]
    fn h0_demo_05_config_validation_probability_over_1() {
        let config = WasmDemoConfig {
            fill_probability: 1.5,
            ..Default::default()
        };
        assert!(matches!(
            config.validate(),
            Err(ConfigError::InvalidProbability(_))
        ));
    }

    // =========================================================================
    // Section 2: PCG RNG Tests (QA 21-30)
    // =========================================================================

    #[test]
    fn h0_rng_01_determinism_same_seed() {
        let mut rng1 = PcgRng::new(42);
        let mut rng2 = PcgRng::new(42);

        for _ in 0..100 {
            assert_eq!(rng1.next_u32(), rng2.next_u32());
        }
    }

    #[test]
    fn h0_rng_02_determinism_different_seeds() {
        let mut rng1 = PcgRng::new(42);
        let mut rng2 = PcgRng::new(123);

        // At least some values should differ
        let mut any_different = false;
        for _ in 0..100 {
            if rng1.next_u32() != rng2.next_u32() {
                any_different = true;
                break;
            }
        }
        assert!(
            any_different,
            "Different seeds should produce different sequences"
        );
    }

    #[test]
    fn h0_rng_03_pixel_hash_determinism() {
        let hash1 = PcgRng::hash_pixel(42, 1000, 5);
        let hash2 = PcgRng::hash_pixel(42, 1000, 5);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn h0_rng_04_pixel_hash_frame_dependency() {
        let hash_frame_0 = PcgRng::hash_pixel(42, 1000, 0);
        let hash_frame_1 = PcgRng::hash_pixel(42, 1000, 1);
        assert_ne!(hash_frame_0, hash_frame_1);
    }

    #[test]
    fn h0_rng_05_should_fill_zero_probability() {
        // With 0.0 probability, should never fill
        for idx in 0..1000 {
            assert!(!PcgRng::should_fill(42, idx, 1, 0.0));
        }
    }

    #[test]
    fn h0_rng_06_should_fill_full_probability() {
        // With 1.0 probability, should always fill
        for idx in 0..1000 {
            assert!(PcgRng::should_fill(42, idx, 1, 1.0));
        }
    }

    #[test]
    fn h0_rng_07_float_range() {
        let mut rng = PcgRng::new(42);
        for _ in 0..1000 {
            let f = rng.next_f32();
            assert!((0.0..1.0).contains(&f));
        }
    }

    #[test]
    fn h0_rng_08_zero_seed_works() {
        // PCG should handle zero seed correctly (unlike some simple LCGs)
        let mut rng = PcgRng::new(0);
        let val1 = rng.next_u32();
        let val2 = rng.next_u32();
        assert_ne!(val1, val2, "Zero seed should still produce varying output");
    }

    // =========================================================================
    // Section 3: GPU Buffer Tests (QA 11-20)
    // =========================================================================

    #[test]
    fn h0_buffer_01_creation_1080p() {
        let buffer = GpuPixelBuffer::new_1080p();
        assert_eq!(buffer.width, 1920);
        assert_eq!(buffer.height, 1080);
        assert_eq!(buffer.total_pixels(), 1920 * 1080);
    }

    #[test]
    fn h0_buffer_02_creation_720p() {
        let buffer = GpuPixelBuffer::new_720p();
        assert_eq!(buffer.width, 1280);
        assert_eq!(buffer.height, 720);
    }

    #[test]
    fn h0_buffer_03_initial_zero_coverage() {
        let buffer = GpuPixelBuffer::new(100, 100, 42);
        let stats = buffer.coverage_stats();
        assert_eq!(stats.covered, 0);
        assert_eq!(stats.percentage, 0.0);
    }

    #[test]
    fn h0_buffer_04_random_fill_increases_coverage() {
        let mut buffer = GpuPixelBuffer::new(100, 100, 42);
        buffer.random_fill_pass(0.1);
        assert!(buffer.coverage_percentage() > 0.0);
    }

    #[test]
    fn h0_buffer_05_fill_convergence() {
        let mut buffer = GpuPixelBuffer::new(50, 50, 42);
        buffer.fill_to_coverage(0.99, 0.1, 1000);
        assert!(
            buffer.coverage_percentage() >= 0.99,
            "Should converge to 99%+ coverage"
        );
    }

    #[test]
    fn h0_buffer_06_deterministic_fill() {
        let mut buffer1 = GpuPixelBuffer::new(50, 50, 42);
        let mut buffer2 = GpuPixelBuffer::new(50, 50, 42);

        for _ in 0..10 {
            buffer1.random_fill_pass(0.1);
            buffer2.random_fill_pass(0.1);
        }

        assert_eq!(
            buffer1.pixels, buffer2.pixels,
            "Same seed should produce same pattern"
        );
    }

    #[test]
    fn h0_buffer_07_reset_clears() {
        let mut buffer = GpuPixelBuffer::new(50, 50, 42);
        buffer.random_fill_pass(0.5);
        assert!(buffer.coverage_percentage() > 0.0);

        buffer.reset();
        assert_eq!(buffer.coverage_percentage(), 0.0);
        assert_eq!(buffer.frame, 0);
    }

    // =========================================================================
    // Section 4: Coverage Statistics Tests (QA 41-50)
    // =========================================================================

    #[test]
    fn h0_stats_01_wilson_ci_bounds() {
        let ci = wilson_confidence_interval(50, 100, 0.95);
        assert!(ci.lower <= 0.50);
        assert!(ci.upper >= 0.50);
        assert!(ci.lower >= 0.0);
        assert!(ci.upper <= 1.0);
    }

    #[test]
    fn h0_stats_02_wilson_ci_empty() {
        let ci = wilson_confidence_interval(0, 0, 0.95);
        assert_eq!(ci.lower, 0.0);
        assert_eq!(ci.upper, 0.0);
    }

    #[test]
    fn h0_stats_03_wilson_ci_zero_coverage() {
        let ci = wilson_confidence_interval(0, 100, 0.95);
        assert!(ci.lower == 0.0);
        assert!(ci.upper > 0.0);
    }

    #[test]
    fn h0_stats_04_wilson_ci_full_coverage() {
        let ci = wilson_confidence_interval(100, 100, 0.95);
        assert!(ci.lower < 1.0);
        // Upper bound should be very close to 1.0 (clamped)
        assert!((ci.upper - 1.0).abs() < 0.001);
    }

    #[test]
    fn h0_stats_05_wilson_ci_narrows_with_samples() {
        let ci_small = wilson_confidence_interval(5, 10, 0.95);
        let ci_large = wilson_confidence_interval(500, 1000, 0.95);

        let width_small = ci_small.upper - ci_small.lower;
        let width_large = ci_large.upper - ci_large.lower;

        assert!(
            width_large < width_small,
            "CI should narrow with more samples"
        );
    }

    #[test]
    fn h0_stats_06_coverage_meets_threshold() {
        let mut buffer = GpuPixelBuffer::new(50, 50, 42);
        buffer.fill_to_coverage(0.8, 0.1, 500);
        let stats = buffer.coverage_stats();
        assert!(stats.meets_threshold(0.8));
    }

    // =========================================================================
    // Section 5: Gap Detection Tests (QA 51-60)
    // =========================================================================

    #[test]
    fn h0_gap_01_empty_buffer_is_one_gap() {
        let buffer = GpuPixelBuffer::new(10, 10, 42);
        let gaps = buffer.find_gaps();
        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0].size, 100);
    }

    #[test]
    fn h0_gap_02_full_buffer_no_gaps() {
        let mut buffer = GpuPixelBuffer::new(10, 10, 42);
        // Fill all pixels
        for pixel in &mut buffer.pixels {
            *pixel = 1.0;
        }
        let gaps = buffer.find_gaps();
        assert!(gaps.is_empty());
    }

    #[test]
    fn h0_gap_03_max_gap_size() {
        let buffer = GpuPixelBuffer::new(10, 10, 42);
        let stats = buffer.coverage_stats();
        assert_eq!(stats.max_gap_size(), 100);
    }

    // =========================================================================
    // Section 6: Demo Lifecycle Tests (QA 61-70)
    // =========================================================================

    #[test]
    fn h0_demo_lifecycle_01_creation() {
        let demo = WasmPixelDemo::new(WasmDemoConfig::test_small());
        assert!(!demo.is_complete());
        assert_eq!(demo.frame_count(), 0);
    }

    #[test]
    fn h0_demo_lifecycle_02_tick_advances() {
        let mut demo = WasmPixelDemo::new(WasmDemoConfig::test_small());
        demo.tick();
        assert_eq!(demo.frame_count(), 1);
    }

    #[test]
    fn h0_demo_lifecycle_03_completes_on_target() {
        let config = WasmDemoConfig {
            width: 10,
            height: 10,
            fill_probability: 0.5,
            target_coverage: 0.5,
            ..Default::default()
        };
        let mut demo = WasmPixelDemo::new(config);

        // Run until complete or max frames
        for _ in 0..1000 {
            demo.tick();
            if demo.is_complete() {
                break;
            }
        }

        assert!(demo.is_complete());
    }

    #[test]
    fn h0_demo_lifecycle_04_reset() {
        let mut demo = WasmPixelDemo::new(WasmDemoConfig::test_small());
        demo.tick();
        demo.tick();

        demo.reset();
        assert!(!demo.is_complete());
        assert_eq!(demo.frame_count(), 0);
        assert_eq!(demo.buffer.coverage_percentage(), 0.0);
    }

    // =========================================================================
    // Section 7: Downsampling Tests (QA 31-40)
    // =========================================================================

    #[test]
    fn h0_downsample_01_correct_size() {
        let buffer = GpuPixelBuffer::new(100, 100, 42);
        let downsampled = buffer.downsample(10, 10);
        assert_eq!(downsampled.len(), 100);
    }

    #[test]
    fn h0_downsample_02_preserves_coverage_ratio() {
        let mut buffer = GpuPixelBuffer::new(100, 100, 42);
        buffer.random_fill_pass(1.0); // Fill all

        let downsampled = buffer.downsample(10, 10);
        let ds_covered = downsampled.iter().filter(|&&v| v > 0.0).count();

        // All cells should be covered
        assert_eq!(ds_covered, 100);
    }

    #[test]
    fn h0_downsample_03_handles_zero_terminal() {
        let buffer = GpuPixelBuffer::new(100, 100, 42);
        let downsampled = buffer.downsample(0, 0);
        assert!(downsampled.is_empty());
    }

    // =========================================================================
    // Section 8: Palette Tests
    // =========================================================================

    #[test]
    fn h0_palette_01_default_is_viridis() {
        let config = WasmDemoConfig::default();
        assert_eq!(config.palette, DemoPalette::Viridis);
    }

    // =========================================================================
    // Section 9: Performance Regression Tests (QA 61-70)
    // =========================================================================

    #[test]
    fn h0_perf_01_1080p_creation_fast() {
        let start = std::time::Instant::now();
        let _buffer = GpuPixelBuffer::new_1080p();
        let elapsed = start.elapsed();

        // Should create in under 5s (generous for loaded systems)
        assert!(
            elapsed.as_secs() < 5,
            "1080p buffer creation took {:?}",
            elapsed
        );
    }

    #[test]
    fn h0_perf_02_fill_pass_reasonable_time() {
        let mut buffer = GpuPixelBuffer::new(100, 100, 42);

        let start = std::time::Instant::now();
        for _ in 0..100 {
            buffer.random_fill_pass(0.01);
        }
        let elapsed = start.elapsed();

        // 100 frames on 10k pixels - generous for loaded systems
        assert!(elapsed.as_secs() < 30, "100 fill passes took {:?}", elapsed);
    }

    // =========================================================================
    // Section 10: Error Handling Tests (QA 81-90)
    // =========================================================================

    #[test]
    fn h0_error_01_config_error_display() {
        let err = ConfigError::InvalidDimensions {
            width: 0,
            height: 100,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Invalid dimensions"));
    }

    #[test]
    fn h0_error_02_probability_clamping() {
        let config = WasmDemoConfig::default().with_fill_probability(1.5);
        assert_eq!(config.fill_probability, 1.0);

        let config = WasmDemoConfig::default().with_fill_probability(-0.5);
        assert_eq!(config.fill_probability, 0.0);
    }

    // =========================================================================
    // Section 11: Additional Configuration Tests (Coverage Boost)
    // =========================================================================

    #[test]
    fn h0_config_06_hd_720p() {
        let config = WasmDemoConfig::hd_720p();
        assert_eq!(config.width, 1280);
        assert_eq!(config.height, 720);
        // Should inherit defaults for other fields
        assert!((config.fill_probability - 0.01).abs() < f32::EPSILON);
        assert_eq!(config.palette, DemoPalette::Viridis);
    }

    #[test]
    fn h0_config_07_test_small() {
        let config = WasmDemoConfig::test_small();
        assert_eq!(config.width, 100);
        assert_eq!(config.height, 100);
        assert!((config.fill_probability - 0.1).abs() < f32::EPSILON);
        assert_eq!(config.seed, 42);
    }

    #[test]
    fn h0_config_08_with_seed() {
        let config = WasmDemoConfig::default().with_seed(12345);
        assert_eq!(config.seed, 12345);
        // Should preserve other defaults
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
    }

    #[test]
    fn h0_config_09_validation_zero_height() {
        let config = WasmDemoConfig {
            height: 0,
            ..Default::default()
        };
        let err = config.validate().unwrap_err();
        assert!(matches!(
            err,
            ConfigError::InvalidDimensions {
                width: 1920,
                height: 0
            }
        ));
    }

    #[test]
    fn h0_config_10_validation_invalid_target_coverage_low() {
        let config = WasmDemoConfig {
            target_coverage: -0.5,
            ..Default::default()
        };
        assert!(matches!(
            config.validate(),
            Err(ConfigError::InvalidTargetCoverage(_))
        ));
    }

    #[test]
    fn h0_config_11_validation_invalid_target_coverage_high() {
        let config = WasmDemoConfig {
            target_coverage: 1.5,
            ..Default::default()
        };
        assert!(matches!(
            config.validate(),
            Err(ConfigError::InvalidTargetCoverage(_))
        ));
    }

    // =========================================================================
    // Section 12: ConfigError Display Tests (Coverage Boost)
    // =========================================================================

    #[test]
    fn h0_error_03_probability_display() {
        let err = ConfigError::InvalidProbability(1.5);
        let msg = format!("{}", err);
        assert!(msg.contains("Invalid probability"));
        assert!(msg.contains("1.5"));
        assert!(msg.contains("0.0..=1.0"));
    }

    #[test]
    fn h0_error_04_target_coverage_display() {
        let err = ConfigError::InvalidTargetCoverage(-0.5);
        let msg = format!("{}", err);
        assert!(msg.contains("Invalid target coverage"));
        assert!(msg.contains("-0.5"));
    }

    #[test]
    fn h0_error_05_config_error_is_error_trait() {
        let err: Box<dyn std::error::Error> = Box::new(ConfigError::InvalidProbability(1.5));
        // Verify it implements std::error::Error
        let _ = err.source(); // Should return None (default impl)
        let _ = format!("{}", err);
    }

    #[test]
    fn h0_error_06_config_error_clone_eq() {
        let err1 = ConfigError::InvalidProbability(1.5);
        let err2 = err1.clone();
        assert_eq!(err1, err2);

        let err3 = ConfigError::InvalidDimensions {
            width: 0,
            height: 100,
        };
        let err4 = err3.clone();
        assert_eq!(err3, err4);
    }

    // =========================================================================
    // Section 13: DemoPalette Tests (Coverage Boost)
    // =========================================================================

    #[test]
    fn h0_palette_02_all_variants() {
        let palettes = [
            DemoPalette::Viridis,
            DemoPalette::Magma,
            DemoPalette::Heat,
            DemoPalette::Grayscale,
        ];

        for palette in &palettes {
            // Test Debug
            let debug = format!("{:?}", palette);
            assert!(!debug.is_empty());

            // Test Clone
            let cloned = *palette;
            assert_eq!(*palette, cloned);
        }
    }

    #[test]
    fn h0_palette_03_equality() {
        assert_eq!(DemoPalette::Magma, DemoPalette::Magma);
        assert_ne!(DemoPalette::Magma, DemoPalette::Heat);
        assert_ne!(DemoPalette::Viridis, DemoPalette::Grayscale);
    }

    // =========================================================================
    // Section 14: GapSeverity Tests (Coverage Boost)
    // =========================================================================

    #[test]
    fn h0_gap_04_severity_info() {
        // Gaps < 25 pixels should be Info
        let mut buffer = GpuPixelBuffer::new(10, 10, 42);
        // Fill most pixels, leaving a small gap of ~10 pixels
        for (idx, pixel) in buffer.pixels.iter_mut().enumerate() {
            if idx >= 10 {
                *pixel = 1.0;
            }
        }

        let gaps = buffer.find_gaps();
        assert!(!gaps.is_empty());
        let small_gap = &gaps[0];
        assert!(small_gap.size < 25);
        assert_eq!(small_gap.severity, GapSeverity::Info);
    }

    #[test]
    fn h0_gap_05_severity_warning() {
        // Gaps 25-100 pixels should be Warning
        let mut buffer = GpuPixelBuffer::new(20, 20, 42);
        // Fill all but 50 pixels (5x10 region)
        for y in 0..20 {
            for x in 0..20 {
                let idx = y * 20 + x;
                if x >= 10 || y >= 5 {
                    buffer.pixels[idx] = 1.0;
                }
            }
        }

        let gaps = buffer.find_gaps();
        assert!(!gaps.is_empty());
        let medium_gap = &gaps[0];
        assert!(medium_gap.size >= 25 && medium_gap.size <= 100);
        assert_eq!(medium_gap.severity, GapSeverity::Warning);
    }

    #[test]
    fn h0_gap_06_severity_critical() {
        // Gaps > 100 pixels should be Critical
        let buffer = GpuPixelBuffer::new(20, 20, 42);
        // All 400 pixels are uncovered
        let gaps = buffer.find_gaps();
        assert_eq!(gaps.len(), 1);
        assert!(gaps[0].size > 100);
        assert_eq!(gaps[0].severity, GapSeverity::Critical);
    }

    #[test]
    fn h0_gap_07_severity_variants() {
        // Test all severity variants for Debug, Clone, Eq
        let severities = [
            GapSeverity::Info,
            GapSeverity::Warning,
            GapSeverity::Critical,
        ];

        for sev in &severities {
            let debug = format!("{:?}", sev);
            assert!(!debug.is_empty());

            let cloned = *sev;
            assert_eq!(*sev, cloned);
        }
    }

    // =========================================================================
    // Section 15: DemoGapRegion Tests (Coverage Boost)
    // =========================================================================

    #[test]
    fn h0_gap_08_region_fields() {
        let region = DemoGapRegion {
            x: 10,
            y: 20,
            width: 30,
            height: 40,
            size: 150,
            severity: GapSeverity::Critical,
        };

        assert_eq!(region.x, 10);
        assert_eq!(region.y, 20);
        assert_eq!(region.width, 30);
        assert_eq!(region.height, 40);
        assert_eq!(region.size, 150);
        assert_eq!(region.severity, GapSeverity::Critical);

        // Test Debug and Clone
        let debug = format!("{:?}", region);
        assert!(debug.contains("DemoGapRegion"));

        let cloned = region.clone();
        assert_eq!(cloned.size, region.size);
    }

    // =========================================================================
    // Section 16: GpuPixelBuffer Additional Tests (Coverage Boost)
    // =========================================================================

    #[test]
    fn h0_buffer_08_gpu_device_name() {
        // GPU device name returns None when gpu feature is disabled
        let name = GpuPixelBuffer::gpu_device_name();
        // Just verify it doesn't panic - result depends on feature flags
        let _ = format!("{:?}", name);
    }

    #[test]
    fn h0_buffer_09_is_using_gpu() {
        let buffer = GpuPixelBuffer::new(100, 100, 42);
        let using_gpu = buffer.is_using_gpu();
        // Without gpu feature, should be false
        #[cfg(not(feature = "gpu"))]
        assert!(!using_gpu);
        // Just verify it returns a valid bool
        let _ = format!("{}", using_gpu);
    }

    #[test]
    fn h0_buffer_10_gpu_available() {
        let available = GpuPixelBuffer::gpu_available();
        // Without gpu feature, should be false
        #[cfg(not(feature = "gpu"))]
        assert!(!available);
        let _ = format!("{}", available);
    }

    #[test]
    fn h0_buffer_11_custom_dimensions() {
        let buffer = GpuPixelBuffer::new(123, 456, 789);
        assert_eq!(buffer.width, 123);
        assert_eq!(buffer.height, 456);
        assert_eq!(buffer.seed, 789);
        assert_eq!(buffer.frame, 0);
        assert_eq!(buffer.total_pixels(), 123 * 456);
    }

    // =========================================================================
    // Section 17: CoverageStats Additional Tests (Coverage Boost)
    // =========================================================================

    #[test]
    fn h0_stats_07_max_gap_size_no_gaps() {
        let mut buffer = GpuPixelBuffer::new(10, 10, 42);
        // Fill all pixels
        for pixel in &mut buffer.pixels {
            *pixel = 1.0;
        }

        let stats = buffer.coverage_stats();
        assert_eq!(stats.max_gap_size(), 0);
        assert!(stats.gaps.is_empty());
    }

    #[test]
    fn h0_stats_08_meets_threshold_boundary() {
        let mut buffer = GpuPixelBuffer::new(100, 100, 42);
        // Fill exactly 80% of pixels
        for i in 0..8000 {
            buffer.pixels[i] = 1.0;
        }

        let stats = buffer.coverage_stats();
        assert!(stats.meets_threshold(0.80));
        assert!(!stats.meets_threshold(0.81));
    }

    #[test]
    fn h0_stats_09_coverage_stats_debug_clone() {
        let buffer = GpuPixelBuffer::new(10, 10, 42);
        let stats = buffer.coverage_stats();

        let debug = format!("{:?}", stats);
        assert!(debug.contains("CoverageStats"));

        let cloned = stats.clone();
        assert_eq!(cloned.covered, stats.covered);
        assert_eq!(cloned.total, stats.total);
    }

    // =========================================================================
    // Section 18: WasmPixelDemo Additional Tests (Coverage Boost)
    // =========================================================================

    #[test]
    fn h0_demo_lifecycle_05_hd_1080p() {
        let demo = WasmPixelDemo::hd_1080p();
        assert_eq!(demo.buffer.width, 1920);
        assert_eq!(demo.buffer.height, 1080);
        assert!(!demo.is_complete());
    }

    #[test]
    fn h0_demo_lifecycle_06_elapsed() {
        let demo = WasmPixelDemo::new(WasmDemoConfig::test_small());
        std::thread::sleep(std::time::Duration::from_millis(10));
        let elapsed = demo.elapsed();
        assert!(elapsed.as_millis() >= 10);
    }

    #[test]
    fn h0_demo_lifecycle_07_stats() {
        let mut demo = WasmPixelDemo::new(WasmDemoConfig::test_small());
        demo.tick();

        let stats = demo.stats();
        assert!(stats.total > 0);
        // After one tick, coverage is tracked (may be 0 or more)
        assert!(stats.covered <= stats.total);
    }

    #[test]
    fn h0_demo_lifecycle_08_tick_when_complete() {
        let config = WasmDemoConfig {
            width: 10,
            height: 10,
            fill_probability: 1.0, // Fill everything
            target_coverage: 0.01, // Very low target
            ..Default::default()
        };
        let mut demo = WasmPixelDemo::new(config);

        demo.tick(); // This should complete
        assert!(demo.is_complete());

        let frame_before = demo.frame_count();
        demo.tick(); // Should return early
        let frame_after = demo.frame_count();

        // Frame count should NOT increase when complete
        assert_eq!(frame_before, frame_after);
    }

    #[test]
    fn h0_demo_lifecycle_09_debug() {
        let demo = WasmPixelDemo::new(WasmDemoConfig::test_small());
        let debug = format!("{:?}", demo);
        assert!(debug.contains("WasmPixelDemo"));
    }

    // =========================================================================
    // Section 19: Wilson CI Additional Tests (Coverage Boost)
    // =========================================================================

    #[test]
    fn h0_wilson_01_confidence_90() {
        let ci = wilson_confidence_interval(50, 100, 0.90);
        assert!(ci.level == 0.90 || (ci.level - 0.90).abs() < 0.01);
        // 90% CI should be narrower than 95% CI
        let ci_95 = wilson_confidence_interval(50, 100, 0.95);
        assert!((ci.upper - ci.lower) < (ci_95.upper - ci_95.lower));
    }

    #[test]
    fn h0_wilson_02_confidence_99() {
        let ci = wilson_confidence_interval(50, 100, 0.99);
        assert!(ci.level == 0.99 || (ci.level - 0.99).abs() < 0.01);
        // 99% CI should be wider than 95% CI
        let ci_95 = wilson_confidence_interval(50, 100, 0.95);
        assert!((ci.upper - ci.lower) > (ci_95.upper - ci_95.lower));
    }

    #[test]
    fn h0_wilson_03_confidence_other() {
        // Non-standard confidence level defaults to 1.96 z-score
        let ci = wilson_confidence_interval(50, 100, 0.85);
        // Should still produce valid bounds
        assert!(ci.lower >= 0.0);
        assert!(ci.upper <= 1.0);
        assert!(ci.lower <= ci.upper);
    }

    // =========================================================================
    // Section 20: Flood Fill Edge Cases (Coverage Boost)
    // =========================================================================

    #[test]
    fn h0_flood_01_single_pixel_gap() {
        let mut buffer = GpuPixelBuffer::new(5, 5, 42);
        // Fill all but center pixel
        for (idx, pixel) in buffer.pixels.iter_mut().enumerate() {
            if idx != 12 {
                // Center of 5x5
                *pixel = 1.0;
            }
        }

        let gaps = buffer.find_gaps();
        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0].size, 1);
        assert_eq!(gaps[0].severity, GapSeverity::Info);
    }

    #[test]
    fn h0_flood_02_corner_gaps() {
        let mut buffer = GpuPixelBuffer::new(10, 10, 42);
        // Fill center, leave corners empty
        for y in 0..10 {
            for x in 0..10 {
                let idx = y * 10 + x;
                if (2..8).contains(&x) && (2..8).contains(&y) {
                    buffer.pixels[idx] = 1.0;
                }
            }
        }

        let gaps = buffer.find_gaps();
        // Should have one connected gap around the border
        assert!(!gaps.is_empty());
    }

    #[test]
    fn h0_flood_03_multiple_isolated_gaps() {
        let mut buffer = GpuPixelBuffer::new(10, 10, 42);
        // Fill everything
        for pixel in &mut buffer.pixels {
            *pixel = 1.0;
        }
        // Create 4 isolated single-pixel gaps
        buffer.pixels[0] = 0.0; // Top-left
        buffer.pixels[9] = 0.0; // Top-right
        buffer.pixels[90] = 0.0; // Bottom-left
        buffer.pixels[99] = 0.0; // Bottom-right

        let gaps = buffer.find_gaps();
        assert_eq!(gaps.len(), 4);
        for gap in &gaps {
            assert_eq!(gap.size, 1);
            assert_eq!(gap.severity, GapSeverity::Info);
        }
    }

    // =========================================================================
    // Section 21: Downsample Edge Cases (Coverage Boost)
    // =========================================================================

    #[test]
    fn h0_downsample_04_exact_multiple() {
        let buffer = GpuPixelBuffer::new(100, 100, 42);
        let downsampled = buffer.downsample(10, 10);
        // Each cell averages 100 source pixels (10x10)
        assert_eq!(downsampled.len(), 100);
    }

    #[test]
    fn h0_downsample_05_non_multiple() {
        let buffer = GpuPixelBuffer::new(100, 100, 42);
        let downsampled = buffer.downsample(7, 7);
        assert_eq!(downsampled.len(), 49);
    }

    #[test]
    fn h0_downsample_06_larger_than_source() {
        // When terminal is larger than source, scale factors become 0
        // which means count stays 0 and result is all zeros
        let buffer = GpuPixelBuffer::new(10, 10, 42);
        let downsampled = buffer.downsample(100, 100);
        assert_eq!(downsampled.len(), 10000);
    }

    // =========================================================================
    // Section 22: RNG Edge Cases (Coverage Boost)
    // =========================================================================

    #[test]
    fn h0_rng_09_hash_pixel_edge_values() {
        // Test with extreme values
        let hash_max = PcgRng::hash_pixel(u32::MAX, u32::MAX, u32::MAX);
        let hash_zero = PcgRng::hash_pixel(0, 0, 0);

        // Hashes should be different
        assert_ne!(hash_max, hash_zero);

        // Should produce valid u32 values (not panic)
        let _ = hash_max.to_string();
        let _ = hash_zero.to_string();
    }

    #[test]
    fn h0_rng_10_should_fill_edge_probability() {
        // Test probability at boundary values
        let near_zero = 0.000001;
        let near_one = 0.999999;

        // At near-zero probability, very few should fill
        let mut fill_count = 0;
        for idx in 0..1000 {
            if PcgRng::should_fill(42, idx, 1, near_zero) {
                fill_count += 1;
            }
        }
        assert!(
            fill_count < 10,
            "Near-zero probability filled {} pixels",
            fill_count
        );

        // At near-one probability, almost all should fill
        let mut fill_count = 0;
        for idx in 0..1000 {
            if PcgRng::should_fill(42, idx, 1, near_one) {
                fill_count += 1;
            }
        }
        assert!(
            fill_count > 990,
            "Near-one probability filled {} pixels",
            fill_count
        );
    }

    // =========================================================================
    // Section 23: Fill to Coverage Edge Cases (Coverage Boost)
    // =========================================================================

    #[test]
    fn h0_fill_01_max_frames_reached() {
        let mut buffer = GpuPixelBuffer::new(100, 100, 42);
        // Very low probability, won't reach 100% in 5 frames
        buffer.fill_to_coverage(1.0, 0.001, 5);
        // Should have run exactly 5 frames
        assert_eq!(buffer.frame, 5);
        // Coverage should be less than 100%
        assert!(buffer.coverage_percentage() < 1.0);
    }

    #[test]
    fn h0_fill_02_early_termination() {
        let mut buffer = GpuPixelBuffer::new(10, 10, 42);
        // High probability, should reach 50% quickly
        buffer.fill_to_coverage(0.5, 1.0, 1000);
        // Should not have run all 1000 frames
        assert!(buffer.frame < 1000);
        // Should have at least 50% coverage
        assert!(buffer.coverage_percentage() >= 0.5);
    }

    // =========================================================================
    // Section 24: Config Clone and Debug (Coverage Boost)
    // =========================================================================

    #[test]
    fn h0_config_12_clone_debug() {
        let config = WasmDemoConfig::default();
        let cloned = config.clone();

        assert_eq!(cloned.width, config.width);
        assert_eq!(cloned.height, config.height);
        assert_eq!(cloned.seed, config.seed);

        let debug = format!("{:?}", config);
        assert!(debug.contains("WasmDemoConfig"));
    }

    // =========================================================================
    // Section 25: PcgRng Clone and Debug (Coverage Boost)
    // =========================================================================

    #[test]
    fn h0_rng_11_clone_debug() {
        let rng = PcgRng::new(42);
        let cloned = rng.clone();

        let debug = format!("{:?}", rng);
        assert!(debug.contains("PcgRng"));

        // Cloned RNG should produce same values
        let mut rng1 = rng;
        let mut rng2 = cloned;
        assert_eq!(rng1.next_u32(), rng2.next_u32());
    }

    // =========================================================================
    // Section 26: GpuPixelBuffer Clone and Debug (Coverage Boost)
    // =========================================================================

    #[test]
    fn h0_buffer_12_clone_debug() {
        let buffer = GpuPixelBuffer::new(10, 10, 42);
        let cloned = buffer.clone();

        assert_eq!(cloned.width, buffer.width);
        assert_eq!(cloned.height, buffer.height);
        assert_eq!(cloned.pixels.len(), buffer.pixels.len());

        let debug = format!("{:?}", buffer);
        assert!(debug.contains("GpuPixelBuffer"));
    }
}
