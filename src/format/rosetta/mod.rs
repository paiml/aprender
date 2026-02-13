//! Rosetta Stone - Universal Model Format Converter
//!
//! Named after the ancient artifact that enabled translation between scripts,
//! this module provides bidirectional conversion between ML model formats.
//!
//! # Supported Formats
//!
//! | Format | Extensions | Quantization |
//! |--------|------------|--------------|
//! | GGUF | `.gguf` | Q4_0, Q4_K_M, Q5_K_M, Q6_K, Q8_0, F16, F32 |
//! | SafeTensors | `.safetensors` | F16, F32, BF16 |
//! | APR | `.apr` | Q4_0, Q8_0, F16, F32 |
//!
//! # Conversion Matrix (6 Direct Paths)
//!
//! ```text
//!     GGUF ←──────→ APR ←──────→ SafeTensors
//!       ↑                              ↑
//!       └──────────────────────────────┘
//! ```
//!
//! # Toyota Way Alignment
//!
//! - **Genchi Genbutsu**: Inspect raw tensor data, not abstractions
//! - **Jidoka**: Stop on any conversion anomaly
//! - **Kaizen**: Multi-step chains for iterative improvement
//! - **Visualization**: Full metadata before/after display
//!
//! # Example
//!
//! ```rust,ignore
//! use aprender::format::rosetta::{RosettaStone, FormatType, ConversionOptions};
//!
//! let rosetta = RosettaStone::new();
//!
//! // Inspect before conversion
//! let inspection = rosetta.inspect("model.gguf")?;
//! println!("{}", inspection);
//!
//! // Convert GGUF to APR
//! let report = rosetta.convert(
//!     "model.gguf",
//!     "model.apr",
//!     ConversionOptions::default()
//! )?;
//!
//! // Verify round-trip
//! let verification = rosetta.verify_roundtrip("model.gguf", FormatType::Apr)?;
//! assert!(verification.is_equivalent);
//! ```
//!
//! # References
//!
//! - Popper, K. (1959). The Logic of Scientific Discovery. Routledge.
//! - Ohno, T. (1988). Toyota Production System. Productivity Press.
//! - Gerganov, G. (2023). GGUF Format Specification. llama.cpp.

use crate::error::{AprenderError, Result};
use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};

/// Bug 212: Check if a path is a sharded SafeTensors index file.
fn is_sharded_index(path: &Path) -> bool {
    path.file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| n.ends_with(".index.json"))
}

// ============================================================================
// Format Types
// ============================================================================

/// Supported model formats for Rosetta Stone conversion
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FormatType {
    /// GGUF - llama.cpp compatible format
    Gguf,
    /// SafeTensors - HuggingFace format
    SafeTensors,
    /// APR - Aprender native format
    Apr,
}

impl FormatType {
    /// Detect format from file extension
    ///
    /// # Errors
    ///
    /// Returns error if extension is not recognized
    pub fn from_extension(path: &Path) -> Result<Self> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .ok_or_else(|| {
                // BUG-PATH-001: Provide helpful error when directory passed instead of file
                let is_dir = path.is_dir();
                let message = if is_dir {
                    // Check if common model files exist in directory
                    let candidates = ["model.gguf", "model.apr", "model.safetensors"];
                    let found: Vec<_> = candidates
                        .iter()
                        .filter(|f| path.join(f).exists())
                        .collect();
                    if found.is_empty() {
                        format!(
                            "Path '{}' is a directory, not a model file. \
                             Expected a file with .gguf, .apr, or .safetensors extension.",
                            path.display()
                        )
                    } else {
                        format!(
                            "Path '{}' is a directory. Did you mean '{}'?",
                            path.display(),
                            path.join(found[0]).display()
                        )
                    }
                } else {
                    format!(
                        "No file extension found in '{}'. Expected .gguf, .apr, or .safetensors",
                        path.display()
                    )
                };
                AprenderError::FormatError { message }
            })?
            .to_lowercase();

        match ext.as_str() {
            "gguf" => Ok(Self::Gguf),
            "safetensors" => Ok(Self::SafeTensors),
            "apr" => Ok(Self::Apr),
            _ => Err(AprenderError::FormatError {
                message: format!("Unknown format extension: .{ext}"),
            }),
        }
    }

    /// Detect format from file magic bytes (Genchi Genbutsu - go and see)
    ///
    /// # Errors
    ///
    /// Returns error if file cannot be read or magic is unknown
    pub fn from_magic(path: &Path) -> Result<Self> {
        use std::fs::File;
        use std::io::Read;

        let mut file = File::open(path).map_err(|e| AprenderError::FormatError {
            message: format!("Cannot open file: {e}"),
        })?;

        let mut magic = [0u8; 8];
        file.read_exact(&mut magic)
            .map_err(|e| AprenderError::FormatError {
                message: format!("Cannot read magic bytes: {e}"),
            })?;

        // GGUF magic: "GGUF" (0x46554747 little-endian)
        if &magic[0..4] == b"GGUF" {
            return Ok(Self::Gguf);
        }

        // SafeTensors magic: starts with JSON header length (little-endian u64)
        // then '{"' if it's a valid SafeTensors file
        // First 8 bytes are the header length as u64
        let header_len = u64::from_le_bytes(magic);
        if header_len < 100_000_000 {
            // Reasonable header size
            // Read next 2 bytes to check for JSON start
            let mut json_start = [0u8; 2];
            if file.read_exact(&mut json_start).is_ok() && &json_start == b"{\"" {
                return Ok(Self::SafeTensors);
            }
        }

        // APR magic: "APR\0" (v1) or "APR2" (v2)
        if &magic[0..3] == b"APR" {
            return Ok(Self::Apr);
        }

        Err(AprenderError::FormatError {
            message: "Unknown file format - magic bytes not recognized".to_string(),
        })
    }

    /// Get canonical file extension
    #[must_use]
    pub const fn extension(&self) -> &'static str {
        match self {
            Self::Gguf => "gguf",
            Self::SafeTensors => "safetensors",
            Self::Apr => "apr",
        }
    }

    /// Check if direct conversion to target format is supported
    #[must_use]
    pub const fn can_convert_to(&self, target: Self) -> bool {
        // All 6 direct paths are supported
        !matches!(
            (self, target),
            (Self::Gguf, Self::Gguf)
                | (Self::SafeTensors, Self::SafeTensors)
                | (Self::Apr, Self::Apr)
        )
    }
}

impl fmt::Display for FormatType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gguf => write!(f, "GGUF"),
            Self::SafeTensors => write!(f, "SafeTensors"),
            Self::Apr => write!(f, "APR"),
        }
    }
}

// ============================================================================
// Conversion Path
// ============================================================================

/// Describes a conversion path between formats
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConversionPath {
    /// Source format
    pub source: FormatType,
    /// Target format
    pub target: FormatType,
    /// Intermediate formats (for multi-step chains)
    pub intermediates: Vec<FormatType>,
}

impl ConversionPath {
    /// Create a direct conversion path
    #[must_use]
    pub const fn direct(source: FormatType, target: FormatType) -> Self {
        Self {
            source,
            target,
            intermediates: Vec::new(),
        }
    }

    /// Create a multi-step conversion chain
    #[must_use]
    pub fn chain(source: FormatType, intermediates: Vec<FormatType>, target: FormatType) -> Self {
        Self {
            source,
            target,
            intermediates,
        }
    }

    /// Get all steps in the conversion (including source and target)
    #[must_use]
    pub fn steps(&self) -> Vec<FormatType> {
        let mut steps = vec![self.source];
        steps.extend(self.intermediates.clone());
        steps.push(self.target);
        steps
    }

    /// Check if this is a round-trip (ends where it started)
    #[must_use]
    pub fn is_roundtrip(&self) -> bool {
        self.source == self.target && !self.intermediates.is_empty()
    }

    /// Detect cycles in the conversion chain
    #[must_use]
    pub fn has_cycle(&self) -> bool {
        let steps = self.steps();
        // Remove first and last (they can be same for round-trip)
        let middle: Vec<_> = steps[1..steps.len() - 1].to_vec();

        // Check for repeated formats in the middle
        let mut seen = std::collections::HashSet::new();
        for fmt in middle {
            if !seen.insert(fmt) {
                return true;
            }
        }
        false
    }
}

impl fmt::Display for ConversionPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let steps = self.steps();
        let path_str: Vec<_> = steps.iter().map(ToString::to_string).collect();
        write!(f, "{}", path_str.join(" → "))
    }
}

// ============================================================================
// Inspection Report
// ============================================================================

/// Tensor metadata from inspection
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Tensor name
    pub name: String,
    /// Data type (e.g., "F32", "Q4_K_M")
    pub dtype: String,
    /// Shape dimensions
    pub shape: Vec<usize>,
    /// Size in bytes
    pub size_bytes: usize,
    /// Statistical summary (min, max, mean, std)
    pub stats: Option<TensorStats>,
}

/// Statistical summary of tensor values
#[derive(Debug, Clone, Copy)]
pub struct TensorStats {
    /// Minimum value
    pub min: f32,
    /// Maximum value
    pub max: f32,
    /// Mean value
    pub mean: f32,
    /// Standard deviation
    pub std: f32,
}

/// Complete inspection report for a model file
#[derive(Debug, Clone)]
pub struct InspectionReport {
    /// Detected format
    pub format: FormatType,
    /// File size in bytes
    pub file_size: usize,
    /// Model metadata (key-value pairs)
    pub metadata: BTreeMap<String, String>,
    /// Tensor information
    pub tensors: Vec<TensorInfo>,
    /// Total parameter count
    pub total_params: usize,
    /// Quantization type if detected
    pub quantization: Option<String>,
    /// Architecture name if detected
    pub architecture: Option<String>,
}

impl fmt::Display for InspectionReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Rosetta Stone Inspection ===")?;
        writeln!(f, "Format: {}", self.format)?;
        writeln!(f, "File Size: {} bytes", self.file_size)?;
        writeln!(f, "Total Parameters: {}", self.total_params)?;

        if let Some(ref arch) = self.architecture {
            writeln!(f, "Architecture: {arch}")?;
        }
        if let Some(ref quant) = self.quantization {
            writeln!(f, "Quantization: {quant}")?;
        }

        writeln!(f, "\n--- Metadata ({} keys) ---", self.metadata.len())?;
        for (k, v) in &self.metadata {
            // Truncate long values
            let display_v = if v.len() > 60 {
                format!("{}...", &v[..60])
            } else {
                v.clone()
            };
            writeln!(f, "  {k}: {display_v}")?;
        }

        writeln!(f, "\n--- Tensors ({} total) ---", self.tensors.len())?;
        for (i, t) in self.tensors.iter().enumerate() {
            if i < 10 || i >= self.tensors.len() - 2 {
                writeln!(
                    f,
                    "  {}: {} {:?} ({} bytes)",
                    t.name, t.dtype, t.shape, t.size_bytes
                )?;
            } else if i == 10 {
                writeln!(f, "  ... ({} more tensors) ...", self.tensors.len() - 12)?;
            }
        }

        Ok(())
    }
}

// ============================================================================
// Conversion Report
// ============================================================================

/// Report from a conversion operation
#[derive(Debug, Clone)]
pub struct ConversionReport {
    /// Conversion path taken
    pub path: ConversionPath,
    /// Source file inspection
    pub source_inspection: InspectionReport,
    /// Target file inspection (after conversion)
    pub target_inspection: InspectionReport,
    /// Warnings during conversion
    pub warnings: Vec<String>,
    /// Conversion time in milliseconds
    pub duration_ms: u64,
    /// Tensors that were modified
    pub modified_tensors: Vec<String>,
    /// Tensors that were dropped
    pub dropped_tensors: Vec<String>,
}

impl ConversionReport {
    /// Check if conversion preserved all tensors
    #[must_use]
    pub fn is_lossless(&self) -> bool {
        self.dropped_tensors.is_empty()
    }

    /// Check if tensor counts match
    #[must_use]
    pub fn tensor_counts_match(&self) -> bool {
        self.source_inspection.tensors.len() == self.target_inspection.tensors.len()
    }
}

// ============================================================================
// Verification Report
// ============================================================================

/// Report from round-trip verification
#[derive(Debug, Clone)]
pub struct VerificationReport {
    /// Whether the round-trip preserved semantic equivalence
    pub is_equivalent: bool,
    /// Maximum absolute difference in any tensor value
    pub max_diff: f32,
    /// Mean absolute difference across all tensors
    pub mean_diff: f32,
    /// Tensor-wise comparison results
    pub tensor_diffs: BTreeMap<String, f32>,
    /// Metadata keys that changed
    pub changed_metadata: Vec<String>,
    /// Tensors that failed comparison
    pub failed_tensors: Vec<String>,
}

impl VerificationReport {
    /// Create a passing verification report
    #[must_use]
    pub fn passing() -> Self {
        Self {
            is_equivalent: true,
            max_diff: 0.0,
            mean_diff: 0.0,
            tensor_diffs: BTreeMap::new(),
            changed_metadata: Vec::new(),
            failed_tensors: Vec::new(),
        }
    }

    /// Check if verification passes within tolerance
    #[must_use]
    pub fn passes_with_tolerance(&self, epsilon: f32) -> bool {
        self.max_diff <= epsilon && self.failed_tensors.is_empty()
    }
}

// ============================================================================
// Validation Report (GH-175, PMAT-180)
// ============================================================================

/// Tensor validation result per APR-SPEC 10.9
#[derive(Debug, Clone)]
pub struct TensorValidation {
    /// Tensor name
    pub name: String,
    /// Whether tensor passes all checks
    pub is_valid: bool,
    /// Number of NaN values detected
    pub nan_count: usize,
    /// Number of Inf values detected
    pub inf_count: usize,
    /// Number of zero values
    pub zero_count: usize,
    /// Total element count
    pub element_count: usize,
    /// Minimum value (excluding NaN/Inf)
    pub min: f32,
    /// Maximum value (excluding NaN/Inf)
    pub max: f32,
    /// Mean value (excluding NaN/Inf)
    pub mean: f32,
    /// Standard deviation
    pub std: f32,
    /// Specific validation failures
    pub failures: Vec<String>,
}

impl TensorValidation {
    /// Check if tensor has any NaN values
    #[must_use]
    pub fn has_nan(&self) -> bool {
        self.nan_count > 0
    }

    /// Check if tensor has any Inf values
    #[must_use]
    pub fn has_inf(&self) -> bool {
        self.inf_count > 0
    }

    /// Check if tensor is all zeros (suspicious)
    #[must_use]
    pub fn is_all_zeros(&self) -> bool {
        self.zero_count == self.element_count
    }
}

/// Complete validation report for a model file
/// Implements APR-SPEC 10.9 Physics Constraints
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// Detected format
    pub format: FormatType,
    /// File path validated
    pub file_path: String,
    /// Overall pass/fail status
    pub is_valid: bool,
    /// Total tensors validated
    pub tensor_count: usize,
    /// Tensors with issues
    pub failed_tensor_count: usize,
    /// Total NaN values across all tensors
    pub total_nan_count: usize,
    /// Total Inf values across all tensors
    pub total_inf_count: usize,
    /// Tensors that are all zeros
    pub all_zero_tensors: Vec<String>,
    /// Per-tensor validation results
    pub tensors: Vec<TensorValidation>,
    /// Validation time in milliseconds
    pub duration_ms: u64,
}

impl ValidationReport {
    /// Check if validation passed (no NaN/Inf, no all-zeros)
    #[must_use]
    pub fn passed(&self) -> bool {
        self.is_valid
    }

    /// Get summary line for CLI output
    #[must_use]
    pub fn summary(&self) -> String {
        if self.is_valid {
            format!(
                "VALID: {} tensors checked, 0 contract violations (PMAT-235)",
                self.tensor_count
            )
        } else {
            let contract_failures: usize = self.tensors.iter().map(|t| t.failures.len()).sum();
            format!(
                "INVALID: {} tensors, {} contract violations, {} NaN, {} Inf, {} all-zeros",
                self.tensor_count,
                contract_failures,
                self.total_nan_count,
                self.total_inf_count,
                self.all_zero_tensors.len()
            )
        }
    }
}

impl fmt::Display for ValidationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Model Validation Report (APR-SPEC 10.9) ===")?;
        writeln!(f, "Format: {}", self.format)?;
        writeln!(f, "File: {}", self.file_path)?;
        writeln!(
            f,
            "Status: {}",
            if self.is_valid { "VALID" } else { "INVALID" }
        )?;
        writeln!(f, "Tensors: {}", self.tensor_count)?;
        writeln!(f, "Total NaN: {}", self.total_nan_count)?;
        writeln!(f, "Total Inf: {}", self.total_inf_count)?;
        writeln!(f, "All-Zero Tensors: {}", self.all_zero_tensors.len())?;
        writeln!(f, "Duration: {} ms", self.duration_ms)?;

        if !self.is_valid {
            writeln!(f, "\n--- Failed Tensors ---")?;
            for tv in &self.tensors {
                if !tv.is_valid {
                    writeln!(
                        f,
                        "  {}: {} NaN, {} Inf, {} zeros / {}",
                        tv.name, tv.nan_count, tv.inf_count, tv.zero_count, tv.element_count
                    )?;
                    for failure in &tv.failures {
                        writeln!(f, "    - {failure}")?;
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Conversion Options
// ============================================================================

/// Options for conversion operations
#[allow(clippy::struct_excessive_bools)] // Options structs commonly have multiple boolean flags
#[derive(Debug, Clone)]
pub struct ConversionOptions {
    /// Target quantization (None = preserve original)
    pub quantization: Option<String>,
    /// Verify after conversion
    pub verify: bool,
    /// Compute tensor statistics
    pub compute_stats: bool,
    /// Tolerance for floating-point comparison
    pub tolerance: f32,
    /// Preserve metadata from source
    pub preserve_metadata: bool,
    /// Add conversion provenance to metadata
    pub add_provenance: bool,
    /// External tokenizer.json for weights-only models (PMAT-232)
    pub tokenizer_path: Option<PathBuf>,
}

impl Default for ConversionOptions {
    fn default() -> Self {
        Self {
            quantization: None,
            verify: true,
            compute_stats: false,
            tolerance: 1e-6,
            preserve_metadata: true,
            add_provenance: true,
            tokenizer_path: None,
        }
    }
}

// ============================================================================
// Rosetta Stone Converter
// ============================================================================

/// Main converter implementing the Rosetta Stone pattern
#[derive(Debug)]
pub struct RosettaStone {
    /// Options for conversion
    options: ConversionOptions,
}

impl RosettaStone {
    /// Create a new Rosetta Stone converter with default options
    #[must_use]
    pub fn new() -> Self {
        Self {
            options: ConversionOptions::default(),
        }
    }

    /// Create with custom options
    #[must_use]
    pub fn with_options(options: ConversionOptions) -> Self {
        Self { options }
    }

    /// Inspect a model file (Genchi Genbutsu - go and see)
    ///
    /// # Errors
    ///
    /// Returns error if file cannot be read or format is unknown
    pub fn inspect<P: AsRef<Path>>(&self, path: P) -> Result<InspectionReport> {
        let path = path.as_ref();

        // Sharded SafeTensors index detection (GH-212, resolved)
        if is_sharded_index(path) {
            return self.inspect_sharded_safetensors(path);
        }

        // Detect format from magic bytes first, fall back to extension
        let format = FormatType::from_magic(path).or_else(|_| FormatType::from_extension(path))?;

        let file_size = std::fs::metadata(path)
            .map(|m| m.len() as usize)
            .unwrap_or(0);

        match format {
            FormatType::Gguf => self.inspect_gguf(path, file_size),
            FormatType::SafeTensors => self.inspect_safetensors(path, file_size),
            FormatType::Apr => self.inspect_apr(path, file_size),
        }
    }

    /// Validate a model file for physics constraints (GH-175, PMAT-180)
    ///
    /// Checks per APR-SPEC 10.9:
    /// - NaN detection (corruption indicator)
    /// - Inf detection (overflow indicator)
    /// - All-zeros detection (uninitialized weights)
    ///
    /// # Errors
    ///
    /// Returns error if file cannot be read or format is unknown
    pub fn validate<P: AsRef<Path>>(&self, path: P) -> Result<ValidationReport> {
        let path = path.as_ref();
        let start = std::time::Instant::now();

        // Detect format
        let format = FormatType::from_magic(path).or_else(|_| FormatType::from_extension(path))?;

        // Dispatch to format-specific validation
        let mut report = match format {
            FormatType::Gguf => self.validate_gguf(path)?,
            FormatType::SafeTensors => self.validate_safetensors(path)?,
            FormatType::Apr => self.validate_apr(path)?,
        };

        report.duration_ms = start.elapsed().as_millis() as u64;
        Ok(report)
    }

    /// Convert a model file to a different format
    ///
    /// # Errors
    ///
    /// Returns error if conversion fails
    pub fn convert<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        source: P,
        target: Q,
        options: Option<ConversionOptions>,
    ) -> Result<ConversionReport> {
        let source = source.as_ref();
        let target = target.as_ref();
        let opts = options.unwrap_or_else(|| self.options.clone());

        let start = std::time::Instant::now();

        // Sharded SafeTensors index detection (GH-212, resolved)
        if is_sharded_index(source) {
            let target_format = FormatType::from_extension(target)?;
            let source_inspection = self.inspect_sharded_safetensors(source)?;

            self.convert_sharded(source, target, target_format, &opts)?;

            let target_inspection = self.inspect(target)?;
            let duration_ms = start.elapsed().as_millis() as u64;

            return Ok(ConversionReport {
                path: ConversionPath::direct(FormatType::SafeTensors, target_format),
                source_inspection,
                target_inspection,
                warnings: Vec::new(),
                duration_ms,
                modified_tensors: Vec::new(),
                dropped_tensors: Vec::new(),
            });
        }

        // Detect formats
        let source_format =
            FormatType::from_magic(source).or_else(|_| FormatType::from_extension(source))?;
        let target_format = FormatType::from_extension(target)?;

        // Inspect source
        let source_inspection = self.inspect(source)?;

        // Perform conversion
        self.convert_internal(source, target, source_format, target_format, &opts)?;

        // Inspect target
        let target_inspection = self.inspect(target)?;

        let duration_ms = start.elapsed().as_millis() as u64;

        Ok(ConversionReport {
            path: ConversionPath::direct(source_format, target_format),
            source_inspection,
            target_inspection,
            warnings: Vec::new(),
            duration_ms,
            modified_tensors: Vec::new(),
            dropped_tensors: Vec::new(),
        })
    }

    /// Execute a multi-step conversion chain
    ///
    /// # Errors
    ///
    /// Returns error if any step fails or cycle detected
    pub fn chain<P: AsRef<Path>>(
        &self,
        source: P,
        chain: &[FormatType],
        work_dir: &Path,
    ) -> Result<Vec<ConversionReport>> {
        if chain.len() < 2 {
            return Err(AprenderError::FormatError {
                message: "Chain must have at least 2 formats".to_string(),
            });
        }

        // Check for cycles (Popperian: The Infinite Loop test)
        let path = ConversionPath::chain(
            chain[0],
            chain[1..chain.len() - 1].to_vec(),
            chain[chain.len() - 1],
        );
        if path.has_cycle() {
            return Err(AprenderError::FormatError {
                message: "Conversion chain contains a cycle".to_string(),
            });
        }

        let source = source.as_ref();
        let mut reports = Vec::new();
        let mut current_path = source.to_path_buf();

        for (i, window) in chain.windows(2).enumerate() {
            let target_format = window[1];
            let target_path = work_dir.join(format!("step_{i}.{}", target_format.extension()));

            let report = self.convert(&current_path, &target_path, None)?;
            reports.push(report);

            current_path = target_path;
        }

        Ok(reports)
    }

    /// Verify round-trip conversion preserves equivalence
    ///
    /// # Errors
    ///
    /// Returns error if verification fails
    pub fn verify_roundtrip<P: AsRef<Path>>(
        &self,
        source: P,
        intermediate: FormatType,
    ) -> Result<VerificationReport> {
        let source = source.as_ref();
        let source_format =
            FormatType::from_magic(source).or_else(|_| FormatType::from_extension(source))?;

        // Create temp directory for intermediate files
        let temp_dir = std::env::temp_dir().join("rosetta_verify");
        std::fs::create_dir_all(&temp_dir).map_err(|e| AprenderError::FormatError {
            message: format!("Cannot create temp dir: {e}"),
        })?;

        // Source → Intermediate
        let intermediate_path = temp_dir.join(format!("intermediate.{}", intermediate.extension()));
        self.convert(source, &intermediate_path, None)?;

        // Intermediate → Source format (round-trip)
        let roundtrip_path = temp_dir.join(format!("roundtrip.{}", source_format.extension()));
        self.convert(&intermediate_path, &roundtrip_path, None)?;

        // Compare source and round-trip
        self.compare_files(source, &roundtrip_path)
    }

    /// Load a tensor as f32 values from any supported format
    ///
    /// Handles dequantization for quantized formats (Q4_K, Q6_K, etc.)
    ///
    /// # Errors
    ///
    /// Returns error if file cannot be read, format is unknown, or tensor not found
    pub fn load_tensor_f32<P: AsRef<Path>>(&self, path: P, tensor_name: &str) -> Result<Vec<f32>> {
        let path = path.as_ref();
        let format = FormatType::from_magic(path).or_else(|_| FormatType::from_extension(path))?;

        match format {
            FormatType::Gguf => self.load_tensor_f32_gguf(path, tensor_name),
            FormatType::SafeTensors => self.load_tensor_f32_safetensors(path, tensor_name),
            FormatType::Apr => self.load_tensor_f32_apr(path, tensor_name),
        }
    }

    fn load_tensor_f32_gguf(&self, path: &Path, tensor_name: &str) -> Result<Vec<f32>> {
        use crate::format::gguf::GgufReader;

        let reader = GgufReader::from_file(path)?;
        let (data, _shape) =
            reader
                .get_tensor_f32(tensor_name)
                .map_err(|e| AprenderError::FormatError {
                    message: format!("Failed to load GGUF tensor '{}': {}", tensor_name, e),
                })?;
        Ok(data)
    }

    fn load_tensor_f32_safetensors(&self, path: &Path, tensor_name: &str) -> Result<Vec<f32>> {
        use crate::serialization::safetensors::MappedSafeTensors;

        let mapped = MappedSafeTensors::open(path).map_err(|e| AprenderError::FormatError {
            message: format!("SafeTensors open failed: {e}"),
        })?;
        mapped
            .get_tensor(tensor_name)
            .map_err(|e| AprenderError::FormatError {
                message: format!("Failed to load SafeTensors tensor '{}': {}", tensor_name, e),
            })
    }

    fn load_tensor_f32_apr(&self, path: &Path, tensor_name: &str) -> Result<Vec<f32>> {
        use crate::format::v2::AprV2Reader;

        let data = std::fs::read(path).map_err(|e| AprenderError::FormatError {
            message: format!("Cannot read APR file: {e}"),
        })?;
        let reader = AprV2Reader::from_bytes(&data).map_err(|e| AprenderError::FormatError {
            message: format!("APR parse failed: {e}"),
        })?;
        reader
            .get_tensor_as_f32(tensor_name)
            .ok_or_else(|| AprenderError::FormatError {
                message: format!("Tensor '{}' not found in APR file", tensor_name),
            })
    }

    // ========================================================================
    // Private Methods
    // ========================================================================

    // ------------------------------------------------------------------------
    // Validation Methods (GH-175, PMAT-180)
    // ------------------------------------------------------------------------

    fn validate_gguf(&self, path: &Path) -> Result<ValidationReport> {
        use crate::format::gguf::GgufReader;

        let reader = GgufReader::from_file(path)?;
        let mut tensors = Vec::new();
        let mut total_nan = 0;
        let mut total_inf = 0;
        let mut all_zero_tensors = Vec::new();

        // Get tensor names from metadata
        let tensor_names: Vec<String> = reader.tensors.iter().map(|t| t.name.clone()).collect();

        for name in &tensor_names {
            // Use GgufReader's dequantization (handles Q4K, Q6K, etc.)
            if let Ok((f32_data, _shape)) = reader.get_tensor_f32(name) {
                let tv = self.compute_tensor_validation(name, &f32_data);

                total_nan += tv.nan_count;
                total_inf += tv.inf_count;
                if tv.is_all_zeros() {
                    all_zero_tensors.push(name.clone());
                }
                tensors.push(tv);
            }
        }

        let failed_count = tensors.iter().filter(|t| !t.is_valid).count();
        let is_valid = failed_count == 0;

        Ok(ValidationReport {
            format: FormatType::Gguf,
            file_path: path.display().to_string(),
            is_valid,
            tensor_count: tensors.len(),
            failed_tensor_count: failed_count,
            total_nan_count: total_nan,
            total_inf_count: total_inf,
            all_zero_tensors,
            tensors,
            duration_ms: 0, // Set by caller
        })
    }

    fn validate_safetensors(&self, path: &Path) -> Result<ValidationReport> {
        use crate::serialization::safetensors::MappedSafeTensors;

        let mapped = MappedSafeTensors::open(path).map_err(|e| AprenderError::FormatError {
            message: format!("SafeTensors open failed: {e}"),
        })?;

        let mut tensors = Vec::new();
        let mut total_nan = 0;
        let mut total_inf = 0;
        let mut all_zero_tensors = Vec::new();

        for name in mapped.tensor_names() {
            // get_tensor returns Result<Vec<f32>, String>
            if let Ok(f32_data) = mapped.get_tensor(name) {
                let tv = self.compute_tensor_validation(name, &f32_data);

                total_nan += tv.nan_count;
                total_inf += tv.inf_count;
                if tv.is_all_zeros() {
                    all_zero_tensors.push(name.to_string());
                }
                tensors.push(tv);
            }
        }

        let failed_count = tensors.iter().filter(|t| !t.is_valid).count();
        let is_valid = failed_count == 0;

        Ok(ValidationReport {
            format: FormatType::SafeTensors,
            file_path: path.display().to_string(),
            is_valid,
            tensor_count: tensors.len(),
            failed_tensor_count: failed_count,
            total_nan_count: total_nan,
            total_inf_count: total_inf,
            all_zero_tensors,
            tensors,
            duration_ms: 0,
        })
    }

    fn validate_apr(&self, path: &Path) -> Result<ValidationReport> {
        use crate::format::v2::AprV2Reader;

        let data = std::fs::read(path).map_err(|e| AprenderError::FormatError {
            message: format!("Cannot read APR file: {e}"),
        })?;

        let reader = AprV2Reader::from_bytes(&data).map_err(|e| AprenderError::FormatError {
            message: format!("APR parse failed: {e}"),
        })?;

        let mut tensors = Vec::new();
        let mut total_nan = 0;
        let mut total_inf = 0;
        let mut all_zero_tensors = Vec::new();

        for name in reader.tensor_names() {
            // Use get_tensor_as_f32 which handles dequantization
            if let Some(f32_data) = reader.get_tensor_as_f32(name) {
                let tv = self.compute_tensor_validation(name, &f32_data);

                total_nan += tv.nan_count;
                total_inf += tv.inf_count;
                if tv.is_all_zeros() {
                    all_zero_tensors.push(name.to_string());
                }
                tensors.push(tv);
            }
        }

        let failed_count = tensors.iter().filter(|t| !t.is_valid).count();
        let is_valid = failed_count == 0;

        Ok(ValidationReport {
            format: FormatType::Apr,
            file_path: path.display().to_string(),
            is_valid,
            tensor_count: tensors.len(),
            failed_tensor_count: failed_count,
            total_nan_count: total_nan,
            total_inf_count: total_inf,
            all_zero_tensors,
            tensors,
            duration_ms: 0,
        })
    }

    fn compute_tensor_validation(&self, name: &str, data: &[f32]) -> TensorValidation {
        let element_count = data.len();
        if element_count == 0 {
            return TensorValidation {
                name: name.to_string(),
                is_valid: true,
                nan_count: 0,
                inf_count: 0,
                zero_count: 0,
                element_count: 0,
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                std: 0.0,
                failures: Vec::new(),
            };
        }

        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut sum = 0.0f64;
        let mut nan_count = 0;
        let mut inf_count = 0;
        let mut zero_count = 0;

        for &v in data {
            if v.is_nan() {
                nan_count += 1;
                continue;
            }
            if v.is_infinite() {
                inf_count += 1;
                continue;
            }
            if v == 0.0 {
                zero_count += 1;
            }
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
            sum += f64::from(v);
        }

        let valid_count = element_count - nan_count - inf_count;
        let mean = if valid_count > 0 {
            (sum / valid_count as f64) as f32
        } else {
            0.0
        };

        // Compute std dev
        let mut var_sum = 0.0f64;
        for &v in data {
            if !v.is_nan() && !v.is_infinite() {
                let diff = f64::from(v) - f64::from(mean);
                var_sum += diff * diff;
            }
        }
        let std = if valid_count > 1 {
            (var_sum / (valid_count - 1) as f64).sqrt() as f32
        } else {
            0.0
        };

        // Collect failures (APR-SPEC 10.9 + PMAT-235 contract gates)
        let mut failures = Vec::new();
        if nan_count > 0 {
            failures.push(format!(
                "[F-DATA-QUALITY-002] {nan_count} NaN values detected"
            ));
        }
        if inf_count > 0 {
            failures.push(format!(
                "[F-DATA-QUALITY-002] {inf_count} Inf values detected"
            ));
        }
        if zero_count == element_count {
            failures.push("[F-DATA-QUALITY-001] All values are zero (uninitialized?)".to_string());
        }

        // Density gate (F-DATA-QUALITY-001)
        // Embedding tensors: >50% zeros indicates corrupt offset loading
        // Weight tensors: >80% zeros indicates uninitialized or zeroed memory
        let zero_pct = if element_count > 0 {
            100.0 * zero_count as f32 / element_count as f32
        } else {
            0.0
        };
        let name_lower = name.to_lowercase();
        let is_embedding = name_lower.contains("embed")
            || name_lower.contains("wte")
            || name_lower.contains("wpe")
            || name_lower.contains("position_embedding");
        // GH-234: lm_head has similar value distribution to embeddings (especially weight-tied)
        let is_lm_head = name_lower.contains("lm_head") || name_lower == "output.weight";
        let density_threshold = if is_embedding || is_lm_head { 50.0 } else { 80.0 };
        if zero_pct > density_threshold && zero_count < element_count {
            failures.push(format!(
                "[F-DATA-QUALITY-001] DENSITY: {zero_pct:.1}% zeros (max {density_threshold}%)"
            ));
        }

        // PMAT-235: L2 norm gate (F-DATA-QUALITY-003)
        let l2_norm = {
            let mut sum_sq = 0.0f64;
            for &v in data {
                if !v.is_nan() && !v.is_infinite() {
                    sum_sq += f64::from(v) * f64::from(v);
                }
            }
            sum_sq.sqrt() as f32
        };
        if valid_count > 0 && l2_norm < 1e-6 {
            failures
                .push("[F-DATA-QUALITY-003] L2 norm ~0: tensor is effectively empty".to_string());
        }

        // PMAT-235: Variation gate (F-DATA-QUALITY-003)
        // Norm and bias tensors are exempt: constant init (e.g., all 1.0 for RMS norm) is correct
        let is_norm_or_bias = name_lower.contains("norm")
            || name_lower.contains("bias")
            || name_lower.contains("ln_");
        if valid_count > 1 && (max - min).abs() < 1e-10 && !min.is_infinite() && !is_norm_or_bias {
            failures
                .push("[F-DATA-QUALITY-003] All values identical: tensor is constant".to_string());
        }

        let is_valid = failures.is_empty();

        TensorValidation {
            name: name.to_string(),
            is_valid,
            nan_count,
            inf_count,
            zero_count,
            element_count,
            min: if min.is_infinite() { 0.0 } else { min },
            max: if max.is_infinite() { 0.0 } else { max },
            mean,
            std,
            failures,
        }
    }

    // ------------------------------------------------------------------------
    // Inspection Methods
    // ------------------------------------------------------------------------

    fn inspect_gguf(&self, path: &Path, file_size: usize) -> Result<InspectionReport> {
        use crate::format::gguf::{load_gguf_raw, GgufRawTensor};

        let result = load_gguf_raw(path)?;

        let mut meta_map: BTreeMap<String, String> = BTreeMap::new();
        // Add config info to metadata
        if let Some(ref arch) = result.model_config.architecture {
            meta_map.insert("general.architecture".to_string(), arch.clone());
        }
        if let Some(num_layers) = result.model_config.num_layers {
            meta_map.insert("n_layers".to_string(), num_layers.to_string());
        }
        if let Some(num_heads) = result.model_config.num_heads {
            meta_map.insert("n_heads".to_string(), num_heads.to_string());
        }
        if let Some(hidden_size) = result.model_config.hidden_size {
            meta_map.insert("n_embd".to_string(), hidden_size.to_string());
        }

        let tensors: Vec<TensorInfo> = result
            .tensors
            .iter()
            .map(|(name, t): (&String, &GgufRawTensor)| TensorInfo {
                name: name.clone(),
                dtype: format!("{}", t.dtype),
                shape: t.shape.clone(),
                size_bytes: t.data.len(),
                stats: None,
            })
            .collect();

        let total_params: usize = tensors
            .iter()
            .map(|t| t.shape.iter().product::<usize>())
            .sum();

        let architecture = result.model_config.architecture.clone();

        let quantization = tensors.first().map(|t| t.dtype.clone());

        Ok(InspectionReport {
            format: FormatType::Gguf,
            file_size,
            metadata: meta_map,
            tensors,
            total_params,
            quantization,
            architecture,
        })
    }

    fn inspect_safetensors(&self, path: &Path, file_size: usize) -> Result<InspectionReport> {
        use crate::serialization::safetensors::{MappedSafeTensors, TensorMetadata};

        let mapped = MappedSafeTensors::open(path).map_err(|e| AprenderError::FormatError {
            message: format!("SafeTensors open failed: {e}"),
        })?;
        let tensor_names = mapped.tensor_names();

        let mut tensors = Vec::new();
        let mut total_params: usize = 0;

        for name in tensor_names {
            if let Some(info) = mapped.get_metadata(name) {
                let info: &TensorMetadata = info;
                let shape: Vec<usize> = info.shape.clone();
                let params: usize = shape.iter().product();
                total_params += params;

                let data_len = info.data_offsets[1] - info.data_offsets[0];

                tensors.push(TensorInfo {
                    name: name.to_string(),
                    dtype: info.dtype.clone(),
                    shape,
                    size_bytes: data_len,
                    stats: None,
                });
            }
        }

        Ok(InspectionReport {
            format: FormatType::SafeTensors,
            file_size,
            metadata: BTreeMap::new(),
            tensors,
            total_params,
            quantization: None,
            architecture: None,
        })
    }

    fn inspect_apr(&self, path: &Path, file_size: usize) -> Result<InspectionReport> {
        use crate::format::v2::AprV2Reader;

        // Read file into bytes
        let data = std::fs::read(path).map_err(|e| AprenderError::FormatError {
            message: format!("Cannot read APR file: {e}"),
        })?;

        let reader = AprV2Reader::from_bytes(&data).map_err(|e| AprenderError::FormatError {
            message: format!("APR parse failed: {e}"),
        })?;

        let meta = reader.metadata();

        let mut metadata: BTreeMap<String, String> = BTreeMap::new();
        metadata.insert("format_version".to_string(), "2".to_string());
        metadata.insert("model_type".to_string(), meta.model_type.clone());
        if let Some(ref name) = meta.name {
            metadata.insert("model_name".to_string(), name.clone());
        }

        // Get tensors from tensor_names + get_tensor
        let tensor_names = reader.tensor_names();
        let mut tensors = Vec::new();
        let mut total_params: usize = 0;

        for name in tensor_names {
            if let Some(entry) = reader.get_tensor(name) {
                let params: usize = entry.shape.iter().product();
                total_params += params;
                tensors.push(TensorInfo {
                    name: entry.name.clone(),
                    dtype: format!("{:?}", entry.dtype),
                    shape: entry.shape.clone(),
                    size_bytes: entry.size as usize,
                    stats: None,
                });
            }
        }

        Ok(InspectionReport {
            format: FormatType::Apr,
            file_size,
            metadata,
            tensors,
            total_params,
            quantization: meta.quantization.as_ref().map(|q| q.quant_type.clone()),
            architecture: meta.architecture.clone(),
        })
    }

    /// Bug 212: Inspect a sharded SafeTensors model via its index.json.
    /// Iterates shard files and aggregates tensor metadata.
    fn inspect_sharded_safetensors(&self, index_path: &Path) -> Result<InspectionReport> {
        use crate::format::sharded::ShardIndex;
        use crate::serialization::safetensors::{MappedSafeTensors, TensorMetadata};

        let content =
            std::fs::read_to_string(index_path).map_err(|e| AprenderError::FormatError {
                message: format!("Failed to read shard index {}: {e}", index_path.display()),
            })?;
        let index = ShardIndex::from_json(&content)?;

        let base_dir = index_path
            .parent()
            .ok_or_else(|| AprenderError::FormatError {
                message: format!(
                    "Cannot determine parent directory of {}",
                    index_path.display()
                ),
            })?;

        let mut tensors = Vec::new();
        let mut total_params: usize = 0;
        let mut total_file_size: usize = 0;

        for shard_file in index.shard_files() {
            let shard_path = base_dir.join(shard_file);
            if !shard_path.exists() {
                continue;
            }

            total_file_size += std::fs::metadata(&shard_path)
                .map(|m| m.len() as usize)
                .unwrap_or(0);

            let mapped =
                MappedSafeTensors::open(&shard_path).map_err(|e| AprenderError::FormatError {
                    message: format!("SafeTensors open failed for shard {shard_file}: {e}"),
                })?;

            for name in mapped.tensor_names() {
                if let Some(info) = mapped.get_metadata(name) {
                    let info: &TensorMetadata = info;
                    let shape: Vec<usize> = info.shape.clone();
                    let params: usize = shape.iter().product();
                    total_params += params;
                    let data_len = info.data_offsets[1] - info.data_offsets[0];

                    tensors.push(TensorInfo {
                        name: name.to_string(),
                        dtype: info.dtype.clone(),
                        shape,
                        size_bytes: data_len,
                        stats: None,
                    });
                }
            }
        }

        Ok(InspectionReport {
            format: FormatType::SafeTensors,
            file_size: total_file_size,
            metadata: BTreeMap::from([("shards".to_string(), index.shard_count().to_string())]),
            tensors,
            total_params,
            quantization: None,
            architecture: None,
        })
    }

    /// Bug 212: Convert a sharded SafeTensors model to any target format.
    /// Routes through import (sharded ST → APR) then converts APR → target.
    fn convert_sharded(
        &self,
        source: &Path,
        target: &Path,
        target_format: FormatType,
        opts: &ConversionOptions,
    ) -> Result<()> {
        use crate::format::converter::{apr_import, ImportOptions};

        let source_str = source.to_string_lossy();
        let effective_tokenizer = opts.tokenizer_path.clone().or_else(|| {
            let sibling = source.with_file_name("tokenizer.json");
            if sibling.exists() {
                Some(sibling)
            } else {
                None
            }
        });
        let import_opts = ImportOptions {
            tokenizer_path: effective_tokenizer,
            allow_no_config: true, // Sharded models may have config.json; let import warn
            ..ImportOptions::default()
        };

        if target_format == FormatType::Apr {
            // Direct: sharded ST → APR via import
            eprintln!(
                "[BUG-212] Converting sharded SafeTensors → APR: {}",
                source.display()
            );
            apr_import(&source_str, target, import_opts)?;
        } else {
            // Sharded ST conversion via intermediate APR
            let temp_apr = std::env::temp_dir().join("rosetta_sharded_temp.apr");
            eprintln!(
                "[BUG-212] Converting sharded SafeTensors → {} (via temp APR): {}",
                target_format,
                source.display()
            );
            apr_import(&source_str, &temp_apr, import_opts)?;
            self.convert_internal(&temp_apr, target, FormatType::Apr, target_format, opts)?;
            let _ = std::fs::remove_file(&temp_apr);
        }

        Ok(())
    }

    #[allow(clippy::self_only_used_in_recursion)] // Self is needed for recursive convert calls
    fn convert_internal(
        &self,
        source: &Path,
        target: &Path,
        source_format: FormatType,
        target_format: FormatType,
        opts: &ConversionOptions,
    ) -> Result<()> {
        use crate::format::converter::{
            apr_export, apr_import, ExportFormat, ExportOptions, ImportOptions, QuantizationType,
        };

        // GH-205 FIX: Map ConversionOptions.quantization to ExportOptions.quantize
        // Previously opts was ignored, causing F32 GGUF export even when quantization requested.
        // Note: Q6_K maps to Q4K since that's what realizar's inference supports.
        let export_quantize =
            opts.quantization
                .as_ref()
                .and_then(|q| match q.to_lowercase().as_str() {
                    "q4_k" | "q4_k_m" | "int4" | "q6_k" => Some(QuantizationType::Q4K),
                    "int8" | "q8_0" => Some(QuantizationType::Int8),
                    "fp16" | "f16" => Some(QuantizationType::Fp16),
                    _ => None,
                });

        match (source_format, target_format) {
            // GGUF/SafeTensors → APR (same conversion path via apr_import)
            // GH-196: Default ImportOptions are permissive (strict=false),
            // so format conversion proceeds with warnings for unverified architectures.
            (FormatType::Gguf | FormatType::SafeTensors, FormatType::Apr) => {
                let source_str = source.to_string_lossy();
                let effective_tokenizer = opts.tokenizer_path.clone().or_else(|| {
                    let sibling = source.with_file_name("tokenizer.json");
                    if sibling.exists() {
                        Some(sibling)
                    } else {
                        None
                    }
                });
                let import_opts = ImportOptions {
                    tokenizer_path: effective_tokenizer,
                    allow_no_config: true,
                    ..ImportOptions::default()
                };
                apr_import(&source_str, target, import_opts)?;
                Ok(())
            }

            // APR → GGUF
            // GH-205 FIX: Default to Q4_K quantization for GGUF export.
            // F32 GGUF files don't work with realizar's fused matmul kernels
            // (see export.rs:532-537 comment). Q4_K is the standard format.
            (FormatType::Apr, FormatType::Gguf) => {
                let gguf_quantize = export_quantize.clone().or(Some(QuantizationType::Q4K)); // Default to Q4K for GGUF
                apr_export(
                    source,
                    target,
                    ExportOptions {
                        format: ExportFormat::Gguf,
                        quantize: gguf_quantize,
                        ..Default::default()
                    },
                )?;
                Ok(())
            }

            // APR → SafeTensors
            (FormatType::Apr, FormatType::SafeTensors) => {
                apr_export(
                    source,
                    target,
                    ExportOptions {
                        format: ExportFormat::SafeTensors,
                        ..Default::default()
                    },
                )?;
                Ok(())
            }

            // GGUF → SafeTensors (via APR)
            (FormatType::Gguf, FormatType::SafeTensors) => {
                let temp_apr = std::env::temp_dir().join("rosetta_temp.apr");
                self.convert_internal(source, &temp_apr, FormatType::Gguf, FormatType::Apr, opts)?;
                self.convert_internal(
                    &temp_apr,
                    target,
                    FormatType::Apr,
                    FormatType::SafeTensors,
                    opts,
                )?;
                let _ = std::fs::remove_file(temp_apr);
                Ok(())
            }

            // SafeTensors → GGUF (via APR)
            (FormatType::SafeTensors, FormatType::Gguf) => {
                let temp_apr = std::env::temp_dir().join("rosetta_temp.apr");
                self.convert_internal(
                    source,
                    &temp_apr,
                    FormatType::SafeTensors,
                    FormatType::Apr,
                    opts,
                )?;
                self.convert_internal(&temp_apr, target, FormatType::Apr, FormatType::Gguf, opts)?;
                let _ = std::fs::remove_file(temp_apr);
                Ok(())
            }

            // Same format - just copy
            (f1, f2) if f1 == f2 => {
                std::fs::copy(source, target).map_err(|e| AprenderError::FormatError {
                    message: format!("Copy failed: {e}"),
                })?;
                Ok(())
            }

            _ => Err(AprenderError::FormatError {
                message: format!("Conversion {source_format} → {target_format} not supported"),
            }),
        }
    }

    fn compare_files(&self, file_a: &Path, file_b: &Path) -> Result<VerificationReport> {
        let inspection_a = self.inspect(file_a)?;
        let inspection_b = self.inspect(file_b)?;

        // Compare tensor counts
        if inspection_a.tensors.len() != inspection_b.tensors.len() {
            return Ok(VerificationReport {
                is_equivalent: false,
                max_diff: f32::INFINITY,
                mean_diff: f32::INFINITY,
                tensor_diffs: BTreeMap::new(),
                changed_metadata: Vec::new(),
                failed_tensors: vec!["Tensor count mismatch".to_string()],
            });
        }

        // Compare tensor statistics (Toyota Way: no SATD, implement now)
        // Uses statistical comparison: if stats match closely, tensors are equivalent
        let mut tensor_diffs = BTreeMap::new();
        let mut max_diff: f32 = 0.0;
        let mut total_diff: f32 = 0.0;
        let mut diff_count: usize = 0;
        let mut failed_tensors = Vec::new();

        for (tensor_a, tensor_b) in inspection_a.tensors.iter().zip(inspection_b.tensors.iter()) {
            // Check tensor names match
            if tensor_a.name != tensor_b.name {
                failed_tensors.push(format!(
                    "Tensor name mismatch: {} vs {}",
                    tensor_a.name, tensor_b.name
                ));
                continue;
            }

            // Check shapes match
            if tensor_a.shape != tensor_b.shape {
                failed_tensors.push(format!(
                    "{}: shape mismatch {:?} vs {:?}",
                    tensor_a.name, tensor_a.shape, tensor_b.shape
                ));
                continue;
            }

            // Compare statistics if available
            match (&tensor_a.stats, &tensor_b.stats) {
                (Some(stats_a), Some(stats_b)) => {
                    let mean_diff = (stats_a.mean - stats_b.mean).abs();
                    let std_diff = (stats_a.std - stats_b.std).abs();
                    let min_diff = (stats_a.min - stats_b.min).abs();
                    let max_val_diff = (stats_a.max - stats_b.max).abs();

                    let tensor_max_diff = mean_diff.max(std_diff).max(min_diff).max(max_val_diff);
                    tensor_diffs.insert(tensor_a.name.clone(), tensor_max_diff);

                    max_diff = max_diff.max(tensor_max_diff);
                    total_diff += tensor_max_diff;
                    diff_count += 1;
                }
                _ => {
                    // No stats available, assume matching if shapes match
                    tensor_diffs.insert(tensor_a.name.clone(), 0.0);
                }
            }
        }

        let mean_diff = if diff_count > 0 {
            total_diff / diff_count as f32
        } else {
            0.0
        };

        // Threshold: max_diff < 1e-4 is considered equivalent (float precision)
        let is_equivalent = failed_tensors.is_empty() && max_diff < 1e-4;

        Ok(VerificationReport {
            is_equivalent,
            max_diff,
            mean_diff,
            tensor_diffs,
            changed_metadata: Vec::new(),
            failed_tensors,
        })
    }
}

impl Default for RosettaStone {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests (Extreme TDD - Popperian Falsification)
// ============================================================================

#[cfg(test)]
mod tests;
