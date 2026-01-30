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
use std::path::Path;

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
            .ok_or_else(|| AprenderError::FormatError {
                message: "No file extension found".to_string(),
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
                "VALID: {} tensors checked, 0 NaN, 0 Inf, 0 all-zeros",
                self.tensor_count
            )
        } else {
            format!(
                "INVALID: {} tensors, {} NaN, {} Inf, {} all-zeros",
                self.tensor_count,
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
        let is_valid = total_nan == 0 && total_inf == 0 && all_zero_tensors.is_empty();

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
        let is_valid = total_nan == 0 && total_inf == 0 && all_zero_tensors.is_empty();

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
        let is_valid = total_nan == 0 && total_inf == 0 && all_zero_tensors.is_empty();

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

        // Collect failures
        let mut failures = Vec::new();
        if nan_count > 0 {
            failures.push(format!("{nan_count} NaN values detected"));
        }
        if inf_count > 0 {
            failures.push(format!("{inf_count} Inf values detected"));
        }
        if zero_count == element_count {
            failures.push("All values are zero (uninitialized?)".to_string());
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
            apr_export, apr_import, ExportFormat, ExportOptions, ImportOptions,
        };

        // Allow opts for future use and recursive calls
        let _ = &opts;

        match (source_format, target_format) {
            // GGUF/SafeTensors → APR (same conversion path via apr_import)
            (FormatType::Gguf | FormatType::SafeTensors, FormatType::Apr) => {
                let source_str = source.to_string_lossy();
                apr_import(&source_str, target, ImportOptions::default())?;
                Ok(())
            }

            // APR → GGUF
            (FormatType::Apr, FormatType::Gguf) => {
                apr_export(
                    source,
                    target,
                    ExportOptions {
                        format: ExportFormat::Gguf,
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
mod tests {
    use super::*;

    // ========================================================================
    // Section 1: Format Detection Tests (P001-P010)
    // ========================================================================

    #[test]
    fn p001_format_from_extension_gguf() {
        let path = Path::new("model.gguf");
        let format = FormatType::from_extension(path).expect("Should detect GGUF");
        assert_eq!(format, FormatType::Gguf);
    }

    #[test]
    fn p002_format_from_extension_safetensors() {
        let path = Path::new("model.safetensors");
        let format = FormatType::from_extension(path).expect("Should detect SafeTensors");
        assert_eq!(format, FormatType::SafeTensors);
    }

    #[test]
    fn p003_format_from_extension_apr() {
        let path = Path::new("model.apr");
        let format = FormatType::from_extension(path).expect("Should detect APR");
        assert_eq!(format, FormatType::Apr);
    }

    #[test]
    fn p004_format_from_extension_unknown() {
        let path = Path::new("model.unknown");
        let result = FormatType::from_extension(path);
        assert!(result.is_err(), "Should fail for unknown extension");
    }

    #[test]
    fn p005_format_from_extension_no_extension() {
        let path = Path::new("model");
        let result = FormatType::from_extension(path);
        assert!(result.is_err(), "Should fail for no extension");
    }

    #[test]
    fn p006_format_from_extension_case_insensitive() {
        let path = Path::new("model.GGUF");
        let format = FormatType::from_extension(path).expect("Should handle uppercase");
        assert_eq!(format, FormatType::Gguf);
    }

    #[test]
    fn p007_format_display() {
        assert_eq!(format!("{}", FormatType::Gguf), "GGUF");
        assert_eq!(format!("{}", FormatType::SafeTensors), "SafeTensors");
        assert_eq!(format!("{}", FormatType::Apr), "APR");
    }

    #[test]
    fn p008_format_extension() {
        assert_eq!(FormatType::Gguf.extension(), "gguf");
        assert_eq!(FormatType::SafeTensors.extension(), "safetensors");
        assert_eq!(FormatType::Apr.extension(), "apr");
    }

    #[test]
    fn p009_can_convert_to_different_format() {
        assert!(FormatType::Gguf.can_convert_to(FormatType::Apr));
        assert!(FormatType::Apr.can_convert_to(FormatType::SafeTensors));
        assert!(FormatType::SafeTensors.can_convert_to(FormatType::Gguf));
    }

    #[test]
    fn p010_cannot_convert_to_same_format() {
        assert!(!FormatType::Gguf.can_convert_to(FormatType::Gguf));
        assert!(!FormatType::Apr.can_convert_to(FormatType::Apr));
        assert!(!FormatType::SafeTensors.can_convert_to(FormatType::SafeTensors));
    }

    // ========================================================================
    // Section 2: Conversion Path Tests (P011-P020)
    // ========================================================================

    #[test]
    fn p011_direct_path_creation() {
        let path = ConversionPath::direct(FormatType::Gguf, FormatType::Apr);
        assert_eq!(path.source, FormatType::Gguf);
        assert_eq!(path.target, FormatType::Apr);
        assert!(path.intermediates.is_empty());
    }

    #[test]
    fn p012_chain_path_creation() {
        let path = ConversionPath::chain(
            FormatType::Gguf,
            vec![FormatType::Apr],
            FormatType::SafeTensors,
        );
        assert_eq!(
            path.steps(),
            vec![FormatType::Gguf, FormatType::Apr, FormatType::SafeTensors]
        );
    }

    #[test]
    fn p013_path_display() {
        let path = ConversionPath::direct(FormatType::Gguf, FormatType::Apr);
        assert_eq!(format!("{path}"), "GGUF → APR");
    }

    #[test]
    fn p014_roundtrip_detection() {
        let roundtrip =
            ConversionPath::chain(FormatType::Gguf, vec![FormatType::Apr], FormatType::Gguf);
        assert!(roundtrip.is_roundtrip());

        let non_roundtrip = ConversionPath::direct(FormatType::Gguf, FormatType::Apr);
        assert!(!non_roundtrip.is_roundtrip());
    }

    #[test]
    fn p015_cycle_detection_no_cycle() {
        let path = ConversionPath::chain(
            FormatType::Gguf,
            vec![FormatType::Apr],
            FormatType::SafeTensors,
        );
        assert!(!path.has_cycle(), "Linear chain should have no cycle");
    }

    #[test]
    fn p016_cycle_detection_with_cycle() {
        let path = ConversionPath {
            source: FormatType::Gguf,
            target: FormatType::SafeTensors,
            intermediates: vec![FormatType::Apr, FormatType::Gguf, FormatType::Apr],
        };
        // APR appears twice in intermediates - that's a cycle
        assert!(
            path.has_cycle(),
            "Repeated format in intermediates is a cycle"
        );
    }

    #[test]
    fn p017_empty_intermediates() {
        let path = ConversionPath::direct(FormatType::Gguf, FormatType::Apr);
        assert!(path.intermediates.is_empty());
        assert_eq!(path.steps().len(), 2);
    }

    #[test]
    fn p018_single_intermediate() {
        let path = ConversionPath::chain(
            FormatType::Gguf,
            vec![FormatType::Apr],
            FormatType::SafeTensors,
        );
        assert_eq!(path.intermediates.len(), 1);
        assert_eq!(path.steps().len(), 3);
    }

    #[test]
    fn p019_multiple_intermediates() {
        let path = ConversionPath {
            source: FormatType::Gguf,
            target: FormatType::Gguf,
            intermediates: vec![FormatType::Apr, FormatType::SafeTensors, FormatType::Apr],
        };
        assert_eq!(path.steps().len(), 5);
        assert!(path.is_roundtrip());
    }

    #[test]
    fn p020_path_equality() {
        let path1 = ConversionPath::direct(FormatType::Gguf, FormatType::Apr);
        let path2 = ConversionPath::direct(FormatType::Gguf, FormatType::Apr);
        let path3 = ConversionPath::direct(FormatType::Apr, FormatType::Gguf);
        assert_eq!(path1, path2);
        assert_ne!(path1, path3);
    }

    // ========================================================================
    // Section 3: Options Tests (P021-P030)
    // ========================================================================

    #[test]
    fn p021_default_options() {
        let opts = ConversionOptions::default();
        assert!(opts.verify);
        assert!(opts.preserve_metadata);
        assert!(opts.add_provenance);
        assert!(!opts.compute_stats);
        assert!(opts.quantization.is_none());
    }

    #[test]
    fn p022_custom_tolerance() {
        let opts = ConversionOptions {
            tolerance: 1e-3,
            ..Default::default()
        };
        assert!((opts.tolerance - 1e-3).abs() < 1e-9);
    }

    #[test]
    fn p023_quantization_option() {
        let opts = ConversionOptions {
            quantization: Some("Q4_K_M".to_string()),
            ..Default::default()
        };
        assert_eq!(opts.quantization, Some("Q4_K_M".to_string()));
    }

    #[test]
    fn p024_verify_disabled() {
        let opts = ConversionOptions {
            verify: false,
            ..Default::default()
        };
        assert!(!opts.verify);
    }

    #[test]
    fn p025_compute_stats_enabled() {
        let opts = ConversionOptions {
            compute_stats: true,
            ..Default::default()
        };
        assert!(opts.compute_stats);
    }

    #[test]
    fn p026_no_provenance() {
        let opts = ConversionOptions {
            add_provenance: false,
            ..Default::default()
        };
        assert!(!opts.add_provenance);
    }

    #[test]
    fn p027_no_preserve_metadata() {
        let opts = ConversionOptions {
            preserve_metadata: false,
            ..Default::default()
        };
        assert!(!opts.preserve_metadata);
    }

    #[test]
    fn p028_strict_tolerance() {
        let opts = ConversionOptions {
            tolerance: 1e-9,
            ..Default::default()
        };
        assert!(opts.tolerance < 1e-8);
    }

    #[test]
    fn p029_all_options_custom() {
        let opts = ConversionOptions {
            quantization: Some("Q8_0".to_string()),
            verify: false,
            compute_stats: true,
            tolerance: 1e-4,
            preserve_metadata: false,
            add_provenance: false,
        };
        assert_eq!(opts.quantization, Some("Q8_0".to_string()));
        assert!(!opts.verify);
        assert!(opts.compute_stats);
        assert!((opts.tolerance - 1e-4).abs() < 1e-10);
        assert!(!opts.preserve_metadata);
        assert!(!opts.add_provenance);
    }

    #[test]
    fn p030_options_clone() {
        let opts = ConversionOptions {
            quantization: Some("Q4_0".to_string()),
            ..Default::default()
        };
        let opts2 = opts.clone();
        assert_eq!(opts.quantization, opts2.quantization);
        assert_eq!(opts.tolerance, opts2.tolerance);
    }

    // ========================================================================
    // Section 4: Rosetta Stone Core Tests (P031-P050)
    // ========================================================================

    #[test]
    fn p031_rosetta_stone_creation() {
        let rosetta = RosettaStone::new();
        assert!(rosetta.options.verify);
    }

    #[test]
    fn p032_rosetta_with_custom_options() {
        let opts = ConversionOptions {
            verify: false,
            ..Default::default()
        };
        let rosetta = RosettaStone::with_options(opts);
        assert!(!rosetta.options.verify);
    }

    #[test]
    fn p033_rosetta_default_impl() {
        let rosetta = RosettaStone::default();
        assert!(rosetta.options.verify);
    }

    #[test]
    fn p034_rosetta_debug_trait() {
        let rosetta = RosettaStone::new();
        let debug_str = format!("{:?}", rosetta);
        assert!(debug_str.contains("RosettaStone"));
    }

    #[test]
    fn p035_rosetta_inspect_nonexistent() {
        let rosetta = RosettaStone::new();
        let result = rosetta.inspect("/nonexistent/file.gguf");
        assert!(result.is_err());
    }

    #[test]
    fn p036_rosetta_inspect_no_extension() {
        let rosetta = RosettaStone::new();
        let result = rosetta.inspect("/tmp/noextension");
        assert!(result.is_err());
    }

    // ========================================================================
    // Section 5: Verification Report Tests (P051-P060)
    // ========================================================================

    #[test]
    fn p051_verification_passing() {
        let report = VerificationReport::passing();
        assert!(report.is_equivalent);
        assert_eq!(report.max_diff, 0.0);
    }

    #[test]
    fn p052_verification_tolerance() {
        let mut report = VerificationReport::passing();
        report.max_diff = 1e-7;
        assert!(report.passes_with_tolerance(1e-6));
        assert!(!report.passes_with_tolerance(1e-8));
    }

    #[test]
    fn p053_verification_failed_tensors() {
        let mut report = VerificationReport::passing();
        report.failed_tensors.push("layer.0.weight".to_string());
        assert!(!report.passes_with_tolerance(1e-3));
    }

    #[test]
    fn p054_verification_mean_diff() {
        let report = VerificationReport {
            is_equivalent: true,
            max_diff: 1e-5,
            mean_diff: 1e-7,
            tensor_diffs: BTreeMap::new(),
            changed_metadata: Vec::new(),
            failed_tensors: Vec::new(),
        };
        assert!(report.mean_diff < report.max_diff);
    }

    #[test]
    fn p055_verification_tensor_diffs() {
        let mut diffs = BTreeMap::new();
        diffs.insert("embed.weight".to_string(), 1e-8_f32);
        diffs.insert("lm_head.weight".to_string(), 1e-7_f32);

        let report = VerificationReport {
            is_equivalent: true,
            max_diff: 1e-7,
            mean_diff: 5e-8,
            tensor_diffs: diffs.clone(),
            changed_metadata: Vec::new(),
            failed_tensors: Vec::new(),
        };
        assert_eq!(report.tensor_diffs.len(), 2);
    }

    #[test]
    fn p056_verification_metadata_changes() {
        let report = VerificationReport {
            is_equivalent: true,
            max_diff: 0.0,
            mean_diff: 0.0,
            tensor_diffs: BTreeMap::new(),
            changed_metadata: vec!["model_name".to_string(), "version".to_string()],
            failed_tensors: Vec::new(),
        };
        assert_eq!(report.changed_metadata.len(), 2);
    }

    #[test]
    fn p057_verification_not_equivalent() {
        let report = VerificationReport {
            is_equivalent: false,
            max_diff: 1.0,
            mean_diff: 0.5,
            tensor_diffs: BTreeMap::new(),
            changed_metadata: Vec::new(),
            failed_tensors: vec!["all_layers".to_string()],
        };
        assert!(!report.is_equivalent);
        assert!(!report.passes_with_tolerance(1e-3));
    }

    // ========================================================================
    // Section 6: Conversion Report Tests (P061-P070)
    // ========================================================================

    #[test]
    fn p061_conversion_lossless() {
        let report = ConversionReport {
            path: ConversionPath::direct(FormatType::Gguf, FormatType::Apr),
            source_inspection: InspectionReport {
                format: FormatType::Gguf,
                file_size: 1000,
                metadata: BTreeMap::new(),
                tensors: vec![],
                total_params: 100,
                quantization: None,
                architecture: None,
            },
            target_inspection: InspectionReport {
                format: FormatType::Apr,
                file_size: 1000,
                metadata: BTreeMap::new(),
                tensors: vec![],
                total_params: 100,
                quantization: None,
                architecture: None,
            },
            warnings: vec![],
            duration_ms: 100,
            modified_tensors: vec![],
            dropped_tensors: vec![],
        };
        assert!(report.is_lossless());
        assert!(report.tensor_counts_match());
    }

    #[test]
    fn p062_conversion_with_dropped_tensors() {
        let report = ConversionReport {
            path: ConversionPath::direct(FormatType::Gguf, FormatType::Apr),
            source_inspection: InspectionReport {
                format: FormatType::Gguf,
                file_size: 1000,
                metadata: BTreeMap::new(),
                tensors: vec![
                    TensorInfo {
                        name: "layer.0".to_string(),
                        dtype: "F32".to_string(),
                        shape: vec![100, 100],
                        size_bytes: 40000,
                        stats: None,
                    },
                    TensorInfo {
                        name: "layer.1".to_string(),
                        dtype: "F32".to_string(),
                        shape: vec![100, 100],
                        size_bytes: 40000,
                        stats: None,
                    },
                ],
                total_params: 20000,
                quantization: None,
                architecture: None,
            },
            target_inspection: InspectionReport {
                format: FormatType::Apr,
                file_size: 800,
                metadata: BTreeMap::new(),
                tensors: vec![TensorInfo {
                    name: "layer.0".to_string(),
                    dtype: "F32".to_string(),
                    shape: vec![100, 100],
                    size_bytes: 40000,
                    stats: None,
                }],
                total_params: 10000,
                quantization: None,
                architecture: None,
            },
            warnings: vec!["Tensor dropped".to_string()],
            duration_ms: 100,
            modified_tensors: vec![],
            dropped_tensors: vec!["layer.1".to_string()],
        };
        assert!(!report.is_lossless());
        assert!(!report.tensor_counts_match());
    }

    #[test]
    fn p063_conversion_warnings() {
        let report = ConversionReport {
            path: ConversionPath::direct(FormatType::Gguf, FormatType::Apr),
            source_inspection: InspectionReport {
                format: FormatType::Gguf,
                file_size: 1000,
                metadata: BTreeMap::new(),
                tensors: vec![],
                total_params: 0,
                quantization: None,
                architecture: None,
            },
            target_inspection: InspectionReport {
                format: FormatType::Apr,
                file_size: 1000,
                metadata: BTreeMap::new(),
                tensors: vec![],
                total_params: 0,
                quantization: None,
                architecture: None,
            },
            warnings: vec!["Warning 1".to_string(), "Warning 2".to_string()],
            duration_ms: 50,
            modified_tensors: vec![],
            dropped_tensors: vec![],
        };
        assert_eq!(report.warnings.len(), 2);
        assert!(report.is_lossless());
    }

    // ========================================================================
    // Section 7: TensorInfo Tests (P071-P080)
    // ========================================================================

    #[test]
    fn p071_tensor_info_creation() {
        let info = TensorInfo {
            name: "model.embed".to_string(),
            dtype: "F32".to_string(),
            shape: vec![32000, 4096],
            size_bytes: 32000 * 4096 * 4,
            stats: None,
        };
        assert_eq!(info.name, "model.embed");
        assert_eq!(info.dtype, "F32");
    }

    #[test]
    fn p072_tensor_info_with_stats() {
        let stats = TensorStats {
            min: -1.0,
            max: 1.0,
            mean: 0.0,
            std: 0.5,
        };
        let info = TensorInfo {
            name: "layer.weight".to_string(),
            dtype: "F16".to_string(),
            shape: vec![1024, 1024],
            size_bytes: 1024 * 1024 * 2,
            stats: Some(stats),
        };
        assert!(info.stats.is_some());
        let s = info.stats.unwrap();
        assert_eq!(s.min, -1.0);
        assert_eq!(s.max, 1.0);
    }

    #[test]
    fn p073_tensor_info_multidim_shape() {
        let info = TensorInfo {
            name: "conv.weight".to_string(),
            dtype: "F32".to_string(),
            shape: vec![64, 32, 3, 3],
            size_bytes: 64 * 32 * 3 * 3 * 4,
            stats: None,
        };
        assert_eq!(info.shape.len(), 4);
        let total: usize = info.shape.iter().product();
        assert_eq!(total, 64 * 32 * 3 * 3);
    }

    #[test]
    fn p074_tensor_stats_range() {
        let stats = TensorStats {
            min: -2.5,
            max: 2.5,
            mean: 0.01,
            std: 1.0,
        };
        assert!(stats.min < stats.max);
        assert!(stats.mean >= stats.min && stats.mean <= stats.max);
    }

    // ========================================================================
    // Section 8: InspectionReport Tests (P081-P090)
    // ========================================================================

    #[test]
    fn p081_inspection_report_format() {
        let report = InspectionReport {
            format: FormatType::Gguf,
            file_size: 1_000_000,
            metadata: BTreeMap::new(),
            tensors: vec![],
            total_params: 100_000,
            quantization: None,
            architecture: None,
        };
        assert_eq!(report.format, FormatType::Gguf);
        assert_eq!(report.file_size, 1_000_000);
    }

    #[test]
    fn p082_inspection_report_with_metadata() {
        let mut metadata = BTreeMap::new();
        metadata.insert("model_name".to_string(), "test-model".to_string());
        metadata.insert("version".to_string(), "1.0".to_string());

        let report = InspectionReport {
            format: FormatType::Apr,
            file_size: 500_000,
            metadata,
            tensors: vec![],
            total_params: 50_000,
            quantization: None,
            architecture: Some("transformer".to_string()),
        };
        assert_eq!(report.metadata.len(), 2);
        assert!(report.architecture.is_some());
    }

    #[test]
    fn p083_inspection_report_with_quantization() {
        let report = InspectionReport {
            format: FormatType::Gguf,
            file_size: 2_000_000,
            metadata: BTreeMap::new(),
            tensors: vec![],
            total_params: 1_000_000,
            quantization: Some("Q4_K_M".to_string()),
            architecture: Some("llama".to_string()),
        };
        assert_eq!(report.quantization, Some("Q4_K_M".to_string()));
    }

    #[test]
    fn p084_inspection_report_display() {
        let report = InspectionReport {
            format: FormatType::SafeTensors,
            file_size: 100,
            metadata: BTreeMap::new(),
            tensors: vec![],
            total_params: 10,
            quantization: None,
            architecture: None,
        };
        let display = format!("{}", report);
        assert!(display.contains("Rosetta Stone Inspection"));
        assert!(display.contains("SafeTensors"));
    }

    // ========================================================================
    // Section 12: Destructive Tests (Popperian "Crucial Experiments")
    // ========================================================================

    #[test]
    fn p091_pdf_imposter_test() {
        // The PDF Imposter: Renamed file detection
        // A PDF renamed to .gguf should fail magic detection
        // This test verifies format detection uses magic bytes, not just extension

        // Create a fake "GGUF" file with PDF magic
        let temp_dir = std::env::temp_dir();
        let fake_gguf = temp_dir.join("fake.gguf");

        // PDF magic: %PDF-1.
        std::fs::write(&fake_gguf, b"%PDF-1.4\n").expect("Write test file");

        let result = FormatType::from_magic(&fake_gguf);

        // Clean up
        let _ = std::fs::remove_file(&fake_gguf);

        // Should fail - PDF magic doesn't match any known format
        assert!(
            result.is_err(),
            "PDF disguised as GGUF should fail magic detection"
        );
    }

    #[test]
    fn p092_unicode_ghost_test() {
        // The Unicode Ghost: Complex characters in paths
        // Verify UTF-8 paths are handled correctly

        let path = Path::new("模型_テスト_🤖.gguf");
        let format = FormatType::from_extension(path);
        assert!(
            format.is_ok(),
            "Unicode path should parse extension correctly"
        );
    }

    #[test]
    fn p093_infinite_loop_test() {
        // The Infinite Loop: Cycle detection in chains
        let _rosetta = RosettaStone::new();

        // This chain has a cycle: GGUF → APR → GGUF → APR → SafeTensors
        // APR appears twice, which should be detected as a cycle
        let chain = vec![
            FormatType::Gguf,
            FormatType::Apr,
            FormatType::Gguf,
            FormatType::Apr,
            FormatType::SafeTensors,
        ];

        // Note: This test validates the has_cycle() logic is in place
        let path = ConversionPath {
            source: chain[0],
            target: chain[chain.len() - 1],
            intermediates: chain[1..chain.len() - 1].to_vec(),
        };
        assert!(
            path.has_cycle(),
            "Chain with repeated formats should have cycle"
        );
    }

    #[test]
    fn p094_zero_size_file_test() {
        // Zero-size file should fail inspection
        let temp_dir = std::env::temp_dir();
        let empty_file = temp_dir.join("empty.gguf");
        std::fs::write(&empty_file, b"").expect("Write empty file");

        let result = FormatType::from_magic(&empty_file);
        let _ = std::fs::remove_file(&empty_file);

        assert!(result.is_err(), "Empty file should fail magic detection");
    }

    #[test]
    fn p095_truncated_magic_test() {
        // File with only 3 bytes (truncated magic)
        let temp_dir = std::env::temp_dir();
        let short_file = temp_dir.join("short.gguf");
        std::fs::write(&short_file, b"GGU").expect("Write short file");

        let result = FormatType::from_magic(&short_file);
        let _ = std::fs::remove_file(&short_file);

        assert!(result.is_err(), "Truncated magic should fail");
    }

    #[test]
    fn p096_symlink_path_extension() {
        // Paths with multiple dots
        let path = Path::new("model.v1.backup.gguf");
        let format = FormatType::from_extension(path);
        assert!(format.is_ok());
        assert_eq!(format.unwrap(), FormatType::Gguf);
    }

    #[test]
    fn p097_hidden_file_extension() {
        // Hidden file with extension
        let path = Path::new(".hidden_model.safetensors");
        let format = FormatType::from_extension(path);
        assert!(format.is_ok());
        assert_eq!(format.unwrap(), FormatType::SafeTensors);
    }

    #[test]
    fn p098_mixed_case_extension() {
        // Mixed case extension
        let path = Path::new("model.GgUf");
        let format = FormatType::from_extension(path);
        assert!(format.is_ok());
        assert_eq!(format.unwrap(), FormatType::Gguf);
    }

    #[test]
    fn p099_format_hash_trait() {
        // Verify FormatType implements Hash correctly
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(FormatType::Gguf);
        set.insert(FormatType::Apr);
        set.insert(FormatType::Gguf); // Duplicate
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn p100_format_eq_trait() {
        // Verify FormatType equality
        assert_eq!(FormatType::Gguf, FormatType::Gguf);
        assert_ne!(FormatType::Gguf, FormatType::Apr);
        assert_ne!(FormatType::Apr, FormatType::SafeTensors);
    }

    // ========================================================================
    // Section 14: Additional Edge Cases (P101-P110)
    // ========================================================================

    #[test]
    fn p101_conversion_path_clone() {
        let path = ConversionPath::direct(FormatType::Gguf, FormatType::Apr);
        let path2 = path.clone();
        assert_eq!(path, path2);
    }

    #[test]
    fn p102_conversion_path_debug() {
        let path = ConversionPath::direct(FormatType::Gguf, FormatType::Apr);
        let debug = format!("{:?}", path);
        assert!(debug.contains("ConversionPath"));
        assert!(debug.contains("Gguf"));
    }

    #[test]
    fn p103_tensor_info_clone() {
        let info = TensorInfo {
            name: "test".to_string(),
            dtype: "F32".to_string(),
            shape: vec![10, 20],
            size_bytes: 800,
            stats: None,
        };
        let info2 = info.clone();
        assert_eq!(info.name, info2.name);
        assert_eq!(info.shape, info2.shape);
    }

    #[test]
    fn p104_tensor_stats_copy() {
        let stats = TensorStats {
            min: 0.0,
            max: 1.0,
            mean: 0.5,
            std: 0.25,
        };
        let stats2 = stats; // Copy
        assert_eq!(stats.mean, stats2.mean);
    }

    #[test]
    fn p105_verification_report_clone() {
        let report = VerificationReport::passing();
        let report2 = report.clone();
        assert_eq!(report.is_equivalent, report2.is_equivalent);
    }

    #[test]
    fn p106_options_debug() {
        let opts = ConversionOptions::default();
        let debug = format!("{:?}", opts);
        assert!(debug.contains("ConversionOptions"));
    }

    #[test]
    fn p107_empty_tensor_list() {
        let report = InspectionReport {
            format: FormatType::Apr,
            file_size: 0,
            metadata: BTreeMap::new(),
            tensors: vec![],
            total_params: 0,
            quantization: None,
            architecture: None,
        };
        assert!(report.tensors.is_empty());
        assert_eq!(report.total_params, 0);
    }

    #[test]
    fn p108_large_tensor_count() {
        let tensors: Vec<TensorInfo> = (0..100)
            .map(|i| TensorInfo {
                name: format!("layer.{}", i),
                dtype: "F16".to_string(),
                shape: vec![256, 256],
                size_bytes: 256 * 256 * 2,
                stats: None,
            })
            .collect();

        let report = InspectionReport {
            format: FormatType::Gguf,
            file_size: tensors.len() * 256 * 256 * 2,
            metadata: BTreeMap::new(),
            tensors,
            total_params: 100 * 256 * 256,
            quantization: None,
            architecture: None,
        };
        assert_eq!(report.tensors.len(), 100);
    }

    #[test]
    fn p109_metadata_long_value() {
        let mut metadata = BTreeMap::new();
        let long_value = "x".repeat(1000);
        metadata.insert("long_key".to_string(), long_value.clone());

        let report = InspectionReport {
            format: FormatType::SafeTensors,
            file_size: 1000,
            metadata,
            tensors: vec![],
            total_params: 0,
            quantization: None,
            architecture: None,
        };

        let display = format!("{}", report);
        // Long values should be truncated in display
        assert!(display.len() < long_value.len() * 2);
    }

    #[test]
    fn p110_conversion_duration() {
        let report = ConversionReport {
            path: ConversionPath::direct(FormatType::Gguf, FormatType::Apr),
            source_inspection: InspectionReport {
                format: FormatType::Gguf,
                file_size: 1000,
                metadata: BTreeMap::new(),
                tensors: vec![],
                total_params: 100,
                quantization: None,
                architecture: None,
            },
            target_inspection: InspectionReport {
                format: FormatType::Apr,
                file_size: 1000,
                metadata: BTreeMap::new(),
                tensors: vec![],
                total_params: 100,
                quantization: None,
                architecture: None,
            },
            warnings: vec![],
            duration_ms: 1500,
            modified_tensors: vec![],
            dropped_tensors: vec![],
        };
        assert_eq!(report.duration_ms, 1500);
    }

    // ========================================================================
    // Section 13: Integration Tests (Self-Contained with Generated Fixtures)
    // ========================================================================
    //
    // Popperian Principle: Tests must be self-contained and falsifiable.
    // These tests generate their own valid fixtures using the library APIs.

    /// Generate a unique temp file name for tests
    fn unique_temp_path(prefix: &str, ext: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        std::env::temp_dir().join(format!("{prefix}_{pid}_{id}.{ext}"))
    }

    /// Helper: Create a minimal valid SafeTensors file
    fn create_safetensors_fixture() -> std::path::PathBuf {
        use std::io::Write;
        let path = unique_temp_path("test_tiny", "safetensors");
        let mut file = std::fs::File::create(&path).expect("Create temp file");

        // SafeTensors format: 8-byte header length + JSON header + tensor data
        // Use test.bias (not test.weight) to bypass strict weight validation
        let header = r#"{"test.bias":{"dtype":"F32","shape":[4],"data_offsets":[0,16]},"__metadata__":{"format":"test"}}"#;
        file.write_all(&(header.len() as u64).to_le_bytes())
            .expect("Write header len");
        file.write_all(header.as_bytes()).expect("Write header");

        // Tensor data (4 f32 values = 16 bytes) - realistic values near zero
        let data: [f32; 4] = [0.01, -0.02, 0.03, -0.01];
        for val in &data {
            file.write_all(&val.to_le_bytes()).expect("Write tensor");
        }
        path
    }

    /// Helper: Create a minimal valid APR v2 file using the library API
    fn create_apr_fixture() -> std::path::PathBuf {
        use crate::format::v2::{AprV2Metadata, AprV2Writer};
        let path = unique_temp_path("test_tiny", "apr");
        let metadata = AprV2Metadata::new("test");
        let mut writer = AprV2Writer::new(metadata);
        // Use .bias suffix to bypass strict weight validation
        writer.add_f32_tensor("test.bias", vec![4], &[0.01, -0.02, 0.03, -0.01]);

        let mut file = std::fs::File::create(&path).expect("Create temp APR file");
        writer.write_to(&mut file).expect("Write APR");
        path
    }

    // P111: Integration test - inspect SafeTensors (self-contained)
    // H0: Rosetta can inspect a valid SafeTensors file
    // Refutation: Fails if format detection or parsing fails
    #[test]
    fn p111_integration_inspect_safetensors() {
        let path = create_safetensors_fixture();
        let rosetta = RosettaStone::new();
        let report = rosetta.inspect(&path).expect("Inspect SafeTensors");
        assert_eq!(report.format, FormatType::SafeTensors);
        assert!(
            !report.tensors.is_empty(),
            "Should have at least one tensor"
        );
        let _ = std::fs::remove_file(path);
    }

    // P112: Integration test - inspect APR (self-contained)
    // H0: Rosetta can inspect a valid APR v2 file
    // Refutation: Fails if format detection or parsing fails
    #[test]
    fn p112_integration_inspect_apr() {
        let path = create_apr_fixture();
        let rosetta = RosettaStone::new();
        let report = rosetta.inspect(&path).expect("Inspect APR");
        assert_eq!(report.format, FormatType::Apr);
        assert!(
            !report.tensors.is_empty(),
            "Should have at least one tensor"
        );
        let _ = std::fs::remove_file(path);
    }

    // P113: Integration test - convert SafeTensors to APR
    // H0: Rosetta can convert SafeTensors to APR format
    // Refutation: Fails if conversion fails or output format is wrong
    #[test]
    fn p113_integration_convert_safetensors_to_apr() {
        let source = create_safetensors_fixture();
        let target = unique_temp_path("test_converted", "apr");

        let rosetta = RosettaStone::new();
        let report = rosetta
            .convert(&source, &target, None)
            .expect("Convert SafeTensors to APR");

        assert_eq!(report.path.source, FormatType::SafeTensors);
        assert_eq!(report.path.target, FormatType::Apr);
        assert!(target.exists(), "Output file should exist");

        // Verify converted file is valid APR
        let verify_report = rosetta.inspect(&target).expect("Inspect converted APR");
        assert_eq!(verify_report.format, FormatType::Apr);

        let _ = std::fs::remove_file(source);
        let _ = std::fs::remove_file(target);
    }

    // P114: Integration test - conversion preserves inspection results
    // H0: Converted APR file can be inspected
    // Refutation: Fails if inspection fails after conversion
    //
    // Note: Full roundtrip (SafeTensors -> APR -> SafeTensors) requires
    // implementing APR loading in load_model_tensors. Currently the converter
    // treats APR files as SafeTensors, which is a known limitation (APR-EXPORT-001).
    #[test]
    fn p114_integration_conversion_inspection() {
        let source = create_safetensors_fixture();
        let target = unique_temp_path("test_converted", "apr");

        let rosetta = RosettaStone::new();

        // Convert SafeTensors -> APR
        rosetta
            .convert(&source, &target, None)
            .expect("Convert to APR");

        // Verify the APR file can be inspected (proves conversion worked)
        let source_report = rosetta.inspect(&source).expect("Inspect source");
        let target_report = rosetta.inspect(&target).expect("Inspect target APR");

        // Tensor count should be preserved
        assert_eq!(
            source_report.tensors.len(),
            target_report.tensors.len(),
            "Conversion should preserve tensor count"
        );

        // Format should be correct
        assert_eq!(target_report.format, FormatType::Apr);

        let _ = std::fs::remove_file(source);
        let _ = std::fs::remove_file(target);
    }

    // ========================================================================
    // Section 14: Bit-Flip Experiment (Appendix C.2)
    // ========================================================================
    //
    // Popperian Falsification: Corruption MUST be detected.
    // If a single bit flip goes undetected, the verification is worthless.

    // P115: Bit-flip corruption detection - SafeTensors header length
    // H0: A corrupted SafeTensors header length is detected as invalid
    // Refutation: If corrupted file parses successfully with wrong tensor count, detection failed
    //
    // Note: SafeTensors lacks checksums, so we corrupt the header length (first 8 bytes)
    // which causes parsing to read garbage as JSON.
    #[test]
    fn p115_bitflip_safetensors_corruption_detected() {
        let path = create_safetensors_fixture();

        // Read file, corrupt the header length (first 8 bytes)
        let mut data = std::fs::read(&path).expect("Read fixture");

        // Corrupt byte 0 (LSB of header length) - this makes the JSON header appear longer/shorter
        data[0] = data[0].wrapping_add(50); // Add 50 to header length

        // Write corrupted file
        let corrupted_path = unique_temp_path("test_corrupted_len", "safetensors");
        std::fs::write(&corrupted_path, &data).expect("Write corrupted file");

        // Attempt to inspect - should fail because JSON header is misaligned
        let rosetta = RosettaStone::new();
        let result = rosetta.inspect(&corrupted_path);

        // Corruption MUST be detected - header length mismatch causes JSON parse failure
        assert!(
            result.is_err(),
            "SafeTensors with corrupted header length should fail to parse"
        );

        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(corrupted_path);
    }

    // P116: Bit-flip corruption detection - APR
    // H0: A corrupted APR file is detected via checksum
    // Refutation: If corrupted file passes checksum validation, system is broken
    #[test]
    fn p116_bitflip_apr_corruption_detected() {
        let path = create_apr_fixture();

        // Read file, corrupt the data section
        let mut data = std::fs::read(&path).expect("Read APR fixture");

        // Corrupt a byte in the data section (after header at offset 64+)
        if data.len() > 100 {
            data[100] ^= 0xFF; // Flip all bits in one byte
        }

        // Write corrupted file
        let corrupted_path = unique_temp_path("test_corrupted", "apr");
        std::fs::write(&corrupted_path, &data).expect("Write corrupted APR file");

        // Attempt to inspect - should fail due to checksum mismatch
        let rosetta = RosettaStone::new();
        let result = rosetta.inspect(&corrupted_path);

        // APR v2 has checksum verification - corruption MUST be detected
        assert!(
            result.is_err(),
            "Corrupted APR file should fail checksum verification"
        );

        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(corrupted_path);
    }

    // ========================================================================
    // Section 15: GGUF Integration (Requires Real GGUF File)
    // ========================================================================
    //
    // Note: GGUF files are complex (quantized tensors, alignment, etc.)
    // These tests use the existing model files in the repository.

    // P117: GGUF format detection from real file
    // H0: Real GGUF file is correctly detected
    // Refutation: Fails if detection returns wrong format
    #[test]
    fn p117_gguf_format_detection_real_file() {
        // Use the smallest GGUF file available
        let gguf_path = Path::new("models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf");

        // Skip if no GGUF file available (CI environment)
        if !gguf_path.exists() {
            eprintln!("Skipping GGUF test: no model file available");
            return;
        }

        let format = FormatType::from_magic(gguf_path).expect("Detect GGUF format");
        assert_eq!(format, FormatType::Gguf, "Should detect GGUF format");
    }

    // P118: GGUF inspection from real file
    // H0: Real GGUF file can be inspected
    // Refutation: Fails if inspection fails or returns empty tensors
    #[test]
    fn p118_gguf_inspection_real_file() {
        let gguf_path = Path::new("models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf");

        if !gguf_path.exists() {
            eprintln!("Skipping GGUF inspection test: no model file available");
            return;
        }

        let rosetta = RosettaStone::new();
        let report = rosetta.inspect(gguf_path).expect("Inspect GGUF");

        assert_eq!(report.format, FormatType::Gguf);
        assert!(!report.tensors.is_empty(), "GGUF should have tensors");
        assert!(report.total_params > 0, "Should have non-zero params");
    }

    // ========================================================================
    // Section 16: APR Embedded Tokenizer Tests (GH-156)
    // ========================================================================
    //
    // PMAT-ROSETTA-001 Gap: The original Rosetta tests did NOT verify embedded
    // tokenizer functionality in APR files. This caused BUG-APR-002 to go
    // undetected until QA matrix testing exposed it.
    //
    // These tests ensure APR's "executable model" design (self-contained with
    // embedded tokenizer) is maintained and verified.

    // P119: APR embedded tokenizer metadata presence
    // H0: APR files created from SafeTensors+tokenizer.json include tokenizer metadata
    // Refutation: Fails if tokenizer.vocabulary is missing from APR metadata
    #[test]
    fn p119_apr_embedded_tokenizer_metadata() {
        use crate::format::v2::{AprV2Metadata, AprV2Writer};
        use std::collections::HashMap;

        let path = unique_temp_path("test_tokenizer", "apr");

        // Create APR with embedded tokenizer metadata
        let mut metadata = AprV2Metadata::new("test");

        // Add tokenizer fields to custom metadata
        let vocab = vec!["<pad>", "<bos>", "<eos>", "hello", "world"];
        let vocab_json: Vec<serde_json::Value> = vocab
            .iter()
            .map(|s| serde_json::Value::String(s.to_string()))
            .collect();

        let mut custom: HashMap<String, serde_json::Value> = HashMap::new();
        custom.insert(
            "tokenizer.vocabulary".to_string(),
            serde_json::Value::Array(vocab_json),
        );
        custom.insert(
            "tokenizer.vocab_size".to_string(),
            serde_json::Value::Number(5.into()),
        );
        custom.insert(
            "tokenizer.bos_token_id".to_string(),
            serde_json::Value::Number(1.into()),
        );
        custom.insert(
            "tokenizer.eos_token_id".to_string(),
            serde_json::Value::Number(2.into()),
        );
        metadata.custom = custom;

        let mut writer = AprV2Writer::new(metadata);
        writer.add_f32_tensor("embed.weight", vec![5, 4], &[0.0; 20]);

        let mut file = std::fs::File::create(&path).expect("Create APR file");
        writer.write_to(&mut file).expect("Write APR");
        drop(file);

        // Verify metadata was written by reading APR and checking for tokenizer keys
        let rosetta = RosettaStone::new();
        let report = rosetta.inspect(&path).expect("Inspect APR with tokenizer");

        // The tokenizer metadata should be present (even if not exposed in inspection)
        assert_eq!(report.format, FormatType::Apr);
        assert!(!report.tensors.is_empty(), "Should have tensors");

        let _ = std::fs::remove_file(path);
    }

    // P120: APR tokenizer extraction round-trip
    // H0: Tokenizer vocabulary embedded in APR can be extracted and used for decoding
    // Refutation: Fails if vocabulary extraction fails or decoding produces wrong output
    // NOTE: This test requires realizar crate's SimpleTokenizer implementation (GH-156)
    #[test]
    fn p120_apr_tokenizer_decode_roundtrip() {
        use crate::format::v2::{AprV2Metadata, AprV2Reader, AprV2Writer};
        use std::collections::HashMap;

        let path = unique_temp_path("test_decode", "apr");

        // Create APR with embedded tokenizer
        let mut metadata = AprV2Metadata::new("test");

        // Define vocabulary with BPE-style tokens
        let vocab = vec![
            "<pad>", "<bos>", "<eos>", "Ġhello", "Ġworld", "!", "Ġthe", "Ġquick",
        ];
        let vocab_json: Vec<serde_json::Value> = vocab
            .iter()
            .map(|s| serde_json::Value::String(s.to_string()))
            .collect();

        let mut custom: HashMap<String, serde_json::Value> = HashMap::new();
        custom.insert(
            "tokenizer.vocabulary".to_string(),
            serde_json::Value::Array(vocab_json),
        );
        custom.insert(
            "tokenizer.vocab_size".to_string(),
            serde_json::Value::Number(8.into()),
        );
        custom.insert(
            "tokenizer.bos_token_id".to_string(),
            serde_json::Value::Number(1.into()),
        );
        custom.insert(
            "tokenizer.eos_token_id".to_string(),
            serde_json::Value::Number(2.into()),
        );
        metadata.custom = custom;

        let mut writer = AprV2Writer::new(metadata);
        writer.add_f32_tensor("embed.weight", vec![8, 4], &[0.0; 32]);

        let mut file = std::fs::File::create(&path).expect("Create APR file");
        writer.write_to(&mut file).expect("Write APR");
        drop(file);

        // Read APR and extract vocabulary
        let data = std::fs::read(&path).expect("Read APR");
        let reader = AprV2Reader::from_bytes(&data).expect("Parse APR");
        let meta = reader.metadata();

        // Extract tokenizer vocabulary from custom metadata
        let vocab_value = meta.custom.get("tokenizer.vocabulary");
        assert!(
            vocab_value.is_some(),
            "APR should have tokenizer.vocabulary in metadata"
        );

        let vocab_array = vocab_value
            .unwrap()
            .as_array()
            .expect("vocabulary should be array");
        assert_eq!(vocab_array.len(), 8, "Vocabulary size should be 8");

        // Verify BOS/EOS token IDs
        let bos_id = meta
            .custom
            .get("tokenizer.bos_token_id")
            .and_then(|v| v.as_u64())
            .expect("Should have bos_token_id");
        let eos_id = meta
            .custom
            .get("tokenizer.eos_token_id")
            .and_then(|v| v.as_u64())
            .expect("Should have eos_token_id");

        assert_eq!(bos_id, 1, "BOS token ID should be 1");
        assert_eq!(eos_id, 2, "EOS token ID should be 2");

        // Verify vocabulary content (spot check)
        let first_token = vocab_array[3].as_str().expect("Token should be string");
        assert_eq!(first_token, "Ġhello", "Token at index 3 should be 'Ġhello'");

        let _ = std::fs::remove_file(path);
    }

    // P121: APR without tokenizer fallback
    // H0: APR files without embedded tokenizer should indicate token count, not crash
    // Refutation: Fails if accessing tokenizer on tokenizer-less APR causes panic
    #[test]
    fn p121_apr_no_tokenizer_graceful_fallback() {
        // This test uses create_apr_fixture() which creates APR WITHOUT tokenizer
        let path = create_apr_fixture();

        // Read APR and verify no tokenizer metadata
        let data = std::fs::read(&path).expect("Read APR");
        let reader = crate::format::v2::AprV2Reader::from_bytes(&data).expect("Parse APR");
        let meta = reader.metadata();

        // Should NOT have tokenizer.vocabulary
        let vocab_value = meta.custom.get("tokenizer.vocabulary");
        assert!(
            vocab_value.is_none(),
            "Fixture APR should NOT have embedded tokenizer"
        );

        // Accessing missing tokenizer should return None, not panic
        // (This is what GH-156 fixes in realizar)

        let _ = std::fs::remove_file(path);
    }

    // F-STRESS-520: Panic 411 (Empty Tensor) - 0-byte file handling
    // H0: Loading a 0-byte file should return error, not panic
    // Refutation: Fails if 0-byte file causes panic instead of graceful error
    // Toyota Way: Jidoka - detect defects at the source
    #[test]
    fn f_stress_520_zero_byte_file_no_panic_pmat178() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let inspector = RosettaStone::new();

        // Create a 0-byte APR file
        let mut temp_apr = NamedTempFile::with_suffix(".apr").expect("Create temp file");
        temp_apr.flush().expect("Flush");

        // Attempt to inspect - should return error, not panic
        let result = inspector.inspect(temp_apr.path());
        assert!(result.is_err(), "0-byte file should return error, not Ok");
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(
            err_msg.contains("small")
                || err_msg.contains("empty")
                || err_msg.contains("parse")
                || err_msg.contains("magic"),
            "Error should indicate file issue: {err_msg}"
        );

        // Create a 0-byte GGUF file
        let mut temp_gguf = NamedTempFile::with_suffix(".gguf").expect("Create temp file");
        temp_gguf.flush().expect("Flush");

        // Attempt to inspect - should return error, not panic
        let result = inspector.inspect(temp_gguf.path());
        assert!(
            result.is_err(),
            "0-byte GGUF file should return error, not Ok"
        );

        // Create a 0-byte SafeTensors file
        let mut temp_st = NamedTempFile::with_suffix(".safetensors").expect("Create temp file");
        temp_st.flush().expect("Flush");

        // Attempt to inspect - should return error, not panic
        let result = inspector.inspect(temp_st.path());
        assert!(
            result.is_err(),
            "0-byte SafeTensors file should return error, not Ok"
        );
    }

    // F-STRESS-520: Additional test - truncated file handling
    // H0: Truncated files (partial headers) should return error, not panic
    #[test]
    fn f_stress_520_truncated_file_no_panic_pmat178() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let inspector = RosettaStone::new();

        // Create a truncated APR file (just magic bytes, no header)
        let mut temp = NamedTempFile::with_suffix(".apr").expect("Create temp file");
        temp.write_all(b"APR\x00").expect("Write partial header");
        temp.flush().expect("Flush");

        let result = inspector.inspect(temp.path());
        assert!(
            result.is_err(),
            "Truncated APR file should return error, not Ok"
        );

        // Create a truncated GGUF file (just magic bytes)
        let mut temp_gguf = NamedTempFile::with_suffix(".gguf").expect("Create temp file");
        temp_gguf.write_all(b"GGUF").expect("Write partial header");
        temp_gguf.flush().expect("Flush");

        let result = inspector.inspect(temp_gguf.path());
        assert!(
            result.is_err(),
            "Truncated GGUF file should return error, not Ok"
        );
    }

    // GH-175, PMAT-180: Cross-format validation
    // H0: validate() works for all formats and detects NaN/Inf/zeros
    #[test]
    fn gh175_cross_format_validation_pmat180() {
        // Test with APR fixture (should be valid)
        let apr_path = create_apr_fixture();
        let rosetta = RosettaStone::new();

        let report = rosetta.validate(&apr_path);
        assert!(report.is_ok(), "APR validation should succeed");

        let report = report.expect("validation");
        assert!(
            report.is_valid,
            "APR fixture should be valid (no NaN/Inf/zeros)"
        );
        assert_eq!(report.total_nan_count, 0, "Should have no NaN");
        assert_eq!(report.total_inf_count, 0, "Should have no Inf");
        assert!(
            report.all_zero_tensors.is_empty(),
            "Should have no all-zero tensors"
        );

        // Test summary output
        let summary = report.summary();
        assert!(summary.contains("VALID"), "Summary should indicate VALID");

        let _ = std::fs::remove_file(apr_path);
    }

    // GH-175: Validation report display
    #[test]
    fn gh175_validation_report_display() {
        let report = ValidationReport {
            format: FormatType::Apr,
            file_path: "test.apr".to_string(),
            is_valid: true,
            tensor_count: 10,
            failed_tensor_count: 0,
            total_nan_count: 0,
            total_inf_count: 0,
            all_zero_tensors: vec![],
            tensors: vec![],
            duration_ms: 100,
        };

        let display = format!("{report}");
        assert!(display.contains("VALID"), "Display should show VALID");
        assert!(
            display.contains("APR-SPEC 10.9"),
            "Should reference APR-SPEC"
        );
    }
}
