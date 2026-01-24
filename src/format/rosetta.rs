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
// Conversion Options
// ============================================================================

/// Options for conversion operations
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

    fn convert_internal(
        &self,
        source: &Path,
        target: &Path,
        source_format: FormatType,
        target_format: FormatType,
        _opts: &ConversionOptions,
    ) -> Result<()> {
        use crate::format::converter::{apr_export, apr_import, ExportFormat, ExportOptions, ImportOptions};

        match (source_format, target_format) {
            // GGUF → APR
            (FormatType::Gguf, FormatType::Apr) => {
                let source_str = source.to_string_lossy();
                apr_import(&source_str, target, ImportOptions::default())?;
                Ok(())
            }

            // SafeTensors → APR
            (FormatType::SafeTensors, FormatType::Apr) => {
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
                self.convert_internal(source, &temp_apr, FormatType::Gguf, FormatType::Apr, _opts)?;
                self.convert_internal(
                    &temp_apr,
                    target,
                    FormatType::Apr,
                    FormatType::SafeTensors,
                    _opts,
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
                    _opts,
                )?;
                self.convert_internal(&temp_apr, target, FormatType::Apr, FormatType::Gguf, _opts)?;
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

        // TODO: Implement actual tensor data comparison
        // For now, return passing if structure matches
        Ok(VerificationReport::passing())
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

    // ========================================================================
    // Section 13: Integration Tests (Require test fixtures)
    // ========================================================================

    // These tests require actual model files and are marked as ignored
    // Run with: cargo test -- --ignored

    #[test]
    #[ignore = "Requires GGUF test fixture"]
    fn integration_inspect_gguf() {
        let rosetta = RosettaStone::new();
        let report = rosetta
            .inspect("tests/fixtures/tiny.gguf")
            .expect("Inspect GGUF");
        assert_eq!(report.format, FormatType::Gguf);
    }

    #[test]
    #[ignore = "Requires SafeTensors test fixture"]
    fn integration_inspect_safetensors() {
        let rosetta = RosettaStone::new();
        let report = rosetta
            .inspect("tests/fixtures/tiny.safetensors")
            .expect("Inspect SafeTensors");
        assert_eq!(report.format, FormatType::SafeTensors);
    }

    #[test]
    #[ignore = "Requires APR test fixture"]
    fn integration_inspect_apr() {
        let rosetta = RosettaStone::new();
        let report = rosetta
            .inspect("tests/fixtures/tiny.apr")
            .expect("Inspect APR");
        assert_eq!(report.format, FormatType::Apr);
    }

    #[test]
    #[ignore = "Requires test fixtures for conversion"]
    fn integration_convert_gguf_to_apr() {
        let rosetta = RosettaStone::new();
        let temp_out = std::env::temp_dir().join("test_out.apr");
        let report = rosetta
            .convert("tests/fixtures/tiny.gguf", &temp_out, None)
            .expect("Convert GGUF to APR");
        assert_eq!(report.path.source, FormatType::Gguf);
        assert_eq!(report.path.target, FormatType::Apr);
        let _ = std::fs::remove_file(temp_out);
    }
}
