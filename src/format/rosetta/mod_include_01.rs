
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
        use std::io::{Read, Seek, SeekFrom};

        let mut file = File::open(path).map_err(|e| AprenderError::FormatError {
            message: format!("Cannot open file: {e}"),
        })?;

        let mut magic = [0u8; 8];
        file.read_exact(&mut magic)
            .map_err(|e| AprenderError::FormatError {
                message: format!("Cannot read magic bytes: {e}"),
            })?;

        // GGUF magic: "GGUF" (0x46554747 little-endian)
        if magic.get(0..4) == Some(b"GGUF") {
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
                // PMAT-264: Detect truncated SafeTensors files early
                let file_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
                let min_size = 8 + header_len;
                if file_size < min_size {
                    return Err(AprenderError::FormatError {
                        message: format!(
                            "Truncated SafeTensors file: header declares {min_size} bytes but file is only {file_size} bytes"
                        ),
                    });
                }
                // PMAT-264: Also verify tensor data section is complete
                let data_section_start = 8 + header_len;
                file.seek(SeekFrom::Start(8))
                    .map_err(|e| AprenderError::FormatError {
                        message: format!("Cannot seek in SafeTensors file: {e}"),
                    })?;
                let mut header_buf = vec![0u8; header_len as usize];
                file.read_exact(&mut header_buf)
                    .map_err(|e| AprenderError::FormatError {
                        message: format!("Cannot read SafeTensors header: {e}"),
                    })?;
                if let Ok(header_json) = serde_json::from_slice::<serde_json::Value>(&header_buf) {
                    if let Some(obj) = header_json.as_object() {
                        let max_data_end = obj
                            .iter()
                            .filter(|(k, _)| *k != "__metadata__")
                            .filter_map(|(_, v)| {
                                v.get("data_offsets")
                                    .and_then(|d| d.as_array())
                                    .and_then(|arr| arr.get(1))
                                    .and_then(|e| e.as_u64())
                            })
                            .max()
                            .unwrap_or(0);
                        let required_size = data_section_start + max_data_end;
                        if file_size < required_size {
                            return Err(AprenderError::FormatError {
                                message: format!(
                                    "Truncated SafeTensors file: tensor data requires {required_size} bytes but file is only {file_size} bytes"
                                ),
                            });
                        }
                    }
                }
                return Ok(Self::SafeTensors);
            }
        }

        // APR magic: "APR\0" (v1) or "APR2" (v2)
        if magic.get(0..3) == Some(b"APR") {
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
                // Use char-boundary-safe truncation
                let truncated = match v.get(..60) {
                    Some(s) => s,
                    None => v.as_str(),
                };
                format!("{truncated}...")
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

/// Internal accumulator for tensor statistics (used by `compute_tensor_validation`).
struct TensorAccum {
    min: f32,
    max: f32,
    sum: f64,
    nan_count: usize,
    inf_count: usize,
    zero_count: usize,
}

include!("check.rs");
include!("mod_include_02.rs");
include!("default_impl.rs");
