
impl AprValidator {
    /// Create new validator
    #[must_use]
    pub fn new() -> Self {
        Self {
            report: ValidationReport::new(),
            tensor_stats: Vec::new(),
        }
    }

    /// Add tensor stats for validation
    pub fn add_tensor_stats(&mut self, stats: TensorStats) {
        self.tensor_stats.push(stats);
    }

    /// Run validation on file bytes
    pub fn validate_bytes(&mut self, data: &[u8]) -> &ValidationReport {
        self.validate_structure(data);
        &self.report
    }

    /// Run all validation checks (tensor-based)
    pub fn validate(&mut self) -> ValidationReport {
        self.validate_tensors();
        std::mem::take(&mut self.report)
    }

    /// Validate tensor statistics (Section B)
    fn validate_tensors(&mut self) {
        // Check 26: No NaNs
        let nan_count: usize = self.tensor_stats.iter().map(|s| s.nan_count).sum();
        let status = if nan_count == 0 {
            CheckStatus::Pass
        } else {
            CheckStatus::Fail(format!("{nan_count} NaN values found across tensors"))
        };
        self.add_check(26, "No NaN values", Category::Physics, status);

        // Check 27: No Infs
        let inf_count: usize = self.tensor_stats.iter().map(|s| s.inf_count).sum();
        let status = if inf_count == 0 {
            CheckStatus::Pass
        } else {
            CheckStatus::Fail(format!("{inf_count} Inf values found across tensors"))
        };
        self.add_check(27, "No Inf values", Category::Physics, status);

        // Check 28: LayerNorm weights valid
        let invalid_ln: Vec<_> = self
            .tensor_stats
            .iter()
            .filter(|s| {
                (s.name.contains("layer_norm") || s.name.contains("ln_"))
                    && (s.name.ends_with(".weight") || s.name.ends_with(".gamma"))
                    && !s.is_valid_layernorm_weight()
            })
            .collect();

        let status = if invalid_ln.is_empty() {
            CheckStatus::Pass
        } else {
            let names: Vec<_> = invalid_ln
                .iter()
                .map(|s| format!("{} (mean={:.4})", s.name, s.mean))
                .collect();
            CheckStatus::Fail(format!("Invalid LayerNorm weights: {}", names.join(", ")))
        };
        self.add_check(28, "LayerNorm weights valid", Category::Physics, status);

        // Check 31: No all-zero tensors
        let zero_tensors: Vec<_> = self
            .tensor_stats
            .iter()
            .filter(|s| !s.is_not_all_zeros())
            .collect();

        let status = if zero_tensors.is_empty() {
            CheckStatus::Pass
        } else {
            let names: Vec<_> = zero_tensors.iter().map(|s| s.name.clone()).collect();
            CheckStatus::Fail(format!("All-zero tensors: {}", names.join(", ")))
        };
        self.add_check(31, "No all-zero tensors", Category::Physics, status);

        // Checks 29-30, 32-50 placeholders
        for id in [29, 30] {
            self.add_check(
                id,
                "Physics check",
                Category::Physics,
                CheckStatus::Skip("Not implemented".to_string()),
            );
        }
        for id in 32..=50 {
            self.add_check(
                id,
                "Physics/Tooling check",
                if id <= 35 {
                    Category::Physics
                } else {
                    Category::Tooling
                },
                CheckStatus::Skip("Not implemented".to_string()),
            );
        }

        // Checks 51-100 placeholders
        for id in 51..=100 {
            self.add_check(
                id,
                "Advanced check",
                if id <= 75 {
                    Category::Tooling
                } else {
                    Category::Conversion
                },
                CheckStatus::Skip("Not implemented".to_string()),
            );
        }
    }

    /// Run Section A: Format & Structural Integrity checks (1-25)
    ///
    /// GH-178: Detect format (APR vs GGUF) and validate appropriately
    fn validate_structure(&mut self, data: &[u8]) {
        // Check 1: Magic bytes valid
        self.check_magic(data);

        // Check 2: Header size fixed (32 bytes for APR, 8+ for GGUF)
        self.check_header_size(data);

        // GH-178: Detect format and validate version accordingly
        if data.len() >= 4 {
            let magic = data.get(0..4).unwrap_or(&[]);
            if magic == b"GGUF" {
                // GGUF format - check version at bytes 4-7 (u32 LE)
                self.check_gguf_version(data);
                // Skip APR-specific flags check for GGUF
                self.add_check(
                    11,
                    "Flags parsed",
                    Category::Structure,
                    CheckStatus::Skip("GGUF format - no APR flags".to_string()),
                );
            } else if data.len() >= 32 {
                // APR format
                if let Ok(header) = AprHeader::parse(data) {
                    self.check_version(&header);
                    self.check_flags(&header);
                }
            }
        }

        // Check 4: Checksum valid (placeholder - need footer)
        self.add_check(
            4,
            "Checksum valid",
            Category::Structure,
            CheckStatus::Skip("Footer not implemented".to_string()),
        );

        // Checks 5-25 are placeholders for now
        for id in 5..=25 {
            self.add_check(
                id,
                "Pending",
                Category::Structure,
                CheckStatus::Skip("Not implemented".to_string()),
            );
        }
    }

    /// Check GGUF version (GH-178)
    ///
    /// GGUF versions 1, 2, and 3 are widely supported by llama.cpp
    fn check_gguf_version(&mut self, data: &[u8]) {
        let status = if data.len() >= 8 {
            let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
            // GH-178: GGUF v1, v2, v3 are all valid
            if (1..=3).contains(&version) {
                CheckStatus::Pass
            } else {
                CheckStatus::Fail(format!(
                    "Unsupported GGUF version: {version} (expected 1-3)"
                ))
            }
        } else {
            CheckStatus::Fail("File too small for GGUF version".to_string())
        };

        self.add_check(3, "Version supported", Category::Structure, status);
    }

    /// Check 1: Magic bytes valid
    ///
    /// GH-178/GH-183: Support both APR and GGUF formats:
    /// - APR: `APR\0` (0x41 0x50 0x52 0x00)
    /// - GGUF: `GGUF` (0x47 0x47 0x55 0x46 = [71, 71, 85, 70])
    fn check_magic(&mut self, data: &[u8]) {
        let status = if let Some(magic) = data.get(0..4) {
            if magic == b"APR\0" {
                CheckStatus::Pass
            } else if magic == b"GGUF" {
                // GH-178: GGUF magic is valid ([71, 71, 85, 70] = "GGUF")
                CheckStatus::Pass
            } else {
                // GH-183: Enhanced error message showing hex and ASCII
                let magic_ascii: String = magic
                    .iter()
                    .map(|&b| if b.is_ascii_graphic() { b as char } else { '.' })
                    .collect();
                CheckStatus::Fail(format!(
                    "Invalid magic: {magic:02X?} (ascii: \"{magic_ascii}\"). Expected APR\\0 or GGUF"
                ))
            }
        } else {
            CheckStatus::Fail("File too small for magic bytes".to_string())
        };

        self.add_check(1, "Magic bytes valid", Category::Structure, status);
    }

    /// Check 2: Header size fixed
    fn check_header_size(&mut self, data: &[u8]) {
        let status = if data.len() >= 32 {
            CheckStatus::Pass
        } else {
            CheckStatus::Fail(format!("Header incomplete: {} bytes", data.len()))
        };

        self.add_check(2, "Header size fixed", Category::Structure, status);
    }

    /// Check 3: Version supported
    fn check_version(&mut self, header: &AprHeader) {
        let status = if header.is_supported_version() {
            CheckStatus::Pass
        } else {
            CheckStatus::Fail(format!(
                "Unsupported version: {}.{}",
                header.version_major, header.version_minor
            ))
        };

        self.add_check(3, "Version supported", Category::Structure, status);
    }

    /// Check flags (11)
    fn check_flags(&mut self, header: &AprHeader) {
        // Check for undefined flag bits
        let known_flags = 0xFF; // Bits 0-7 are defined
        let unknown = header.flags & !known_flags;

        let status = if unknown == 0 {
            CheckStatus::Pass
        } else {
            CheckStatus::Warn(format!("Unknown flag bits: 0x{unknown:08X}"))
        };

        self.add_check(11, "Flags parsed", Category::Structure, status);
    }

    /// Add a check to the report
    fn add_check(&mut self, id: u8, name: &'static str, category: Category, status: CheckStatus) {
        let points = u8::from(status.is_pass());
        self.report.add_check(ValidationCheck {
            id,
            name,
            category,
            status,
            points,
        });
    }

    /// Get the validation report
    #[must_use]
    pub fn report(&self) -> &ValidationReport {
        &self.report
    }
}

impl Default for AprValidator {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// POKA-YOKE: Extensible Model Validation (APR-POKA-001)
// Toyota Way - Mistake-proofing with self-describing quality scores
// ============================================================================

/// Poka-yoke gate result
#[derive(Debug, Clone)]
pub struct Gate {
    /// Gate name (e.g., "`filterbank_present`")
    pub name: &'static str,
    /// Whether gate passed
    pub passed: bool,
    /// Points awarded (0 if failed)
    pub points: u8,
    /// Max points possible
    pub max_points: u8,
    /// Error message if failed
    pub error: Option<String>,
}

impl Gate {
    /// Create a passing gate
    #[must_use]
    pub fn pass(name: &'static str, points: u8) -> Self {
        Self {
            name,
            passed: true,
            points,
            max_points: points,
            error: None,
        }
    }

    /// Create a failing gate with actionable error
    #[must_use]
    pub fn fail(name: &'static str, max_points: u8, error: impl Into<String>) -> Self {
        Self {
            name,
            passed: false,
            points: 0,
            max_points,
            error: Some(error.into()),
        }
    }
}

/// Poka-yoke validation result
#[derive(Debug, Clone, Default)]
pub struct PokaYokeResult {
    /// All gates evaluated
    pub gates: Vec<Gate>,
    /// Total score (0-100)
    pub score: u8,
    /// Maximum possible score
    pub max_score: u8,
}

impl PokaYokeResult {
    /// Create empty result
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create result from a vector of gates (bulk construction)
    ///
    /// # Example
    ///
    /// ```rust
    /// use aprender::format::validation::{Gate, PokaYokeResult};
    ///
    /// let gates = vec![
    ///     Gate::pass("check_a", 30),
    ///     Gate::pass("check_b", 40),
    ///     Gate::fail("check_c", 30, "Fix: implement check_c"),
    /// ];
    /// let result = PokaYokeResult::from_gates(gates);
    /// assert_eq!(result.score, 70); // 70/100
    /// assert_eq!(result.grade(), "C");
    /// ```
    #[must_use]
    pub fn from_gates(gates: Vec<Gate>) -> Self {
        let max_score: u8 = gates
            .iter()
            .map(|g| g.max_points)
            .fold(0u8, u8::saturating_add);
        let total_points: u16 = gates.iter().map(|g| u16::from(g.points)).sum();
        let max_points: u16 = gates.iter().map(|g| u16::from(g.max_points)).sum();
        let score = if max_points > 0 {
            ((total_points * 100) / max_points).min(100) as u8
        } else {
            0
        };
        Self {
            gates,
            score,
            max_score,
        }
    }

    /// Add a gate result
    pub fn add_gate(&mut self, gate: Gate) {
        self.max_score = self.max_score.saturating_add(gate.max_points);
        self.gates.push(gate);
        self.recalculate_score();
    }

    /// Recalculate score from gates
    fn recalculate_score(&mut self) {
        let total_points: u16 = self.gates.iter().map(|g| u16::from(g.points)).sum();
        let max_points: u16 = self.gates.iter().map(|g| u16::from(g.max_points)).sum();
        self.score = if max_points > 0 {
            ((total_points * 100) / max_points).min(100) as u8
        } else {
            0
        };
    }

    /// Get letter grade
    #[must_use]
    pub fn grade(&self) -> &'static str {
        match self.score {
            95..=100 => "A+",
            90..=94 => "A",
            85..=89 => "B+",
            80..=84 => "B",
            75..=79 => "C+",
            70..=74 => "C",
            60..=69 => "D",
            _ => "F",
        }
    }

    /// Check if validation passed (score >= 60)
    #[must_use]
    pub fn passed(&self) -> bool {
        self.score >= 60
    }

    /// Get all failed gates
    #[must_use]
    pub fn failed_gates(&self) -> Vec<&Gate> {
        self.gates.iter().filter(|g| !g.passed).collect()
    }

    /// Get actionable error summary
    #[must_use]
    pub fn error_summary(&self) -> String {
        let errors: Vec<String> = self
            .failed_gates()
            .iter()
            .filter_map(|g| g.error.as_ref().map(|e| format!("- {}: {}", g.name, e)))
            .collect();
        if errors.is_empty() {
            String::new()
        } else {
            format!("Poka-yoke validation failed:\n{}", errors.join("\n"))
        }
    }
}
