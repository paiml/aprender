
// ============================================================================
// Knowledge Distillation Types (spec §6.3)
// ============================================================================

/// Distillation method used (spec §6.3.1)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistillMethod {
    /// KL divergence on final logits (Hinton2015)
    Standard,
    /// Intermediate layer matching
    Progressive,
    /// Multiple teachers weighted average
    Ensemble,
}

/// Teacher model provenance for audit trails (spec §6.3.2)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeacherProvenance {
    /// SHA256 hash of teacher .apr file
    pub hash: String,
    /// Ed25519 signature of teacher (if signed)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
    /// Teacher model type
    pub model_type: ModelType,
    /// Teacher parameter count
    pub param_count: u64,
    /// For ensemble: multiple teachers
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ensemble_teachers: Option<Vec<TeacherProvenance>>,
}

/// Distillation hyperparameters (spec §6.3.2)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationParams {
    /// Temperature for softening distributions (typically 2.0-5.0)
    pub temperature: f32,
    /// Weight for soft vs hard loss (α in loss formula)
    pub alpha: f32,
    /// For progressive: weight for hidden vs logit loss (β)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub beta: Option<f32>,
    /// Training epochs for distillation
    pub epochs: u32,
    /// Final distillation loss achieved
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_loss: Option<f32>,
}

/// Layer mapping for progressive distillation (spec §6.3.2)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerMapping {
    /// Student layer index
    pub student_layer: usize,
    /// Teacher layer index
    pub teacher_layer: usize,
    /// Weight for this layer's loss
    pub weight: f32,
}

/// Complete distillation provenance (spec §6.3.2)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationInfo {
    /// Distillation method used
    pub method: DistillMethod,
    /// Teacher model provenance
    pub teacher: TeacherProvenance,
    /// Distillation hyperparameters
    pub params: DistillationParams,
    /// Optional: layer mapping for progressive distillation
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer_mapping: Option<Vec<LayerMapping>>,
}

// ============================================================================
// Commercial License Types (spec §9)
// ============================================================================

/// License tier levels (spec §9.1)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LicenseTier {
    /// Personal/individual use
    Personal,
    /// Team/organization use (limited seats)
    Team,
    /// Enterprise use (unlimited seats, priority support)
    Enterprise,
    /// Academic/research use (non-commercial)
    Academic,
}

/// Commercial license information (spec §9.1)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LicenseInfo {
    /// Unique license identifier (UUID v4)
    pub uuid: String,
    /// Hash of the license certificate (cryptographically bound)
    pub hash: String,
    /// License expiration date (ISO 8601) - None for perpetual
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expiry: Option<String>,
    /// Maximum concurrent seats - None for unlimited
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seats: Option<u32>,
    /// Licensee name/organization
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub licensee: Option<String>,
    /// License tier
    pub tier: LicenseTier,
}

impl Default for Metadata {
    fn default() -> Self {
        Self {
            created_at: chrono_lite_now(),
            aprender_version: env!("CARGO_PKG_VERSION").to_string(),
            model_name: None,
            description: None,
            training: None,
            hyperparameters: HashMap::new(),
            metrics: HashMap::new(),
            custom: HashMap::new(),
            distillation: None,
            distillation_info: None,
            license: None,
            model_card: None,
        }
    }
}

/// Simple ISO 8601 timestamp (no chrono dependency)
pub(crate) fn chrono_lite_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = duration.as_secs();
    // Convert to rough ISO 8601 (good enough for metadata)
    format!("{secs}")
}

/// Options for saving models
#[derive(Debug, Clone, Default)]
pub struct SaveOptions {
    /// Compression algorithm
    pub compression: Compression,
    /// Additional metadata
    pub metadata: Metadata,
    /// Quality score from Poka-yoke validation (APR-POKA-001)
    /// - None: no validation performed (score=0 in file)
    /// - Some(0): explicit failure - save will be REFUSED (Jidoka)
    /// - Some(1-59): validation failed but allowed to save
    /// - Some(60-100): validation passed
    pub quality_score: Option<u8>,
}

impl SaveOptions {
    /// Create with default compression
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set compression algorithm
    #[must_use]
    pub fn with_compression(mut self, compression: Compression) -> Self {
        self.compression = compression;
        self
    }

    /// Set model name
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.metadata.model_name = Some(name.into());
        self
    }

    /// Set description
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.metadata.description = Some(desc.into());
        self
    }

    /// Set distillation info (spec §6.3)
    #[must_use]
    pub fn with_distillation_info(mut self, info: DistillationInfo) -> Self {
        self.metadata.distillation_info = Some(info);
        self
    }

    /// Set license info (spec §9.1)
    #[must_use]
    pub fn with_license(mut self, license: LicenseInfo) -> Self {
        self.metadata.license = Some(license);
        self
    }

    /// Set model card (spec §11)
    #[must_use]
    pub fn with_model_card(mut self, card: ModelCard) -> Self {
        self.metadata.model_card = Some(card);
        self
    }

    /// Set quality score from Poka-yoke validation (APR-POKA-001)
    ///
    /// # Jidoka (Stop the Line)
    /// - Score 0 will cause `save()` to REFUSE the write
    /// - Score 1-59 allows save with warning
    /// - Score 60-100 is passing
    #[must_use]
    pub fn with_quality_score(mut self, score: u8) -> Self {
        self.quality_score = Some(score);
        self
    }

    /// Set quality score from `PokaYokeResult` (APR-POKA-001)
    #[must_use]
    pub fn with_poka_yoke_result(mut self, result: &PokaYokeResult) -> Self {
        self.quality_score = Some(result.score);
        self
    }
}

/// Model information (from inspection)
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)] // Bools represent independent flag states
pub struct ModelInfo {
    /// Model type
    pub model_type: ModelType,
    /// Format version
    pub format_version: (u8, u8),
    /// Metadata
    pub metadata: Metadata,
    /// Compressed payload size
    pub payload_size: usize,
    /// Uncompressed payload size
    pub uncompressed_size: usize,
    /// Is encrypted
    pub encrypted: bool,
    /// Is signed
    pub signed: bool,
    /// Is streaming
    pub streaming: bool,
    /// Has commercial license block
    pub licensed: bool,
    /// Uses trueno-native 64-byte aligned tensors
    pub trueno_native: bool,
    /// Contains quantized tensors
    pub quantized: bool,
    /// Has model card metadata (spec §11)
    pub has_model_card: bool,
}
