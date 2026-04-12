
impl ModelConfig {
    /// Extract org from hf_repo
    #[must_use]
    pub fn hf_org(&self) -> String {
        self.hf_repo
            .split('/')
            .next()
            .unwrap_or("unknown")
            .to_string()
    }

    /// Extract name from hf_repo
    #[must_use]
    pub fn hf_name(&self) -> String {
        self.hf_repo
            .split('/')
            .nth(1)
            .unwrap_or(&self.hf_repo)
            .to_string()
    }

    /// Populate expected architectural parameters from a family contract (PMAT-269).
    ///
    /// This method derives expected values from the family YAML size_variants,
    /// enabling YAML-driven test matrix generation.
    ///
    /// # Arguments
    /// * `contract` - The family contract to derive values from
    /// * `size` - The size variant key (e.g., "0.5b", "7b")
    ///
    /// # Returns
    /// `true` if the size variant was found and values were populated,
    /// `false` if the size variant doesn't exist in the contract.
    pub fn populate_from_family_contract(
        &mut self,
        contract: &crate::family_contract::FamilyContract,
        size: &str,
    ) -> bool {
        let Some(variant) = contract.get_size_variant(size) else {
            return false;
        };

        self.family = Some(contract.family.clone());
        self.size_variant = Some(size.to_string());
        self.expected_hidden_dim = Some(variant.hidden_dim);
        self.expected_num_layers = Some(variant.num_layers);
        self.expected_num_heads = variant.num_heads;
        self.expected_num_kv_heads = variant.num_kv_heads;
        self.expected_vocab_size = variant.vocab_size;
        self.expected_intermediate_dim = variant.intermediate_dim;

        // PMAT-270: Auto-set size_category from family YAML if not explicitly set
        // Only override if the current size_category is the default (Tiny)
        if self.size_category == SizeCategory::default() {
            if let Some(category_str) = contract.get_size_category(size) {
                if let Ok(category) = SizeCategory::from_str_lowercase(category_str) {
                    self.size_category = category;
                }
            }
        }

        true
    }

    /// Check if this config has expected architectural parameters set.
    #[must_use]
    pub fn has_expected_params(&self) -> bool {
        self.expected_hidden_dim.is_some()
            || self.expected_num_layers.is_some()
            || self.expected_num_heads.is_some()
    }

    /// Validate that the model matches expected architectural parameters.
    ///
    /// Returns a list of mismatches if any parameters don't match.
    #[must_use]
    pub fn validate_architecture(
        &self,
        hidden_dim: u32,
        num_layers: u32,
        num_heads: Option<u32>,
        num_kv_heads: Option<u32>,
    ) -> Vec<String> {
        let mut mismatches = Vec::new();

        if let Some(expected) = self.expected_hidden_dim {
            if expected != hidden_dim {
                mismatches.push(format!(
                    "hidden_dim mismatch: expected {expected}, got {hidden_dim}"
                ));
            }
        }

        if let Some(expected) = self.expected_num_layers {
            if expected != num_layers {
                mismatches.push(format!(
                    "num_layers mismatch: expected {expected}, got {num_layers}"
                ));
            }
        }

        if let (Some(expected), Some(actual)) = (self.expected_num_heads, num_heads) {
            if expected != actual {
                mismatches.push(format!(
                    "num_heads mismatch: expected {expected}, got {actual}"
                ));
            }
        }

        if let (Some(expected), Some(actual)) = (self.expected_num_kv_heads, num_kv_heads) {
            if expected != actual {
                mismatches.push(format!(
                    "num_kv_heads mismatch: expected {expected}, got {actual}"
                ));
            }
        }

        mismatches
    }
}

/// Return the default model formats: gguf, safetensors, and apr
fn default_formats() -> Vec<Format> {
    vec![Format::Gguf, Format::SafeTensors, Format::Apr]
}

/// Return the default quantization list containing q4_k_m
fn default_quantizations() -> Vec<String> {
    vec!["q4_k_m".to_string()]
}

/// Test matrix configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestMatrix {
    /// Modalities to test
    #[serde(default = "default_modalities")]
    pub modalities: Vec<Modality>,
    /// Backends to test
    #[serde(default = "default_backends")]
    pub backends: Vec<Backend>,
    /// Number of scenarios per combination
    #[serde(default = "default_scenario_count")]
    pub scenario_count: usize,
    /// Architecture-specific prompts (optional; falls back to default if absent)
    #[serde(default)]
    pub prompts: Option<Vec<String>>,
}

/// Return the default test modalities: run, chat, and serve
fn default_modalities() -> Vec<Modality> {
    vec![Modality::Run, Modality::Chat, Modality::Serve]
}

/// Return the default backends: cpu and gpu
fn default_backends() -> Vec<Backend> {
    vec![Backend::Cpu, Backend::Gpu]
}

/// Return the default scenario count of 100
fn default_scenario_count() -> usize {
    100
}

impl Default for TestMatrix {
    /// Create a TestMatrix with default modalities, backends, and scenario count
    fn default() -> Self {
        Self {
            modalities: default_modalities(),
            backends: default_backends(),
            scenario_count: default_scenario_count(),
            prompts: None,
        }
    }
}

/// Property test definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyTest {
    /// Test name
    pub name: String,
    /// Generator expression
    pub generator: String,
    /// Oracle expression
    pub oracle: String,
    /// Number of test cases
    #[serde(default = "default_proptest_count")]
    pub count: usize,
}

/// Return the default property test case count of 100
fn default_proptest_count() -> usize {
    100
}

/// Falsification gate definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FalsificationGate {
    /// Gate ID (e.g., "F-QUAL-001")
    pub id: String,
    /// Description
    pub description: String,
    /// Condition expression
    pub condition: String,
    /// Severity (P0, P1, P2)
    #[serde(default = "default_severity")]
    pub severity: String,
}

/// Return the default gate severity of P1
fn default_severity() -> String {
    "P1".to_string()
}

/// State machine for complex workflows
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateMachine {
    /// Initial state
    pub initial: String,
    /// State definitions
    pub states: HashMap<String, State>,
}

/// State in a state machine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct State {
    /// Actions to execute on entering this state
    #[serde(default)]
    pub on_enter: Vec<Action>,
    /// Actions to execute on exiting this state
    #[serde(default)]
    pub on_exit: Vec<Action>,
    /// Transitions from this state
    #[serde(default)]
    pub transitions: Vec<Transition>,
}

/// Action to execute
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Action {
    /// Action name or command
    pub action: String,
}

/// Transition between states
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transition {
    /// Event that triggers this transition
    pub event: String,
    /// Target state
    pub target: String,
    /// Optional action to execute
    pub action: Option<String>,
    /// Guard conditions
    #[serde(default)]
    pub guards: Vec<String>,
}

/// A single step in a playbook
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaybookStep {
    /// Step name
    pub name: String,
    /// Command to execute
    pub command: String,
    /// Timeout in milliseconds
    #[serde(default = "default_timeout")]
    pub timeout_ms: u64,
    /// Expected exit code
    #[serde(default)]
    pub expected_exit_code: i32,
    /// Expected output patterns
    #[serde(default)]
    pub expected_patterns: Vec<String>,
    /// Forbidden output patterns
    #[serde(default)]
    pub forbidden_patterns: Vec<String>,
}

/// Return the default step timeout of 60 seconds
fn default_timeout() -> u64 {
    60000 // 60 seconds
}

/// Differential test configuration (GH-188, PMAT-114, PMAT-201, PMAT-202)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DifferentialTestConfig {
    /// Format validation configuration (GH-186 prevention)
    #[serde(default)]
    pub format_validation: Option<FormatValidationConfig>,
    /// Tensor diff configuration
    #[serde(default)]
    pub tensor_diff: Option<TensorDiffConfig>,
    /// Inference comparison configuration
    #[serde(default)]
    pub inference_compare: Option<InferenceCompareConfig>,
    /// Fingerprint configuration (PMAT-201)
    #[serde(default)]
    pub fingerprint: Option<FingerprintConfig>,
    /// Validate stats configuration (PMAT-202)
    #[serde(default)]
    pub validate_stats: Option<ValidateStatsConfig>,
}

/// Format validation configuration (GH-186 prevention)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormatValidationConfig {
    /// Enable format validation
    #[serde(default, deserialize_with = "deserialize_bool_or_string")]
    pub enabled: bool,
    /// Checks to run: dtype_mapping, tensor_alignment, header_integrity
    #[serde(default)]
    pub checks: Vec<String>,
    /// Gates to verify
    #[serde(default)]
    pub gates: Vec<String>,
}

/// Tensor diff configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorDiffConfig {
    /// Enable tensor diff
    #[serde(default, deserialize_with = "deserialize_bool_or_string")]
    pub enabled: bool,
    /// Filter pattern for tensor names
    #[serde(default)]
    pub filter: Option<String>,
    /// Gates to verify
    #[serde(default)]
    pub gates: Vec<String>,
}

/// Inference comparison configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceCompareConfig {
    /// Enable inference comparison
    #[serde(default, deserialize_with = "deserialize_bool_or_string")]
    pub enabled: bool,
    /// Prompt to use for comparison
    #[serde(default)]
    pub prompt: Option<String>,
    /// Maximum tokens to generate
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    /// Tolerance for logit comparison
    #[serde(default = "default_tolerance")]
    pub tolerance: f64,
    /// Gates to verify
    #[serde(default)]
    pub gates: Vec<String>,
}

/// Return the default max tokens of 10 for inference comparison
fn default_max_tokens() -> usize {
    10
}

/// Return the default logit comparison tolerance of 1e-5
fn default_tolerance() -> f64 {
    1e-5
}
