
/// Create a demo APR v2 model for testing
pub(crate) fn create_demo_apr_model(_input_dim: usize) -> Result<AprModel, RealizarError> {
    use crate::apr::TensorEntry;

    // Create minimal APR v2 file
    let metadata = r#"{"model_type":"demo","name":"demo-model"}"#;
    let tensor_index: Vec<TensorEntry> = vec![TensorEntry {
        name: "weight".to_string(),
        dtype: "F32".to_string(),
        shape: vec![4],
        offset: 0,
        size: 16,
    }];
    let tensor_index_json = serde_json::to_vec(&tensor_index).unwrap_or_default();
    let tensor_data: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
    let tensor_bytes: Vec<u8> = tensor_data.iter().flat_map(|f| f.to_le_bytes()).collect();

    // Calculate offsets (64-byte aligned)
    let metadata_offset = HEADER_SIZE as u64;
    let metadata_size = metadata.len() as u32;
    let tensor_index_offset =
        ((metadata_offset as usize + metadata.len()).div_ceil(64) * 64) as u64;
    let data_offset =
        ((tensor_index_offset as usize + tensor_index_json.len()).div_ceil(64) * 64) as u64;

    let mut data = vec![0u8; data_offset as usize + tensor_bytes.len()];

    // Header (64 bytes)
    data[0..4].copy_from_slice(&MAGIC);
    data[4] = 2; // Version major
    data[5] = 0; // Version minor
    data[6..8].copy_from_slice(&0u16.to_le_bytes()); // Flags
    data[8..12].copy_from_slice(&1u32.to_le_bytes()); // Tensor count
    data[12..20].copy_from_slice(&metadata_offset.to_le_bytes());
    data[20..24].copy_from_slice(&metadata_size.to_le_bytes());
    data[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
    data[32..40].copy_from_slice(&data_offset.to_le_bytes());
    // Checksum at 40..44 (leave as 0 for now)

    // Metadata
    data[metadata_offset as usize..metadata_offset as usize + metadata.len()]
        .copy_from_slice(metadata.as_bytes());

    // Tensor index
    data[tensor_index_offset as usize..tensor_index_offset as usize + tensor_index_json.len()]
        .copy_from_slice(&tensor_index_json);

    // Tensor data
    data[data_offset as usize..data_offset as usize + tensor_bytes.len()]
        .copy_from_slice(&tensor_bytes);

    AprModel::from_bytes(data)
}

// Basic API types moved to types.rs (PMAT-COMPLY)

// ============================================================================
// OpenAI-Compatible API Types (per spec §5.4)
// ============================================================================

/// OpenAI-compatible chat completion request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    /// Model ID to use
    pub model: String,
    /// Chat messages
    pub messages: Vec<ChatMessage>,
    /// Maximum tokens to generate
    #[serde(default)]
    pub max_tokens: Option<usize>,
    /// Sampling temperature
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Nucleus sampling
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Number of completions to generate
    #[serde(default = "default_n")]
    pub n: usize,
    /// Stream responses
    #[serde(default)]
    pub stream: bool,
    /// Stop sequences
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    /// User identifier
    #[serde(default)]
    pub user: Option<String>,
}

fn default_n() -> usize {
    1
}

/// Chat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Role: "system", "user", "assistant"
    pub role: String,
    /// Message content
    pub content: String,
    /// Optional name
    #[serde(default)]
    pub name: Option<String>,
}

/// OpenAI-compatible chat completion response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    /// Unique request ID
    pub id: String,
    /// Object type
    pub object: String,
    /// Creation timestamp
    pub created: i64,
    /// Model used
    pub model: String,
    /// Choices array
    pub choices: Vec<ChatChoice>,
    /// Token usage statistics
    pub usage: Usage,
    /// Brick-level trace data (tensor operations) - only present when X-Trace-Level: brick
    #[serde(skip_serializing_if = "Option::is_none")]
    pub brick_trace: Option<TraceData>,
    /// Step-level trace data (forward pass steps) - only present when X-Trace-Level: step
    #[serde(skip_serializing_if = "Option::is_none")]
    pub step_trace: Option<TraceData>,
    /// Layer-level trace data (attention, MLP) - only present when X-Trace-Level: layer
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer_trace: Option<TraceData>,
}

/// Provenance of trace timing data (GH-92: truth-in-reporting)
///
/// Distinguishes measured data from estimates to prevent fabricated trace output.
/// Every `TraceData` instance MUST declare its provenance.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TraceProvenance {
    /// Real per-operation timing from BrickProfiler instrumentation
    Measured,
    /// Only the wall-clock total is real; no per-op breakdown available
    WallClockTotal,
    /// Values are statistical estimates (e.g., from sampling or heuristics)
    #[default]
    Estimated,
}

/// Trace data for debugging inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceData {
    /// Trace level that was requested
    pub level: String,
    /// Number of operations traced
    pub operations: usize,
    /// Total time in microseconds
    pub total_time_us: u64,
    /// Per-operation timing breakdown
    pub breakdown: Vec<TraceOperation>,
    /// Data provenance — how these values were obtained (GH-92)
    #[serde(default)]
    pub provenance: TraceProvenance,
}

/// Individual traced operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceOperation {
    /// Operation name
    pub name: String,
    /// Time in microseconds
    pub time_us: u64,
    /// Additional details
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<String>,
}

/// Build trace data based on X-Trace-Level header
///
/// Returns (brick_trace, step_trace, layer_trace) tuple based on requested level.
/// Only reports wall-clock totals — per-operation breakdown requires `apr profile`
/// with BrickProfiler instrumentation. We refuse to fabricate per-op estimates.
#[must_use]
pub fn build_trace_data(
    trace_level: Option<&str>,
    latency_us: u64,
    prompt_tokens: usize,
    completion_tokens: usize,
    num_layers: usize,
) -> (Option<TraceData>, Option<TraceData>, Option<TraceData>) {
    match trace_level {
        Some("brick") => (
            Some(TraceData {
                level: "brick".to_string(),
                operations: completion_tokens,
                total_time_us: latency_us,
                breakdown: vec![
                    TraceOperation {
                        name: "total_inference".to_string(),
                        time_us: latency_us,
                        details: Some(format!(
                            "{} prompt + {} completion tokens, {} layers. \
                             Per-op breakdown not available — use `apr profile` for real brick-level telemetry",
                            prompt_tokens, completion_tokens, num_layers
                        )),
                    },
                ],
                provenance: TraceProvenance::WallClockTotal,
            }),
            None,
            None,
        ),
        Some("step") => (
            None,
            Some(TraceData {
                level: "step".to_string(),
                operations: completion_tokens,
                total_time_us: latency_us,
                breakdown: vec![
                    TraceOperation {
                        name: "total_inference".to_string(),
                        time_us: latency_us,
                        details: Some(format!(
                            "{} prompt + {} completion tokens, {} layers. \
                             Step-level breakdown not instrumented — use `apr profile` for real timing",
                            prompt_tokens, completion_tokens, num_layers
                        )),
                    },
                ],
                provenance: TraceProvenance::WallClockTotal,
            }),
            None,
        ),
        Some("layer") => (
            None,
            None,
            Some(TraceData {
                level: "layer".to_string(),
                operations: num_layers,
                total_time_us: latency_us,
                breakdown: vec![
                    TraceOperation {
                        name: "total_inference".to_string(),
                        time_us: latency_us,
                        details: Some(format!(
                            "{} layers, {} tokens. \
                             Per-layer breakdown not instrumented — use `apr profile --granular` for real per-layer timing",
                            num_layers, prompt_tokens + completion_tokens
                        )),
                    },
                ],
                provenance: TraceProvenance::WallClockTotal,
            }),
        ),
        _ => (None, None, None),
    }
}

/// Chat completion choice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChoice {
    /// Choice index
    pub index: usize,
    /// Generated message
    pub message: ChatMessage,
    /// Finish reason
    pub finish_reason: String,
}

/// Token usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    /// Prompt tokens
    pub prompt_tokens: usize,
    /// Completion tokens
    pub completion_tokens: usize,
    /// Total tokens
    pub total_tokens: usize,
}

/// OpenAI-compatible models list response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIModelsResponse {
    /// Object type
    pub object: String,
    /// Model list
    pub data: Vec<OpenAIModel>,
}

/// OpenAI model info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIModel {
    /// Model ID
    pub id: String,
    /// Object type
    pub object: String,
    /// Created timestamp
    pub created: i64,
    /// Owner
    pub owned_by: String,
}

// ============================================================================
// OpenAI Streaming Types (SSE)
// ============================================================================

/// Streaming chat completion chunk (SSE format)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    /// Unique request ID
    pub id: String,
    /// Object type (always "chat.completion.chunk")
    pub object: String,
    /// Creation timestamp
    pub created: i64,
    /// Model used
    pub model: String,
    /// Choices array with deltas
    pub choices: Vec<ChatChunkChoice>,
}

/// Streaming choice with delta
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChunkChoice {
    /// Choice index
    pub index: usize,
    /// Delta content (partial message)
    pub delta: ChatDelta,
    /// Finish reason (None until done)
    pub finish_reason: Option<String>,
}

/// Delta content for streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatDelta {
    /// Role (only in first chunk)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    /// Content chunk
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

impl ChatCompletionChunk {
    /// Create a new chunk with content
    fn new(id: &str, model: &str, content: Option<String>, finish_reason: Option<String>) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0),
            model: model.to_string(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: ChatDelta {
                    role: if content.is_none() && finish_reason.is_none() {
                        Some("assistant".to_string())
                    } else {
                        None
                    },
                    content,
                },
                finish_reason,
            }],
        }
    }

    /// Create initial chunk with role only
    fn initial(id: &str, model: &str) -> Self {
        Self::new(id, model, None, None)
    }

    /// Create content chunk
    fn content(id: &str, model: &str, text: &str) -> Self {
        Self::new(id, model, Some(text.to_string()), None)
    }

    /// Create final chunk with finish reason
    fn done(id: &str, model: &str) -> Self {
        Self::new(id, model, None, Some("stop".to_string()))
    }
}

// ============================================================================
// APR-Specific API Types (spec §15.1)
// ============================================================================

/// APR prediction request (classification/regression)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictRequest {
    /// Model ID (optional, uses default if not specified)
    #[serde(default)]
    pub model: Option<String>,
    /// Input features as flat array
    pub features: Vec<f32>,
    /// Feature names (optional, for explainability)
    #[serde(default)]
    pub feature_names: Option<Vec<String>>,
    /// Return top-k predictions for classification
    #[serde(default)]
    pub top_k: Option<usize>,
    /// Include confidence scores
    #[serde(default = "default_true")]
    pub include_confidence: bool,
}

pub(crate) fn default_true() -> bool {
    true
}

/// APR prediction response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictResponse {
    /// Request ID for audit trail
    pub request_id: String,
    /// Model ID used
    pub model: String,
    /// Prediction result (class label or regression value)
    pub prediction: serde_json::Value,
    /// Confidence score (0.0-1.0) for classification
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    /// Top-k predictions with probabilities
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k_predictions: Option<Vec<PredictionWithScore>>,
    /// Latency in milliseconds
    pub latency_ms: f64,
}

/// Prediction with confidence score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionWithScore {
    /// Class label or value
    pub label: String,
    /// Probability/confidence
    pub score: f32,
}

/// APR explanation request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplainRequest {
    /// Model ID (optional)
    #[serde(default)]
    pub model: Option<String>,
    /// Input features
    pub features: Vec<f32>,
    /// Feature names (required for meaningful explanations)
    pub feature_names: Vec<String>,
    /// Number of top features to include
    #[serde(default = "default_top_k_features")]
    pub top_k_features: usize,
    /// Explanation method (shap, lime, attention)
    #[serde(default = "default_explain_method")]
    pub method: String,
}

pub(crate) fn default_top_k_features() -> usize {
    5
}

pub(crate) fn default_explain_method() -> String {
    "shap".to_string()
}
