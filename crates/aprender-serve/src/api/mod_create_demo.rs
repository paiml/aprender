
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
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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
    /// Top-k sampling (aprender extension; NOT OpenAI standard but llama-style APIs include it).
    /// qwen3-moe-sampling-v1 V1_001: top_k=1 forces greedy regardless of temperature.
    #[serde(default)]
    pub top_k: Option<usize>,
    /// Repetition penalty (aprender extension; mirrors Candle's apply_repeat_penalty).
    /// qwen3-moe-repetition-penalty-v1: 1.0 = no penalty; 1.1-1.3 = standard chat presets.
    #[serde(default)]
    pub repeat_penalty: Option<f32>,
    /// Repetition penalty window (how many recent tokens get penalized).
    /// qwen3-moe-repetition-penalty-v1: 0 = no penalty; 64-128 = standard.
    #[serde(default)]
    pub repeat_last_n: Option<usize>,
    /// Random seed for reproducible sampling.
    /// qwen3-moe-sampling-v1 V1_002: same seed → same tokens.
    #[serde(default)]
    pub seed: Option<u64>,
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
    /// PMAT-801: OpenAI tool/function definitions the model may call.
    /// When `Some`, the non-streaming chat handler runs the generated text
    /// through `grammar::ToolCallParser` and populates `tool_calls` on the
    /// response. When `None`, the response is byte-identical to pre-PMAT-801
    /// behavior (the whole tool-calling path is gated on `tools.is_some()`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<OpenAiTool>>,
    /// PMAT-801: OpenAI `tool_choice` ("auto" | "none" | "required" |
    /// `{"type":"function","function":{"name":"..."}}`). `"none"` skips parsing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<OpenAiToolChoice>,
}

fn default_n() -> usize {
    1
}

/// Chat message
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Role: "system", "user", "assistant", "tool"
    pub role: String,
    /// Message content.
    ///
    /// PMAT-801: kept REQUIRED on deserialize (a request message missing
    /// `content` is rejected 422, preserving the pre-PMAT-801 contract). The
    /// response-side assistant tool-call message sets `content: String::new()`
    /// directly in Rust, so it does not need a serde default. (Optional/null
    /// content for assistant tool-call history on the REQUEST side is a
    /// follow-up — it would require relaxing this field to Option.)
    pub content: String,
    /// Optional name
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// PMAT-801: tool calls emitted by the assistant (response side) — and
    /// echoed back by clients on a follow-up turn (request round-trip).
    /// Serialized only when present so non-tool messages are byte-identical.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ResponseToolCall>>,
    /// PMAT-801: when this message carries a tool RESULT (role "tool"), the id
    /// of the `tool_call` it answers. Lets tool results round-trip through the API.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

// ============================================================================
// PMAT-801: OpenAI tool-calling wire types (request in, response out)
// ============================================================================

/// OpenAI request tool entry: `{"type":"function","function":{...}}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiTool {
    /// Always "function" in the OpenAI spec.
    #[serde(rename = "type", default = "default_function_type")]
    pub tool_type: String,
    /// The function definition.
    pub function: OpenAiFunctionDef,
}

/// OpenAI function definition inside a tool entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiFunctionDef {
    /// Function name (valid identifier).
    pub name: String,
    /// Human-readable description.
    #[serde(default)]
    pub description: String,
    /// JSON Schema for the parameters (opaque — we only need the names for the
    /// `ToolCallParser`; full schema-constrained decoding is a follow-up).
    #[serde(default)]
    pub parameters: Option<serde_json::Value>,
}

fn default_function_type() -> String {
    "function".to_string()
}

/// OpenAI `tool_choice`: a string mode or a specific-function object.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OpenAiToolChoice {
    /// "auto" | "none" | "required"
    Mode(String),
    /// `{"type":"function","function":{"name":"..."}}`
    Specific {
        /// Always "function".
        #[serde(rename = "type")]
        choice_type: String,
        /// The chosen function (only `name` is used).
        function: OpenAiToolChoiceFunction,
    },
}

/// The `function` object inside a specific `tool_choice`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiToolChoiceFunction {
    /// Name of the function the model must call.
    pub name: String,
}

impl OpenAiToolChoice {
    /// Map to the `grammar::ToolChoice` library type.
    #[must_use]
    pub fn to_grammar(&self) -> crate::grammar::ToolChoice {
        use crate::grammar::ToolChoice;
        match self {
            Self::Mode(m) => match m.as_str() {
                "none" => ToolChoice::None,
                "required" => ToolChoice::Required,
                _ => ToolChoice::Auto,
            },
            Self::Specific { function, .. } => ToolChoice::Specific(function.name.clone()),
        }
    }
}

/// Tool call as it appears in the OpenAI response message:
/// `{"id":"call_0","type":"function","function":{"name":"...","arguments":"<json-string>"}}`.
///
/// `arguments` is a JSON STRING (not a nested object) per the OpenAI spec — the
/// `grammar::ToolCall` library already stores arguments as a string, so this
/// passes through unchanged.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResponseToolCall {
    /// Unique tool-call id, e.g. "call_0".
    pub id: String,
    /// Always "function".
    #[serde(rename = "type")]
    pub call_type: String,
    /// The called function (name + arguments-as-string).
    pub function: ResponseFunctionCall,
}

/// The `function` payload inside a response tool call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResponseFunctionCall {
    /// Function name.
    pub name: String,
    /// Arguments as a JSON-encoded STRING (OpenAI wire format).
    pub arguments: String,
}

impl From<crate::grammar::ToolCall> for ResponseToolCall {
    fn from(tc: crate::grammar::ToolCall) -> Self {
        Self {
            id: tc.id,
            call_type: "function".to_string(),
            function: ResponseFunctionCall {
                name: tc.name,
                arguments: tc.arguments,
            },
        }
    }
}

impl OpenAiTool {
    /// Map this OpenAI tool to the `grammar::ToolDefinition` library type.
    ///
    /// The parser only needs the tool NAME to recognise a tool call in the
    /// generated text; parameter schema is preserved opaquely but not used for
    /// constrained decoding (a follow-up). We extract top-level property names
    /// from the JSON Schema when present so `required_params` is non-empty.
    #[must_use]
    pub fn to_grammar(&self) -> crate::grammar::ToolDefinition {
        crate::grammar::ToolDefinition {
            name: self.function.name.clone(),
            description: self.function.description.clone(),
            parameters: extract_tool_parameters(self.function.parameters.as_ref()),
        }
    }
}

/// Extract `ToolParameter`s from an OpenAI JSON-Schema `parameters` object.
/// Best-effort: maps top-level `properties` keys to string params and marks
/// those listed in `required` as required. Returns empty on absent/invalid schema.
fn extract_tool_parameters(
    schema: Option<&serde_json::Value>,
) -> Vec<crate::grammar::ToolParameter> {
    use crate::grammar::{ToolParameter, ToolParameterType};
    let Some(schema) = schema else {
        return Vec::new();
    };
    let Some(props) = schema.get("properties").and_then(|p| p.as_object()) else {
        return Vec::new();
    };
    let required: std::collections::HashSet<&str> = schema
        .get("required")
        .and_then(|r| r.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();
    props
        .iter()
        .map(|(name, spec)| {
            let description = spec
                .get("description")
                .and_then(|d| d.as_str())
                .unwrap_or_default()
                .to_string();
            ToolParameter {
                name: name.clone(),
                description,
                param_type: ToolParameterType::String,
                required: required.contains(name.as_str()),
                default: None,
            }
        })
        .collect()
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

    /// Create the terminal chunk, carrying the finish reason the generation
    /// actually produced.
    ///
    /// The previous `done(id, model)` hardcoded `"stop"`, so a stream truncated
    /// by `max_tokens` reported `"stop"` while the non-streaming path on the very
    /// same request reported `"length"`. Clients that implement continuation
    /// ("if finish_reason == 'length', ask for more") therefore never triggered
    /// and silently truncated.
    fn done_with_reason(id: &str, model: &str, finish_reason: &str) -> Self {
        Self::new(id, model, None, Some(finish_reason.to_string()))
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
