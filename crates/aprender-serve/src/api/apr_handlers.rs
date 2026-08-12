//! APR-specific API handlers
//!
//! Extracted from api/mod.rs (PMAT-802) to reduce module size.
//! Contains prediction, explanation, and audit handlers for APR models.

use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};

use super::{
    AppState, AuditResponse, ErrorResponse, ExplainRequest, ExplainResponse, PredictRequest,
    PredictResponse, PredictionWithScore, ShapExplanation,
};

// ============================================================================
// APR-Specific API Handlers (spec §15.1)
// ============================================================================

/// APR prediction handler (/v1/predict)
///
/// Handles classification and regression predictions for APR models.
/// APR v2 prediction handler - tensor-based inference
///
/// Note: APR v2 uses tensor-based access rather than direct predict().
/// For LLM inference, use the /generate endpoint instead.
// serde_json::json!() uses infallible unwrap
#[allow(clippy::disallowed_methods)]
pub(crate) async fn apr_predict_handler(
    State(state): State<AppState>,
    Json(request): Json<PredictRequest>,
) -> Result<Json<PredictResponse>, (StatusCode, Json<ErrorResponse>)> {
    let start = std::time::Instant::now();

    // Validate input features
    if request.features.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Input features cannot be empty".to_string(),
            }),
        ));
    }

    // Get APR model from state
    let apr_model = state.apr_model.as_ref().ok_or_else(|| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                // aprender#2376(7): this used to read "Use AppState::demo() or load
                // a .apr model." — an internal Rust constructor, instructing an HTTP
                // client to call something it has no access to. Name what the
                // OPERATOR must do instead, and what this server actually holds.
                error: format!(
                    "No APR model loaded: /v1/predict requires a .apr model, and this \
                     server was started with a {} model. Use /generate or \
                     /v1/completions for text generation.",
                    state.model_format()
                ),
            }),
        )
    })?;

    // Log request to audit trail
    let model_name = apr_model
        .metadata()
        .name
        .clone()
        .unwrap_or_else(|| "unknown".to_string());
    let request_id = state
        .audit_logger
        .log_request(&model_name, &[request.features.len()]);

    // APR v2 uses tensor-based inference
    // For simple regression/classification, we need a weights tensor
    let output = apr_model
        .get_tensor_f32("weights")
        .or_else(|_| apr_model.get_tensor_f32("output"))
        .map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: format!("Inference failed: {e}. Use /generate for LLM inference."),
                }),
            )
        })?;

    // Simple linear prediction: output = features * weights (demo only)
    let output: Vec<f32> = if output.len() == request.features.len() {
        vec![request
            .features
            .iter()
            .zip(output.iter())
            .map(|(f, w)| f * w)
            .sum()]
    } else {
        // Just return first few weights as output
        output.into_iter().take(10).collect()
    };

    // Convert output to prediction (regression or classification)
    let prediction = if output.len() == 1 {
        // Regression: single value
        serde_json::json!(output[0])
    } else {
        // Classification: argmax for class label
        let max_idx = output
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i);
        serde_json::json!(format!("class_{}", max_idx))
    };

    // Compute confidence (for classification: max probability after softmax)
    let confidence = if output.len() > 1 {
        // Softmax then take max
        let max_val = output.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = output.iter().map(|x| (x - max_val).exp()).sum();
        let probs: Vec<f32> = output
            .iter()
            .map(|x| (x - max_val).exp() / exp_sum)
            .collect();
        probs.into_iter().fold(0.0_f32, f32::max)
    } else {
        // Regression: use 1.0 confidence
        1.0
    };

    // Top-k predictions (for classification)
    let top_k_predictions = request.top_k.map(|k| {
        if output.len() > 1 {
            // Compute softmax
            let max_val = output.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = output.iter().map(|x| (x - max_val).exp()).sum();
            let mut probs: Vec<(usize, f32)> = output
                .iter()
                .enumerate()
                .map(|(i, x)| (i, (x - max_val).exp() / exp_sum))
                .collect();
            probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            probs
                .into_iter()
                .take(k)
                .map(|(i, score)| PredictionWithScore {
                    label: format!("class_{}", i),
                    score,
                })
                .collect()
        } else {
            // Regression: no top-k
            vec![PredictionWithScore {
                label: format!("{:.4}", output[0]),
                score: 1.0,
            }]
        }
    });

    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

    // Log response to audit trail
    state.audit_logger.log_response(
        request_id,
        prediction.clone(),
        start.elapsed(),
        Some(confidence),
    );

    Ok(Json(PredictResponse {
        request_id: request_id.to_string(),
        model: request.model.unwrap_or_else(|| "default".to_string()),
        prediction,
        confidence: if request.include_confidence {
            Some(confidence)
        } else {
            None
        },
        top_k_predictions,
        latency_ms,
    }))
}

/// APR explanation handler (/v1/explain)
///
/// Returns SHAP-based feature importance explanations for APR models.
// serde_json::json!() uses infallible unwrap
#[allow(clippy::disallowed_methods)]
pub(crate) async fn apr_explain_handler(
    State(state): State<AppState>,
    Json(request): Json<ExplainRequest>,
) -> Result<Json<ExplainResponse>, (StatusCode, Json<ErrorResponse>)> {
    let start = std::time::Instant::now();
    let request_id = uuid::Uuid::new_v4().to_string();

    // Validate inputs
    if request.features.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Input features cannot be empty".to_string(),
            }),
        ));
    }

    if request.feature_names.len() != request.features.len() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: format!(
                    "Feature names count ({}) must match features count ({})",
                    request.feature_names.len(),
                    request.features.len()
                ),
            }),
        ));
    }

    // aprender#2375(2): this returned FABRICATED explanations with HTTP 200.
    //
    // The SHAP values were derived from the feature INDEX — `0.1 - i * 0.02` —
    // so they did not depend on the feature VALUES, and `prediction` was the
    // literal 0.95. `State` was bound as `_state`, so the response was
    // identical whether a model was loaded or not. A caller integrating
    // against it gets numbers shaped exactly like an explanation, with nothing
    // behind them.
    //
    // Kernel SHAP needs a BACKGROUND DATASET to compute expected values
    // (`ShapExplainer::new(background, model_fn)` in aprender::interpret), and
    // `ExplainRequest` carries none — there is no honest way to compute this
    // from one request. So it fails, and says what is missing.
    //
    // Same rule the CLI now follows for `apr trace --reference` (#2407): an
    // advertised surface whose implementation is a stub must FAIL, not emit
    // something plausible and return success.
    //
    // Model presence is checked FIRST so the caller gets the most specific
    // error, matching how /v1/predict reports the same condition.
    let _apr_model = state.apr_model.as_ref().ok_or_else(|| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                error: format!(
                    "No APR model loaded: /v1/explain requires a .apr model, and this \
                     server is serving {}.",
                    state.model_format()
                ),
            }),
        )
    })?;

    let _ = (start, request_id, request.top_k_features);
    Err((
        StatusCode::NOT_IMPLEMENTED,
        Json(ErrorResponse {
            error: format!(
                "/v1/explain is not implemented: computing {} attributions requires a \
                 background dataset to establish expected values, which this request does \
                 not carry. Until it does, this endpoint returns an error rather than \
                 fabricated values (it previously returned index-derived SHAP numbers and \
                 a hardcoded prediction of 0.95 with HTTP 200).",
                request.method
            ),
        }),
    ))
}

/// APR audit handler (/v1/audit/:request_id)
///
/// Retrieves the audit record for a given request ID.
/// Real implementation using AuditLogger - NOT a stub.
pub(crate) async fn apr_audit_handler(
    State(state): State<AppState>,
    Path(request_id): Path<String>,
) -> Result<Json<AuditResponse>, (StatusCode, Json<ErrorResponse>)> {
    // Validate request_id format (should be UUID)
    if uuid::Uuid::parse_str(&request_id).is_err() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: format!("Invalid request ID format: {}", request_id),
            }),
        ));
    }

    // Flush buffer to ensure all records are available
    let _ = state.audit_logger.flush();

    // Search for the record in the audit sink
    let records = state.audit_sink.records();
    let record = records
        .into_iter()
        .find(|r| r.request_id == request_id)
        .ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: format!("Audit record not found for request_id: {}", request_id),
                }),
            )
        })?;

    Ok(Json(AuditResponse { record }))
}
