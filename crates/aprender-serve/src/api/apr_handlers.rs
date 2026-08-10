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

/// 503 for "this endpoint needs a tabular APR estimator and none is resident".
///
/// The pre-fix message was `"No APR model loaded. Use AppState::demo() or load a
/// .apr model."`. It was wrong twice over: it was emitted verbatim while the
/// server was serving a `.apr` file (the startup banner even logs `Detected
/// format: APR / APR loaded: 339 tensors`), and it told an HTTP client to call
/// `AppState::demo()` — an internal Rust constructor no client can reach. The
/// real condition is narrower: `/v1/predict` and `/v1/explain` serve tabular
/// estimators (a `weights`/`output` vector), and a language model loaded from
/// `.apr`/GGUF is not one. Say that, and point at the endpoint that does serve it.
fn no_estimator_error(state: &AppState, endpoint: &str) -> (StatusCode, Json<ErrorResponse>) {
    let hint = if state.model_loaded() {
        " The loaded model is a language model; use /v1/chat/completions or /v1/completions instead."
    } else {
        " Start the server with a tabular .apr estimator to enable it."
    };
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(ErrorResponse {
            error: format!(
                "{endpoint} serves tabular APR estimator models (a 'weights' or 'output' \
                 tensor); no such model is loaded.{hint}"
            ),
        }),
    )
}

/// Exact Shapley values for the linear estimator `/v1/predict` evaluates.
///
/// `/v1/predict` computes `f(x) = Σ wᵢ·xᵢ` from the model's `weights`/`output`
/// tensor. For a linear model against an all-zero baseline the Shapley value of
/// feature `i` is exactly `φᵢ = wᵢ·xᵢ` (Lundberg & Lee 2017, §4.1 "Linear SHAP"),
/// with `base_value = f(0) = 0`, so local accuracy `Σφᵢ + base_value == f(x)`
/// holds by construction — and the returned `prediction` is the SAME number
/// `/v1/predict` returns for the same features.
///
/// This replaces a hardcoded `0.1 - i*0.02` ramp and a literal `prediction: 0.95`
/// that were a pure function of the feature INDEX: three wildly different feature
/// vectors produced byte-identical SHAP values and the same 0.95, with HTTP 200.
pub(crate) fn linear_shap_attributions(features: &[f32], weights: &[f32]) -> (Vec<f32>, f32) {
    let shap_values: Vec<f32> = features
        .iter()
        .zip(weights.iter())
        .map(|(x, w)| x * w)
        .collect();
    let prediction = shap_values.iter().sum();
    (shap_values, prediction)
}

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
    let apr_model = state
        .apr_model
        .as_ref()
        .ok_or_else(|| no_estimator_error(&state, "/v1/predict"))?;

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
/// Returns SHAP feature attributions computed FROM the loaded APR estimator —
/// `φᵢ = wᵢ·xᵢ` against a zero baseline, the exact Shapley values for the linear
/// model `/v1/predict` evaluates. When no such estimator is resident the endpoint
/// fails closed with 503, exactly like its sibling `/v1/predict`.
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

    // Only "shap" is implemented. `method` was previously parsed and then dropped
    // on the floor, so `method: "lime"` returned the same numbers relabelled as a
    // LIME explanation. Reject what we cannot compute rather than mislabel it.
    if request.method != "shap" {
        return Err((
            StatusCode::NOT_IMPLEMENTED,
            Json(ErrorResponse {
                error: format!(
                    "Explanation method '{}' is not implemented; only 'shap' is supported.",
                    request.method
                ),
            }),
        ));
    }

    // Fail closed when there is no estimator to attribute to — the sibling
    // /v1/predict already 503s under exactly this condition.
    let apr_model = state
        .apr_model
        .as_ref()
        .ok_or_else(|| no_estimator_error(&state, "/v1/explain"))?;

    // Same tensor lookup /v1/predict uses, so both endpoints explain the SAME model.
    let weights = apr_model
        .get_tensor_f32("weights")
        .or_else(|_| apr_model.get_tensor_f32("output"))
        .map_err(|_| no_estimator_error(&state, "/v1/explain"))?;

    if weights.len() != request.features.len() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: format!(
                    "Feature count ({}) must match the model's weight count ({})",
                    request.features.len(),
                    weights.len()
                ),
            }),
        ));
    }

    let (shap_values, predicted) = linear_shap_attributions(&request.features, &weights);

    let explanation = ShapExplanation {
        // f(0) = 0 for the linear estimator /v1/predict evaluates, so local
        // accuracy (Σφᵢ + base_value == prediction) holds exactly.
        base_value: 0.0,
        shap_values: shap_values.clone(),
        feature_names: request.feature_names.clone(),
        prediction: predicted,
    };

    // Build summary from top features
    let mut feature_importance: Vec<_> = request
        .feature_names
        .iter()
        .zip(shap_values.iter())
        .collect();
    feature_importance.sort_by(|a, b| {
        b.1.abs()
            .partial_cmp(&a.1.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let top_features: Vec<_> = feature_importance
        .iter()
        .take(request.top_k_features)
        .collect();

    let summary = if top_features.is_empty() {
        "No significant features found.".to_string()
    } else {
        let feature_strs: Vec<String> = top_features
            .iter()
            .map(|(name, val)| {
                let direction = if **val > 0.0 { "+" } else { "-" };
                format!("{} ({})", name, direction)
            })
            .collect();
        format!("Top contributing features: {}", feature_strs.join(", "))
    };

    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

    Ok(Json(ExplainResponse {
        request_id,
        model: request.model.unwrap_or_else(|| "default".to_string()),
        prediction: serde_json::json!(predicted),
        // No calibrated confidence exists for a regression output. The old literal
        // 0.95 was indistinguishable from a real one to any consumer; omit it.
        confidence: None,
        explanation,
        summary,
        latency_ms,
    }))
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
