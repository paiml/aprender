
/// Run prediction on a loaded model, returning (prediction, optional_probabilities)
#[cfg(feature = "aprender-serve")]
fn run_model_prediction(
    model: &LoadedModel,
    input: &Matrix<f32>,
    return_probs: bool,
) -> HttpResult<(f32, Option<Vec<f32>>)> {
    let map_err = |e: aprender::AprenderError| {
        http_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Inference error: {e}"),
            "E_INFERENCE_ERROR",
        )
    };

    match model {
        LoadedModel::LogisticRegression(lr) => {
            let pred = first_pred!(&lr.predict(input));
            let probs = return_probs.then(|| {
                let p1 = lr
                    .predict_proba(input)
                    .as_slice()
                    .first()
                    .copied()
                    .unwrap_or(0.5);
                vec![1.0 - p1, p1]
            });
            Ok((pred, probs))
        },
        LoadedModel::KNearestNeighbors(knn) => {
            Ok((first_pred!(&knn.predict(input).map_err(map_err)?), None))
        },
        LoadedModel::GaussianNB(nb) => {
            let pred = first_pred!(&nb.predict(input).map_err(map_err)?);
            let probs = if return_probs {
                nb.predict_proba(input).map_err(map_err)?.first().cloned()
            } else {
                None
            };
            Ok((pred, probs))
        },
        LoadedModel::LinearSVM(svm) => {
            Ok((first_pred!(&svm.predict(input).map_err(map_err)?), None))
        },
        LoadedModel::DecisionTreeClassifier(dt) => Ok((first_pred!(&dt.predict(input)), None)),
        LoadedModel::RandomForestClassifier(rf) => {
            let pred = first_pred!(&rf.predict(input));
            let probs = return_probs.then(|| {
                let m = rf.predict_proba(input);
                (0..m.n_cols()).map(|j| m.get(0, j)).collect()
            });
            Ok((pred, probs))
        },
        LoadedModel::GradientBoostingClassifier(gb) => {
            let pred = first_pred!(&gb.predict(input).map_err(map_err)?);
            let probs = if return_probs {
                gb.predict_proba(input).map_err(map_err)?.first().cloned()
            } else {
                None
            };
            Ok((pred, probs))
        },
        LoadedModel::LinearRegression(lr) => Ok((
            lr.predict(input).as_slice().first().copied().unwrap_or(0.0),
            None,
        )),
    }
}

/// Create router for aprender model serving
///
/// Per spec §5.1: HTTP API endpoint schema
pub fn create_serve_router(state: ServeState) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/ready", get(ready_handler))
        .route("/predict", post(predict_handler))
        .route("/predict/batch", post(batch_predict_handler))
        .route("/models", get(models_handler))
        .route("/metrics", get(metrics_handler))
        .with_state(state)
}

/// Health check handler
///
/// Per spec §5.1: Liveness probe
async fn health_handler() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

/// Readiness check handler
///
/// Per spec §5.1: Readiness probe (model loaded)
async fn ready_handler(State(state): State<ServeState>) -> Json<ReadyResponse> {
    let model_loaded = state.has_model();
    Json(ReadyResponse {
        ready: model_loaded,
        model_loaded,
        model_name: state.model_name,
    })
}

/// Predict handler
///
/// Per spec §5.1: Single prediction endpoint
/// Per spec §8.1: Target p50 latency <10ms (actual: ~0.5µs)
#[cfg(feature = "aprender-serve")]
async fn predict_handler(
    State(state): State<ServeState>,
    Json(payload): Json<PredictRequest>,
) -> Result<Json<PredictResponse>, (StatusCode, Json<ErrorResponse>)> {
    use std::sync::atomic::Ordering;

    state.request_count.fetch_add(1, Ordering::Relaxed);

    let model = require_model(&state)?;
    validate_dimensions(state.input_dim, payload.features.len())?;

    let start = Instant::now();
    let n_features = payload.features.len();
    let input = Matrix::from_vec(1, n_features, payload.features.clone()).map_err(|e| {
        http_error(
            StatusCode::BAD_REQUEST,
            format!("Matrix error: {e}"),
            "E_MATRIX_ERROR",
        )
    })?;

    let return_probs = payload
        .options
        .as_ref()
        .is_some_and(|o| o.return_probabilities);
    let (prediction, probabilities) = run_model_prediction(model, &input, return_probs)?;

    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

    Ok(Json(PredictResponse {
        prediction,
        probabilities,
        latency_ms,
        model_version: state.model_version.clone(),
    }))
}

/// Predict handler (fallback when aprender-serve not enabled)
#[cfg(not(feature = "aprender-serve"))]
async fn predict_handler(
    State(_state): State<ServeState>,
    Json(_payload): Json<PredictRequest>,
) -> Result<Json<PredictResponse>, (StatusCode, Json<ErrorResponse>)> {
    Err((
        StatusCode::NOT_IMPLEMENTED,
        Json(ErrorResponse {
            error: "aprender-serve feature not enabled".to_string(),
            code: Some("E_NOT_IMPLEMENTED".to_string()),
        }),
    ))
}

/// Batch predict handler
///
/// Per spec §5.3: Batch inference
#[cfg(feature = "aprender-serve")]
async fn batch_predict_handler(
    State(state): State<ServeState>,
    Json(payload): Json<BatchPredictRequest>,
) -> Result<Json<BatchPredictResponse>, (StatusCode, Json<ErrorResponse>)> {
    use std::sync::atomic::Ordering;

    let model = require_model(&state)?;

    if payload.instances.is_empty() {
        return Err(http_error(
            StatusCode::BAD_REQUEST,
            "Empty batch",
            "E_EMPTY_BATCH",
        ));
    }

    let batch_start = Instant::now();
    let mut predictions = Vec::with_capacity(payload.instances.len());

    state
        .request_count
        .fetch_add(payload.instances.len() as u64, Ordering::Relaxed);

    for instance in &payload.instances {
        validate_dimensions(state.input_dim, instance.features.len())?;

        let start = Instant::now();
        let n_features = instance.features.len();
        let input = Matrix::from_vec(1, n_features, instance.features.clone()).map_err(|e| {
            http_error(
                StatusCode::BAD_REQUEST,
                format!("Matrix error: {e}"),
                "E_MATRIX_ERROR",
            )
        })?;

        // Batch doesn't return probabilities to keep response smaller
        let (prediction, probabilities) = run_model_prediction(model, &input, false)?;
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        predictions.push(PredictResponse {
            prediction,
            probabilities,
            latency_ms,
            model_version: state.model_version.clone(),
        });
    }

    let total_latency_ms = batch_start.elapsed().as_secs_f64() * 1000.0;

    Ok(Json(BatchPredictResponse {
        predictions,
        total_latency_ms,
    }))
}

/// Batch predict handler (fallback when aprender-serve not enabled)
#[cfg(not(feature = "aprender-serve"))]
async fn batch_predict_handler(
    State(_state): State<ServeState>,
    Json(_payload): Json<BatchPredictRequest>,
) -> Result<Json<BatchPredictResponse>, (StatusCode, Json<ErrorResponse>)> {
    Err((
        StatusCode::NOT_IMPLEMENTED,
        Json(ErrorResponse {
            error: "aprender-serve feature not enabled".to_string(),
            code: Some("E_NOT_IMPLEMENTED".to_string()),
        }),
    ))
}

/// Models list handler
///
/// Per spec §5.1: List loaded models endpoint
async fn models_handler(
    State(state): State<ServeState>,
) -> Result<Json<ModelsResponse>, (StatusCode, Json<ErrorResponse>)> {
    let models = if state.has_model() {
        vec![ModelInfo {
            id: "default".to_string(),
            model_type: state.model_name.clone(),
            version: state.model_version,
            loaded: true,
        }]
    } else {
        vec![]
    };

    Ok(Json(ModelsResponse { models }))
}

/// Metrics handler
///
/// Per spec §5.1: Prometheus metrics endpoint
async fn metrics_handler(State(state): State<ServeState>) -> String {
    use std::sync::atomic::Ordering;

    let request_count = state.request_count.load(Ordering::Relaxed);
    let model_loaded = i32::from(state.has_model());

    format!(
        "# HELP requests_total Total number of inference requests\n\
         # TYPE requests_total counter\n\
         requests_total {request_count}\n\
         # HELP model_loaded Whether a model is loaded (1=yes, 0=no)\n\
         # TYPE model_loaded gauge\n\
         model_loaded {model_loaded}\n\
         # HELP input_dimension Expected input feature dimension\n\
         # TYPE input_dimension gauge\n\
         input_dimension {}\n",
        state.input_dim
    )
}
