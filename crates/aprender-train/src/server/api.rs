//! API router and server setup
//!
//! Configures axum routes and runs the HTTP server.

use crate::server::{
    handlers::{
        create_experiment, create_run, get_experiment, get_run, health_check, list_experiments,
        log_metrics, log_params, update_run,
    },
    state::AppState,
    Result, ServerConfig, ServerError,
};
use axum::{
    routing::{get, patch, post},
    Router,
};
use std::net::SocketAddr;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;

/// Tracking server for experiment management
pub struct TrackingServer {
    config: ServerConfig,
    state: AppState,
}

impl TrackingServer {
    /// Create a new tracking server
    pub fn new(config: ServerConfig) -> Self {
        let state = AppState::new(config.clone());
        Self { config, state }
    }

    /// Build the router
    pub fn router(&self) -> Router {
        let mut app = Router::new()
            // Health check
            .route("/health", get(health_check))
            // Experiments
            .route("/api/v1/experiments", post(create_experiment))
            .route("/api/v1/experiments", get(list_experiments))
            .route("/api/v1/experiments/{id}", get(get_experiment))
            // Runs
            .route("/api/v1/runs", post(create_run))
            .route("/api/v1/runs/{id}", get(get_run))
            .route("/api/v1/runs/{id}", patch(update_run))
            .route("/api/v1/runs/{id}/params", post(log_params))
            .route("/api/v1/runs/{id}/metrics", post(log_metrics))
            // State
            .with_state(self.state.clone())
            // Tracing
            .layer(TraceLayer::new_for_http());

        // Add CORS if enabled
        if self.config.cors_enabled {
            let cors = CorsLayer::new().allow_origin(Any).allow_methods(Any).allow_headers(Any);
            app = app.layer(cors);
        }

        app
    }

    /// Run the server
    pub async fn run(&self) -> Result<()> {
        let addr = self.config.address;
        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .map_err(|e| ServerError::Bind(e.to_string()))?;

        println!("🚀 Entrenar tracking server running on http://{addr}");

        axum::serve(listener, self.router()).await.map_err(ServerError::Io)?;

        Ok(())
    }

    /// Get the configured address
    pub fn address(&self) -> SocketAddr {
        self.config.address
    }

    /// Get the current state (for testing)
    pub fn state(&self) -> &AppState {
        &self.state
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    fn test_server() -> TrackingServer {
        TrackingServer::new(ServerConfig::default())
    }

    #[tokio::test]
    async fn test_tracking_server_new() {
        let server = test_server();
        assert_eq!(server.address().port(), 5000);
    }

    #[tokio::test]
    async fn test_health_endpoint() {
        let server = test_server();
        let app = server.router();

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .expect("operation should succeed"),
            )
            .await
            .expect("operation should succeed");

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_create_experiment_endpoint() {
        let server = test_server();
        let app = server.router();

        let body = r#"{"name": "test-experiment"}"#;
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/v1/experiments")
                    .header("Content-Type", "application/json")
                    .body(Body::from(body))
                    .expect("operation should succeed"),
            )
            .await
            .expect("operation should succeed");

        assert_eq!(response.status(), StatusCode::CREATED);
    }

    #[tokio::test]
    async fn test_list_experiments_endpoint() {
        let server = test_server();
        let app = server.router();

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/v1/experiments")
                    .body(Body::empty())
                    .expect("operation should succeed"),
            )
            .await
            .expect("operation should succeed");

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_get_experiment_not_found() {
        let server = test_server();
        let app = server.router();

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/v1/experiments/nonexistent")
                    .body(Body::empty())
                    .expect("operation should succeed"),
            )
            .await
            .expect("operation should succeed");

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_create_run_endpoint() {
        let server = test_server();

        // First create an experiment
        let exp = server
            .state
            .storage
            .create_experiment("test", None, None)
            .expect("operation should succeed");

        let app = server.router();
        let body = format!(r#"{{"experiment_id": "{}"}}"#, exp.id);

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/v1/runs")
                    .header("Content-Type", "application/json")
                    .body(Body::from(body))
                    .expect("operation should succeed"),
            )
            .await
            .expect("operation should succeed");

        assert_eq!(response.status(), StatusCode::CREATED);
    }

    #[tokio::test]
    async fn test_log_params_endpoint() {
        let server = test_server();

        // Create experiment and run
        let exp = server
            .state
            .storage
            .create_experiment("test", None, None)
            .expect("operation should succeed");
        let run =
            server.state.storage.create_run(&exp.id, None, None).expect("operation should succeed");

        let app = server.router();
        let body = r#"{"params": {"lr": 0.001, "batch_size": 32}}"#;

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/v1/runs/{}/params", run.id))
                    .header("Content-Type", "application/json")
                    .body(Body::from(body))
                    .expect("operation should succeed"),
            )
            .await
            .expect("operation should succeed");

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_log_metrics_endpoint() {
        let server = test_server();

        // Create experiment and run
        let exp = server
            .state
            .storage
            .create_experiment("test", None, None)
            .expect("operation should succeed");
        let run =
            server.state.storage.create_run(&exp.id, None, None).expect("operation should succeed");

        let app = server.router();
        let body = r#"{"metrics": {"loss": 0.5, "accuracy": 0.9}, "step": 100}"#;

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/v1/runs/{}/metrics", run.id))
                    .header("Content-Type", "application/json")
                    .body(Body::from(body))
                    .expect("operation should succeed"),
            )
            .await
            .expect("operation should succeed");

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_update_run_endpoint() {
        let server = test_server();

        // Create experiment and run
        let exp = server
            .state
            .storage
            .create_experiment("test", None, None)
            .expect("operation should succeed");
        let run =
            server.state.storage.create_run(&exp.id, None, None).expect("operation should succeed");

        let app = server.router();
        let body = r#"{"status": "completed"}"#;

        let response = app
            .oneshot(
                Request::builder()
                    .method("PATCH")
                    .uri(format!("/api/v1/runs/{}", run.id))
                    .header("Content-Type", "application/json")
                    .body(Body::from(body))
                    .expect("operation should succeed"),
            )
            .await
            .expect("operation should succeed");

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_cors_enabled() {
        let config = ServerConfig::default();
        assert!(config.cors_enabled);

        let server = TrackingServer::new(config);
        let _app = server.router();
        // Router builds successfully with CORS
    }

    #[tokio::test]
    async fn test_cors_disabled() {
        let config = ServerConfig::default().without_cors();
        assert!(!config.cors_enabled);

        let server = TrackingServer::new(config);
        let _app = server.router();
        // Router builds successfully without CORS
    }
}
