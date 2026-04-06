//! Web visualization for simular.
//!
//! WebSocket streaming server using axum.
//!
//! This module is only available with the `web` feature.

use std::sync::Arc;

use axum::{
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    response::{Html, IntoResponse},
    routing::get,
    Router,
};
use tokio::sync::{broadcast, RwLock};

use super::SimMetrics;
use crate::engine::{SimState, SimTime};
use crate::error::SimResult;

/// Shared state for the web server.
#[derive(Clone)]
pub struct WebState {
    /// Broadcast channel for simulation updates.
    tx: broadcast::Sender<String>,
    /// Connected client count.
    client_count: Arc<RwLock<usize>>,
}

impl Default for WebState {
    fn default() -> Self {
        Self::new()
    }
}

impl WebState {
    /// Create new web state.
    #[must_use]
    pub fn new() -> Self {
        let (tx, _) = broadcast::channel(100);
        Self {
            tx,
            client_count: Arc::new(RwLock::new(0)),
        }
    }

    /// Get a broadcast receiver for simulation updates.
    #[must_use]
    pub fn subscribe(&self) -> broadcast::Receiver<String> {
        self.tx.subscribe()
    }

    /// Get the number of connected clients.
    pub async fn client_count(&self) -> usize {
        *self.client_count.read().await
    }
}

/// Web visualization server.
pub struct WebVisualization {
    /// Server state.
    state: WebState,
    /// Server port.
    port: u16,
}

impl WebVisualization {
    /// Create new web visualization server.
    #[must_use]
    pub fn new(port: u16) -> Self {
        Self {
            state: WebState::new(),
            port,
        }
    }

    /// Get the router for the web server.
    pub fn router(&self) -> Router {
        let state = self.state.clone();
        Router::new()
            .route(
                "/ws",
                get(move |ws: WebSocketUpgrade| {
                    let state = state.clone();
                    async move { ws.on_upgrade(move |socket| handle_socket(socket, state)) }
                }),
            )
            .route("/", get(index_handler))
            .route("/health", get(health_handler))
    }

    /// Broadcast simulation state to all connected clients.
    ///
    /// # Errors
    ///
    /// Returns error if serialization fails.
    pub fn broadcast(
        &self,
        state: &SimState,
        time: SimTime,
        metrics: &SimMetrics,
    ) -> SimResult<()> {
        let payload = WebPayload {
            time: time.as_secs_f64(),
            body_count: state.num_bodies(),
            kinetic_energy: state.kinetic_energy(),
            potential_energy: state.potential_energy(),
            total_energy: state.total_energy(),
            metrics: metrics.clone(),
        };

        let json = serde_json::to_string(&payload).map_err(|e| {
            crate::error::SimError::serialization(format!("JSON serialization failed: {e}"))
        })?;

        // Ignore send errors (no subscribers is fine)
        let _ = self.state.tx.send(json);
        Ok(())
    }

    /// Get the number of connected clients.
    pub async fn client_count(&self) -> usize {
        self.state.client_count().await
    }

    /// Get the server port.
    #[must_use]
    pub const fn port(&self) -> u16 {
        self.port
    }

    /// Get a broadcast receiver for simulation updates.
    #[must_use]
    pub fn subscribe(&self) -> broadcast::Receiver<String> {
        self.state.subscribe()
    }
}

/// Handle individual WebSocket connection.
async fn handle_socket(mut socket: WebSocket, state: WebState) {
    // Increment client count
    {
        let mut count = state.client_count.write().await;
        *count += 1;
    }

    // Subscribe to broadcast channel
    let mut rx = state.tx.subscribe();

    // Send updates to client
    loop {
        tokio::select! {
            // Forward broadcast messages to client
            result = rx.recv() => {
                match result {
                    Ok(msg) => {
                        if socket.send(Message::Text(msg)).await.is_err() {
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(_)) => {
                        // Lagged receiver — dropped messages are expected for live viz
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        break;
                    }
                }
            }
            // Handle incoming messages from client
            result = socket.recv() => {
                match result {
                    Some(Ok(Message::Close(_)) | Err(_)) | None => break,
                    Some(Ok(Message::Ping(data))) => {
                        if socket.send(Message::Pong(data)).await.is_err() {
                            break;
                        }
                    }
                    Some(Ok(_)) => {}
                }
            }
        }
    }

    // Decrement client count
    {
        let mut count = state.client_count.write().await;
        *count = count.saturating_sub(1);
    }
}

/// Index handler (simple HTML page).
async fn index_handler() -> Html<&'static str> {
    Html(INDEX_HTML)
}

/// Health check handler.
async fn health_handler() -> impl IntoResponse {
    "{\"status\":\"ok\"}"
}

/// Payload sent to WebSocket clients.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WebPayload {
    /// Simulation time in seconds.
    pub time: f64,
    /// Number of bodies.
    pub body_count: usize,
    /// Kinetic energy.
    pub kinetic_energy: f64,
    /// Potential energy.
    pub potential_energy: f64,
    /// Total energy.
    pub total_energy: f64,
    /// Full metrics.
    pub metrics: SimMetrics,
}

/// Simple HTML page for visualization.
const INDEX_HTML: &str = r#"<!DOCTYPE html>
<html>
<head>
    <title>Simular Visualization</title>
    <style>
        body { font-family: monospace; background: #1a1a2e; color: #eee; padding: 20px; }
        .metric { margin: 10px 0; }
        .value { color: #00ff88; }
        #status { color: #ff6b6b; }
        #status.connected { color: #00ff88; }
    </style>
</head>
<body>
    <h1>Simular Visualization</h1>
    <div id="status">Disconnected</div>
    <div class="metric">Time: <span id="time" class="value">0.0</span>s</div>
    <div class="metric">Bodies: <span id="bodies" class="value">0</span></div>
    <div class="metric">Total Energy: <span id="energy" class="value">0.0</span></div>
    <div class="metric">Kinetic: <span id="ke" class="value">0.0</span></div>
    <div class="metric">Potential: <span id="pe" class="value">0.0</span></div>
    <script>
        const ws = new WebSocket(`ws://${location.host}/ws`);
        ws.onopen = () => {
            document.getElementById('status').textContent = 'Connected';
            document.getElementById('status').className = 'connected';
        };
        ws.onclose = () => {
            document.getElementById('status').textContent = 'Disconnected';
            document.getElementById('status').className = '';
        };
        ws.onmessage = (e) => {
            const data = JSON.parse(e.data);
            document.getElementById('time').textContent = data.time.toFixed(4);
            document.getElementById('bodies').textContent = data.body_count;
            document.getElementById('energy').textContent = data.total_energy.toFixed(6);
            document.getElementById('ke').textContent = data.kinetic_energy.toFixed(6);
            document.getElementById('pe').textContent = data.potential_energy.toFixed(6);
        };
    </script>
</body>
</html>"#;

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::engine::{SimState, SimTime};

    #[test]
    fn test_web_state_new() {
        let state = WebState::new();
        assert!(state.tx.receiver_count() == 0);
    }

    #[test]
    fn test_web_state_default() {
        let state = WebState::default();
        assert!(state.tx.receiver_count() == 0);
    }

    #[test]
    fn test_web_state_subscribe() {
        let state = WebState::new();
        let _rx1 = state.subscribe();
        assert_eq!(state.tx.receiver_count(), 1);

        let _rx2 = state.subscribe();
        assert_eq!(state.tx.receiver_count(), 2);
    }

    #[test]
    fn test_web_state_clone() {
        let state = WebState::new();
        let cloned = state.clone();
        // Both share the same broadcast channel
        let _rx = state.subscribe();
        assert_eq!(cloned.tx.receiver_count(), 1);
    }

    #[tokio::test]
    async fn test_web_state_client_count_initial() {
        let state = WebState::new();
        assert_eq!(state.client_count().await, 0);
    }

    #[test]
    fn test_web_visualization_new() {
        let viz = WebVisualization::new(8080);
        assert_eq!(viz.port(), 8080);
    }

    #[test]
    fn test_web_visualization_port_zero() {
        let viz = WebVisualization::new(0);
        assert_eq!(viz.port(), 0);
    }

    #[test]
    fn test_web_visualization_router() {
        let viz = WebVisualization::new(8080);
        let _router = viz.router();
        // Router was created successfully
    }

    #[test]
    fn test_web_visualization_subscribe() {
        let viz = WebVisualization::new(8080);
        let _rx = viz.subscribe();
        // Subscribed successfully
    }

    #[tokio::test]
    async fn test_web_visualization_client_count() {
        let viz = WebVisualization::new(8080);
        assert_eq!(viz.client_count().await, 0);
    }

    #[test]
    fn test_web_visualization_broadcast() {
        let viz = WebVisualization::new(8080);
        let mut rx = viz.subscribe();

        let state = SimState::default();
        let time = SimTime::from_secs(1.0);
        let metrics = SimMetrics::new();

        let result = viz.broadcast(&state, time, &metrics);
        assert!(result.is_ok());

        // Should be able to receive the broadcast
        let msg = rx.try_recv();
        assert!(msg.is_ok());
        let json = msg.unwrap();
        assert!(json.contains("\"time\":1.0"));
    }

    #[test]
    fn test_web_visualization_broadcast_no_subscribers() {
        let viz = WebVisualization::new(8080);
        // No subscribers - should still succeed (just drops message)

        let state = SimState::default();
        let time = SimTime::from_secs(1.0);
        let metrics = SimMetrics::new();

        let result = viz.broadcast(&state, time, &metrics);
        assert!(result.is_ok());
    }

    #[test]
    fn test_web_payload_serialize() {
        let payload = WebPayload {
            time: 1.0,
            body_count: 2,
            kinetic_energy: 10.0,
            potential_energy: -5.0,
            total_energy: 5.0,
            metrics: SimMetrics::new(),
        };

        let json = serde_json::to_string(&payload).ok();
        assert!(json.is_some());
        assert!(json.as_ref().map_or(false, |j| j.contains("\"time\":1.0")));
    }

    #[test]
    fn test_web_payload_deserialize() {
        let json = r#"{"time":1.0,"body_count":2,"kinetic_energy":10.0,"potential_energy":-5.0,"total_energy":5.0,"metrics":{"time":0.0,"step":0,"steps_per_second":0.0,"total_energy":null,"kinetic_energy":null,"potential_energy":null,"energy_drift":null,"body_count":0,"jidoka_warnings":0,"jidoka_errors":0,"memory_bytes":0,"custom":{}}}"#;

        let payload: WebPayload = serde_json::from_str(json).unwrap();
        assert!((payload.time - 1.0).abs() < f64::EPSILON);
        assert_eq!(payload.body_count, 2);
    }

    #[test]
    fn test_web_payload_clone() {
        let payload = WebPayload {
            time: 1.0,
            body_count: 2,
            kinetic_energy: 10.0,
            potential_energy: -5.0,
            total_energy: 5.0,
            metrics: SimMetrics::new(),
        };

        let cloned = payload.clone();
        assert!((cloned.time - 1.0).abs() < f64::EPSILON);
        assert_eq!(cloned.body_count, 2);
    }

    #[test]
    fn test_web_payload_debug() {
        let payload = WebPayload {
            time: 1.0,
            body_count: 2,
            kinetic_energy: 10.0,
            potential_energy: -5.0,
            total_energy: 5.0,
            metrics: SimMetrics::new(),
        };

        let debug_str = format!("{:?}", payload);
        assert!(debug_str.contains("WebPayload"));
        assert!(debug_str.contains("time: 1.0"));
    }

    #[tokio::test]
    async fn test_health_handler() {
        let response = health_handler().await;
        let body = response.into_response();
        // Handler returns a response
        let _ = body;
    }

    #[tokio::test]
    async fn test_index_handler() {
        let response = index_handler().await;
        let html = response.0;
        assert!(html.contains("Simular Visualization"));
        assert!(html.contains("WebSocket"));
    }

    #[test]
    fn test_index_html_content() {
        assert!(INDEX_HTML.contains("<!DOCTYPE html>"));
        assert!(INDEX_HTML.contains("Simular Visualization"));
        assert!(INDEX_HTML.contains("ws://"));
        assert!(INDEX_HTML.contains("Total Energy"));
    }

    #[test]
    fn test_broadcast_channel_capacity() {
        let state = WebState::new();
        // Channel has capacity of 100
        for i in 0..100 {
            let _ = state.tx.send(format!("msg{i}"));
        }
        // Should not panic
    }

    #[test]
    fn test_web_visualization_broadcast_multiple() {
        let viz = WebVisualization::new(8080);
        let mut rx = viz.subscribe();

        let state = SimState::default();
        let metrics = SimMetrics::new();

        // Multiple broadcasts
        for i in 0..5 {
            let time = SimTime::from_secs(i as f64);
            let result = viz.broadcast(&state, time, &metrics);
            assert!(result.is_ok());
        }

        // Should receive all broadcasts
        for i in 0..5 {
            let msg = rx.try_recv();
            assert!(msg.is_ok(), "Should receive broadcast {i}");
        }
    }

    #[test]
    fn test_web_payload_default_metrics() {
        let payload = WebPayload {
            time: 0.0,
            body_count: 0,
            kinetic_energy: 0.0,
            potential_energy: 0.0,
            total_energy: 0.0,
            metrics: SimMetrics::new(),
        };

        let json = serde_json::to_string(&payload).unwrap();
        assert!(json.contains("\"time\":0.0"));
        assert!(json.contains("\"body_count\":0"));
    }

    #[test]
    fn test_web_state_multiple_subscribers() {
        let state = WebState::new();
        let mut rx1 = state.subscribe();
        let mut rx2 = state.subscribe();
        let mut rx3 = state.subscribe();

        // Send a message
        let _ = state.tx.send("test".to_string());

        // All subscribers should receive
        assert!(rx1.try_recv().is_ok());
        assert!(rx2.try_recv().is_ok());
        assert!(rx3.try_recv().is_ok());
    }

    #[tokio::test]
    async fn test_client_count_thread_safe() {
        let state = WebState::new();
        let state_clone = state.clone();

        // Concurrent reads should work
        let (count1, count2) = tokio::join!(state.client_count(), state_clone.client_count());

        assert_eq!(count1, 0);
        assert_eq!(count2, 0);
    }

    #[test]
    fn test_web_payload_with_bodies() {
        let mut sim_state = SimState::default();
        sim_state.add_body(
            1.0,
            crate::engine::state::Vec3::new(1.0, 0.0, 0.0),
            crate::engine::state::Vec3::new(0.0, 1.0, 0.0),
        );
        sim_state.add_body(
            2.0,
            crate::engine::state::Vec3::new(-1.0, 0.0, 0.0),
            crate::engine::state::Vec3::new(0.0, -0.5, 0.0),
        );

        let viz = WebVisualization::new(8080);
        let mut rx = viz.subscribe();

        let result = viz.broadcast(&sim_state, SimTime::from_secs(1.0), &SimMetrics::new());
        assert!(result.is_ok());

        let msg = rx.try_recv().unwrap();
        assert!(msg.contains("\"body_count\":2"));
    }

    #[test]
    fn test_router_routes_exist() {
        let viz = WebVisualization::new(8080);
        let router = viz.router();

        // Router should have routes configured
        // We can't easily test route matching without a full request,
        // but we verify the router was constructed
        let _ = router;
    }

    #[test]
    fn test_web_payload_energy_values() {
        let payload = WebPayload {
            time: 10.5,
            body_count: 3,
            kinetic_energy: 100.5,
            potential_energy: -50.25,
            total_energy: 50.25,
            metrics: SimMetrics::new(),
        };

        assert!((payload.kinetic_energy - 100.5).abs() < f64::EPSILON);
        assert!((payload.potential_energy - (-50.25)).abs() < f64::EPSILON);
        assert!((payload.total_energy - 50.25).abs() < f64::EPSILON);
    }
}
