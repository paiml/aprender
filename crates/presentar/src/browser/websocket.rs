//! WebSocket client for real-time data streaming.
//!
//! This module provides a browser-native WebSocket client that integrates
//! with Presentar's streaming infrastructure.
//!
//! # Example
//!
//! ```ignore
//! use presentar::browser::websocket::WebSocketClient;
//!
//! let client = WebSocketClient::new("wss://api.example.com/stream");
//! client.connect()?;
//! client.subscribe("metrics/cpu", |data| {
//!     console_log!("CPU: {}", data);
//! });
//! ```

use presentar_core::streaming::{
    ConnectionState, DataStream, RateLimiter, StreamConfig, StreamMessage, StreamSubscription,
};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::time::Duration;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{CloseEvent, ErrorEvent, MessageEvent, WebSocket};

/// WebSocket client for WASM environments.
pub struct WebSocketClient {
    /// Inner socket (None when disconnected)
    socket: Rc<RefCell<Option<WebSocket>>>,
    /// Connection URL
    url: String,
    /// Stream manager
    stream: Rc<RefCell<DataStream>>,
    /// Rate limiter for outbound messages
    rate_limiter: Rc<RefCell<RateLimiter>>,
    /// Message handlers by subscription ID
    handlers: Rc<RefCell<HashMap<String, Box<dyn Fn(serde_json::Value)>>>>,
    /// Error handlers
    error_handlers: Rc<RefCell<Vec<Box<dyn Fn(&str)>>>>,
    /// State change handlers
    state_handlers: Rc<RefCell<Vec<Box<dyn Fn(ConnectionState)>>>>,
    /// Reconnect timer handle
    reconnect_handle: Rc<RefCell<Option<i32>>>,
    /// Heartbeat timer handle
    heartbeat_handle: Rc<RefCell<Option<i32>>>,
    /// Configuration
    config: StreamConfig,
}

impl WebSocketClient {
    /// Create a new WebSocket client.
    #[must_use]
    pub fn new(url: impl Into<String>) -> Self {
        let url = url.into();
        let config = StreamConfig::new(&url);
        Self {
            socket: Rc::new(RefCell::new(None)),
            url,
            stream: Rc::new(RefCell::new(DataStream::new(config.clone()))),
            rate_limiter: Rc::new(RefCell::new(RateLimiter::new(100, Duration::from_secs(1)))),
            handlers: Rc::new(RefCell::new(HashMap::new())),
            error_handlers: Rc::new(RefCell::new(Vec::new())),
            state_handlers: Rc::new(RefCell::new(Vec::new())),
            reconnect_handle: Rc::new(RefCell::new(None)),
            heartbeat_handle: Rc::new(RefCell::new(None)),
            config,
        }
    }

    /// Create with custom config.
    #[must_use]
    pub fn with_config(config: StreamConfig) -> Self {
        let url = config.url.clone();
        Self {
            socket: Rc::new(RefCell::new(None)),
            url,
            stream: Rc::new(RefCell::new(DataStream::new(config.clone()))),
            rate_limiter: Rc::new(RefCell::new(RateLimiter::new(100, Duration::from_secs(1)))),
            handlers: Rc::new(RefCell::new(HashMap::new())),
            error_handlers: Rc::new(RefCell::new(Vec::new())),
            state_handlers: Rc::new(RefCell::new(Vec::new())),
            reconnect_handle: Rc::new(RefCell::new(None)),
            heartbeat_handle: Rc::new(RefCell::new(None)),
            config,
        }
    }

    /// Get the current connection state.
    #[must_use]
    pub fn state(&self) -> ConnectionState {
        self.stream.borrow().state()
    }

    /// Check if connected.
    #[must_use]
    pub fn is_connected(&self) -> bool {
        self.stream.borrow().state().is_active()
    }

    /// Connect to the WebSocket server.
    ///
    /// # Errors
    ///
    /// Returns an error if the WebSocket cannot be created.
    pub fn connect(&self) -> Result<(), WebSocketError> {
        if self.is_connected() {
            return Ok(());
        }

        self.set_state(ConnectionState::Connecting);

        let ws = WebSocket::new(&self.url)
            .map_err(|e| WebSocketError::ConnectionFailed(format!("{:?}", e)))?;

        ws.set_binary_type(web_sys::BinaryType::Arraybuffer);

        // Set up event handlers
        self.setup_open_handler(&ws);
        self.setup_message_handler(&ws);
        self.setup_error_handler(&ws);
        self.setup_close_handler(&ws);

        *self.socket.borrow_mut() = Some(ws);

        Ok(())
    }

    /// Disconnect from the server.
    pub fn disconnect(&self) {
        self.cancel_timers();

        if let Some(ws) = self.socket.borrow().as_ref() {
            let _ = ws.close();
        }

        *self.socket.borrow_mut() = None;
        self.set_state(ConnectionState::Disconnected);
    }

    /// Subscribe to a data source.
    pub fn subscribe<F>(&self, source: impl Into<String>, handler: F) -> String
    where
        F: Fn(serde_json::Value) + 'static,
    {
        let sub = StreamSubscription::new(source);
        let id = sub.id.clone();

        self.handlers
            .borrow_mut()
            .insert(id.clone(), Box::new(handler));
        self.stream.borrow().subscribe(sub);

        self.flush_outbox();
        id
    }

    /// Subscribe with transform and interval.
    pub fn subscribe_with_options<F>(
        &self,
        source: impl Into<String>,
        transform: Option<&str>,
        interval_ms: Option<u64>,
        handler: F,
    ) -> String
    where
        F: Fn(serde_json::Value) + 'static,
    {
        let mut sub = StreamSubscription::new(source);

        if let Some(t) = transform {
            sub = sub.with_transform(t);
        }
        if let Some(ms) = interval_ms {
            sub = sub.with_interval(ms);
        }

        let id = sub.id.clone();
        self.handlers
            .borrow_mut()
            .insert(id.clone(), Box::new(handler));
        self.stream.borrow().subscribe(sub);

        self.flush_outbox();
        id
    }

    /// Unsubscribe from a data source.
    pub fn unsubscribe(&self, id: &str) {
        self.handlers.borrow_mut().remove(id);
        self.stream.borrow().unsubscribe(id);
        self.flush_outbox();
    }

    /// Add an error handler.
    pub fn on_error<F>(&self, handler: F)
    where
        F: Fn(&str) + 'static,
    {
        self.error_handlers.borrow_mut().push(Box::new(handler));
    }

    /// Add a state change handler.
    pub fn on_state_change<F>(&self, handler: F)
    where
        F: Fn(ConnectionState) + 'static,
    {
        self.state_handlers.borrow_mut().push(Box::new(handler));
    }

    /// Get cached data for a subscription.
    #[must_use]
    pub fn get_data(&self, id: &str) -> Option<serde_json::Value> {
        self.stream.borrow().get_data(id)
    }

    /// Send a raw message.
    ///
    /// # Errors
    ///
    /// Returns error if not connected or rate limited.
    pub fn send(&self, msg: StreamMessage) -> Result<(), WebSocketError> {
        if !self.is_connected() {
            return Err(WebSocketError::NotConnected);
        }

        let now = js_sys::Date::now() as u64;
        if !self.rate_limiter.borrow_mut().check(now) {
            return Err(WebSocketError::RateLimited);
        }

        self.send_internal(&msg)
    }

    // === Private Methods ===

    fn set_state(&self, state: ConnectionState) {
        self.stream.borrow().set_state(state);

        for handler in self.state_handlers.borrow().iter() {
            handler(state);
        }
    }

    fn setup_open_handler(&self, ws: &WebSocket) {
        let stream = self.stream.clone();
        let socket = self.socket.clone();
        let state_handlers = self.state_handlers.clone();
        let config = self.config.clone();
        let heartbeat_handle = self.heartbeat_handle.clone();

        let onopen = Closure::<dyn FnMut()>::new(move || {
            stream.borrow().set_state(ConnectionState::Connected);
            stream.borrow().reset_reconnect_attempts();
            stream.borrow().resubscribe_all();

            // Notify handlers
            for handler in state_handlers.borrow().iter() {
                handler(ConnectionState::Connected);
            }

            // Flush pending messages
            let messages = stream.borrow().take_outbox();
            if let Some(ws) = socket.borrow().as_ref() {
                for msg in messages {
                    if let Ok(json) = serde_json::to_string(&msg) {
                        let _ = ws.send_with_str(&json);
                    }
                }
            }

            // Start heartbeat
            let heartbeat_interval = config.heartbeat_interval.as_millis() as i32;
            let socket_clone = socket.clone();
            let heartbeat_cb = Closure::<dyn FnMut()>::new(move || {
                if let Some(ws) = socket_clone.borrow().as_ref() {
                    let ping = StreamMessage::ping(js_sys::Date::now() as u64);
                    if let Ok(json) = serde_json::to_string(&ping) {
                        let _ = ws.send_with_str(&json);
                    }
                }
            });

            if let Some(window) = web_sys::window() {
                if let Ok(id) = window.set_interval_with_callback_and_timeout_and_arguments_0(
                    heartbeat_cb.as_ref().unchecked_ref(),
                    heartbeat_interval,
                ) {
                    *heartbeat_handle.borrow_mut() = Some(id);
                }
            }
            heartbeat_cb.forget();
        });

        ws.set_onopen(Some(onopen.as_ref().unchecked_ref()));
        onopen.forget();
    }

    fn setup_message_handler(&self, ws: &WebSocket) {
        let stream = self.stream.clone();
        let handlers = self.handlers.clone();
        let socket = self.socket.clone();

        let onmessage = Closure::<dyn FnMut(MessageEvent)>::new(move |e: MessageEvent| {
            if let Ok(txt) = e.data().dyn_into::<js_sys::JsString>() {
                let s: String = txt.into();
                if let Ok(msg) = serde_json::from_str::<StreamMessage>(&s) {
                    // Handle response messages (like pong)
                    if let Some(response) = stream.borrow().handle_message(msg.clone()) {
                        if let Some(ws) = socket.borrow().as_ref() {
                            if let Ok(json) = serde_json::to_string(&response) {
                                let _ = ws.send_with_str(&json);
                            }
                        }
                    }

                    // Dispatch data to handlers
                    if let StreamMessage::Data { id, payload, .. } = msg {
                        if let Some(handler) = handlers.borrow().get(&id) {
                            handler(payload);
                        }
                    }
                }
            }
        });

        ws.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));
        onmessage.forget();
    }

    fn setup_error_handler(&self, ws: &WebSocket) {
        let error_handlers = self.error_handlers.clone();

        let onerror = Closure::<dyn FnMut(ErrorEvent)>::new(move |e: ErrorEvent| {
            let msg = e.message();
            for handler in error_handlers.borrow().iter() {
                handler(&msg);
            }
        });

        ws.set_onerror(Some(onerror.as_ref().unchecked_ref()));
        onerror.forget();
    }

    fn setup_close_handler(&self, ws: &WebSocket) {
        let stream = self.stream.clone();
        let socket = self.socket.clone();
        let state_handlers = self.state_handlers.clone();
        let reconnect_handle = self.reconnect_handle.clone();
        let heartbeat_handle = self.heartbeat_handle.clone();
        let url = self.url.clone();
        let config = self.config.clone();

        let onclose = Closure::<dyn FnMut(CloseEvent)>::new(move |e: CloseEvent| {
            // Cancel heartbeat
            if let Some(id) = heartbeat_handle.borrow_mut().take() {
                if let Some(window) = web_sys::window() {
                    window.clear_interval_with_handle(id);
                }
            }

            let was_clean = e.was_clean();

            if was_clean {
                stream.borrow().set_state(ConnectionState::Disconnected);
                for handler in state_handlers.borrow().iter() {
                    handler(ConnectionState::Disconnected);
                }
            } else if stream.borrow().should_reconnect() {
                stream.borrow().set_state(ConnectionState::Reconnecting);
                for handler in state_handlers.borrow().iter() {
                    handler(ConnectionState::Reconnecting);
                }

                // Schedule reconnect
                let delay = stream.borrow().reconnect_delay();
                stream.borrow().increment_reconnect_attempts();

                let stream_clone = stream.clone();
                let socket_clone = socket.clone();
                let state_handlers_clone = state_handlers.clone();
                let url_clone = url.clone();
                let config_clone = config.clone();
                let heartbeat_handle_clone = heartbeat_handle.clone();

                let reconnect_cb = Closure::<dyn FnMut()>::new(move || {
                    // Try to reconnect
                    if let Ok(new_ws) = WebSocket::new(&url_clone) {
                        new_ws.set_binary_type(web_sys::BinaryType::Arraybuffer);

                        // Setup handlers for new socket (simplified - in real code would share logic)
                        let stream_inner = stream_clone.clone();
                        let state_handlers_inner = state_handlers_clone.clone();
                        let socket_inner = socket_clone.clone();
                        let config_inner = config_clone.clone();
                        let heartbeat_inner = heartbeat_handle_clone.clone();

                        let onopen = Closure::<dyn FnMut()>::new(move || {
                            stream_inner.borrow().set_state(ConnectionState::Connected);
                            stream_inner.borrow().reset_reconnect_attempts();
                            stream_inner.borrow().resubscribe_all();

                            for handler in state_handlers_inner.borrow().iter() {
                                handler(ConnectionState::Connected);
                            }

                            // Flush pending
                            let messages = stream_inner.borrow().take_outbox();
                            if let Some(ws) = socket_inner.borrow().as_ref() {
                                for msg in messages {
                                    if let Ok(json) = serde_json::to_string(&msg) {
                                        let _ = ws.send_with_str(&json);
                                    }
                                }
                            }

                            // Restart heartbeat
                            let heartbeat_interval =
                                config_inner.heartbeat_interval.as_millis() as i32;
                            let socket_hb = socket_inner.clone();
                            let heartbeat_fn = Closure::<dyn FnMut()>::new(move || {
                                if let Some(ws) = socket_hb.borrow().as_ref() {
                                    let ping = StreamMessage::ping(js_sys::Date::now() as u64);
                                    if let Ok(json) = serde_json::to_string(&ping) {
                                        let _ = ws.send_with_str(&json);
                                    }
                                }
                            });

                            if let Some(window) = web_sys::window() {
                                if let Ok(id) = window
                                    .set_interval_with_callback_and_timeout_and_arguments_0(
                                        heartbeat_fn.as_ref().unchecked_ref(),
                                        heartbeat_interval,
                                    )
                                {
                                    *heartbeat_inner.borrow_mut() = Some(id);
                                }
                            }
                            heartbeat_fn.forget();
                        });

                        new_ws.set_onopen(Some(onopen.as_ref().unchecked_ref()));
                        onopen.forget();

                        *socket_clone.borrow_mut() = Some(new_ws);
                    }
                });

                if let Some(window) = web_sys::window() {
                    if let Ok(id) = window.set_timeout_with_callback_and_timeout_and_arguments_0(
                        reconnect_cb.as_ref().unchecked_ref(),
                        delay.as_millis() as i32,
                    ) {
                        *reconnect_handle.borrow_mut() = Some(id);
                    }
                }
                reconnect_cb.forget();
            } else {
                stream.borrow().set_state(ConnectionState::Failed);
                for handler in state_handlers.borrow().iter() {
                    handler(ConnectionState::Failed);
                }
            }
        });

        ws.set_onclose(Some(onclose.as_ref().unchecked_ref()));
        onclose.forget();
    }

    fn flush_outbox(&self) {
        if !self.is_connected() {
            return;
        }

        let messages = self.stream.borrow().take_outbox();
        if let Some(ws) = self.socket.borrow().as_ref() {
            for msg in messages {
                if let Ok(json) = serde_json::to_string(&msg) {
                    let _ = ws.send_with_str(&json);
                }
            }
        }
    }

    fn send_internal(&self, msg: &StreamMessage) -> Result<(), WebSocketError> {
        let json = serde_json::to_string(msg)
            .map_err(|e| WebSocketError::SerializationFailed(e.to_string()))?;

        if let Some(ws) = self.socket.borrow().as_ref() {
            ws.send_with_str(&json)
                .map_err(|e| WebSocketError::SendFailed(format!("{:?}", e)))?;
        } else {
            return Err(WebSocketError::NotConnected);
        }

        Ok(())
    }

    fn cancel_timers(&self) {
        if let Some(window) = web_sys::window() {
            if let Some(id) = self.reconnect_handle.borrow_mut().take() {
                window.clear_timeout_with_handle(id);
            }
            if let Some(id) = self.heartbeat_handle.borrow_mut().take() {
                window.clear_interval_with_handle(id);
            }
        }
    }
}

impl Drop for WebSocketClient {
    fn drop(&mut self) {
        self.disconnect();
    }
}

/// WebSocket errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WebSocketError {
    /// Failed to create connection
    ConnectionFailed(String),
    /// Not connected
    NotConnected,
    /// Rate limited
    RateLimited,
    /// Failed to serialize message
    SerializationFailed(String),
    /// Failed to send message
    SendFailed(String),
}

impl std::fmt::Display for WebSocketError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConnectionFailed(msg) => write!(f, "connection failed: {msg}"),
            Self::NotConnected => write!(f, "not connected"),
            Self::RateLimited => write!(f, "rate limited"),
            Self::SerializationFailed(msg) => write!(f, "serialization failed: {msg}"),
            Self::SendFailed(msg) => write!(f, "send failed: {msg}"),
        }
    }
}

impl std::error::Error for WebSocketError {}

// ============================================================================
// Tests (non-WASM only since we can't run browser tests in regular cargo test)
// ============================================================================

#[cfg(test)]
#[cfg(not(target_arch = "wasm32"))]
mod tests {
    use super::*;

    // Test WebSocketError Display
    #[test]
    fn test_error_display() {
        assert_eq!(WebSocketError::NotConnected.to_string(), "not connected");
        assert_eq!(WebSocketError::RateLimited.to_string(), "rate limited");
        assert_eq!(
            WebSocketError::ConnectionFailed("timeout".to_string()).to_string(),
            "connection failed: timeout"
        );
    }

    #[test]
    fn test_error_eq() {
        assert_eq!(WebSocketError::NotConnected, WebSocketError::NotConnected);
        assert_ne!(WebSocketError::NotConnected, WebSocketError::RateLimited);
    }
}

// WASM-specific tests
#[cfg(test)]
#[cfg(target_arch = "wasm32")]
mod wasm_tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    fn test_websocket_client_creation() {
        let client = WebSocketClient::new("wss://example.com/ws");
        assert_eq!(client.state(), ConnectionState::Disconnected);
        assert!(!client.is_connected());
    }

    #[wasm_bindgen_test]
    fn test_websocket_client_with_config() {
        let config = StreamConfig::new("wss://example.com/ws")
            .with_heartbeat(Duration::from_secs(60))
            .with_buffer_size(2048);
        let client = WebSocketClient::with_config(config);
        assert_eq!(client.state(), ConnectionState::Disconnected);
    }

    #[wasm_bindgen_test]
    fn test_subscribe_when_disconnected() {
        let client = WebSocketClient::new("wss://example.com/ws");
        let id = client.subscribe("metrics/cpu", |_data| {});
        assert!(id.starts_with("sub_"));
    }

    #[wasm_bindgen_test]
    fn test_unsubscribe() {
        let client = WebSocketClient::new("wss://example.com/ws");
        let id = client.subscribe("metrics/cpu", |_data| {});
        client.unsubscribe(&id);
        // No panic means success
    }

    #[wasm_bindgen_test]
    fn test_send_when_disconnected() {
        let client = WebSocketClient::new("wss://example.com/ws");
        let result = client.send(StreamMessage::ping(12345));
        assert!(matches!(result, Err(WebSocketError::NotConnected)));
    }

    #[wasm_bindgen_test]
    fn test_disconnect_when_not_connected() {
        let client = WebSocketClient::new("wss://example.com/ws");
        client.disconnect(); // Should not panic
        assert_eq!(client.state(), ConnectionState::Disconnected);
    }

    #[wasm_bindgen_test]
    fn test_on_error_handler() {
        let client = WebSocketClient::new("wss://example.com/ws");
        client.on_error(|msg| {
            web_sys::console::log_1(&format!("Error: {}", msg).into());
        });
        // No panic means success
    }

    #[wasm_bindgen_test]
    fn test_on_state_change_handler() {
        let client = WebSocketClient::new("wss://example.com/ws");
        client.on_state_change(|state| {
            web_sys::console::log_1(&format!("State: {:?}", state).into());
        });
        // No panic means success
    }

    #[wasm_bindgen_test]
    fn test_get_data_empty() {
        let client = WebSocketClient::new("wss://example.com/ws");
        assert!(client.get_data("nonexistent").is_none());
    }

    #[wasm_bindgen_test]
    fn test_subscribe_with_options() {
        let client = WebSocketClient::new("wss://example.com/ws");
        let id =
            client.subscribe_with_options("metrics/cpu", Some("rate()"), Some(1000), |_data| {});
        assert!(id.starts_with("sub_"));
    }
}
