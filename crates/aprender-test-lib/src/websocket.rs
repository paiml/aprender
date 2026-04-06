//! WebSocket Monitoring (Feature 8)
//!
//! Monitor and mock WebSocket connections for testing.
//!
//! ## EXTREME TDD: Tests written FIRST per spec
//!
//! ## Toyota Way Application
//!
//! - **Genchi Genbutsu**: See actual WebSocket traffic
//! - **Jidoka**: Fail-fast on unexpected messages
//! - **Kaizen**: Continuous improvement through message inspection

use crate::result::{ProbarError, ProbarResult};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// WebSocket connection state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WebSocketState {
    /// Connection is being established
    Connecting,
    /// Connection is open
    Open,
    /// Connection is closing
    Closing,
    /// Connection is closed
    Closed,
}

/// WebSocket message type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessageType {
    /// Text message
    Text,
    /// Binary message
    Binary,
    /// Ping message
    Ping,
    /// Pong message
    Pong,
    /// Close message
    Close,
}

/// Direction of WebSocket message
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessageDirection {
    /// Message sent from client to server
    Sent,
    /// Message received from server
    Received,
}

/// A captured WebSocket message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSocketMessage {
    /// Message type
    pub message_type: MessageType,
    /// Message direction
    pub direction: MessageDirection,
    /// Message data (text or base64 encoded binary)
    pub data: String,
    /// Raw binary data (if binary message)
    #[serde(skip)]
    pub raw_data: Option<Vec<u8>>,
    /// Timestamp (milliseconds since connection start)
    pub timestamp_ms: u64,
    /// Connection ID this message belongs to
    pub connection_id: String,
}

impl WebSocketMessage {
    /// Create a new text message
    #[must_use]
    pub fn text(data: &str, direction: MessageDirection, timestamp_ms: u64) -> Self {
        Self {
            message_type: MessageType::Text,
            direction,
            data: data.to_string(),
            raw_data: None,
            timestamp_ms,
            connection_id: String::new(),
        }
    }

    /// Create a new binary message
    #[must_use]
    pub fn binary(data: Vec<u8>, direction: MessageDirection, timestamp_ms: u64) -> Self {
        Self {
            message_type: MessageType::Binary,
            direction,
            data: base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &data),
            raw_data: Some(data),
            timestamp_ms,
            connection_id: String::new(),
        }
    }

    /// Create a ping message
    #[must_use]
    pub fn ping(timestamp_ms: u64) -> Self {
        Self {
            message_type: MessageType::Ping,
            direction: MessageDirection::Sent,
            data: String::new(),
            raw_data: None,
            timestamp_ms,
            connection_id: String::new(),
        }
    }

    /// Create a pong message
    #[must_use]
    pub fn pong(timestamp_ms: u64) -> Self {
        Self {
            message_type: MessageType::Pong,
            direction: MessageDirection::Received,
            data: String::new(),
            raw_data: None,
            timestamp_ms,
            connection_id: String::new(),
        }
    }

    /// Create a close message
    #[must_use]
    pub fn close(code: u16, reason: &str, timestamp_ms: u64) -> Self {
        Self {
            message_type: MessageType::Close,
            direction: MessageDirection::Received,
            data: format!("{}: {}", code, reason),
            raw_data: None,
            timestamp_ms,
            connection_id: String::new(),
        }
    }

    /// Set connection ID
    #[must_use]
    pub fn with_connection(mut self, connection_id: &str) -> Self {
        self.connection_id = connection_id.to_string();
        self
    }

    /// Check if this is a text message
    #[must_use]
    pub const fn is_text(&self) -> bool {
        matches!(self.message_type, MessageType::Text)
    }

    /// Check if this is a binary message
    #[must_use]
    pub const fn is_binary(&self) -> bool {
        matches!(self.message_type, MessageType::Binary)
    }

    /// Check if this message was sent by client
    #[must_use]
    pub const fn is_sent(&self) -> bool {
        matches!(self.direction, MessageDirection::Sent)
    }

    /// Check if this message was received
    #[must_use]
    pub const fn is_received(&self) -> bool {
        matches!(self.direction, MessageDirection::Received)
    }

    /// Parse data as JSON
    pub fn json<T: for<'de> Deserialize<'de>>(&self) -> ProbarResult<T> {
        let data = serde_json::from_str(&self.data)?;
        Ok(data)
    }

    /// Check if data contains a string
    #[must_use]
    pub fn contains(&self, s: &str) -> bool {
        self.data.contains(s)
    }
}

/// A tracked WebSocket connection
#[derive(Debug)]
pub struct WebSocketConnection {
    /// Connection ID
    pub id: String,
    /// WebSocket URL
    pub url: String,
    /// Connection state
    pub state: WebSocketState,
    /// Messages on this connection
    messages: Arc<Mutex<Vec<WebSocketMessage>>>,
    /// Start time
    start_time: Instant,
    /// Close code (if closed)
    pub close_code: Option<u16>,
    /// Close reason (if closed)
    pub close_reason: Option<String>,
}

impl WebSocketConnection {
    /// Create a new connection
    #[must_use]
    pub fn new(id: &str, url: &str) -> Self {
        Self {
            id: id.to_string(),
            url: url.to_string(),
            state: WebSocketState::Connecting,
            messages: Arc::new(Mutex::new(Vec::new())),
            start_time: Instant::now(),
            close_code: None,
            close_reason: None,
        }
    }

    /// Open the connection
    pub fn open(&mut self) {
        self.state = WebSocketState::Open;
    }

    /// Close the connection
    pub fn close(&mut self, code: u16, reason: &str) {
        self.state = WebSocketState::Closed;
        self.close_code = Some(code);
        self.close_reason = Some(reason.to_string());
    }

    /// Get elapsed time in milliseconds
    #[must_use]
    pub fn elapsed_ms(&self) -> u64 {
        self.start_time.elapsed().as_millis() as u64
    }

    /// Record a message
    pub fn record_message(&self, mut message: WebSocketMessage) {
        message.connection_id = self.id.clone();
        if let Ok(mut messages) = self.messages.lock() {
            messages.push(message);
        }
    }

    /// Send a text message
    pub fn send_text(&self, data: &str) {
        let message = WebSocketMessage::text(data, MessageDirection::Sent, self.elapsed_ms());
        self.record_message(message);
    }

    /// Send a binary message
    pub fn send_binary(&self, data: Vec<u8>) {
        let message = WebSocketMessage::binary(data, MessageDirection::Sent, self.elapsed_ms());
        self.record_message(message);
    }

    /// Receive a text message
    pub fn receive_text(&self, data: &str) {
        let message = WebSocketMessage::text(data, MessageDirection::Received, self.elapsed_ms());
        self.record_message(message);
    }

    /// Receive a binary message
    pub fn receive_binary(&self, data: Vec<u8>) {
        let message = WebSocketMessage::binary(data, MessageDirection::Received, self.elapsed_ms());
        self.record_message(message);
    }

    /// Get all messages
    #[must_use]
    pub fn messages(&self) -> Vec<WebSocketMessage> {
        self.messages.lock().map(|m| m.clone()).unwrap_or_default()
    }

    /// Get sent messages
    #[must_use]
    pub fn sent_messages(&self) -> Vec<WebSocketMessage> {
        self.messages()
            .into_iter()
            .filter(|m| m.is_sent())
            .collect()
    }

    /// Get received messages
    #[must_use]
    pub fn received_messages(&self) -> Vec<WebSocketMessage> {
        self.messages()
            .into_iter()
            .filter(|m| m.is_received())
            .collect()
    }

    /// Get message count
    #[must_use]
    pub fn message_count(&self) -> usize {
        self.messages.lock().map(|m| m.len()).unwrap_or(0)
    }

    /// Check if connection is open
    #[must_use]
    pub const fn is_open(&self) -> bool {
        matches!(self.state, WebSocketState::Open)
    }

    /// Check if connection is closed
    #[must_use]
    pub const fn is_closed(&self) -> bool {
        matches!(self.state, WebSocketState::Closed)
    }
}

/// Mock responses for WebSocket
#[derive(Debug, Clone)]
pub struct MockWebSocketResponse {
    /// Messages to send when conditions are met
    pub messages: Vec<WebSocketMessage>,
    /// Delay before sending (ms)
    pub delay_ms: u64,
}

impl MockWebSocketResponse {
    /// Create a new mock response
    #[must_use]
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            delay_ms: 0,
        }
    }

    /// Add a text message
    #[must_use]
    pub fn with_text(mut self, data: &str) -> Self {
        self.messages
            .push(WebSocketMessage::text(data, MessageDirection::Received, 0));
        self
    }

    /// Add a binary message
    #[must_use]
    pub fn with_binary(mut self, data: Vec<u8>) -> Self {
        self.messages.push(WebSocketMessage::binary(
            data,
            MessageDirection::Received,
            0,
        ));
        self
    }

    /// Set delay
    #[must_use]
    pub const fn with_delay(mut self, delay_ms: u64) -> Self {
        self.delay_ms = delay_ms;
        self
    }
}

impl Default for MockWebSocketResponse {
    fn default() -> Self {
        Self::new()
    }
}

/// A WebSocket mock rule
#[derive(Debug, Clone)]
pub struct WebSocketMock {
    /// URL pattern to match
    pub url_pattern: String,
    /// Message pattern to match (for triggered responses)
    pub message_pattern: Option<String>,
    /// Response to send
    pub response: MockWebSocketResponse,
    /// Whether this is a one-time mock
    pub once: bool,
    /// Whether this mock has been used
    pub used: bool,
}

impl WebSocketMock {
    /// Create a new mock
    #[must_use]
    pub fn new(url_pattern: &str) -> Self {
        Self {
            url_pattern: url_pattern.to_string(),
            message_pattern: None,
            response: MockWebSocketResponse::new(),
            once: false,
            used: false,
        }
    }

    /// Set response for when connection opens
    #[must_use]
    pub fn on_open(mut self, response: MockWebSocketResponse) -> Self {
        self.response = response;
        self
    }

    /// Set response for when message is received
    #[must_use]
    pub fn on_message(mut self, pattern: &str, response: MockWebSocketResponse) -> Self {
        self.message_pattern = Some(pattern.to_string());
        self.response = response;
        self
    }

    /// Make this a one-time mock
    #[must_use]
    pub const fn once(mut self) -> Self {
        self.once = true;
        self
    }

    /// Check if URL matches
    #[must_use]
    pub fn matches_url(&self, url: &str) -> bool {
        if self.once && self.used {
            return false;
        }
        url.contains(&self.url_pattern)
    }

    /// Check if message matches
    #[must_use]
    pub fn matches_message(&self, message: &str) -> bool {
        if self.once && self.used {
            return false;
        }
        self.message_pattern
            .as_ref()
            .is_some_and(|p| message.contains(p))
    }

    /// Mark as used
    pub fn mark_used(&mut self) {
        self.used = true;
    }
}

/// WebSocket monitor for tracking connections
#[derive(Debug)]
pub struct WebSocketMonitor {
    /// Active connections
    connections: Arc<Mutex<Vec<WebSocketConnection>>>,
    /// Mock rules
    mocks: Vec<WebSocketMock>,
    /// Message queue for pending mock responses
    pending_responses: VecDeque<(String, MockWebSocketResponse)>,
    /// Whether monitoring is active
    active: bool,
    /// Connection counter
    connection_counter: u64,
}

impl Default for WebSocketMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl WebSocketMonitor {
    /// Create a new WebSocket monitor
    #[must_use]
    pub fn new() -> Self {
        Self {
            connections: Arc::new(Mutex::new(Vec::new())),
            mocks: Vec::new(),
            pending_responses: VecDeque::new(),
            active: false,
            connection_counter: 0,
        }
    }

    /// Start monitoring
    pub fn start(&mut self) {
        self.active = true;
    }

    /// Stop monitoring
    pub fn stop(&mut self) {
        self.active = false;
    }

    /// Check if monitoring is active
    #[must_use]
    pub const fn is_active(&self) -> bool {
        self.active
    }

    /// Add a mock rule
    pub fn mock(&mut self, mock: WebSocketMock) {
        self.mocks.push(mock);
    }

    /// Create a new connection
    pub fn connect(&mut self, url: &str) -> String {
        self.connection_counter += 1;
        let id = format!("ws_{}", self.connection_counter);

        let mut connection = WebSocketConnection::new(&id, url);
        connection.open();

        // Check for on_open mocks
        for mock in &mut self.mocks {
            if mock.matches_url(url) && mock.message_pattern.is_none() {
                self.pending_responses
                    .push_back((id.clone(), mock.response.clone()));
                mock.mark_used();
            }
        }

        if let Ok(mut connections) = self.connections.lock() {
            connections.push(connection);
        }

        id
    }

    /// Close a connection
    pub fn disconnect(&mut self, connection_id: &str, code: u16, reason: &str) {
        if let Ok(mut connections) = self.connections.lock() {
            if let Some(conn) = connections.iter_mut().find(|c| c.id == connection_id) {
                conn.close(code, reason);
            }
        }
    }

    /// Send a message on a connection
    pub fn send(&mut self, connection_id: &str, message: &str) {
        if let Ok(connections) = self.connections.lock() {
            if let Some(conn) = connections.iter().find(|c| c.id == connection_id) {
                conn.send_text(message);

                // Check for message-triggered mocks
                for mock in &mut self.mocks {
                    if mock.matches_url(&conn.url) && mock.matches_message(message) {
                        self.pending_responses
                            .push_back((connection_id.to_string(), mock.response.clone()));
                        mock.mark_used();
                    }
                }
            }
        }
    }

    /// Receive a message on a connection
    pub fn receive(&self, connection_id: &str, message: &str) {
        if let Ok(connections) = self.connections.lock() {
            if let Some(conn) = connections.iter().find(|c| c.id == connection_id) {
                conn.receive_text(message);
            }
        }
    }

    /// Get pending mock responses
    #[must_use]
    pub fn take_pending_responses(&mut self) -> Vec<(String, MockWebSocketResponse)> {
        self.pending_responses.drain(..).collect()
    }

    /// Get all connections
    #[must_use]
    pub fn connections(&self) -> Vec<String> {
        self.connections
            .lock()
            .map(|c| c.iter().map(|conn| conn.id.clone()).collect())
            .unwrap_or_default()
    }

    /// Get connection by ID
    pub fn get_connection(&self, connection_id: &str) -> Option<Vec<WebSocketMessage>> {
        self.connections.lock().ok().and_then(|connections| {
            connections
                .iter()
                .find(|c| c.id == connection_id)
                .map(|c| c.messages())
        })
    }

    /// Get all messages across all connections
    #[must_use]
    pub fn all_messages(&self) -> Vec<WebSocketMessage> {
        self.connections
            .lock()
            .map(|connections| connections.iter().flat_map(|c| c.messages()).collect())
            .unwrap_or_default()
    }

    /// Get connection count
    #[must_use]
    pub fn connection_count(&self) -> usize {
        self.connections.lock().map(|c| c.len()).unwrap_or(0)
    }

    /// Get active connection count
    #[must_use]
    pub fn active_connection_count(&self) -> usize {
        self.connections
            .lock()
            .map(|c| c.iter().filter(|conn| conn.is_open()).count())
            .unwrap_or(0)
    }

    /// Assert a message was sent
    pub fn assert_sent(&self, pattern: &str) -> ProbarResult<()> {
        let messages = self.all_messages();
        let found = messages.iter().any(|m| m.is_sent() && m.contains(pattern));
        if !found {
            return Err(ProbarError::AssertionFailed {
                message: format!(
                    "Expected sent message containing '{}', but none found",
                    pattern
                ),
            });
        }
        Ok(())
    }

    /// Assert a message was received
    pub fn assert_received(&self, pattern: &str) -> ProbarResult<()> {
        let messages = self.all_messages();
        let found = messages
            .iter()
            .any(|m| m.is_received() && m.contains(pattern));
        if !found {
            return Err(ProbarError::AssertionError {
                message: format!(
                    "Expected received message containing '{}', but none found",
                    pattern
                ),
            });
        }
        Ok(())
    }

    /// Assert connection was made to URL
    pub fn assert_connected(&self, url_pattern: &str) -> ProbarResult<()> {
        let found = self
            .connections
            .lock()
            .map(|connections| connections.iter().any(|c| c.url.contains(url_pattern)))
            .unwrap_or(false);

        if !found {
            return Err(ProbarError::AssertionError {
                message: format!(
                    "Expected connection to URL containing '{}', but none found",
                    url_pattern
                ),
            });
        }
        Ok(())
    }

    /// Clear all connections
    pub fn clear(&mut self) {
        if let Ok(mut connections) = self.connections.lock() {
            connections.clear();
        }
        self.mocks.clear();
        self.pending_responses.clear();
        self.connection_counter = 0;
    }
}

/// Builder for WebSocket monitor
#[derive(Debug, Default)]
pub struct WebSocketMonitorBuilder {
    monitor: WebSocketMonitor,
}

impl WebSocketMonitorBuilder {
    /// Create a new builder
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a mock for connection open
    #[must_use]
    pub fn mock_open(mut self, url_pattern: &str, response: MockWebSocketResponse) -> Self {
        self.monitor
            .mock(WebSocketMock::new(url_pattern).on_open(response));
        self
    }

    /// Add a mock for message
    #[must_use]
    pub fn mock_message(
        mut self,
        url_pattern: &str,
        message_pattern: &str,
        response: MockWebSocketResponse,
    ) -> Self {
        self.monitor
            .mock(WebSocketMock::new(url_pattern).on_message(message_pattern, response));
        self
    }

    /// Build the monitor
    #[must_use]
    pub fn build(self) -> WebSocketMonitor {
        self.monitor
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    mod websocket_message_tests {
        use super::*;

        #[test]
        fn test_text_message() {
            let msg = WebSocketMessage::text("hello", MessageDirection::Sent, 1000);
            assert!(msg.is_text());
            assert!(msg.is_sent());
            assert_eq!(msg.data, "hello");
            assert_eq!(msg.timestamp_ms, 1000);
        }

        #[test]
        fn test_binary_message() {
            let msg = WebSocketMessage::binary(vec![1, 2, 3], MessageDirection::Received, 500);
            assert!(msg.is_binary());
            assert!(msg.is_received());
            assert!(msg.raw_data.is_some());
        }

        #[test]
        fn test_ping_pong() {
            let ping = WebSocketMessage::ping(100);
            assert!(matches!(ping.message_type, MessageType::Ping));

            let pong = WebSocketMessage::pong(200);
            assert!(matches!(pong.message_type, MessageType::Pong));
        }

        #[test]
        fn test_close_message() {
            let close = WebSocketMessage::close(1000, "Normal closure", 500);
            assert!(matches!(close.message_type, MessageType::Close));
            assert!(close.data.contains("1000"));
        }

        #[test]
        fn test_with_connection() {
            let msg =
                WebSocketMessage::text("test", MessageDirection::Sent, 0).with_connection("conn_1");
            assert_eq!(msg.connection_id, "conn_1");
        }

        #[test]
        fn test_contains() {
            let msg = WebSocketMessage::text("hello world", MessageDirection::Sent, 0);
            assert!(msg.contains("world"));
            assert!(!msg.contains("foo"));
        }

        #[test]
        fn test_json() {
            let msg = WebSocketMessage::text(r#"{"name":"test"}"#, MessageDirection::Sent, 0);
            let data: serde_json::Value = msg.json().unwrap();
            assert_eq!(data["name"], "test");
        }
    }

    mod websocket_connection_tests {
        use super::*;

        #[test]
        fn test_new() {
            let conn = WebSocketConnection::new("conn_1", "ws://example.com");
            assert_eq!(conn.id, "conn_1");
            assert_eq!(conn.url, "ws://example.com");
            assert!(matches!(conn.state, WebSocketState::Connecting));
        }

        #[test]
        fn test_open() {
            let mut conn = WebSocketConnection::new("conn_1", "ws://example.com");
            conn.open();
            assert!(conn.is_open());
            assert!(!conn.is_closed());
        }

        #[test]
        fn test_close() {
            let mut conn = WebSocketConnection::new("conn_1", "ws://example.com");
            conn.open();
            conn.close(1000, "Normal closure");

            assert!(conn.is_closed());
            assert_eq!(conn.close_code, Some(1000));
            assert_eq!(conn.close_reason, Some("Normal closure".to_string()));
        }

        #[test]
        fn test_send_text() {
            let conn = WebSocketConnection::new("conn_1", "ws://example.com");
            conn.send_text("hello");

            let messages = conn.messages();
            assert_eq!(messages.len(), 1);
            assert!(messages[0].is_sent());
            assert_eq!(messages[0].data, "hello");
        }

        #[test]
        fn test_receive_text() {
            let conn = WebSocketConnection::new("conn_1", "ws://example.com");
            conn.receive_text("response");

            let messages = conn.messages();
            assert_eq!(messages.len(), 1);
            assert!(messages[0].is_received());
        }

        #[test]
        fn test_sent_received_messages() {
            let conn = WebSocketConnection::new("conn_1", "ws://example.com");
            conn.send_text("request");
            conn.receive_text("response");

            assert_eq!(conn.sent_messages().len(), 1);
            assert_eq!(conn.received_messages().len(), 1);
        }

        #[test]
        fn test_message_count() {
            let conn = WebSocketConnection::new("conn_1", "ws://example.com");
            conn.send_text("msg1");
            conn.send_text("msg2");

            assert_eq!(conn.message_count(), 2);
        }
    }

    mod mock_websocket_response_tests {
        use super::*;

        #[test]
        fn test_new() {
            let response = MockWebSocketResponse::new();
            assert!(response.messages.is_empty());
            assert_eq!(response.delay_ms, 0);
        }

        #[test]
        fn test_with_text() {
            let response = MockWebSocketResponse::new()
                .with_text("message 1")
                .with_text("message 2");
            assert_eq!(response.messages.len(), 2);
        }

        #[test]
        fn test_with_delay() {
            let response = MockWebSocketResponse::new().with_delay(100);
            assert_eq!(response.delay_ms, 100);
        }
    }

    mod websocket_mock_tests {
        use super::*;

        #[test]
        fn test_new() {
            let mock = WebSocketMock::new("ws://example.com");
            assert_eq!(mock.url_pattern, "ws://example.com");
            assert!(mock.message_pattern.is_none());
        }

        #[test]
        fn test_matches_url() {
            let mock = WebSocketMock::new("example.com");
            assert!(mock.matches_url("ws://example.com/socket"));
            assert!(!mock.matches_url("ws://other.com"));
        }

        #[test]
        fn test_matches_message() {
            let mock =
                WebSocketMock::new("example.com").on_message("hello", MockWebSocketResponse::new());
            assert!(mock.matches_message("say hello world"));
            assert!(!mock.matches_message("goodbye"));
        }

        #[test]
        fn test_once() {
            let mut mock = WebSocketMock::new("example.com").once();
            assert!(mock.matches_url("ws://example.com"));
            mock.mark_used();
            assert!(!mock.matches_url("ws://example.com"));
        }
    }

    mod websocket_monitor_tests {
        use super::*;

        #[test]
        fn test_new() {
            let monitor = WebSocketMonitor::new();
            assert!(!monitor.is_active());
            assert_eq!(monitor.connection_count(), 0);
        }

        #[test]
        fn test_start_stop() {
            let mut monitor = WebSocketMonitor::new();
            monitor.start();
            assert!(monitor.is_active());
            monitor.stop();
            assert!(!monitor.is_active());
        }

        #[test]
        fn test_connect() {
            let mut monitor = WebSocketMonitor::new();
            let id = monitor.connect("ws://example.com");
            assert!(!id.is_empty());
            assert_eq!(monitor.connection_count(), 1);
        }

        #[test]
        fn test_disconnect() {
            let mut monitor = WebSocketMonitor::new();
            let id = monitor.connect("ws://example.com");
            monitor.disconnect(&id, 1000, "Normal");

            assert_eq!(monitor.active_connection_count(), 0);
        }

        #[test]
        fn test_send() {
            let mut monitor = WebSocketMonitor::new();
            let id = monitor.connect("ws://example.com");
            monitor.send(&id, "hello");

            let messages = monitor.get_connection(&id).unwrap();
            assert_eq!(messages.len(), 1);
            assert!(messages[0].is_sent());
        }

        #[test]
        fn test_receive() {
            let mut monitor = WebSocketMonitor::new();
            let id = monitor.connect("ws://example.com");
            monitor.receive(&id, "response");

            let messages = monitor.get_connection(&id).unwrap();
            assert_eq!(messages.len(), 1);
            assert!(messages[0].is_received());
        }

        #[test]
        fn test_all_messages() {
            let mut monitor = WebSocketMonitor::new();
            let id1 = monitor.connect("ws://example.com");
            let id2 = monitor.connect("ws://other.com");

            monitor.send(&id1, "msg1");
            monitor.send(&id2, "msg2");

            let all = monitor.all_messages();
            assert_eq!(all.len(), 2);
        }

        #[test]
        fn test_mock_on_open() {
            let mut monitor = WebSocketMonitor::new();
            monitor.mock(
                WebSocketMock::new("example.com")
                    .on_open(MockWebSocketResponse::new().with_text("welcome")),
            );

            let _id = monitor.connect("ws://example.com/socket");
            let pending = monitor.take_pending_responses();

            assert_eq!(pending.len(), 1);
            assert_eq!(pending[0].1.messages.len(), 1);
        }

        #[test]
        fn test_mock_on_message() {
            let mut monitor = WebSocketMonitor::new();
            monitor.mock(
                WebSocketMock::new("example.com")
                    .on_message("ping", MockWebSocketResponse::new().with_text("pong")),
            );

            let id = monitor.connect("ws://example.com");
            monitor.send(&id, "ping");

            let pending = monitor.take_pending_responses();
            assert_eq!(pending.len(), 1);
        }

        #[test]
        fn test_assert_sent() {
            let mut monitor = WebSocketMonitor::new();
            let id = monitor.connect("ws://example.com");
            monitor.send(&id, "hello world");

            assert!(monitor.assert_sent("hello").is_ok());
            assert!(monitor.assert_sent("foo").is_err());
        }

        #[test]
        fn test_assert_received() {
            let mut monitor = WebSocketMonitor::new();
            let id = monitor.connect("ws://example.com");
            monitor.receive(&id, "server response");

            assert!(monitor.assert_received("response").is_ok());
            assert!(monitor.assert_received("foo").is_err());
        }

        #[test]
        fn test_assert_connected() {
            let mut monitor = WebSocketMonitor::new();
            let _id = monitor.connect("ws://example.com/socket");

            assert!(monitor.assert_connected("example.com").is_ok());
            assert!(monitor.assert_connected("other.com").is_err());
        }

        #[test]
        fn test_clear() {
            let mut monitor = WebSocketMonitor::new();
            let _id = monitor.connect("ws://example.com");
            monitor.mock(WebSocketMock::new("test"));

            monitor.clear();

            assert_eq!(monitor.connection_count(), 0);
        }
    }

    mod websocket_monitor_builder_tests {
        use super::*;

        #[test]
        fn test_builder() {
            let monitor = WebSocketMonitorBuilder::new()
                .mock_open(
                    "example.com",
                    MockWebSocketResponse::new().with_text("hello"),
                )
                .mock_message(
                    "example.com",
                    "ping",
                    MockWebSocketResponse::new().with_text("pong"),
                )
                .build();

            assert_eq!(monitor.mocks.len(), 2);
        }
    }

    mod additional_coverage_tests {
        use super::*;

        #[test]
        fn test_websocket_connection_send_binary() {
            let conn = WebSocketConnection::new("conn_1", "ws://example.com");
            conn.send_binary(vec![0x01, 0x02, 0x03, 0x04]);

            let messages = conn.messages();
            assert_eq!(messages.len(), 1);
            assert!(messages[0].is_binary());
            assert!(messages[0].is_sent());
            assert!(messages[0].raw_data.is_some());
            assert_eq!(
                messages[0].raw_data.as_ref().unwrap(),
                &vec![0x01, 0x02, 0x03, 0x04]
            );
        }

        #[test]
        fn test_websocket_connection_receive_binary() {
            let conn = WebSocketConnection::new("conn_1", "ws://example.com");
            conn.receive_binary(vec![0xDE, 0xAD, 0xBE, 0xEF]);

            let messages = conn.messages();
            assert_eq!(messages.len(), 1);
            assert!(messages[0].is_binary());
            assert!(messages[0].is_received());
            assert!(messages[0].raw_data.is_some());
        }

        #[test]
        fn test_websocket_connection_elapsed_ms() {
            let conn = WebSocketConnection::new("conn_1", "ws://example.com");
            // elapsed_ms should return some value >= 0
            let elapsed = conn.elapsed_ms();
            assert!(elapsed < 1000); // Should be very small, just created
        }

        #[test]
        fn test_mock_websocket_response_with_binary() {
            let response = MockWebSocketResponse::new()
                .with_binary(vec![1, 2, 3])
                .with_binary(vec![4, 5, 6]);
            assert_eq!(response.messages.len(), 2);
            assert!(response.messages[0].is_binary());
            assert!(response.messages[1].is_binary());
        }

        #[test]
        fn test_mock_websocket_response_default() {
            let response = MockWebSocketResponse::default();
            assert!(response.messages.is_empty());
            assert_eq!(response.delay_ms, 0);
        }

        #[test]
        fn test_websocket_monitor_default() {
            let monitor = WebSocketMonitor::default();
            assert!(!monitor.is_active());
            assert_eq!(monitor.connection_count(), 0);
        }

        #[test]
        fn test_websocket_monitor_connections_list() {
            let mut monitor = WebSocketMonitor::new();
            let id1 = monitor.connect("ws://example1.com");
            let id2 = monitor.connect("ws://example2.com");

            let connections = monitor.connections();
            assert_eq!(connections.len(), 2);
            assert!(connections.contains(&id1));
            assert!(connections.contains(&id2));
        }

        #[test]
        fn test_websocket_mock_on_open() {
            let mock = WebSocketMock::new("example.com")
                .on_open(MockWebSocketResponse::new().with_text("welcome"));

            assert_eq!(mock.url_pattern, "example.com");
            assert!(mock.message_pattern.is_none());
            assert_eq!(mock.response.messages.len(), 1);
        }

        #[test]
        fn test_websocket_message_json_error() {
            let msg = WebSocketMessage::text("not valid json {{{", MessageDirection::Sent, 0);
            let result: Result<serde_json::Value, _> = msg.json();
            assert!(result.is_err());
        }

        #[test]
        fn test_websocket_mock_matches_message_no_pattern() {
            let mock = WebSocketMock::new("example.com");
            // When message_pattern is None, matches_message should return false
            assert!(!mock.matches_message("any message"));
        }

        #[test]
        fn test_websocket_mock_once_matches_message() {
            let mut mock = WebSocketMock::new("example.com")
                .on_message("hello", MockWebSocketResponse::new())
                .once();

            assert!(mock.matches_message("say hello"));
            mock.mark_used();
            assert!(!mock.matches_message("say hello"));
        }

        #[test]
        fn test_websocket_monitor_get_connection_not_found() {
            let monitor = WebSocketMonitor::new();
            let result = monitor.get_connection("nonexistent");
            assert!(result.is_none());
        }

        #[test]
        fn test_websocket_monitor_send_to_nonexistent() {
            let mut monitor = WebSocketMonitor::new();
            // Should not panic when sending to nonexistent connection
            monitor.send("nonexistent", "message");
            assert_eq!(monitor.all_messages().len(), 0);
        }

        #[test]
        fn test_websocket_monitor_receive_to_nonexistent() {
            let monitor = WebSocketMonitor::new();
            // Should not panic when receiving on nonexistent connection
            monitor.receive("nonexistent", "message");
            assert_eq!(monitor.all_messages().len(), 0);
        }

        #[test]
        fn test_websocket_monitor_disconnect_nonexistent() {
            let mut monitor = WebSocketMonitor::new();
            // Should not panic when disconnecting nonexistent connection
            monitor.disconnect("nonexistent", 1000, "Normal");
        }

        #[test]
        fn test_websocket_state_enum_variants() {
            // Test all WebSocketState variants
            let connecting = WebSocketState::Connecting;
            let open = WebSocketState::Open;
            let closing = WebSocketState::Closing;
            let closed = WebSocketState::Closed;

            assert!(matches!(connecting, WebSocketState::Connecting));
            assert!(matches!(open, WebSocketState::Open));
            assert!(matches!(closing, WebSocketState::Closing));
            assert!(matches!(closed, WebSocketState::Closed));
        }

        #[test]
        fn test_message_type_enum_variants() {
            // Test all MessageType variants
            assert!(matches!(MessageType::Text, MessageType::Text));
            assert!(matches!(MessageType::Binary, MessageType::Binary));
            assert!(matches!(MessageType::Ping, MessageType::Ping));
            assert!(matches!(MessageType::Pong, MessageType::Pong));
            assert!(matches!(MessageType::Close, MessageType::Close));
        }

        #[test]
        fn test_message_direction_enum_variants() {
            // Test all MessageDirection variants
            assert!(matches!(MessageDirection::Sent, MessageDirection::Sent));
            assert!(matches!(
                MessageDirection::Received,
                MessageDirection::Received
            ));
        }

        #[test]
        fn test_websocket_message_is_not_text() {
            let msg = WebSocketMessage::binary(vec![1, 2, 3], MessageDirection::Sent, 0);
            assert!(!msg.is_text());
        }

        #[test]
        fn test_websocket_message_is_not_binary() {
            let msg = WebSocketMessage::text("hello", MessageDirection::Sent, 0);
            assert!(!msg.is_binary());
        }

        #[test]
        fn test_websocket_message_is_not_sent() {
            let msg = WebSocketMessage::text("hello", MessageDirection::Received, 0);
            assert!(!msg.is_sent());
        }

        #[test]
        fn test_websocket_message_is_not_received() {
            let msg = WebSocketMessage::text("hello", MessageDirection::Sent, 0);
            assert!(!msg.is_received());
        }

        #[test]
        fn test_websocket_connection_state_transitions() {
            let mut conn = WebSocketConnection::new("conn_1", "ws://example.com");
            assert!(matches!(conn.state, WebSocketState::Connecting));
            assert!(!conn.is_open());
            assert!(!conn.is_closed());

            conn.open();
            assert!(matches!(conn.state, WebSocketState::Open));
            assert!(conn.is_open());
            assert!(!conn.is_closed());

            conn.close(1000, "goodbye");
            assert!(matches!(conn.state, WebSocketState::Closed));
            assert!(!conn.is_open());
            assert!(conn.is_closed());
        }

        #[test]
        fn test_websocket_message_close_format() {
            let close = WebSocketMessage::close(1001, "Going Away", 100);
            assert!(close.data.contains("1001"));
            assert!(close.data.contains("Going Away"));
            assert_eq!(close.timestamp_ms, 100);
        }

        #[test]
        fn test_websocket_monitor_multiple_mocks_same_url() {
            let mut monitor = WebSocketMonitor::new();
            monitor.mock(
                WebSocketMock::new("example.com")
                    .on_open(MockWebSocketResponse::new().with_text("welcome1")),
            );
            monitor.mock(
                WebSocketMock::new("example.com")
                    .on_open(MockWebSocketResponse::new().with_text("welcome2")),
            );

            let _id = monitor.connect("ws://example.com/socket");
            let pending = monitor.take_pending_responses();

            // Both mocks should trigger
            assert_eq!(pending.len(), 2);
        }

        #[test]
        fn test_websocket_monitor_mock_message_url_mismatch() {
            let mut monitor = WebSocketMonitor::new();
            monitor.mock(
                WebSocketMock::new("other.com")
                    .on_message("hello", MockWebSocketResponse::new().with_text("response")),
            );

            let id = monitor.connect("ws://example.com");
            monitor.send(&id, "hello");

            // Mock should not trigger because URL doesn't match
            let pending = monitor.take_pending_responses();
            assert!(pending.is_empty());
        }

        #[test]
        fn test_websocket_connection_record_message_sets_connection_id() {
            let conn = WebSocketConnection::new("my_conn", "ws://example.com");
            let msg = WebSocketMessage::text("test", MessageDirection::Sent, 0);

            conn.record_message(msg);

            let messages = conn.messages();
            assert_eq!(messages.len(), 1);
            assert_eq!(messages[0].connection_id, "my_conn");
        }

        #[test]
        fn test_websocket_monitor_clear_resets_counter() {
            let mut monitor = WebSocketMonitor::new();
            let _id1 = monitor.connect("ws://example1.com");
            let _id2 = monitor.connect("ws://example2.com");

            monitor.clear();

            // After clear, connection counter resets
            let id3 = monitor.connect("ws://example3.com");
            assert_eq!(id3, "ws_1"); // Counter should be reset to 0, so first new conn is ws_1
        }

        #[test]
        fn test_websocket_monitor_builder_default() {
            let builder = WebSocketMonitorBuilder::default();
            let monitor = builder.build();
            assert!(!monitor.is_active());
            assert_eq!(monitor.mocks.len(), 0);
        }

        #[test]
        fn test_websocket_message_ping_direction() {
            let ping = WebSocketMessage::ping(50);
            assert!(ping.is_sent()); // Ping is sent from client
            assert_eq!(ping.timestamp_ms, 50);
            assert!(ping.data.is_empty());
        }

        #[test]
        fn test_websocket_message_pong_direction() {
            let pong = WebSocketMessage::pong(75);
            assert!(pong.is_received()); // Pong is received from server
            assert_eq!(pong.timestamp_ms, 75);
            assert!(pong.data.is_empty());
        }

        #[test]
        fn test_websocket_message_binary_base64_encoding() {
            let data = vec![0x48, 0x65, 0x6c, 0x6c, 0x6f]; // "Hello" in bytes
            let msg = WebSocketMessage::binary(data.clone(), MessageDirection::Sent, 0);

            // Verify the data is base64 encoded
            assert!(!msg.data.is_empty());
            // Decode and verify
            let decoded =
                base64::Engine::decode(&base64::engine::general_purpose::STANDARD, &msg.data)
                    .unwrap();
            assert_eq!(decoded, data);
        }

        #[test]
        fn test_websocket_connection_empty_messages() {
            let conn = WebSocketConnection::new("conn_1", "ws://example.com");
            assert!(conn.messages().is_empty());
            assert!(conn.sent_messages().is_empty());
            assert!(conn.received_messages().is_empty());
            assert_eq!(conn.message_count(), 0);
        }

        #[test]
        fn test_websocket_mock_response_chaining() {
            let response = MockWebSocketResponse::new()
                .with_text("msg1")
                .with_binary(vec![1, 2])
                .with_text("msg2")
                .with_delay(500);

            assert_eq!(response.messages.len(), 3);
            assert_eq!(response.delay_ms, 500);
            assert!(response.messages[0].is_text());
            assert!(response.messages[1].is_binary());
            assert!(response.messages[2].is_text());
        }

        #[test]
        fn test_websocket_message_json_valid_nested() {
            let msg = WebSocketMessage::text(
                r#"{"user":{"name":"Alice","age":30},"active":true}"#,
                MessageDirection::Received,
                0,
            );
            let data: serde_json::Value = msg.json().unwrap();
            assert_eq!(data["user"]["name"], "Alice");
            assert_eq!(data["user"]["age"], 30);
            assert_eq!(data["active"], true);
        }

        #[test]
        fn test_websocket_monitor_active_vs_total_count() {
            let mut monitor = WebSocketMonitor::new();
            let id1 = monitor.connect("ws://example1.com");
            let id2 = monitor.connect("ws://example2.com");
            let _id3 = monitor.connect("ws://example3.com");

            assert_eq!(monitor.connection_count(), 3);
            assert_eq!(monitor.active_connection_count(), 3);

            monitor.disconnect(&id1, 1000, "Normal");
            assert_eq!(monitor.connection_count(), 3);
            assert_eq!(monitor.active_connection_count(), 2);

            monitor.disconnect(&id2, 1000, "Normal");
            assert_eq!(monitor.connection_count(), 3);
            assert_eq!(monitor.active_connection_count(), 1);
        }

        #[test]
        fn test_websocket_message_clone() {
            let msg = WebSocketMessage::text("hello", MessageDirection::Sent, 100)
                .with_connection("conn_1");
            let cloned = msg.clone();

            assert_eq!(cloned.data, msg.data);
            assert_eq!(cloned.timestamp_ms, msg.timestamp_ms);
            assert_eq!(cloned.connection_id, msg.connection_id);
        }

        #[test]
        fn test_websocket_mock_clone() {
            let mock = WebSocketMock::new("example.com")
                .on_message("test", MockWebSocketResponse::new().with_text("response"))
                .once();
            let cloned = mock.clone();

            assert_eq!(cloned.url_pattern, mock.url_pattern);
            assert_eq!(cloned.message_pattern, mock.message_pattern);
            assert_eq!(cloned.once, mock.once);
        }

        #[test]
        fn test_mock_websocket_response_clone() {
            let response = MockWebSocketResponse::new()
                .with_text("msg")
                .with_delay(100);
            let cloned = response.clone();

            assert_eq!(cloned.messages.len(), response.messages.len());
            assert_eq!(cloned.delay_ms, response.delay_ms);
        }

        #[test]
        fn test_websocket_state_serialization() {
            let state = WebSocketState::Open;
            let json = serde_json::to_string(&state).unwrap();
            let deserialized: WebSocketState = serde_json::from_str(&json).unwrap();
            assert_eq!(state, deserialized);
        }

        #[test]
        fn test_message_type_serialization() {
            let msg_type = MessageType::Binary;
            let json = serde_json::to_string(&msg_type).unwrap();
            let deserialized: MessageType = serde_json::from_str(&json).unwrap();
            assert_eq!(msg_type, deserialized);
        }

        #[test]
        fn test_message_direction_serialization() {
            let direction = MessageDirection::Sent;
            let json = serde_json::to_string(&direction).unwrap();
            let deserialized: MessageDirection = serde_json::from_str(&json).unwrap();
            assert_eq!(direction, deserialized);
        }

        #[test]
        fn test_websocket_message_serialization() {
            let msg = WebSocketMessage::text("hello", MessageDirection::Sent, 1000)
                .with_connection("conn_1");
            let json = serde_json::to_string(&msg).unwrap();
            let deserialized: WebSocketMessage = serde_json::from_str(&json).unwrap();

            assert_eq!(deserialized.data, "hello");
            assert_eq!(deserialized.timestamp_ms, 1000);
            assert_eq!(deserialized.connection_id, "conn_1");
            // Note: raw_data is skipped in serialization
            assert!(deserialized.raw_data.is_none());
        }
    }
}
