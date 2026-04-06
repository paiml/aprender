#![allow(
    clippy::unwrap_used,
    clippy::disallowed_methods,
    clippy::comparison_chain,
    clippy::match_same_arms
)]
//! Data streaming and live updates for Presentar.
//!
//! This module provides infrastructure for real-time data updates:
//! - Subscription management for data sources
//! - Message protocol for updates
//! - Reconnection and backpressure handling
//! - Integration with expression executor for live transforms
//!
//! # Example
//!
//! ```
//! use presentar_core::streaming::{
//!     DataStream, StreamConfig, StreamMessage, StreamSubscription,
//! };
//!
//! // Create a subscription
//! let sub = StreamSubscription::new("metrics/cpu")
//!     .with_interval(1000)  // 1 second
//!     .with_transform("rate()");
//!
//! // Handle incoming messages
//! fn handle_message(msg: StreamMessage) {
//!     match msg {
//!         StreamMessage::Data { payload, .. } => println!("Got data: {:?}", payload),
//!         StreamMessage::Error { message, .. } => eprintln!("Error: {}", message),
//!         _ => {}
//!     }
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Stream connection state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConnectionState {
    /// Not connected
    #[default]
    Disconnected,
    /// Attempting to connect
    Connecting,
    /// Connected and ready
    Connected,
    /// Connection lost, attempting reconnection
    Reconnecting,
    /// Permanently failed
    Failed,
}

impl ConnectionState {
    /// Check if the stream can send/receive messages.
    #[must_use]
    pub const fn is_active(&self) -> bool {
        matches!(self, Self::Connected)
    }

    /// Check if the stream is trying to connect.
    #[must_use]
    pub const fn is_connecting(&self) -> bool {
        matches!(self, Self::Connecting | Self::Reconnecting)
    }
}

/// Stream message types for the protocol.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamMessage {
    /// Subscribe to a data source
    Subscribe {
        /// Subscription ID
        id: String,
        /// Data source path
        source: String,
        /// Optional transform expression
        #[serde(skip_serializing_if = "Option::is_none")]
        transform: Option<String>,
        /// Refresh interval in milliseconds
        #[serde(skip_serializing_if = "Option::is_none")]
        interval_ms: Option<u64>,
    },
    /// Unsubscribe from a data source
    Unsubscribe {
        /// Subscription ID
        id: String,
    },
    /// Data update from server
    Data {
        /// Subscription ID this data belongs to
        id: String,
        /// The payload data
        payload: serde_json::Value,
        /// Sequence number for ordering
        #[serde(default)]
        seq: u64,
        /// Timestamp of the data
        #[serde(skip_serializing_if = "Option::is_none")]
        timestamp: Option<u64>,
    },
    /// Error message
    Error {
        /// Related subscription ID (if any)
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        /// Error message
        message: String,
        /// Error code
        #[serde(skip_serializing_if = "Option::is_none")]
        code: Option<i32>,
    },
    /// Acknowledgment
    Ack {
        /// Subscription ID being acknowledged
        id: String,
        /// Status message
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
    },
    /// Heartbeat/ping
    Ping {
        /// Timestamp
        timestamp: u64,
    },
    /// Heartbeat response
    Pong {
        /// Echo of ping timestamp
        timestamp: u64,
    },
}

impl StreamMessage {
    /// Create a subscribe message.
    #[must_use]
    pub fn subscribe(id: impl Into<String>, source: impl Into<String>) -> Self {
        Self::Subscribe {
            id: id.into(),
            source: source.into(),
            transform: None,
            interval_ms: None,
        }
    }

    /// Create a subscribe message with transform.
    #[must_use]
    pub fn subscribe_with_transform(
        id: impl Into<String>,
        source: impl Into<String>,
        transform: impl Into<String>,
    ) -> Self {
        Self::Subscribe {
            id: id.into(),
            source: source.into(),
            transform: Some(transform.into()),
            interval_ms: None,
        }
    }

    /// Create an unsubscribe message.
    #[must_use]
    pub fn unsubscribe(id: impl Into<String>) -> Self {
        Self::Unsubscribe { id: id.into() }
    }

    /// Create a data message.
    #[must_use]
    pub fn data(id: impl Into<String>, payload: serde_json::Value, seq: u64) -> Self {
        Self::Data {
            id: id.into(),
            payload,
            seq,
            timestamp: None,
        }
    }

    /// Create an error message.
    #[must_use]
    pub fn error(message: impl Into<String>) -> Self {
        Self::Error {
            id: None,
            message: message.into(),
            code: None,
        }
    }

    /// Create an error message for a subscription.
    #[must_use]
    pub fn error_for(id: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Error {
            id: Some(id.into()),
            message: message.into(),
            code: None,
        }
    }

    /// Create an ack message.
    #[must_use]
    pub fn ack(id: impl Into<String>) -> Self {
        Self::Ack {
            id: id.into(),
            status: None,
        }
    }

    /// Create a ping message.
    #[must_use]
    pub fn ping(timestamp: u64) -> Self {
        Self::Ping { timestamp }
    }

    /// Create a pong message.
    #[must_use]
    pub fn pong(timestamp: u64) -> Self {
        Self::Pong { timestamp }
    }

    /// Get the subscription ID if this message has one.
    #[must_use]
    pub fn subscription_id(&self) -> Option<&str> {
        match self {
            Self::Subscribe { id, .. }
            | Self::Unsubscribe { id }
            | Self::Data { id, .. }
            | Self::Ack { id, .. } => Some(id),
            Self::Error { id, .. } => id.as_deref(),
            Self::Ping { .. } | Self::Pong { .. } => None,
        }
    }
}

/// Subscription to a data source.
#[derive(Debug, Clone)]
pub struct StreamSubscription {
    /// Unique subscription ID
    pub id: String,
    /// Data source path
    pub source: String,
    /// Transform expression to apply
    pub transform: Option<String>,
    /// Refresh interval
    pub interval: Option<Duration>,
    /// Whether this subscription is active
    pub active: bool,
    /// Last received sequence number
    pub last_seq: u64,
    /// Error count for backoff
    pub error_count: u32,
}

impl StreamSubscription {
    /// Create a new subscription.
    #[must_use]
    pub fn new(source: impl Into<String>) -> Self {
        let source = source.into();
        let id = format!("sub_{}", Self::hash_source(&source));
        Self {
            id,
            source,
            transform: None,
            interval: None,
            active: false,
            last_seq: 0,
            error_count: 0,
        }
    }

    /// Create with explicit ID.
    #[must_use]
    pub fn with_id(id: impl Into<String>, source: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            source: source.into(),
            transform: None,
            interval: None,
            active: false,
            last_seq: 0,
            error_count: 0,
        }
    }

    /// Set refresh interval in milliseconds.
    #[must_use]
    pub fn with_interval(mut self, ms: u64) -> Self {
        self.interval = Some(Duration::from_millis(ms));
        self
    }

    /// Set transform expression.
    #[must_use]
    pub fn with_transform(mut self, transform: impl Into<String>) -> Self {
        self.transform = Some(transform.into());
        self
    }

    /// Convert to subscribe message.
    #[must_use]
    pub fn to_message(&self) -> StreamMessage {
        StreamMessage::Subscribe {
            id: self.id.clone(),
            source: self.source.clone(),
            transform: self.transform.clone(),
            interval_ms: self.interval.map(|d| d.as_millis() as u64),
        }
    }

    /// Simple hash for generating IDs.
    fn hash_source(s: &str) -> u64 {
        let mut hash: u64 = 5381;
        for byte in s.bytes() {
            hash = hash.wrapping_mul(33).wrapping_add(u64::from(byte));
        }
        hash
    }
}

/// Configuration for stream connection.
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// WebSocket URL
    pub url: String,
    /// Reconnection settings
    pub reconnect: ReconnectConfig,
    /// Heartbeat interval
    pub heartbeat_interval: Duration,
    /// Message buffer size
    pub buffer_size: usize,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            url: String::new(),
            reconnect: ReconnectConfig::default(),
            heartbeat_interval: Duration::from_secs(30),
            buffer_size: 1024,
        }
    }
}

impl StreamConfig {
    /// Create with URL.
    #[must_use]
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            ..Default::default()
        }
    }

    /// Set reconnection config.
    #[must_use]
    pub fn with_reconnect(mut self, config: ReconnectConfig) -> Self {
        self.reconnect = config;
        self
    }

    /// Set heartbeat interval.
    #[must_use]
    pub fn with_heartbeat(mut self, interval: Duration) -> Self {
        self.heartbeat_interval = interval;
        self
    }

    /// Set buffer size.
    #[must_use]
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }
}

/// Reconnection configuration.
#[derive(Debug, Clone)]
pub struct ReconnectConfig {
    /// Whether to auto-reconnect
    pub enabled: bool,
    /// Initial delay before first reconnect
    pub initial_delay: Duration,
    /// Maximum delay between reconnects
    pub max_delay: Duration,
    /// Multiplier for exponential backoff
    pub backoff_multiplier: f32,
    /// Maximum reconnection attempts (None = infinite)
    pub max_attempts: Option<u32>,
}

impl Default for ReconnectConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            initial_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            max_attempts: None,
        }
    }
}

impl ReconnectConfig {
    /// Calculate delay for a given attempt number.
    #[must_use]
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        if !self.enabled {
            return Duration::ZERO;
        }

        let delay_ms = self.initial_delay.as_millis() as f32
            * self.backoff_multiplier.powi(attempt.min(20) as i32);

        let delay = Duration::from_millis(delay_ms.min(self.max_delay.as_millis() as f32) as u64);
        delay.min(self.max_delay)
    }

    /// Check if we should attempt reconnection.
    #[must_use]
    pub fn should_reconnect(&self, attempt: u32) -> bool {
        if !self.enabled {
            return false;
        }
        match self.max_attempts {
            Some(max) => attempt < max,
            None => true,
        }
    }
}

/// Callback type for data updates.
pub type DataCallback = Box<dyn Fn(&str, &serde_json::Value) + Send + Sync>;

/// Callback type for errors.
pub type ErrorCallback = Box<dyn Fn(&str) + Send + Sync>;

/// Callback type for connection state changes.
pub type StateCallback = Box<dyn Fn(ConnectionState) + Send + Sync>;

/// Data stream manager.
///
/// Manages subscriptions and handles incoming data.
#[derive(Default)]
pub struct DataStream {
    /// Active subscriptions
    subscriptions: Arc<Mutex<HashMap<String, StreamSubscription>>>,
    /// Connection state
    state: Arc<Mutex<ConnectionState>>,
    /// Pending outbound messages
    outbox: Arc<Mutex<Vec<StreamMessage>>>,
    /// Last received data per subscription
    data_cache: Arc<Mutex<HashMap<String, serde_json::Value>>>,
    /// Reconnection attempt count
    reconnect_attempts: Arc<Mutex<u32>>,
    /// Config
    config: StreamConfig,
}

impl DataStream {
    /// Create a new data stream.
    #[must_use]
    pub fn new(config: StreamConfig) -> Self {
        Self {
            subscriptions: Arc::new(Mutex::new(HashMap::new())),
            state: Arc::new(Mutex::new(ConnectionState::Disconnected)),
            outbox: Arc::new(Mutex::new(Vec::new())),
            data_cache: Arc::new(Mutex::new(HashMap::new())),
            reconnect_attempts: Arc::new(Mutex::new(0)),
            config,
        }
    }

    /// Get connection state.
    #[must_use]
    pub fn state(&self) -> ConnectionState {
        *self.state.lock().expect("state mutex not poisoned")
    }

    /// Set connection state.
    pub fn set_state(&self, state: ConnectionState) {
        *self.state.lock().expect("state mutex not poisoned") = state;
    }

    /// Subscribe to a data source.
    pub fn subscribe(&self, subscription: StreamSubscription) -> String {
        let id = subscription.id.clone();
        let msg = subscription.to_message();

        self.subscriptions
            .lock()
            .expect("subscriptions mutex not poisoned")
            .insert(id.clone(), subscription);

        self.outbox
            .lock()
            .expect("outbox mutex not poisoned")
            .push(msg);
        id
    }

    /// Unsubscribe from a data source.
    pub fn unsubscribe(&self, id: &str) {
        self.subscriptions
            .lock()
            .expect("subscriptions mutex not poisoned")
            .remove(id);
        self.data_cache
            .lock()
            .expect("cache mutex not poisoned")
            .remove(id);
        self.outbox
            .lock()
            .expect("outbox mutex not poisoned")
            .push(StreamMessage::unsubscribe(id));
    }

    /// Get subscription by ID.
    #[must_use]
    pub fn get_subscription(&self, id: &str) -> Option<StreamSubscription> {
        self.subscriptions
            .lock()
            .expect("subscriptions mutex not poisoned")
            .get(id)
            .cloned()
    }

    /// Get all active subscriptions.
    #[must_use]
    pub fn subscriptions(&self) -> Vec<StreamSubscription> {
        self.subscriptions
            .lock()
            .expect("subscriptions mutex not poisoned")
            .values()
            .cloned()
            .collect()
    }

    /// Get cached data for a subscription.
    #[must_use]
    pub fn get_data(&self, id: &str) -> Option<serde_json::Value> {
        self.data_cache
            .lock()
            .expect("cache mutex not poisoned")
            .get(id)
            .cloned()
    }

    /// Handle an incoming message.
    pub fn handle_message(&self, msg: StreamMessage) -> Option<StreamMessage> {
        match msg {
            StreamMessage::Data {
                id, payload, seq, ..
            } => {
                // Update subscription state
                if let Some(sub) = self
                    .subscriptions
                    .lock()
                    .expect("subscriptions mutex not poisoned")
                    .get_mut(&id)
                {
                    sub.last_seq = seq;
                    sub.active = true;
                    sub.error_count = 0;
                }
                // Cache data
                self.data_cache
                    .lock()
                    .expect("cache mutex not poisoned")
                    .insert(id, payload);
                None
            }
            StreamMessage::Ack { id, .. } => {
                if let Some(sub) = self
                    .subscriptions
                    .lock()
                    .expect("subscriptions mutex not poisoned")
                    .get_mut(&id)
                {
                    sub.active = true;
                }
                None
            }
            StreamMessage::Error { id, .. } => {
                if let Some(ref id) = id {
                    if let Some(sub) = self
                        .subscriptions
                        .lock()
                        .expect("subscriptions mutex not poisoned")
                        .get_mut(id)
                    {
                        sub.error_count += 1;
                    }
                }
                None
            }
            StreamMessage::Ping { timestamp } => Some(StreamMessage::pong(timestamp)),
            StreamMessage::Pong { .. } => None,
            _ => None,
        }
    }

    /// Take pending outbound messages.
    #[must_use]
    pub fn take_outbox(&self) -> Vec<StreamMessage> {
        std::mem::take(&mut *self.outbox.lock().expect("outbox mutex not poisoned"))
    }

    /// Queue an outbound message.
    pub fn send(&self, msg: StreamMessage) {
        self.outbox
            .lock()
            .expect("outbox mutex not poisoned")
            .push(msg);
    }

    /// Get reconnection delay based on current attempts.
    #[must_use]
    pub fn reconnect_delay(&self) -> Duration {
        let attempts = *self
            .reconnect_attempts
            .lock()
            .expect("reconnect mutex not poisoned");
        self.config.reconnect.delay_for_attempt(attempts)
    }

    /// Increment reconnection attempts.
    pub fn increment_reconnect_attempts(&self) {
        *self
            .reconnect_attempts
            .lock()
            .expect("reconnect mutex not poisoned") += 1;
    }

    /// Reset reconnection attempts.
    pub fn reset_reconnect_attempts(&self) {
        *self
            .reconnect_attempts
            .lock()
            .expect("reconnect mutex not poisoned") = 0;
    }

    /// Check if we should try to reconnect.
    #[must_use]
    pub fn should_reconnect(&self) -> bool {
        let attempts = *self
            .reconnect_attempts
            .lock()
            .expect("reconnect mutex not poisoned");
        self.config.reconnect.should_reconnect(attempts)
    }

    /// Resubscribe all subscriptions (after reconnect).
    pub fn resubscribe_all(&self) {
        let subs = self
            .subscriptions
            .lock()
            .expect("subscriptions mutex not poisoned")
            .clone();
        let mut outbox = self.outbox.lock().expect("outbox mutex not poisoned");
        for sub in subs.values() {
            outbox.push(sub.to_message());
        }
    }

    /// Number of active subscriptions.
    #[must_use]
    pub fn subscription_count(&self) -> usize {
        self.subscriptions
            .lock()
            .expect("subscriptions mutex not poisoned")
            .len()
    }

    /// Clear all subscriptions and cache.
    pub fn clear(&self) {
        self.subscriptions
            .lock()
            .expect("subscriptions mutex not poisoned")
            .clear();
        self.data_cache
            .lock()
            .expect("cache mutex not poisoned")
            .clear();
        self.outbox
            .lock()
            .expect("outbox mutex not poisoned")
            .clear();
    }
}

/// Rate limiter for backpressure handling.
#[derive(Debug)]
pub struct RateLimiter {
    /// Maximum messages per window
    max_messages: usize,
    /// Window duration
    window: Duration,
    /// Message timestamps
    timestamps: Vec<u64>,
}

impl RateLimiter {
    /// Create a new rate limiter.
    #[must_use]
    pub fn new(max_messages: usize, window: Duration) -> Self {
        Self {
            max_messages,
            window,
            timestamps: Vec::with_capacity(max_messages),
        }
    }

    /// Check if a message is allowed and record it.
    pub fn check(&mut self, now: u64) -> bool {
        let window_start = now.saturating_sub(self.window.as_millis() as u64);

        // Remove expired timestamps (>= to keep timestamps at window boundary)
        self.timestamps.retain(|&ts| ts >= window_start);

        if self.timestamps.len() < self.max_messages {
            self.timestamps.push(now);
            true
        } else {
            false
        }
    }

    /// Get the number of messages in the current window.
    #[must_use]
    pub fn current_count(&self) -> usize {
        self.timestamps.len()
    }

    /// Reset the rate limiter.
    pub fn reset(&mut self) {
        self.timestamps.clear();
    }

    /// Check if the limiter is at capacity.
    #[must_use]
    pub fn is_at_capacity(&self) -> bool {
        self.timestamps.len() >= self.max_messages
    }
}

impl Default for RateLimiter {
    fn default() -> Self {
        Self::new(100, Duration::from_secs(1))
    }
}

/// Message buffer for ordering and deduplication.
#[derive(Debug, Default)]
pub struct MessageBuffer {
    /// Buffer of messages by subscription ID
    buffers: HashMap<String, SubscriptionBuffer>,
}

#[derive(Debug, Default)]
struct SubscriptionBuffer {
    /// Last processed sequence number
    last_seq: u64,
    /// Buffered out-of-order messages
    pending: Vec<(u64, serde_json::Value)>,
}

impl MessageBuffer {
    /// Create a new message buffer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Process a message and return it if it's the next in sequence.
    pub fn process(
        &mut self,
        id: &str,
        seq: u64,
        payload: serde_json::Value,
    ) -> Option<serde_json::Value> {
        let buffer = self.buffers.entry(id.to_string()).or_default();

        if seq == buffer.last_seq + 1 {
            // This is the next expected message
            buffer.last_seq = seq;

            // Check for any buffered messages that are now in order
            let mut result = Some(payload);
            while let Some(pos) = buffer
                .pending
                .iter()
                .position(|(s, _)| *s == buffer.last_seq + 1)
            {
                let (next_seq, next_payload) = buffer.pending.remove(pos);
                buffer.last_seq = next_seq;
                // Replace result with latest (or could accumulate)
                result = Some(next_payload);
            }
            result
        } else if seq > buffer.last_seq + 1 {
            // Out of order - buffer it
            buffer.pending.push((seq, payload));
            None
        } else {
            // Duplicate or old message - ignore
            None
        }
    }

    /// Get the last processed sequence for a subscription.
    #[must_use]
    pub fn last_seq(&self, id: &str) -> u64 {
        self.buffers.get(id).map_or(0, |b| b.last_seq)
    }

    /// Get the number of pending messages for a subscription.
    #[must_use]
    pub fn pending_count(&self, id: &str) -> usize {
        self.buffers.get(id).map_or(0, |b| b.pending.len())
    }

    /// Clear buffer for a subscription.
    pub fn clear(&mut self, id: &str) {
        self.buffers.remove(id);
    }

    /// Clear all buffers.
    pub fn clear_all(&mut self) {
        self.buffers.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // ConnectionState Tests
    // =========================================================================

    #[test]
    fn test_connection_state_default() {
        let state = ConnectionState::default();
        assert_eq!(state, ConnectionState::Disconnected);
    }

    #[test]
    fn test_connection_state_is_active() {
        assert!(!ConnectionState::Disconnected.is_active());
        assert!(!ConnectionState::Connecting.is_active());
        assert!(ConnectionState::Connected.is_active());
        assert!(!ConnectionState::Reconnecting.is_active());
        assert!(!ConnectionState::Failed.is_active());
    }

    #[test]
    fn test_connection_state_is_connecting() {
        assert!(!ConnectionState::Disconnected.is_connecting());
        assert!(ConnectionState::Connecting.is_connecting());
        assert!(!ConnectionState::Connected.is_connecting());
        assert!(ConnectionState::Reconnecting.is_connecting());
        assert!(!ConnectionState::Failed.is_connecting());
    }

    // =========================================================================
    // StreamMessage Tests
    // =========================================================================

    #[test]
    fn test_stream_message_subscribe() {
        let msg = StreamMessage::subscribe("sub1", "metrics/cpu");
        if let StreamMessage::Subscribe { id, source, .. } = msg {
            assert_eq!(id, "sub1");
            assert_eq!(source, "metrics/cpu");
        } else {
            panic!("Expected Subscribe message");
        }
    }

    #[test]
    fn test_stream_message_subscribe_with_transform() {
        let msg = StreamMessage::subscribe_with_transform("sub1", "metrics/cpu", "rate()");
        if let StreamMessage::Subscribe { transform, .. } = msg {
            assert_eq!(transform, Some("rate()".to_string()));
        } else {
            panic!("Expected Subscribe message");
        }
    }

    #[test]
    fn test_stream_message_unsubscribe() {
        let msg = StreamMessage::unsubscribe("sub1");
        if let StreamMessage::Unsubscribe { id } = msg {
            assert_eq!(id, "sub1");
        } else {
            panic!("Expected Unsubscribe message");
        }
    }

    #[test]
    fn test_stream_message_data() {
        let msg = StreamMessage::data("sub1", serde_json::json!({"value": 42}), 5);
        if let StreamMessage::Data {
            id, payload, seq, ..
        } = msg
        {
            assert_eq!(id, "sub1");
            assert_eq!(payload, serde_json::json!({"value": 42}));
            assert_eq!(seq, 5);
        } else {
            panic!("Expected Data message");
        }
    }

    #[test]
    fn test_stream_message_error() {
        let msg = StreamMessage::error("connection failed");
        if let StreamMessage::Error { message, id, .. } = msg {
            assert_eq!(message, "connection failed");
            assert!(id.is_none());
        } else {
            panic!("Expected Error message");
        }
    }

    #[test]
    fn test_stream_message_error_for() {
        let msg = StreamMessage::error_for("sub1", "invalid source");
        if let StreamMessage::Error { message, id, .. } = msg {
            assert_eq!(message, "invalid source");
            assert_eq!(id, Some("sub1".to_string()));
        } else {
            panic!("Expected Error message");
        }
    }

    #[test]
    fn test_stream_message_ack() {
        let msg = StreamMessage::ack("sub1");
        if let StreamMessage::Ack { id, .. } = msg {
            assert_eq!(id, "sub1");
        } else {
            panic!("Expected Ack message");
        }
    }

    #[test]
    fn test_stream_message_ping_pong() {
        let ping = StreamMessage::ping(12345);
        let pong = StreamMessage::pong(12345);

        if let StreamMessage::Ping { timestamp } = ping {
            assert_eq!(timestamp, 12345);
        } else {
            panic!("Expected Ping");
        }

        if let StreamMessage::Pong { timestamp } = pong {
            assert_eq!(timestamp, 12345);
        } else {
            panic!("Expected Pong");
        }
    }

    #[test]
    fn test_stream_message_subscription_id() {
        assert_eq!(
            StreamMessage::subscribe("sub1", "x").subscription_id(),
            Some("sub1")
        );
        assert_eq!(
            StreamMessage::unsubscribe("sub2").subscription_id(),
            Some("sub2")
        );
        assert_eq!(
            StreamMessage::data("sub3", serde_json::json!({}), 0).subscription_id(),
            Some("sub3")
        );
        assert_eq!(StreamMessage::error("msg").subscription_id(), None);
        assert_eq!(
            StreamMessage::error_for("sub4", "msg").subscription_id(),
            Some("sub4")
        );
        assert!(StreamMessage::ping(0).subscription_id().is_none());
        assert!(StreamMessage::pong(0).subscription_id().is_none());
    }

    #[test]
    fn test_stream_message_serialize() {
        let msg = StreamMessage::data("sub1", serde_json::json!({"x": 1}), 42);
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"type\":\"data\""));
        assert!(json.contains("\"id\":\"sub1\""));
        assert!(json.contains("\"seq\":42"));
    }

    #[test]
    fn test_stream_message_deserialize() {
        let json = r#"{"type":"subscribe","id":"s1","source":"data/x"}"#;
        let msg: StreamMessage = serde_json::from_str(json).unwrap();
        if let StreamMessage::Subscribe { id, source, .. } = msg {
            assert_eq!(id, "s1");
            assert_eq!(source, "data/x");
        } else {
            panic!("Expected Subscribe");
        }
    }

    // =========================================================================
    // StreamSubscription Tests
    // =========================================================================

    #[test]
    fn test_subscription_new() {
        let sub = StreamSubscription::new("metrics/cpu");
        assert_eq!(sub.source, "metrics/cpu");
        assert!(sub.id.starts_with("sub_"));
        assert!(!sub.active);
    }

    #[test]
    fn test_subscription_with_id() {
        let sub = StreamSubscription::with_id("my-sub", "data/x");
        assert_eq!(sub.id, "my-sub");
        assert_eq!(sub.source, "data/x");
    }

    #[test]
    fn test_subscription_with_interval() {
        let sub = StreamSubscription::new("x").with_interval(1000);
        assert_eq!(sub.interval, Some(Duration::from_millis(1000)));
    }

    #[test]
    fn test_subscription_with_transform() {
        let sub = StreamSubscription::new("x").with_transform("rate() | limit(10)");
        assert_eq!(sub.transform, Some("rate() | limit(10)".to_string()));
    }

    #[test]
    fn test_subscription_to_message() {
        let sub = StreamSubscription::with_id("sub1", "metrics")
            .with_interval(5000)
            .with_transform("mean()");

        let msg = sub.to_message();
        if let StreamMessage::Subscribe {
            id,
            source,
            transform,
            interval_ms,
        } = msg
        {
            assert_eq!(id, "sub1");
            assert_eq!(source, "metrics");
            assert_eq!(transform, Some("mean()".to_string()));
            assert_eq!(interval_ms, Some(5000));
        } else {
            panic!("Expected Subscribe");
        }
    }

    // =========================================================================
    // StreamConfig Tests
    // =========================================================================

    #[test]
    fn test_stream_config_default() {
        let config = StreamConfig::default();
        assert!(config.url.is_empty());
        assert!(config.reconnect.enabled);
        assert_eq!(config.heartbeat_interval, Duration::from_secs(30));
    }

    #[test]
    fn test_stream_config_new() {
        let config = StreamConfig::new("ws://localhost:8080");
        assert_eq!(config.url, "ws://localhost:8080");
    }

    #[test]
    fn test_stream_config_builder() {
        let config = StreamConfig::new("ws://x")
            .with_heartbeat(Duration::from_secs(10))
            .with_buffer_size(2048);

        assert_eq!(config.heartbeat_interval, Duration::from_secs(10));
        assert_eq!(config.buffer_size, 2048);
    }

    // =========================================================================
    // ReconnectConfig Tests
    // =========================================================================

    #[test]
    fn test_reconnect_config_default() {
        let config = ReconnectConfig::default();
        assert!(config.enabled);
        assert_eq!(config.initial_delay, Duration::from_millis(500));
        assert_eq!(config.max_delay, Duration::from_secs(30));
        assert!(config.max_attempts.is_none());
    }

    #[test]
    fn test_reconnect_delay_for_attempt() {
        let config = ReconnectConfig {
            enabled: true,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            backoff_multiplier: 2.0,
            max_attempts: None,
        };

        assert_eq!(config.delay_for_attempt(0), Duration::from_millis(100));
        assert_eq!(config.delay_for_attempt(1), Duration::from_millis(200));
        assert_eq!(config.delay_for_attempt(2), Duration::from_millis(400));
        assert_eq!(config.delay_for_attempt(3), Duration::from_millis(800));
    }

    #[test]
    fn test_reconnect_delay_capped() {
        let config = ReconnectConfig {
            enabled: true,
            initial_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(5),
            backoff_multiplier: 10.0,
            max_attempts: None,
        };

        // After just a couple attempts, should be capped at max
        assert_eq!(config.delay_for_attempt(5), Duration::from_secs(5));
    }

    #[test]
    fn test_reconnect_disabled() {
        let config = ReconnectConfig {
            enabled: false,
            ..Default::default()
        };

        assert_eq!(config.delay_for_attempt(0), Duration::ZERO);
        assert!(!config.should_reconnect(0));
    }

    #[test]
    fn test_reconnect_max_attempts() {
        let config = ReconnectConfig {
            enabled: true,
            max_attempts: Some(3),
            ..Default::default()
        };

        assert!(config.should_reconnect(0));
        assert!(config.should_reconnect(1));
        assert!(config.should_reconnect(2));
        assert!(!config.should_reconnect(3));
        assert!(!config.should_reconnect(4));
    }

    // =========================================================================
    // DataStream Tests
    // =========================================================================

    #[test]
    fn test_data_stream_new() {
        let stream = DataStream::new(StreamConfig::new("ws://x"));
        assert_eq!(stream.state(), ConnectionState::Disconnected);
        assert_eq!(stream.subscription_count(), 0);
    }

    #[test]
    fn test_data_stream_subscribe() {
        let stream = DataStream::new(StreamConfig::default());
        let sub = StreamSubscription::with_id("sub1", "metrics");

        let id = stream.subscribe(sub);
        assert_eq!(id, "sub1");
        assert_eq!(stream.subscription_count(), 1);

        let outbox = stream.take_outbox();
        assert_eq!(outbox.len(), 1);
        assert!(matches!(outbox[0], StreamMessage::Subscribe { .. }));
    }

    #[test]
    fn test_data_stream_unsubscribe() {
        let stream = DataStream::new(StreamConfig::default());
        stream.subscribe(StreamSubscription::with_id("sub1", "x"));
        let _ = stream.take_outbox(); // Clear

        stream.unsubscribe("sub1");
        assert_eq!(stream.subscription_count(), 0);

        let outbox = stream.take_outbox();
        assert_eq!(outbox.len(), 1);
        assert!(matches!(outbox[0], StreamMessage::Unsubscribe { .. }));
    }

    #[test]
    fn test_data_stream_handle_data() {
        let stream = DataStream::new(StreamConfig::default());
        stream.subscribe(StreamSubscription::with_id("sub1", "x"));

        let msg = StreamMessage::data("sub1", serde_json::json!({"val": 42}), 1);
        stream.handle_message(msg);

        let data = stream.get_data("sub1");
        assert_eq!(data, Some(serde_json::json!({"val": 42})));

        let sub = stream.get_subscription("sub1").unwrap();
        assert!(sub.active);
        assert_eq!(sub.last_seq, 1);
    }

    #[test]
    fn test_data_stream_handle_ack() {
        let stream = DataStream::new(StreamConfig::default());
        stream.subscribe(StreamSubscription::with_id("sub1", "x"));

        stream.handle_message(StreamMessage::ack("sub1"));

        let sub = stream.get_subscription("sub1").unwrap();
        assert!(sub.active);
    }

    #[test]
    fn test_data_stream_handle_error() {
        let stream = DataStream::new(StreamConfig::default());
        stream.subscribe(StreamSubscription::with_id("sub1", "x"));

        stream.handle_message(StreamMessage::error_for("sub1", "fail"));

        let sub = stream.get_subscription("sub1").unwrap();
        assert_eq!(sub.error_count, 1);
    }

    #[test]
    fn test_data_stream_handle_ping() {
        let stream = DataStream::new(StreamConfig::default());
        let response = stream.handle_message(StreamMessage::ping(12345));

        assert!(matches!(
            response,
            Some(StreamMessage::Pong { timestamp: 12345 })
        ));
    }

    #[test]
    fn test_data_stream_reconnect_logic() {
        let stream = DataStream::new(StreamConfig::default());

        assert!(stream.should_reconnect());
        assert_eq!(stream.reconnect_delay(), Duration::from_millis(500));

        stream.increment_reconnect_attempts();
        assert!(stream.should_reconnect());
        assert_eq!(stream.reconnect_delay(), Duration::from_millis(1000));

        stream.reset_reconnect_attempts();
        assert_eq!(stream.reconnect_delay(), Duration::from_millis(500));
    }

    #[test]
    fn test_data_stream_resubscribe_all() {
        let stream = DataStream::new(StreamConfig::default());
        stream.subscribe(StreamSubscription::with_id("sub1", "a"));
        stream.subscribe(StreamSubscription::with_id("sub2", "b"));
        let _ = stream.take_outbox(); // Clear

        stream.resubscribe_all();

        let outbox = stream.take_outbox();
        assert_eq!(outbox.len(), 2);
    }

    #[test]
    fn test_data_stream_clear() {
        let stream = DataStream::new(StreamConfig::default());
        stream.subscribe(StreamSubscription::with_id("sub1", "a"));
        stream.handle_message(StreamMessage::data("sub1", serde_json::json!(1), 1));

        stream.clear();

        assert_eq!(stream.subscription_count(), 0);
        assert!(stream.get_data("sub1").is_none());
    }

    // =========================================================================
    // RateLimiter Tests
    // =========================================================================

    #[test]
    fn test_rate_limiter_allows_under_limit() {
        let mut limiter = RateLimiter::new(5, Duration::from_secs(1));

        for i in 0..5 {
            assert!(limiter.check(i * 100), "message {i} should be allowed");
        }
    }

    #[test]
    fn test_rate_limiter_blocks_over_limit() {
        let mut limiter = RateLimiter::new(3, Duration::from_secs(1));

        assert!(limiter.check(0));
        assert!(limiter.check(100));
        assert!(limiter.check(200));
        assert!(!limiter.check(300)); // Over limit
    }

    #[test]
    fn test_rate_limiter_window_expiry() {
        let mut limiter = RateLimiter::new(2, Duration::from_millis(100));

        assert!(limiter.check(0));
        assert!(limiter.check(50));
        assert!(!limiter.check(60)); // Over limit

        // After window expires
        assert!(limiter.check(200)); // Old messages expired
    }

    #[test]
    fn test_rate_limiter_current_count() {
        let mut limiter = RateLimiter::new(10, Duration::from_secs(1));

        assert_eq!(limiter.current_count(), 0);
        limiter.check(0);
        assert_eq!(limiter.current_count(), 1);
        limiter.check(100);
        assert_eq!(limiter.current_count(), 2);
    }

    #[test]
    fn test_rate_limiter_reset() {
        let mut limiter = RateLimiter::new(2, Duration::from_secs(1));

        limiter.check(0);
        limiter.check(100);
        assert!(limiter.is_at_capacity());

        limiter.reset();
        assert_eq!(limiter.current_count(), 0);
        assert!(!limiter.is_at_capacity());
    }

    #[test]
    fn test_rate_limiter_default() {
        let limiter = RateLimiter::default();
        assert_eq!(limiter.max_messages, 100);
    }

    // =========================================================================
    // MessageBuffer Tests
    // =========================================================================

    #[test]
    fn test_message_buffer_in_order() {
        let mut buffer = MessageBuffer::new();

        let r1 = buffer.process("sub1", 1, serde_json::json!(1));
        let r2 = buffer.process("sub1", 2, serde_json::json!(2));
        let r3 = buffer.process("sub1", 3, serde_json::json!(3));

        assert_eq!(r1, Some(serde_json::json!(1)));
        assert_eq!(r2, Some(serde_json::json!(2)));
        assert_eq!(r3, Some(serde_json::json!(3)));
    }

    #[test]
    fn test_message_buffer_out_of_order() {
        let mut buffer = MessageBuffer::new();

        // Receive seq 1, then 3, then 2
        let r1 = buffer.process("sub1", 1, serde_json::json!(1));
        let r3 = buffer.process("sub1", 3, serde_json::json!(3)); // Buffered
        let r2 = buffer.process("sub1", 2, serde_json::json!(2)); // Triggers flush

        assert_eq!(r1, Some(serde_json::json!(1)));
        assert!(r3.is_none()); // Buffered
        assert_eq!(r2, Some(serde_json::json!(3))); // Returns latest after reorder
    }

    #[test]
    fn test_message_buffer_duplicate() {
        let mut buffer = MessageBuffer::new();

        buffer.process("sub1", 1, serde_json::json!(1));
        let dup = buffer.process("sub1", 1, serde_json::json!("dup"));

        assert!(dup.is_none()); // Duplicate ignored
    }

    #[test]
    fn test_message_buffer_last_seq() {
        let mut buffer = MessageBuffer::new();

        assert_eq!(buffer.last_seq("sub1"), 0);
        buffer.process("sub1", 1, serde_json::json!(1));
        assert_eq!(buffer.last_seq("sub1"), 1);
        buffer.process("sub1", 2, serde_json::json!(2));
        assert_eq!(buffer.last_seq("sub1"), 2);
    }

    #[test]
    fn test_message_buffer_pending_count() {
        let mut buffer = MessageBuffer::new();

        buffer.process("sub1", 1, serde_json::json!(1));
        buffer.process("sub1", 3, serde_json::json!(3)); // Skip 2
        buffer.process("sub1", 4, serde_json::json!(4)); // Skip 2

        assert_eq!(buffer.pending_count("sub1"), 2);
    }

    #[test]
    fn test_message_buffer_clear() {
        let mut buffer = MessageBuffer::new();

        buffer.process("sub1", 1, serde_json::json!(1));
        buffer.process("sub2", 1, serde_json::json!(2));

        buffer.clear("sub1");
        assert_eq!(buffer.last_seq("sub1"), 0);
        assert_eq!(buffer.last_seq("sub2"), 1);
    }

    #[test]
    fn test_message_buffer_clear_all() {
        let mut buffer = MessageBuffer::new();

        buffer.process("sub1", 1, serde_json::json!(1));
        buffer.process("sub2", 1, serde_json::json!(2));

        buffer.clear_all();
        assert_eq!(buffer.last_seq("sub1"), 0);
        assert_eq!(buffer.last_seq("sub2"), 0);
    }

    #[test]
    fn test_message_buffer_multiple_subscriptions() {
        let mut buffer = MessageBuffer::new();

        buffer.process("sub1", 1, serde_json::json!("a"));
        buffer.process("sub2", 1, serde_json::json!("b"));
        buffer.process("sub1", 2, serde_json::json!("c"));

        assert_eq!(buffer.last_seq("sub1"), 2);
        assert_eq!(buffer.last_seq("sub2"), 1);
    }

    // =========================================================================
    // Additional Edge Case Tests
    // =========================================================================

    #[test]
    fn test_connection_state_debug() {
        assert_eq!(format!("{:?}", ConnectionState::Connected), "Connected");
        assert_eq!(format!("{:?}", ConnectionState::Failed), "Failed");
    }

    #[test]
    fn test_connection_state_clone() {
        let state = ConnectionState::Reconnecting;
        let cloned = state;
        assert_eq!(state, cloned);
    }

    #[test]
    fn test_stream_message_debug() {
        let msg = StreamMessage::ping(12345);
        let debug = format!("{msg:?}");
        assert!(debug.contains("Ping"));
    }

    #[test]
    fn test_stream_message_clone() {
        let msg = StreamMessage::data("sub1", serde_json::json!({"x": 1}), 5);
        let cloned = msg.clone();
        assert_eq!(msg, cloned);
    }

    #[test]
    fn test_stream_subscription_clone() {
        let sub = StreamSubscription::with_id("sub1", "metrics")
            .with_interval(1000)
            .with_transform("rate()");
        let cloned = sub;
        assert_eq!(cloned.id, "sub1");
        assert_eq!(cloned.source, "metrics");
        assert_eq!(cloned.transform, Some("rate()".to_string()));
    }

    #[test]
    fn test_stream_subscription_debug() {
        let sub = StreamSubscription::new("test");
        let debug = format!("{sub:?}");
        assert!(debug.contains("StreamSubscription"));
    }

    #[test]
    fn test_stream_subscription_hash_consistency() {
        // Same source should produce same hash
        let sub1 = StreamSubscription::new("metrics/cpu");
        let sub2 = StreamSubscription::new("metrics/cpu");
        assert_eq!(sub1.id, sub2.id);
    }

    #[test]
    fn test_stream_subscription_hash_different() {
        let sub1 = StreamSubscription::new("metrics/cpu");
        let sub2 = StreamSubscription::new("metrics/memory");
        assert_ne!(sub1.id, sub2.id);
    }

    #[test]
    fn test_stream_config_debug() {
        let config = StreamConfig::default();
        let debug = format!("{config:?}");
        assert!(debug.contains("StreamConfig"));
    }

    #[test]
    fn test_stream_config_clone() {
        let config = StreamConfig::new("ws://test")
            .with_buffer_size(2048)
            .with_heartbeat(Duration::from_secs(60));
        let cloned = config;
        assert_eq!(cloned.url, "ws://test");
        assert_eq!(cloned.buffer_size, 2048);
    }

    #[test]
    fn test_stream_config_with_reconnect() {
        let reconnect = ReconnectConfig {
            enabled: false,
            max_attempts: Some(5),
            ..Default::default()
        };
        let config = StreamConfig::new("ws://x").with_reconnect(reconnect);
        assert!(!config.reconnect.enabled);
        assert_eq!(config.reconnect.max_attempts, Some(5));
    }

    #[test]
    fn test_reconnect_config_debug() {
        let config = ReconnectConfig::default();
        let debug = format!("{config:?}");
        assert!(debug.contains("ReconnectConfig"));
    }

    #[test]
    fn test_reconnect_config_clone() {
        let config = ReconnectConfig {
            max_attempts: Some(10),
            ..Default::default()
        };
        let cloned = config;
        assert_eq!(cloned.max_attempts, Some(10));
    }

    #[test]
    fn test_reconnect_delay_large_attempt() {
        let config = ReconnectConfig::default();
        // Large attempt number should be capped by max(20)
        let delay = config.delay_for_attempt(100);
        assert!(delay <= config.max_delay);
    }

    #[test]
    fn test_data_stream_default() {
        let stream = DataStream::default();
        assert_eq!(stream.state(), ConnectionState::Disconnected);
        assert_eq!(stream.subscription_count(), 0);
    }

    #[test]
    fn test_data_stream_set_state() {
        let stream = DataStream::default();
        stream.set_state(ConnectionState::Connected);
        assert_eq!(stream.state(), ConnectionState::Connected);
        stream.set_state(ConnectionState::Failed);
        assert_eq!(stream.state(), ConnectionState::Failed);
    }

    #[test]
    fn test_data_stream_send() {
        let stream = DataStream::default();
        stream.send(StreamMessage::ping(100));
        stream.send(StreamMessage::ping(200));

        let outbox = stream.take_outbox();
        assert_eq!(outbox.len(), 2);
    }

    #[test]
    fn test_data_stream_get_nonexistent_subscription() {
        let stream = DataStream::default();
        assert!(stream.get_subscription("nonexistent").is_none());
    }

    #[test]
    fn test_data_stream_get_nonexistent_data() {
        let stream = DataStream::default();
        assert!(stream.get_data("nonexistent").is_none());
    }

    #[test]
    fn test_data_stream_subscriptions_list() {
        let stream = DataStream::default();
        stream.subscribe(StreamSubscription::with_id("sub1", "a"));
        stream.subscribe(StreamSubscription::with_id("sub2", "b"));

        let subs = stream.subscriptions();
        assert_eq!(subs.len(), 2);
    }

    #[test]
    fn test_data_stream_handle_pong() {
        let stream = DataStream::default();
        let response = stream.handle_message(StreamMessage::pong(12345));
        assert!(response.is_none());
    }

    #[test]
    fn test_data_stream_handle_subscribe() {
        let stream = DataStream::default();
        // Subscribe messages from server side are ignored
        let response = stream.handle_message(StreamMessage::subscribe("sub1", "metrics"));
        assert!(response.is_none());
    }

    #[test]
    fn test_data_stream_handle_error_no_id() {
        let stream = DataStream::default();
        // Error without ID doesn't affect any subscription
        let response = stream.handle_message(StreamMessage::error("general error"));
        assert!(response.is_none());
    }

    #[test]
    fn test_data_stream_handle_error_unknown_id() {
        let stream = DataStream::default();
        // Error for unknown subscription
        let response = stream.handle_message(StreamMessage::error_for("unknown", "error"));
        assert!(response.is_none());
    }

    #[test]
    fn test_data_stream_handle_data_unknown_subscription() {
        let stream = DataStream::default();
        // Data for unknown subscription still gets cached
        stream.handle_message(StreamMessage::data("unknown", serde_json::json!(42), 1));
        assert_eq!(stream.get_data("unknown"), Some(serde_json::json!(42)));
    }

    #[test]
    fn test_rate_limiter_debug() {
        let limiter = RateLimiter::new(10, Duration::from_secs(1));
        let debug = format!("{limiter:?}");
        assert!(debug.contains("RateLimiter"));
    }

    #[test]
    fn test_rate_limiter_at_boundary() {
        let mut limiter = RateLimiter::new(3, Duration::from_millis(100));

        // All at time 0
        assert!(limiter.check(0));
        assert!(limiter.check(0));
        assert!(limiter.check(0));
        assert!(!limiter.check(0)); // Over limit at same time

        // Exactly at window boundary - should keep messages
        assert!(!limiter.check(100)); // At boundary, old ones still valid

        // Past window boundary
        assert!(limiter.check(101)); // Window expired
    }

    #[test]
    fn test_message_buffer_debug() {
        let buffer = MessageBuffer::new();
        let debug = format!("{buffer:?}");
        assert!(debug.contains("MessageBuffer"));
    }

    #[test]
    fn test_message_buffer_old_message() {
        let mut buffer = MessageBuffer::new();

        // Process messages 1, 2, 3 in order
        buffer.process("sub1", 1, serde_json::json!(1));
        buffer.process("sub1", 2, serde_json::json!(2));
        buffer.process("sub1", 3, serde_json::json!(3));

        // Old message (seq 1 when we're at 3) should be ignored
        let old = buffer.process("sub1", 1, serde_json::json!("old"));
        assert!(old.is_none());
        assert_eq!(buffer.last_seq("sub1"), 3);
    }

    #[test]
    fn test_message_buffer_large_gap() {
        let mut buffer = MessageBuffer::new();

        buffer.process("sub1", 1, serde_json::json!(1));
        // Skip many sequence numbers
        buffer.process("sub1", 100, serde_json::json!(100)); // Buffered

        assert_eq!(buffer.last_seq("sub1"), 1);
        assert_eq!(buffer.pending_count("sub1"), 1);
    }

    #[test]
    fn test_message_buffer_nonexistent_subscription() {
        let buffer = MessageBuffer::new();
        assert_eq!(buffer.last_seq("nonexistent"), 0);
        assert_eq!(buffer.pending_count("nonexistent"), 0);
    }

    #[test]
    fn test_stream_message_serialize_all_variants() {
        let messages = vec![
            StreamMessage::subscribe("s1", "source"),
            StreamMessage::subscribe_with_transform("s2", "source", "rate()"),
            StreamMessage::unsubscribe("s1"),
            StreamMessage::data("s1", serde_json::json!({"x": 1}), 5),
            StreamMessage::error("msg"),
            StreamMessage::error_for("s1", "msg"),
            StreamMessage::ack("s1"),
            StreamMessage::ping(1000),
            StreamMessage::pong(1000),
        ];

        for msg in messages {
            let json = serde_json::to_string(&msg).unwrap();
            let parsed: StreamMessage = serde_json::from_str(&json).unwrap();
            assert_eq!(msg, parsed);
        }
    }

    #[test]
    fn test_stream_subscription_empty_source() {
        let sub = StreamSubscription::new("");
        assert!(sub.id.starts_with("sub_"));
        assert_eq!(sub.source, "");
    }

    #[test]
    fn test_stream_subscription_unicode_source() {
        let sub = StreamSubscription::new("数据/指标");
        assert!(sub.id.starts_with("sub_"));
        assert_eq!(sub.source, "数据/指标");
    }

    #[test]
    fn test_data_stream_multiple_data_updates() {
        let stream = DataStream::default();
        stream.subscribe(StreamSubscription::with_id("sub1", "x"));

        // Multiple updates should update cache
        stream.handle_message(StreamMessage::data("sub1", serde_json::json!(1), 1));
        assert_eq!(stream.get_data("sub1"), Some(serde_json::json!(1)));

        stream.handle_message(StreamMessage::data("sub1", serde_json::json!(2), 2));
        assert_eq!(stream.get_data("sub1"), Some(serde_json::json!(2)));

        let sub = stream.get_subscription("sub1").unwrap();
        assert_eq!(sub.last_seq, 2);
        assert_eq!(sub.error_count, 0);
    }

    #[test]
    fn test_reconnect_infinite_attempts() {
        let config = ReconnectConfig {
            enabled: true,
            max_attempts: None,
            ..Default::default()
        };

        // Should always reconnect with infinite attempts
        assert!(config.should_reconnect(0));
        assert!(config.should_reconnect(100));
        assert!(config.should_reconnect(1000));
        assert!(config.should_reconnect(u32::MAX - 1));
    }
}
