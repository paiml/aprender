//! Advanced messaging patterns for distributed coordination.
//!
//! Provides publish-subscribe (PUB/SUB) and push-pull (PUSH/PULL) messaging
//! patterns for coordinating distributed tasks.
//!
//! # PUB/SUB Pattern
//!
//! One publisher broadcasts messages to multiple subscribers:
//!
//! ```no_run
//! use repartir::messaging::{PubSubChannel, Message};
//!
//! #[tokio::main]
//! async fn main() -> repartir::error::Result<()> {
//!     let channel = PubSubChannel::new();
//!
//!     // Subscribe to a topic
//!     let mut subscriber = channel.subscribe("events").await;
//!
//!     // Publish a message
//!     channel.publish("events", Message::text("Task completed!")).await?;
//!
//!     // Receive message
//!     if let Ok(msg) = subscriber.recv().await {
//!         println!("Received: {}", msg.as_text()?);
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! # PUSH/PULL Pattern
//!
//! Multiple producers push work, multiple consumers pull (load balancing):
//!
//! ```no_run
//! use repartir::messaging::{PushPullChannel, Message};
//!
//! #[tokio::main]
//! async fn main() -> repartir::error::Result<()> {
//!     let channel = PushPullChannel::new(100);
//!
//!     // Producer pushes work
//!     channel.push(Message::bytes(b"work item".to_vec())).await?;
//!
//!     // Consumer pulls work
//!     if let Some(msg) = channel.pull().await {
//!         println!("Processing work");
//!     }
//!
//!     Ok(())
//! }
//! ```

use crate::error::{RepartirError, Result};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{broadcast, mpsc, RwLock};
use tracing::{debug, info};

/// A message that can be sent through messaging channels.
#[derive(Clone, Debug)]
pub struct Message {
    /// Message content.
    data: MessageData,
    /// Optional metadata.
    metadata: HashMap<String, String>,
}

/// Message content types.
#[derive(Clone, Debug)]
enum MessageData {
    /// Text message (UTF-8 string).
    Text(String),
    /// Binary message (raw bytes).
    Bytes(Vec<u8>),
}

impl Message {
    /// Creates a new text message.
    #[must_use]
    pub fn text<S: Into<String>>(content: S) -> Self {
        Self { data: MessageData::Text(content.into()), metadata: HashMap::new() }
    }

    /// Creates a new binary message.
    #[must_use]
    pub fn bytes(content: Vec<u8>) -> Self {
        Self { data: MessageData::Bytes(content), metadata: HashMap::new() }
    }

    /// Adds metadata to the message.
    #[must_use]
    pub fn with_metadata<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Returns the message as text.
    ///
    /// # Errors
    ///
    /// Returns an error if the message is not a text message.
    pub fn as_text(&self) -> Result<&str> {
        match &self.data {
            MessageData::Text(s) => Ok(s),
            MessageData::Bytes(_) => {
                Err(RepartirError::InvalidTask { reason: "Message is not text".to_string() })
            }
        }
    }

    /// Returns the message as bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        match &self.data {
            MessageData::Text(s) => s.as_bytes(),
            MessageData::Bytes(b) => b,
        }
    }

    /// Returns metadata value for the given key.
    #[must_use]
    pub fn get_metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(String::as_str)
    }

    /// Returns all metadata.
    #[must_use]
    pub const fn metadata(&self) -> &HashMap<String, String> {
        &self.metadata
    }
}

/// Publish-Subscribe channel.
///
/// Supports multiple publishers broadcasting to multiple subscribers.
/// Each subscriber receives a copy of all published messages on their topics.
///
/// # Example
///
/// ```no_run
/// use repartir::messaging::{PubSubChannel, Message};
///
/// #[tokio::main]
/// async fn main() -> repartir::error::Result<()> {
///     let channel = PubSubChannel::new();
///
///     // Multiple subscribers
///     let mut sub1 = channel.subscribe("events").await;
///     let mut sub2 = channel.subscribe("events").await;
///
///     // Publish to all subscribers
///     channel.publish("events", Message::text("Broadcast!")).await?;
///
///     // Both receive the message
///     assert!(sub1.recv().await.is_ok());
///     assert!(sub2.recv().await.is_ok());
///
///     Ok(())
/// }
/// ```
pub struct PubSubChannel {
    /// Topic-specific broadcast channels.
    topics: Arc<RwLock<HashMap<String, broadcast::Sender<Message>>>>,
    /// Broadcast channel capacity.
    capacity: usize,
}

impl PubSubChannel {
    /// Creates a new PUB/SUB channel with default capacity (1000 messages).
    #[must_use]
    pub fn new() -> Self {
        Self::with_capacity(1000)
    }

    /// Creates a new PUB/SUB channel with the specified capacity per topic.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        info!("Creating PUB/SUB channel with capacity {capacity}");
        Self { topics: Arc::new(RwLock::new(HashMap::new())), capacity }
    }

    /// Subscribes to a topic.
    ///
    /// Returns a receiver that will receive all messages published to this topic.
    pub async fn subscribe(&self, topic: &str) -> broadcast::Receiver<Message> {
        let sender = {
            let mut topics = self.topics.write().await;
            topics
                .entry(topic.to_string())
                .or_insert_with(|| {
                    debug!("Creating new topic: {topic}");
                    broadcast::channel(self.capacity).0
                })
                .clone()
        };

        sender.subscribe()
    }

    /// Publishes a message to a topic.
    ///
    /// All subscribers to this topic will receive the message.
    ///
    /// # Errors
    ///
    /// Returns an error if no subscribers exist for the topic.
    pub async fn publish(&self, topic: &str, message: Message) -> Result<usize> {
        let topics = self.topics.read().await;

        if let Some(sender) = topics.get(topic) {
            let count = sender.send(message).map_err(|_| RepartirError::InvalidTask {
                reason: format!("No active subscribers for topic: {topic}"),
            })?;
            debug!("Published message to {count} subscribers on topic '{topic}'");
            Ok(count)
        } else {
            Err(RepartirError::InvalidTask { reason: format!("Topic does not exist: {topic}") })
        }
    }

    /// Returns the number of active topics.
    pub async fn topic_count(&self) -> usize {
        self.topics.read().await.len()
    }

    /// Returns the number of subscribers for a specific topic.
    pub async fn subscriber_count(&self, topic: &str) -> usize {
        let topics = self.topics.read().await;
        topics.get(topic).map_or(0, broadcast::Sender::receiver_count)
    }
}

impl Default for PubSubChannel {
    fn default() -> Self {
        Self::new()
    }
}

/// Push-Pull channel for work distribution.
///
/// Multiple producers can push work items, and multiple consumers can pull them.
/// Each message is delivered to exactly one consumer (load balancing).
///
/// # Example
///
/// ```no_run
/// use repartir::messaging::{PushPullChannel, Message};
///
/// #[tokio::main]
/// async fn main() -> repartir::error::Result<()> {
///     let channel = PushPullChannel::new(100);
///
///     // Producer
///     channel.push(Message::text("Work item 1")).await?;
///     channel.push(Message::text("Work item 2")).await?;
///
///     // Consumer
///     let work1 = channel.pull().await;
///     let work2 = channel.pull().await;
///
///     assert!(work1.is_some());
///     assert!(work2.is_some());
///
///     Ok(())
/// }
/// ```
pub struct PushPullChannel {
    /// Unbounded MPSC channel for work distribution.
    sender: mpsc::Sender<Message>,
    /// Receiver for pulling work.
    receiver: Arc<RwLock<mpsc::Receiver<Message>>>,
}

impl PushPullChannel {
    /// Creates a new PUSH/PULL channel with the specified capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        info!("Creating PUSH/PULL channel with capacity {capacity}");
        let (sender, receiver) = mpsc::channel(capacity);
        Self { sender, receiver: Arc::new(RwLock::new(receiver)) }
    }

    /// Pushes a message to the channel.
    ///
    /// # Errors
    ///
    /// Returns an error if the channel is full or closed.
    pub async fn push(&self, message: Message) -> Result<()> {
        self.sender.send(message).await.map_err(|_| RepartirError::InvalidTask {
            reason: "Channel closed or full".to_string(),
        })?;
        debug!("Message pushed to PUSH/PULL channel");
        Ok(())
    }

    /// Pulls a message from the channel.
    ///
    /// Blocks until a message is available or the channel is closed.
    pub async fn pull(&self) -> Option<Message> {
        let mut receiver = self.receiver.write().await;
        receiver.recv().await
    }

    /// Returns a sender that can be cloned and shared across threads.
    #[must_use]
    pub fn sender(&self) -> mpsc::Sender<Message> {
        self.sender.clone()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::disallowed_methods)]
mod tests {
    use super::*;

    #[test]
    fn test_message_text() {
        let msg = Message::text("Hello");
        assert_eq!(msg.as_text().unwrap(), "Hello");
        assert_eq!(msg.as_bytes(), b"Hello");
    }

    #[test]
    fn test_message_bytes() {
        let msg = Message::bytes(vec![1, 2, 3]);
        assert_eq!(msg.as_bytes(), &[1, 2, 3]);
        assert!(msg.as_text().is_err());
    }

    #[test]
    fn test_message_metadata() {
        let msg =
            Message::text("test").with_metadata("key1", "value1").with_metadata("key2", "value2");

        assert_eq!(msg.get_metadata("key1"), Some("value1"));
        assert_eq!(msg.get_metadata("key2"), Some("value2"));
        assert_eq!(msg.get_metadata("key3"), None);
        assert_eq!(msg.metadata().len(), 2);
    }

    #[tokio::test]
    async fn test_pubsub_basic() {
        let channel = PubSubChannel::new();
        let mut sub = channel.subscribe("test").await;

        channel.publish("test", Message::text("Hello")).await.unwrap();

        let msg = sub.recv().await.unwrap();
        assert_eq!(msg.as_text().unwrap(), "Hello");
    }

    #[tokio::test]
    async fn test_pubsub_multiple_subscribers() {
        let channel = PubSubChannel::new();
        let mut sub1 = channel.subscribe("events").await;
        let mut sub2 = channel.subscribe("events").await;

        assert_eq!(channel.subscriber_count("events").await, 2);

        channel.publish("events", Message::text("Broadcast")).await.unwrap();

        let msg1 = sub1.recv().await.unwrap();
        let msg2 = sub2.recv().await.unwrap();

        assert_eq!(msg1.as_text().unwrap(), "Broadcast");
        assert_eq!(msg2.as_text().unwrap(), "Broadcast");
    }

    #[tokio::test]
    async fn test_pubsub_topic_isolation() {
        let channel = PubSubChannel::new();
        let mut sub1 = channel.subscribe("topic1").await;
        let mut sub2 = channel.subscribe("topic2").await;

        channel.publish("topic1", Message::text("Message 1")).await.unwrap();

        // sub1 receives, sub2 doesn't
        assert!(sub1.try_recv().is_ok());
        assert!(sub2.try_recv().is_err()); // No message
    }

    #[tokio::test]
    async fn test_pushpull_basic() {
        let channel = PushPullChannel::new(10);

        channel.push(Message::text("Work 1")).await.unwrap();
        channel.push(Message::text("Work 2")).await.unwrap();

        let msg1 = channel.pull().await.unwrap();
        let msg2 = channel.pull().await.unwrap();

        assert_eq!(msg1.as_text().unwrap(), "Work 1");
        assert_eq!(msg2.as_text().unwrap(), "Work 2");
    }

    #[tokio::test]
    async fn test_pushpull_single_consumer() {
        let channel = PushPullChannel::new(10);

        // Multiple producers
        channel.push(Message::text("Item 1")).await.unwrap();
        channel.push(Message::text("Item 2")).await.unwrap();
        channel.push(Message::text("Item 3")).await.unwrap();

        // Single consumer gets all items
        assert!(channel.pull().await.is_some());
        assert!(channel.pull().await.is_some());
        assert!(channel.pull().await.is_some());
    }

    #[tokio::test]
    async fn test_pushpull_sender_clone() {
        let channel = PushPullChannel::new(10);
        let sender = channel.sender();

        // Use cloned sender
        sender.send(Message::text("Test")).await.unwrap();

        let msg = channel.pull().await.unwrap();
        assert_eq!(msg.as_text().unwrap(), "Test");
    }

    #[tokio::test]
    async fn test_pubsub_default() {
        let channel = PubSubChannel::default();
        let mut sub = channel.subscribe("test").await;

        channel.publish("test", Message::text("Default channel")).await.unwrap();

        let msg = sub.recv().await.unwrap();
        assert_eq!(msg.as_text().unwrap(), "Default channel");
    }

    #[tokio::test]
    async fn test_pubsub_no_subscribers() {
        let channel = PubSubChannel::new();

        // Publishing to a non-existent topic should error
        let result = channel.publish("empty_topic", Message::text("Nobody listening")).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_pubsub_subscriber_count() {
        let channel = PubSubChannel::new();

        assert_eq!(channel.subscriber_count("nonexistent").await, 0);

        let _sub1 = channel.subscribe("topic1").await;
        assert_eq!(channel.subscriber_count("topic1").await, 1);

        let _sub2 = channel.subscribe("topic1").await;
        assert_eq!(channel.subscriber_count("topic1").await, 2);
    }

    #[tokio::test]
    async fn test_pubsub_topic_count() {
        let channel = PubSubChannel::new();

        assert_eq!(channel.topic_count().await, 0);

        let _sub1 = channel.subscribe("topic1").await;
        assert_eq!(channel.topic_count().await, 1);

        let _sub2 = channel.subscribe("topic2").await;
        assert_eq!(channel.topic_count().await, 2);
    }

    #[test]
    fn test_message_empty_text() {
        let msg = Message::text("");
        assert_eq!(msg.as_text().unwrap(), "");
        assert_eq!(msg.as_bytes(), b"");
    }

    #[test]
    fn test_message_empty_bytes() {
        let msg = Message::bytes(vec![]);
        assert_eq!(msg.as_bytes(), &[] as &[u8]);
        // Bytes messages always error when calling as_text()
        assert!(msg.as_text().is_err());
    }

    #[tokio::test]
    async fn test_pushpull_empty_channel() {
        use tokio::time::{timeout, Duration};

        let channel = PushPullChannel::new(10);

        // Try to pull from empty channel with timeout
        let result = timeout(Duration::from_millis(10), channel.pull()).await;
        assert!(result.is_err()); // Should timeout
    }
}
