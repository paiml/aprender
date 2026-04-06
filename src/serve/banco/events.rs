//! Real-time event bus for WebSocket notifications.
//!
//! Events are broadcast to all connected WebSocket clients.
//! Producers (handlers) call `EventBus::emit()` — all subscribers receive the event.

use serde::Serialize;
use tokio::sync::broadcast;

/// Event types emitted by Banco operations.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", content = "data")]
#[serde(rename_all = "snake_case")]
pub enum BancoEvent {
    /// Model loaded into slot.
    ModelLoaded { model_id: String, format: String },
    /// Model unloaded from slot.
    ModelUnloaded,
    /// Training run started.
    TrainingStarted { run_id: String, method: String },
    /// Training metric emitted.
    TrainingMetric { run_id: String, step: u64, loss: f32 },
    /// Training run completed.
    TrainingComplete { run_id: String },
    /// File uploaded.
    FileUploaded { file_id: String, name: String },
    /// RAG index updated.
    RagIndexed { doc_count: usize, chunk_count: usize },
    /// Model merge completed.
    MergeComplete { merge_id: String, strategy: String },
    /// System status change.
    SystemEvent { message: String },
}

/// Broadcast event bus — multiple producers, multiple consumers.
pub struct EventBus {
    sender: broadcast::Sender<String>,
}

impl EventBus {
    /// Create a new event bus with the given channel capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        let (sender, _) = broadcast::channel(capacity);
        Self { sender }
    }

    /// Emit an event to all subscribers.
    pub fn emit(&self, event: &BancoEvent) {
        if let Ok(json) = serde_json::to_string(event) {
            // Ignore send errors — no subscribers is fine
            let _ = self.sender.send(json);
        }
    }

    /// Subscribe to events. Returns a receiver for consuming events.
    pub fn subscribe(&self) -> broadcast::Receiver<String> {
        self.sender.subscribe()
    }

    /// Get the current number of subscribers.
    #[must_use]
    pub fn subscriber_count(&self) -> usize {
        self.sender.receiver_count()
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new(256)
    }
}
