//! Conversation persistence for Banco.
//!
//! Stores conversations as JSONL files in `~/.banco/conversations/`.
//! Each conversation has an ID, title, and append-only message log.

use crate::serve::templates::ChatMessage;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

/// Metadata for a single conversation (stored in index).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationMeta {
    pub id: String,
    pub title: String,
    pub model: String,
    pub created: u64,
    pub updated: u64,
    pub message_count: usize,
}

/// A full conversation with messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    pub meta: ConversationMeta,
    pub messages: Vec<ChatMessage>,
}

/// In-memory conversation store with optional disk persistence.
pub struct ConversationStore {
    conversations: RwLock<HashMap<String, Conversation>>,
    data_dir: Option<PathBuf>,
    counter: std::sync::atomic::AtomicU64,
}

impl ConversationStore {
    /// Create an in-memory-only store (for testing).
    #[must_use]
    pub fn in_memory() -> Arc<Self> {
        Arc::new(Self {
            conversations: RwLock::new(HashMap::new()),
            data_dir: None,
            counter: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Create a store backed by `~/.banco/conversations/`.
    #[must_use]
    pub fn with_data_dir(dir: PathBuf) -> Arc<Self> {
        let _ = std::fs::create_dir_all(&dir);

        // Load existing conversations from JSONL files
        let mut conversations = HashMap::new();
        let mut max_seq = 0u64;
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("jsonl") {
                    let conv_id =
                        path.file_stem().and_then(|s| s.to_str()).unwrap_or("unknown").to_string();

                    // Read messages from JSONL
                    let mut messages = Vec::new();
                    if let Ok(content) = std::fs::read_to_string(&path) {
                        for line in content.lines() {
                            if let Ok(msg) = serde_json::from_str::<ChatMessage>(line) {
                                messages.push(msg);
                            }
                        }
                    }

                    // Extract sequence from ID
                    if let Some(seq_str) = conv_id.rsplit('-').next() {
                        if let Ok(seq) = seq_str.parse::<u64>() {
                            max_seq = max_seq.max(seq + 1);
                        }
                    }

                    let title = messages
                        .first()
                        .filter(|m| matches!(m.role, crate::serve::templates::Role::User))
                        .map(|m| auto_title(&m.content))
                        .unwrap_or_else(|| "Loaded conversation".to_string());

                    let conv = Conversation {
                        meta: ConversationMeta {
                            id: conv_id.clone(),
                            title,
                            model: "unknown".to_string(),
                            created: epoch_secs(),
                            updated: epoch_secs(),
                            message_count: messages.len(),
                        },
                        messages,
                    };
                    conversations.insert(conv_id, conv);
                }
            }
        }

        let loaded = conversations.len();
        if loaded > 0 {
            eprintln!("[banco] Loaded {loaded} conversations from {}", dir.display());
        }

        Arc::new(Self {
            conversations: RwLock::new(conversations),
            data_dir: Some(dir),
            counter: std::sync::atomic::AtomicU64::new(max_seq),
        })
    }

    /// Create a new conversation, returning its ID.
    pub fn create(&self, model: &str) -> String {
        let seq = self.counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let id = format!("conv-{}-{seq}", epoch_secs());
        let meta = ConversationMeta {
            id: id.clone(),
            title: "New conversation".to_string(),
            model: model.to_string(),
            created: epoch_secs(),
            updated: epoch_secs(),
            message_count: 0,
        };
        let conv = Conversation { meta, messages: Vec::new() };
        if let Ok(mut store) = self.conversations.write() {
            store.insert(id.clone(), conv);
        }
        id
    }

    /// Append a message to a conversation. Auto-titles on first user message.
    pub fn append(&self, id: &str, message: ChatMessage) -> Result<(), ConversationError> {
        let mut store = self.conversations.write().map_err(|_| ConversationError::LockPoisoned)?;
        let conv = store.get_mut(id).ok_or(ConversationError::NotFound(id.to_string()))?;

        // Auto-title from first user message
        if conv.messages.is_empty()
            && conv.meta.title == "New conversation"
            && matches!(message.role, crate::serve::templates::Role::User)
        {
            conv.meta.title = auto_title(&message.content);
        }

        conv.messages.push(message);
        conv.meta.message_count = conv.messages.len();
        conv.meta.updated = epoch_secs();

        // Persist to disk if configured
        if let Some(ref dir) = self.data_dir {
            let path = dir.join(format!("{id}.jsonl"));
            let json = serde_json::to_string(&conv.messages.last().expect("just pushed"))
                .unwrap_or_default();
            let _ = std::fs::OpenOptions::new().create(true).append(true).open(path).and_then(
                |mut f| {
                    use std::io::Write;
                    writeln!(f, "{json}")
                },
            );
        }

        Ok(())
    }

    /// List all conversations (most recent first).
    #[must_use]
    pub fn list(&self) -> Vec<ConversationMeta> {
        let store = self.conversations.read().unwrap_or_else(|e| e.into_inner());
        let mut metas: Vec<ConversationMeta> = store.values().map(|c| c.meta.clone()).collect();
        metas.sort_by(|a, b| b.updated.cmp(&a.updated));
        metas
    }

    /// Get a conversation by ID.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<Conversation> {
        let store = self.conversations.read().unwrap_or_else(|e| e.into_inner());
        store.get(id).cloned()
    }

    /// Rename a conversation.
    pub fn rename(&self, id: &str, title: &str) -> Result<(), ConversationError> {
        let mut store = self.conversations.write().map_err(|_| ConversationError::LockPoisoned)?;
        let conv = store.get_mut(id).ok_or(ConversationError::NotFound(id.to_string()))?;
        conv.meta.title = title.to_string();
        conv.meta.updated = epoch_secs();
        Ok(())
    }

    /// Delete a conversation by ID.
    pub fn delete(&self, id: &str) -> Result<(), ConversationError> {
        let mut store = self.conversations.write().map_err(|_| ConversationError::LockPoisoned)?;
        store.remove(id).ok_or(ConversationError::NotFound(id.to_string()))?;

        // Remove from disk
        if let Some(ref dir) = self.data_dir {
            let _ = std::fs::remove_file(dir.join(format!("{id}.jsonl")));
        }

        Ok(())
    }

    /// Number of conversations.
    #[must_use]
    pub fn len(&self) -> usize {
        self.conversations.read().map(|s| s.len()).unwrap_or(0)
    }

    /// Check if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Search conversations by content (case-insensitive substring match).
    #[must_use]
    pub fn search(&self, query: &str) -> Vec<ConversationMeta> {
        let store = self.conversations.read().unwrap_or_else(|e| e.into_inner());
        let query_lower = query.to_lowercase();
        let mut results: Vec<ConversationMeta> = store
            .values()
            .filter(|c| {
                c.meta.title.to_lowercase().contains(&query_lower)
                    || c.messages.iter().any(|m| m.content.to_lowercase().contains(&query_lower))
            })
            .map(|c| c.meta.clone())
            .collect();
        results.sort_by(|a, b| b.updated.cmp(&a.updated));
        results
    }

    /// Export all conversations as a JSON-serializable vec.
    #[must_use]
    pub fn export_all(&self) -> Vec<Conversation> {
        let store = self.conversations.read().unwrap_or_else(|e| e.into_inner());
        let mut convs: Vec<Conversation> = store.values().cloned().collect();
        convs.sort_by(|a, b| b.meta.updated.cmp(&a.meta.updated));
        convs
    }

    /// Import conversations, merging by ID (existing conversations are overwritten).
    /// Returns the number of conversations imported.
    pub fn import_all(&self, conversations: Vec<Conversation>) -> usize {
        let mut store = self.conversations.write().unwrap_or_else(|e| e.into_inner());
        let count = conversations.len();
        for conv in conversations {
            store.insert(conv.meta.id.clone(), conv);
        }
        count
    }
}

/// Auto-generate a title from the first user message (first 5 words).
fn auto_title(content: &str) -> String {
    let words: Vec<&str> = content.split_whitespace().take(5).collect();
    if words.is_empty() {
        "New conversation".to_string()
    } else {
        let mut title = words.join(" ");
        if content.split_whitespace().count() > 5 {
            title.push_str("...");
        }
        title
    }
}

fn epoch_secs() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs()
}

/// Conversation errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConversationError {
    NotFound(String),
    LockPoisoned,
}

impl std::fmt::Display for ConversationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(id) => write!(f, "Conversation not found: {id}"),
            Self::LockPoisoned => write!(f, "Internal lock error"),
        }
    }
}

impl std::error::Error for ConversationError {}
