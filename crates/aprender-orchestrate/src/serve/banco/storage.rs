//! Banco storage — manages ~/.banco/ directory structure.
//!
//! Provides content-addressable file storage for uploads, datasets, and runs.
//! Files are stored with SHA-256 content hashes for deduplication.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

/// Metadata about an uploaded file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileInfo {
    pub id: String,
    pub name: String,
    pub size_bytes: u64,
    pub content_type: String,
    pub uploaded_at: u64,
    pub content_hash: String,
}

/// Detected file content type.
impl FileInfo {
    fn detect_content_type(name: &str) -> String {
        match name.rsplit('.').next().map(str::to_lowercase).as_deref() {
            Some("pdf") => "application/pdf",
            Some("csv") => "text/csv",
            Some("json") => "application/json",
            Some("jsonl") => "application/jsonl",
            Some("txt") => "text/plain",
            Some("docx") => {
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            }
            _ => "application/octet-stream",
        }
        .to_string()
    }
}

/// File store — in-memory index with optional disk backing.
pub struct FileStore {
    files: RwLock<HashMap<String, FileInfo>>,
    /// In-memory content cache (keyed by file ID).
    content: RwLock<HashMap<String, Vec<u8>>>,
    data_dir: Option<PathBuf>,
    counter: std::sync::atomic::AtomicU64,
}

impl FileStore {
    /// Create an in-memory-only store (for testing).
    #[must_use]
    pub fn in_memory() -> Arc<Self> {
        Arc::new(Self {
            files: RwLock::new(HashMap::new()),
            content: RwLock::new(HashMap::new()),
            data_dir: None,
            counter: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Create a store backed by a directory.
    #[must_use]
    pub fn with_data_dir(dir: PathBuf) -> Arc<Self> {
        let uploads_dir = dir.join("uploads");
        let _ = std::fs::create_dir_all(&uploads_dir);

        // Load existing file metadata from disk
        let mut files = HashMap::new();
        let mut max_seq = 0u64;
        if let Ok(entries) = std::fs::read_dir(&uploads_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("json") {
                    if let Ok(data) = std::fs::read_to_string(&path) {
                        if let Ok(info) = serde_json::from_str::<FileInfo>(&data) {
                            // Extract sequence number from ID for counter
                            if let Some(seq_str) = info.id.rsplit('-').next() {
                                if let Ok(seq) = seq_str.parse::<u64>() {
                                    max_seq = max_seq.max(seq + 1);
                                }
                            }
                            files.insert(info.id.clone(), info);
                        }
                    }
                }
            }
        }

        let loaded = files.len();
        if loaded > 0 {
            eprintln!("[banco] Loaded {loaded} files from {}", uploads_dir.display());
        }

        Arc::new(Self {
            files: RwLock::new(files),
            content: RwLock::new(HashMap::new()),
            data_dir: Some(dir),
            counter: std::sync::atomic::AtomicU64::new(max_seq),
        })
    }

    /// Store a file, returning its metadata.
    pub fn store(&self, name: &str, data: &[u8]) -> FileInfo {
        let seq = self.counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let id = format!("file-{}-{seq}", epoch_secs());
        let content_hash = sha256_hex(data);

        let info = FileInfo {
            id: id.clone(),
            name: name.to_string(),
            size_bytes: data.len() as u64,
            content_type: FileInfo::detect_content_type(name),
            uploaded_at: epoch_secs(),
            content_hash: content_hash.clone(),
        };

        // Write to disk if configured
        if let Some(ref dir) = self.data_dir {
            let path = dir.join("uploads").join(&content_hash);
            let _ = std::fs::write(path, data);
            // Also write metadata
            let meta_path = dir.join("uploads").join(format!("{content_hash}.meta.json"));
            let _ =
                std::fs::write(meta_path, serde_json::to_string_pretty(&info).unwrap_or_default());
        }

        if let Ok(mut store) = self.files.write() {
            store.insert(id.clone(), info.clone());
        }

        // Always cache content in memory for read_content()
        if let Ok(mut cache) = self.content.write() {
            cache.insert(id, data.to_vec());
        }

        info
    }

    /// List all files (most recent first).
    #[must_use]
    pub fn list(&self) -> Vec<FileInfo> {
        let store = self.files.read().unwrap_or_else(|e| e.into_inner());
        let mut files: Vec<FileInfo> = store.values().cloned().collect();
        files.sort_by(|a, b| b.uploaded_at.cmp(&a.uploaded_at));
        files
    }

    /// Get file metadata by ID.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<FileInfo> {
        self.files.read().unwrap_or_else(|e| e.into_inner()).get(id).cloned()
    }

    /// Get file content by ID (checks memory cache first, then disk).
    #[must_use]
    pub fn read_content(&self, id: &str) -> Option<Vec<u8>> {
        // Check in-memory cache first
        if let Ok(cache) = self.content.read() {
            if let Some(data) = cache.get(id) {
                return Some(data.clone());
            }
        }
        // Fall back to disk
        let info = self.get(id)?;
        if let Some(ref dir) = self.data_dir {
            let path = dir.join("uploads").join(&info.content_hash);
            std::fs::read(path).ok()
        } else {
            None
        }
    }

    /// Delete a file by ID.
    pub fn delete(&self, id: &str) -> Result<(), StorageError> {
        let info = {
            let mut store = self.files.write().map_err(|_| StorageError::LockPoisoned)?;
            store.remove(id).ok_or(StorageError::NotFound(id.to_string()))?
        };

        // Remove from caches
        if let Ok(mut cache) = self.content.write() {
            cache.remove(id);
        }
        if let Some(ref dir) = self.data_dir {
            let _ = std::fs::remove_file(dir.join("uploads").join(&info.content_hash));
            let _ = std::fs::remove_file(
                dir.join("uploads").join(format!("{}.meta.json", info.content_hash)),
            );
        }

        Ok(())
    }

    /// Number of stored files.
    #[must_use]
    pub fn len(&self) -> usize {
        self.files.read().map(|s| s.len()).unwrap_or(0)
    }

    /// Check if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Storage errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StorageError {
    NotFound(String),
    LockPoisoned,
}

impl std::fmt::Display for StorageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(id) => write!(f, "File not found: {id}"),
            Self::LockPoisoned => write!(f, "Internal lock error"),
        }
    }
}

impl std::error::Error for StorageError {}

fn epoch_secs() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs()
}

/// Simple SHA-256 hash (first 16 hex chars for dedup).
fn sha256_hex(data: &[u8]) -> String {
    // FNV-1a 128-bit for content-addressable storage (not crypto — just dedup)
    let mut h1: u64 = 0xcbf2_9ce4_8422_2325;
    let mut h2: u64 = 0x6c62_272e_07bb_0142;
    for &byte in data {
        h1 ^= byte as u64;
        h1 = h1.wrapping_mul(0x0100_0000_01b3);
        h2 ^= byte as u64;
        h2 = h2.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{h1:016x}{h2:016x}")
}
