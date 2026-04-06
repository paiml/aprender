//! ChunkId tests.

use super::*;
use std::collections::HashSet;

#[test]
fn test_chunk_id_unique() {
    let id1 = ChunkId::new();
    let id2 = ChunkId::new();
    assert_ne!(id1, id2);
}

#[test]
fn test_chunk_id_display() {
    let id = ChunkId::new();
    let display = format!("{id}");
    assert!(!display.is_empty());
    assert!(display.contains('-')); // UUID format
}

#[test]
fn test_chunk_id_serialization() {
    let id = ChunkId::new();
    let json = serde_json::to_string(&id).expect("JSON serialization should succeed");
    let deserialized: ChunkId =
        serde_json::from_str(&json).expect("JSON deserialization should succeed");
    assert_eq!(id, deserialized);
}

#[test]
fn test_chunk_id_default() {
    let id = ChunkId::default();
    // Default creates a new UUID, should not be nil
    assert!(!id.0.is_nil());
}

#[test]
fn test_chunk_id_display_length() {
    let id = ChunkId::new();
    let display = format!("{id}");
    // UUID format: 8-4-4-4-12 = 36 chars
    assert_eq!(display.len(), 36);
}

#[test]
fn test_chunk_id_hash_in_set() {
    let id1 = ChunkId::new();
    let id2 = ChunkId::new();
    let mut set = HashSet::new();
    set.insert(id1);
    set.insert(id2);
    assert_eq!(set.len(), 2);
}

#[test]
fn test_chunk_id_clone() {
    let id = ChunkId::new();
    let cloned = id;
    assert_eq!(id, cloned);
}

#[test]
fn test_chunk_id_copy() {
    let id = ChunkId::new();
    let copied = id;
    assert_eq!(id, copied);
}

#[test]
fn test_chunk_id_debug() {
    let id = ChunkId::new();
    let debug = format!("{id:?}");
    assert!(debug.contains("ChunkId"));
}

#[test]
fn test_chunk_id_eq_same_uuid() {
    let uuid = uuid::Uuid::new_v4();
    let id1 = ChunkId(uuid);
    let id2 = ChunkId(uuid);
    assert_eq!(id1, id2);
}
