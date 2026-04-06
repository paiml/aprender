//! Checkpoint management for persistent task state.
#![allow(clippy::missing_errors_doc, clippy::unused_async)]
//!
//! This module provides checkpointing capabilities for long-running tasks,
//! enabling restart from the last checkpoint on failure.
//!
//! # Example
//!
//! ```rust,no_run
//! use repartir::checkpoint::{CheckpointManager, TaskState};
//! use std::path::PathBuf;
//!
//! # async fn example() -> repartir::error::Result<()> {
//! let manager = CheckpointManager::new(PathBuf::from("./checkpoints"))?;
//!
//! // Create a checkpoint
//! let task_id = uuid::Uuid::new_v4();
//! let state = TaskState {
//!     task_id,
//!     iteration: 42,
//!     data: vec![1, 2, 3, 4],
//!     timestamp: std::time::SystemTime::now(),
//! };
//!
//! let checkpoint_id = manager.checkpoint(task_id, &state).await?;
//!
//! // Restore from checkpoint
//! let restored = manager.restore(task_id).await?;
//! assert_eq!(restored.unwrap().iteration, 42);
//! # Ok(())
//! # }
//! ```

use crate::error::{RepartirError, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::SystemTime;
use uuid::Uuid;

#[cfg(feature = "checkpoint")]
use trueno_db::storage::StorageEngine;

#[cfg(feature = "checkpoint")]
use arrow::array::{ArrayRef, BinaryArray, StringArray, TimestampMicrosecondArray, UInt64Array};
#[cfg(feature = "checkpoint")]
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
#[cfg(feature = "checkpoint")]
use arrow::record_batch::RecordBatch;
#[cfg(feature = "checkpoint")]
use parquet::arrow::arrow_reader::ArrowReaderBuilder;
#[cfg(feature = "checkpoint")]
use parquet::arrow::arrow_writer::ArrowWriter;
#[cfg(feature = "checkpoint")]
use parquet::file::properties::WriterProperties;
#[cfg(feature = "checkpoint")]
use std::fs::File;
#[cfg(feature = "checkpoint")]
use std::sync::Arc;

/// Unique identifier for a checkpoint
pub type CheckpointId = Uuid;

/// Task state that can be checkpointed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskState {
    /// Unique task identifier
    pub task_id: Uuid,
    /// Current iteration number
    pub iteration: u64,
    /// Serialized application state
    pub data: Vec<u8>,
    /// Timestamp when checkpoint was created
    pub timestamp: SystemTime,
}

/// Metadata about a checkpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// Checkpoint identifier
    pub checkpoint_id: CheckpointId,
    /// Task identifier
    pub task_id: Uuid,
    /// Iteration number
    pub iteration: u64,
    /// Size of checkpoint data in bytes
    pub size_bytes: usize,
    /// Creation timestamp
    pub created_at: SystemTime,
}

/// Manages persistent checkpoints for tasks
pub struct CheckpointManager {
    checkpoint_dir: PathBuf,
    #[cfg(feature = "checkpoint")]
    _storage: Option<StorageEngine>,
}

impl CheckpointManager {
    /// Create a new checkpoint manager
    ///
    /// # Arguments
    ///
    /// * `checkpoint_dir` - Directory to store checkpoint files
    ///
    /// # Errors
    ///
    /// Returns error if directory cannot be created
    pub fn new(checkpoint_dir: PathBuf) -> Result<Self> {
        // Create checkpoint directory if it doesn't exist
        std::fs::create_dir_all(&checkpoint_dir).map_err(|e| RepartirError::InvalidTask {
            reason: format!("Failed to create checkpoint directory: {e}"),
        })?;

        Ok(Self {
            checkpoint_dir,
            #[cfg(feature = "checkpoint")]
            _storage: None,
        })
    }

    /// Create a checkpoint for a task (v2.0: Parquet format)
    ///
    /// # Arguments
    ///
    /// * `task_id` - Unique task identifier
    /// * `state` - Task state to checkpoint
    ///
    /// # Returns
    ///
    /// Returns the checkpoint ID on success
    ///
    /// # Errors
    ///
    /// Returns error if checkpoint cannot be written
    #[cfg(feature = "checkpoint")]
    pub async fn checkpoint(&self, task_id: Uuid, state: &TaskState) -> Result<CheckpointId> {
        let checkpoint_id = Uuid::new_v4();

        // Create task directory
        let task_dir = self.checkpoint_dir.join(task_id.to_string());
        std::fs::create_dir_all(&task_dir).map_err(|e| RepartirError::InvalidTask {
            reason: format!("Failed to create task checkpoint directory: {e}"),
        })?;

        // Write to Parquet: checkpoints/<task_id>/<checkpoint_id>.parquet
        let checkpoint_path = task_dir.join(format!("{checkpoint_id}.parquet"));
        self.write_parquet(&checkpoint_path, checkpoint_id, state)?;

        Ok(checkpoint_id)
    }

    /// Write checkpoint to Parquet format
    #[cfg(feature = "checkpoint")]
    #[allow(clippy::unused_self)]
    fn write_parquet(
        &self,
        path: &PathBuf,
        checkpoint_id: CheckpointId,
        state: &TaskState,
    ) -> Result<()> {
        // Define Parquet schema
        let schema = Arc::new(Schema::new(vec![
            Field::new("checkpoint_id", DataType::Utf8, false),
            Field::new("task_id", DataType::Utf8, false),
            Field::new("iteration", DataType::UInt64, false),
            Field::new("timestamp_micros", DataType::Timestamp(TimeUnit::Microsecond, None), false),
            Field::new("data", DataType::Binary, false),
        ]));

        // Convert SystemTime to microseconds since UNIX epoch
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let timestamp_micros = state
            .timestamp
            .duration_since(SystemTime::UNIX_EPOCH)
            .map_err(|e| RepartirError::InvalidTask {
                reason: format!("Failed to convert timestamp: {e}"),
            })?
            .as_micros() as i64;

        // Create arrays
        let checkpoint_id_array = StringArray::from(vec![checkpoint_id.to_string()]);
        let task_id_array = StringArray::from(vec![state.task_id.to_string()]);
        let iteration_array = UInt64Array::from(vec![state.iteration]);
        let timestamp_array = TimestampMicrosecondArray::from(vec![timestamp_micros]);
        let data_array = BinaryArray::from(vec![state.data.as_slice()]);

        // Create record batch
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(checkpoint_id_array) as ArrayRef,
                Arc::new(task_id_array) as ArrayRef,
                Arc::new(iteration_array) as ArrayRef,
                Arc::new(timestamp_array) as ArrayRef,
                Arc::new(data_array) as ArrayRef,
            ],
        )
        .map_err(|e| RepartirError::InvalidTask {
            reason: format!("Failed to create record batch: {e}"),
        })?;

        // Write to Parquet file with compression
        let file = File::create(path).map_err(|e| RepartirError::InvalidTask {
            reason: format!("Failed to create parquet file: {e}"),
        })?;

        let props = WriterProperties::builder()
            .set_compression(parquet::basic::Compression::SNAPPY)
            .build();

        let mut writer = ArrowWriter::try_new(file, schema, Some(props)).map_err(|e| {
            RepartirError::InvalidTask { reason: format!("Failed to create parquet writer: {e}") }
        })?;

        writer.write(&batch).map_err(|e| RepartirError::InvalidTask {
            reason: format!("Failed to write parquet batch: {e}"),
        })?;

        writer.close().map_err(|e| RepartirError::InvalidTask {
            reason: format!("Failed to close parquet writer: {e}"),
        })?;

        Ok(())
    }

    /// Checkpoint without trueno-db feature (minimal implementation)
    #[cfg(not(feature = "checkpoint"))]
    pub async fn checkpoint(&self, task_id: Uuid, state: &TaskState) -> Result<CheckpointId> {
        let checkpoint_id = Uuid::new_v4();

        // Serialize state to JSON
        let serialized = serde_json::to_vec(&state).map_err(|e| RepartirError::InvalidTask {
            reason: format!("Failed to serialize checkpoint: {e}"),
        })?;

        // Write to file
        let task_dir = self.checkpoint_dir.join(task_id.to_string());
        std::fs::create_dir_all(&task_dir).map_err(|e| RepartirError::InvalidTask {
            reason: format!("Failed to create task checkpoint directory: {e}"),
        })?;

        let checkpoint_path = task_dir.join(format!("{checkpoint_id}.json"));
        std::fs::write(&checkpoint_path, &serialized).map_err(|e| RepartirError::InvalidTask {
            reason: format!("Failed to write checkpoint: {e}"),
        })?;

        Ok(checkpoint_id)
    }

    /// Restore task from last checkpoint (v2.0: supports Parquet and JSON)
    ///
    /// # Arguments
    ///
    /// * `task_id` - Unique task identifier
    ///
    /// # Returns
    ///
    /// Returns the most recent task state, or None if no checkpoints exist
    ///
    /// # Errors
    ///
    /// Returns error if checkpoint cannot be read
    pub async fn restore(&self, task_id: Uuid) -> Result<Option<TaskState>> {
        let task_dir = self.checkpoint_dir.join(task_id.to_string());

        // Check if task has any checkpoints
        if !task_dir.exists() {
            return Ok(None);
        }

        // Find most recent checkpoint (prefer Parquet, fall back to JSON)
        let mut checkpoints: Vec<_> = std::fs::read_dir(&task_dir)
            .map_err(|e| RepartirError::InvalidTask {
                reason: format!("Failed to read checkpoint directory: {e}"),
            })?
            .filter_map(std::result::Result::ok)
            .filter(|entry| {
                let path = entry.path();
                path.extension().is_some_and(|ext| ext == "parquet" || ext == "json")
            })
            .collect();

        if checkpoints.is_empty() {
            return Ok(None);
        }

        // Sort by filename (checkpoint ID)
        checkpoints.sort_by_key(std::fs::DirEntry::path);

        // Read most recent checkpoint
        let latest = checkpoints.last().ok_or_else(|| RepartirError::InvalidTask {
            reason: "No checkpoints found".to_string(),
        })?;

        let latest_path = latest.path();
        let extension = latest_path.extension().and_then(|s| s.to_str()).ok_or_else(|| {
            RepartirError::InvalidTask { reason: "Invalid checkpoint file extension".to_string() }
        })?;

        // Read based on file format
        match extension {
            #[cfg(feature = "checkpoint")]
            "parquet" => self.read_parquet(&latest_path),
            "json" => {
                let data = std::fs::read(&latest_path).map_err(|e| RepartirError::InvalidTask {
                    reason: format!("Failed to read checkpoint: {e}"),
                })?;

                let state =
                    serde_json::from_slice(&data).map_err(|e| RepartirError::InvalidTask {
                        reason: format!("Failed to deserialize checkpoint: {e}"),
                    })?;

                Ok(Some(state))
            }
            #[cfg(not(feature = "checkpoint"))]
            "parquet" => Err(RepartirError::InvalidTask {
                reason: "Parquet checkpoints require 'checkpoint' feature".to_string(),
            }),
            _ => Err(RepartirError::InvalidTask {
                reason: format!("Unsupported checkpoint format: {extension}"),
            }),
        }
    }

    /// Read checkpoint from Parquet format
    #[cfg(feature = "checkpoint")]
    #[allow(clippy::unused_self)]
    fn read_parquet(&self, path: &PathBuf) -> Result<Option<TaskState>> {
        let file = File::open(path).map_err(|e| RepartirError::InvalidTask {
            reason: format!("Failed to open parquet file: {e}"),
        })?;

        let builder = ArrowReaderBuilder::try_new(file).map_err(|e| {
            RepartirError::InvalidTask { reason: format!("Failed to create parquet reader: {e}") }
        })?;

        let mut reader = builder.build().map_err(|e| RepartirError::InvalidTask {
            reason: format!("Failed to build parquet reader: {e}"),
        })?;

        // Read first batch (checkpoints are single-row)
        let batch = reader
            .next()
            .ok_or_else(|| RepartirError::InvalidTask { reason: "Empty parquet file".to_string() })?
            .map_err(|e| RepartirError::InvalidTask {
                reason: format!("Failed to read parquet batch: {e}"),
            })?;

        // Extract fields
        let task_id_array =
            batch.column(1).as_any().downcast_ref::<StringArray>().ok_or_else(|| {
                RepartirError::InvalidTask { reason: "Invalid task_id column type".to_string() }
            })?;

        let iteration_array =
            batch.column(2).as_any().downcast_ref::<UInt64Array>().ok_or_else(|| {
                RepartirError::InvalidTask { reason: "Invalid iteration column type".to_string() }
            })?;

        let timestamp_array =
            batch.column(3).as_any().downcast_ref::<TimestampMicrosecondArray>().ok_or_else(
                || RepartirError::InvalidTask {
                    reason: "Invalid timestamp column type".to_string(),
                },
            )?;

        let data_array =
            batch.column(4).as_any().downcast_ref::<BinaryArray>().ok_or_else(|| {
                RepartirError::InvalidTask { reason: "Invalid data column type".to_string() }
            })?;

        // Convert to TaskState
        let task_id = Uuid::parse_str(task_id_array.value(0)).map_err(|e| {
            RepartirError::InvalidTask { reason: format!("Failed to parse task_id: {e}") }
        })?;

        let iteration = iteration_array.value(0);

        let timestamp_micros = timestamp_array.value(0);
        #[allow(clippy::cast_sign_loss)]
        let timestamp =
            SystemTime::UNIX_EPOCH + std::time::Duration::from_micros(timestamp_micros as u64);

        let data = data_array.value(0).to_vec();

        Ok(Some(TaskState { task_id, iteration, data, timestamp }))
    }

    /// List all checkpoints for a task (v2.0: supports Parquet and JSON)
    ///
    /// # Arguments
    ///
    /// * `task_id` - Unique task identifier
    ///
    /// # Returns
    ///
    /// Returns a list of checkpoint metadata, sorted by creation time
    ///
    /// # Errors
    ///
    /// Returns error if checkpoints cannot be listed
    pub async fn list_checkpoints(&self, task_id: Uuid) -> Result<Vec<CheckpointMetadata>> {
        let task_dir = self.checkpoint_dir.join(task_id.to_string());

        if !task_dir.exists() {
            return Ok(Vec::new());
        }

        let mut metadata = Vec::new();

        for entry in std::fs::read_dir(&task_dir).map_err(|e| RepartirError::InvalidTask {
            reason: format!("Failed to read checkpoint directory: {e}"),
        })? {
            let entry = entry.map_err(|e| RepartirError::InvalidTask {
                reason: format!("Failed to read directory entry: {e}"),
            })?;

            let path = entry.path();
            let ext = path.extension().and_then(|e| e.to_str());

            let checkpoint_id = path
                .file_stem()
                .and_then(|s| s.to_str())
                .and_then(|s| Uuid::parse_str(s).ok())
                .ok_or_else(|| RepartirError::InvalidTask {
                    reason: "Invalid checkpoint filename".to_string(),
                })?;

            match ext {
                #[cfg(feature = "checkpoint")]
                Some("parquet") => {
                    // Read Parquet metadata
                    if let Some(state) = self.read_parquet(&path)? {
                        #[allow(clippy::cast_possible_truncation)]
                        let size_bytes =
                            std::fs::metadata(&path).map(|m| m.len() as usize).unwrap_or(0);

                        metadata.push(CheckpointMetadata {
                            checkpoint_id,
                            task_id,
                            iteration: state.iteration,
                            size_bytes,
                            created_at: state.timestamp,
                        });
                    }
                }
                Some("json") => {
                    // Read JSON metadata
                    let data = std::fs::read(&path).map_err(|e| RepartirError::InvalidTask {
                        reason: format!("Failed to read checkpoint: {e}"),
                    })?;

                    let state: TaskState =
                        serde_json::from_slice(&data).map_err(|e| RepartirError::InvalidTask {
                            reason: format!("Failed to deserialize checkpoint: {e}"),
                        })?;

                    metadata.push(CheckpointMetadata {
                        checkpoint_id,
                        task_id,
                        iteration: state.iteration,
                        size_bytes: data.len(),
                        created_at: state.timestamp,
                    });
                }
                _ => {} // Ignore unknown formats
            }
        }

        // Sort by creation time
        metadata.sort_by_key(|m| m.created_at);

        Ok(metadata)
    }

    /// Delete old checkpoints based on retention policy
    ///
    /// # Arguments
    ///
    /// * `retention_days` - Keep checkpoints newer than this many days
    ///
    /// # Returns
    ///
    /// Returns the number of checkpoints deleted
    ///
    /// # Errors
    ///
    /// Returns error if checkpoints cannot be deleted
    pub async fn cleanup(&self, retention_days: u32) -> Result<usize> {
        let retention_duration =
            std::time::Duration::from_secs(u64::from(retention_days) * 24 * 3600);
        let cutoff_time = SystemTime::now().checked_sub(retention_duration).ok_or_else(|| {
            RepartirError::InvalidTask { reason: "Invalid retention duration".to_string() }
        })?;

        let mut deleted_count = 0;

        // Iterate over all task directories
        for entry in
            std::fs::read_dir(&self.checkpoint_dir).map_err(|e| RepartirError::InvalidTask {
                reason: format!("Failed to read checkpoint directory: {e}"),
            })?
        {
            let entry = entry.map_err(|e| RepartirError::InvalidTask {
                reason: format!("Failed to read directory entry: {e}"),
            })?;

            if entry.path().is_dir() {
                // This is a task directory
                if let Ok(task_id) = Uuid::parse_str(&entry.file_name().to_string_lossy()) {
                    let checkpoints = self.list_checkpoints(task_id).await?;

                    for checkpoint in checkpoints {
                        if checkpoint.created_at < cutoff_time {
                            // Delete old checkpoint (try both .parquet and .json)
                            let task_path = self.checkpoint_dir.join(task_id.to_string());

                            let parquet_path =
                                task_path.join(format!("{}.parquet", checkpoint.checkpoint_id));
                            let json_path =
                                task_path.join(format!("{}.json", checkpoint.checkpoint_id));

                            // Try to delete parquet first, fall back to json
                            let deleted = if parquet_path.exists() {
                                std::fs::remove_file(&parquet_path).map_err(|e| {
                                    RepartirError::InvalidTask {
                                        reason: format!("Failed to delete parquet checkpoint: {e}"),
                                    }
                                })?;
                                true
                            } else if json_path.exists() {
                                std::fs::remove_file(&json_path).map_err(|e| {
                                    RepartirError::InvalidTask {
                                        reason: format!("Failed to delete json checkpoint: {e}"),
                                    }
                                })?;
                                true
                            } else {
                                false
                            };

                            if deleted {
                                deleted_count += 1;
                            }
                        }
                    }
                }
            }
        }

        Ok(deleted_count)
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    clippy::cast_possible_truncation
)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_checkpoint_and_restore() {
        let temp_dir = std::env::temp_dir().join(format!("repartir_test_{}", Uuid::new_v4()));
        let manager = CheckpointManager::new(temp_dir.clone()).unwrap();

        let task_id = Uuid::new_v4();
        let state = TaskState {
            task_id,
            iteration: 42,
            data: vec![1, 2, 3, 4, 5],
            timestamp: SystemTime::now(),
        };

        // Create checkpoint
        let checkpoint_id = manager.checkpoint(task_id, &state).await.unwrap();
        assert!(!checkpoint_id.is_nil());

        // Restore checkpoint
        let restored = manager.restore(task_id).await.unwrap();
        assert!(restored.is_some());

        let restored_state = restored.unwrap();
        assert_eq!(restored_state.task_id, task_id);
        assert_eq!(restored_state.iteration, 42);
        assert_eq!(restored_state.data, vec![1, 2, 3, 4, 5]);

        // Cleanup
        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[tokio::test]
    async fn test_restore_nonexistent() {
        let temp_dir = std::env::temp_dir().join(format!("repartir_test_{}", Uuid::new_v4()));
        let manager = CheckpointManager::new(temp_dir.clone()).unwrap();

        let task_id = Uuid::new_v4();
        let restored = manager.restore(task_id).await.unwrap();
        assert!(restored.is_none());

        // Cleanup
        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[tokio::test]
    async fn test_list_checkpoints() {
        let temp_dir = std::env::temp_dir().join(format!("repartir_test_{}", Uuid::new_v4()));
        let manager = CheckpointManager::new(temp_dir.clone()).unwrap();

        let task_id = Uuid::new_v4();

        // Create multiple checkpoints
        for i in 0..3 {
            let state = TaskState {
                task_id,
                iteration: i,
                data: vec![i as u8],
                timestamp: SystemTime::now(),
            };
            manager.checkpoint(task_id, &state).await.unwrap();
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        // List checkpoints
        let checkpoints = manager.list_checkpoints(task_id).await.unwrap();
        assert_eq!(checkpoints.len(), 3);

        // Verify sorted by creation time
        for (i, checkpoint) in checkpoints.iter().enumerate() {
            assert_eq!(checkpoint.iteration, i as u64);
        }

        // Cleanup
        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[tokio::test]
    async fn test_cleanup() {
        let temp_dir = std::env::temp_dir().join(format!("repartir_test_{}", Uuid::new_v4()));
        let manager = CheckpointManager::new(temp_dir.clone()).unwrap();

        let task_id = Uuid::new_v4();

        // Create checkpoint with old timestamp
        let old_state = TaskState {
            task_id,
            iteration: 1,
            data: vec![1],
            timestamp: SystemTime::now().checked_sub(Duration::from_secs(10 * 24 * 3600)).unwrap(),
        };
        manager.checkpoint(task_id, &old_state).await.unwrap();

        // Create checkpoint with recent timestamp
        let new_state =
            TaskState { task_id, iteration: 2, data: vec![2], timestamp: SystemTime::now() };
        manager.checkpoint(task_id, &new_state).await.unwrap();

        // Cleanup checkpoints older than 7 days
        let deleted = manager.cleanup(7).await.unwrap();
        assert_eq!(deleted, 1);

        // Verify only recent checkpoint remains
        let checkpoints = manager.list_checkpoints(task_id).await.unwrap();
        assert_eq!(checkpoints.len(), 1);
        assert_eq!(checkpoints[0].iteration, 2);

        // Cleanup
        std::fs::remove_dir_all(&temp_dir).ok();
    }
}
