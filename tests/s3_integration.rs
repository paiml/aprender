//! S3 integration tests for alimentar.
//!
//! These tests require a running S3-compatible service (`MinIO` recommended).
//!
//! To run these tests:
//!
//! 1. Start `MinIO`: ```bash docker run -p 9000:9000 -p 9001:9001 \ -e
//!    MINIO_ROOT_USER=minioadmin \ -e MINIO_ROOT_PASSWORD=minioadmin \
//!    minio/minio server /data --console-address ":9001" ```
//!
//! 2. Create a test bucket: ```bash mc alias set local http://localhost:9000
//!    minioadmin minioadmin mc mb local/alimentar-test ```
//!
//! 3. Run tests with the s3 feature: ```bash cargo test --features s3
//!    s3_integration -- --ignored ```

#![cfg(feature = "s3")]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::uninlined_format_args,
    clippy::cast_lossless,
    clippy::similar_names,
    clippy::doc_markdown
)]

use std::sync::Arc;

use alimentar::{
    backend::{CredentialSource, S3Backend, StorageBackend},
    registry::{DatasetMetadata, Registry},
    ArrowDataset, Dataset,
};
use arrow::{
    array::{Float64Array, Int32Array, StringArray},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use bytes::Bytes;

const TEST_BUCKET: &str = "alimentar-test";
const TEST_REGION: &str = "us-east-1";
const TEST_ENDPOINT: &str = "http://localhost:9000";
const TEST_ACCESS_KEY: &str = "minioadmin";
const TEST_SECRET_KEY: &str = "minioadmin";

fn create_test_backend() -> Result<S3Backend, alimentar::Error> {
    S3Backend::new(
        TEST_BUCKET,
        TEST_REGION,
        Some(TEST_ENDPOINT.to_string()),
        CredentialSource::Static {
            access_key: TEST_ACCESS_KEY.to_string(),
            secret_key: TEST_SECRET_KEY.to_string(),
        },
    )
}

fn create_test_dataset(rows: usize) -> ArrowDataset {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("score", DataType::Float64, false),
    ]));

    let ids: Vec<i32> = (0..rows as i32).collect();
    let names: Vec<String> = ids.iter().map(|i| format!("item_{}", i)).collect();
    let scores: Vec<f64> = ids.iter().map(|i| *i as f64 * 1.5).collect();

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(StringArray::from(names)),
            Arc::new(Float64Array::from(scores)),
        ],
    )
    .ok()
    .unwrap_or_else(|| panic!("Should create batch"));

    ArrowDataset::from_batch(batch)
        .ok()
        .unwrap_or_else(|| panic!("Should create dataset"))
}

fn unique_key(prefix: &str) -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{}_{}_{:?}", prefix, ts, std::thread::current().id())
}

#[test]
#[ignore = "Requires MinIO running"]
fn test_s3_backend_put_and_get() {
    let backend = create_test_backend()
        .ok()
        .unwrap_or_else(|| panic!("Should create S3 backend"));

    let key = unique_key("test_data");
    let data = Bytes::from("Hello, S3!");

    // Put
    backend
        .put(&key, data.clone())
        .ok()
        .unwrap_or_else(|| panic!("Should put"));

    // Get
    let retrieved = backend
        .get(&key)
        .ok()
        .unwrap_or_else(|| panic!("Should get"));

    assert_eq!(retrieved, data);

    // Cleanup
    let _ = backend.delete(&key);
}

#[test]
#[ignore = "Requires MinIO running"]
fn test_s3_backend_exists() {
    let backend = create_test_backend()
        .ok()
        .unwrap_or_else(|| panic!("Should create S3 backend"));

    let key = unique_key("exists_test");

    // Should not exist initially
    let exists = backend
        .exists(&key)
        .ok()
        .unwrap_or_else(|| panic!("Should check exists"));
    assert!(!exists);

    // Put data
    backend
        .put(&key, Bytes::from("test"))
        .ok()
        .unwrap_or_else(|| panic!("Should put"));

    // Should exist now
    let exists = backend
        .exists(&key)
        .ok()
        .unwrap_or_else(|| panic!("Should check exists"));
    assert!(exists);

    // Cleanup
    let _ = backend.delete(&key);
}

#[test]
#[ignore = "Requires MinIO running"]
fn test_s3_backend_size() {
    let backend = create_test_backend()
        .ok()
        .unwrap_or_else(|| panic!("Should create S3 backend"));

    let key = unique_key("size_test");
    let data = Bytes::from("12345678901234567890"); // 20 bytes

    backend
        .put(&key, data)
        .ok()
        .unwrap_or_else(|| panic!("Should put"));

    let size = backend
        .size(&key)
        .ok()
        .unwrap_or_else(|| panic!("Should get size"));

    assert_eq!(size, 20);

    // Cleanup
    let _ = backend.delete(&key);
}

#[test]
#[ignore = "Requires MinIO running"]
fn test_s3_backend_list() {
    let backend = create_test_backend()
        .ok()
        .unwrap_or_else(|| panic!("Should create S3 backend"));

    let prefix = unique_key("list_test");
    let key1 = format!("{}/file1.txt", prefix);
    let key2 = format!("{}/file2.txt", prefix);
    let key3 = format!("{}/subdir/file3.txt", prefix);

    // Put test files
    backend
        .put(&key1, Bytes::from("1"))
        .ok()
        .unwrap_or_else(|| panic!("Should put"));
    backend
        .put(&key2, Bytes::from("2"))
        .ok()
        .unwrap_or_else(|| panic!("Should put"));
    backend
        .put(&key3, Bytes::from("3"))
        .ok()
        .unwrap_or_else(|| panic!("Should put"));

    // List with prefix
    let keys = backend
        .list(&format!("{}/", prefix))
        .ok()
        .unwrap_or_else(|| panic!("Should list"));

    assert_eq!(keys.len(), 3);
    assert!(keys.contains(&key1));
    assert!(keys.contains(&key2));
    assert!(keys.contains(&key3));

    // Cleanup
    let _ = backend.delete(&key1);
    let _ = backend.delete(&key2);
    let _ = backend.delete(&key3);
}

#[test]
#[ignore = "Requires MinIO running"]
fn test_s3_backend_delete() {
    let backend = create_test_backend()
        .ok()
        .unwrap_or_else(|| panic!("Should create S3 backend"));

    let key = unique_key("delete_test");

    backend
        .put(&key, Bytes::from("to delete"))
        .ok()
        .unwrap_or_else(|| panic!("Should put"));

    assert!(backend.exists(&key).ok().unwrap_or(false));

    backend
        .delete(&key)
        .ok()
        .unwrap_or_else(|| panic!("Should delete"));

    assert!(!backend.exists(&key).ok().unwrap_or(true));
}

#[test]
#[ignore = "Requires MinIO running"]
fn test_s3_registry_workflow() {
    let backend = create_test_backend()
        .ok()
        .unwrap_or_else(|| panic!("Should create S3 backend"));

    let prefix = unique_key("registry_test");
    let index_path = format!("{}/registry-index.json", prefix);

    let registry = Registry::with_index_path(Box::new(backend), &index_path);

    // Initialize
    registry
        .init()
        .ok()
        .unwrap_or_else(|| panic!("Should init"));

    // Create and publish dataset
    let dataset = create_test_dataset(50);
    let metadata = DatasetMetadata {
        description: "S3 integration test dataset".to_string(),
        license: "MIT".to_string(),
        tags: vec!["test".to_string(), "s3".to_string()],
        source: None,
        citation: None,
        sha256: None,
    };

    registry
        .publish("s3-test-data", "1.0.0", &dataset, metadata)
        .ok()
        .unwrap_or_else(|| panic!("Should publish"));

    // List datasets
    let list = registry
        .list()
        .ok()
        .unwrap_or_else(|| panic!("Should list"));

    assert_eq!(list.len(), 1);
    assert_eq!(list[0].name, "s3-test-data");

    // Pull dataset
    let pulled = registry
        .pull("s3-test-data", Some("1.0.0"))
        .ok()
        .unwrap_or_else(|| panic!("Should pull"));

    assert_eq!(pulled.len(), 50);

    // Search
    let results = registry
        .search("integration")
        .ok()
        .unwrap_or_else(|| panic!("Should search"));

    assert_eq!(results.len(), 1);

    // Search by tags
    let results = registry
        .search_tags(&["s3"])
        .ok()
        .unwrap_or_else(|| panic!("Should search tags"));

    assert_eq!(results.len(), 1);

    // Note: We don't clean up S3 in tests - bucket should be ephemeral
    // or cleaned up separately
}

#[test]
#[ignore = "Requires MinIO running"]
fn test_s3_registry_versioning() {
    let backend = create_test_backend()
        .ok()
        .unwrap_or_else(|| panic!("Should create S3 backend"));

    let prefix = unique_key("version_test");
    let index_path = format!("{}/registry-index.json", prefix);

    let registry = Registry::with_index_path(Box::new(backend), &index_path);
    registry
        .init()
        .ok()
        .unwrap_or_else(|| panic!("Should init"));

    let metadata = DatasetMetadata {
        description: "Versioned dataset".to_string(),
        license: "MIT".to_string(),
        tags: vec![],
        source: None,
        citation: None,
        sha256: None,
    };

    // Publish v1
    let dataset_v1 = create_test_dataset(10);
    registry
        .publish("versioned-data", "1.0.0", &dataset_v1, metadata.clone())
        .ok()
        .unwrap_or_else(|| panic!("Should publish v1"));

    // Publish v2
    let dataset_v2 = create_test_dataset(20);
    registry
        .publish("versioned-data", "2.0.0", &dataset_v2, metadata)
        .ok()
        .unwrap_or_else(|| panic!("Should publish v2"));

    // Check info
    let info = registry
        .get_info("versioned-data")
        .ok()
        .unwrap_or_else(|| panic!("Should get info"));

    assert_eq!(info.versions.len(), 2);
    assert_eq!(info.latest, "2.0.0");

    // Pull specific version
    let v1 = registry
        .pull("versioned-data", Some("1.0.0"))
        .ok()
        .unwrap_or_else(|| panic!("Should pull v1"));
    assert_eq!(v1.len(), 10);

    // Pull latest
    let latest = registry
        .pull("versioned-data", None)
        .ok()
        .unwrap_or_else(|| panic!("Should pull latest"));
    assert_eq!(latest.len(), 20);
}

#[test]
#[ignore = "Requires MinIO running"]
fn test_s3_large_dataset() {
    let backend = create_test_backend()
        .ok()
        .unwrap_or_else(|| panic!("Should create S3 backend"));

    let prefix = unique_key("large_test");
    let index_path = format!("{}/registry-index.json", prefix);

    let registry = Registry::with_index_path(Box::new(backend), &index_path);
    registry
        .init()
        .ok()
        .unwrap_or_else(|| panic!("Should init"));

    // Create larger dataset (10k rows)
    let dataset = create_test_dataset(10_000);

    let metadata = DatasetMetadata {
        description: "Large dataset test".to_string(),
        license: "MIT".to_string(),
        tags: vec!["large".to_string()],
        source: None,
        citation: None,
        sha256: None,
    };

    // Publish
    registry
        .publish("large-data", "1.0.0", &dataset, metadata)
        .ok()
        .unwrap_or_else(|| panic!("Should publish large dataset"));

    // Pull and verify
    let pulled = registry
        .pull("large-data", Some("1.0.0"))
        .ok()
        .unwrap_or_else(|| panic!("Should pull large dataset"));

    assert_eq!(pulled.len(), 10_000);

    // Verify data integrity
    let batch = pulled.get_batch(0);
    assert!(batch.is_some());
}

#[test]
#[ignore = "Requires MinIO running"]
fn test_s3_backend_bucket_name() {
    let backend = create_test_backend()
        .ok()
        .unwrap_or_else(|| panic!("Should create S3 backend"));

    assert_eq!(backend.bucket(), TEST_BUCKET);
}

#[test]
#[ignore = "Requires MinIO running"]
fn test_s3_registry_delete_version() {
    let backend = create_test_backend()
        .ok()
        .unwrap_or_else(|| panic!("Should create S3 backend"));

    let prefix = unique_key("delete_version_test");
    let index_path = format!("{}/registry-index.json", prefix);

    let registry = Registry::with_index_path(Box::new(backend), &index_path);
    registry
        .init()
        .ok()
        .unwrap_or_else(|| panic!("Should init"));

    let dataset = create_test_dataset(5);
    let metadata = DatasetMetadata::with_description("Delete test");

    // Publish two versions
    registry
        .publish("delete-test", "1.0.0", &dataset, metadata.clone())
        .ok()
        .unwrap_or_else(|| panic!("Should publish"));

    registry
        .publish("delete-test", "2.0.0", &dataset, metadata)
        .ok()
        .unwrap_or_else(|| panic!("Should publish"));

    // Delete v1
    registry
        .delete("delete-test", "1.0.0")
        .ok()
        .unwrap_or_else(|| panic!("Should delete"));

    // Verify v1 is gone
    let info = registry
        .get_info("delete-test")
        .ok()
        .unwrap_or_else(|| panic!("Should get info"));

    assert!(!info.versions.contains(&"1.0.0".to_string()));
    assert!(info.versions.contains(&"2.0.0".to_string()));
}
