# Cloud Storage

Store artifacts in local filesystem, S3, Azure Blob Storage, or Google Cloud Storage with content-addressable hashing.

## Toyota Principle: Heijunka

Level the storage load across backends with consistent abstraction layer.

## Quick Start

```rust
use entrenar::storage::{LocalBackend, ArtifactBackend};

// Local filesystem backend
let backend = LocalBackend::new("./artifacts");

// Store artifact
let hash = backend.put("model.safetensors", &model_bytes)?;
println!("Stored with hash: {}", hash);

// Retrieve artifact
let data = backend.get(&hash)?;

// Check existence
if backend.exists(&hash)? {
    println!("Artifact exists");
}
```

## Supported Backends

### Local Filesystem

```rust
use entrenar::storage::LocalBackend;

let backend = LocalBackend::new("./artifacts");

// With subdirectory organization
let backend = LocalBackend::new("./artifacts")
    .with_subdirs(true);  // Organizes by hash prefix
```

### Amazon S3

```rust
use entrenar::storage::{S3Config, BackendConfig};

let config = S3Config {
    bucket: "my-ml-artifacts".to_string(),
    region: Some("us-east-1".to_string()),
    endpoint: None,  // Use default AWS endpoint
    access_key: Some("AKIA...".to_string()),
    secret_key: Some("secret...".to_string()),
};

// Use environment variables instead (recommended)
let config = S3Config {
    bucket: "my-ml-artifacts".to_string(),
    region: Some("us-east-1".to_string()),
    endpoint: None,
    access_key: None,  // Uses AWS_ACCESS_KEY_ID
    secret_key: None,  // Uses AWS_SECRET_ACCESS_KEY
};
```

### Azure Blob Storage

```rust
use entrenar::storage::AzureConfig;

let config = AzureConfig {
    container: "ml-artifacts".to_string(),
    account: "myaccount".to_string(),
    access_key: Some("key...".to_string()),
};

// Uses AZURE_STORAGE_KEY if access_key is None
```

### Google Cloud Storage

```rust
use entrenar::storage::GCSConfig;

let config = GCSConfig {
    bucket: "ml-artifacts".to_string(),
    project: Some("my-project".to_string()),
    credentials_path: Some("/path/to/credentials.json".to_string()),
};

// Uses GOOGLE_APPLICATION_CREDENTIALS if credentials_path is None
```

## Unified Backend Config

```rust
use entrenar::storage::BackendConfig;

// From environment or config file
let config = BackendConfig::from_env()?;

// Explicit backend selection
let config = BackendConfig::S3(S3Config {
    bucket: "my-bucket".to_string(),
    region: Some("us-east-1".to_string()),
    endpoint: None,
    access_key: None,
    secret_key: None,
});
```

## Content-Addressable Storage

All backends use SHA-256 content hashing:

```rust
use entrenar::storage::ArtifactBackend;

// Same content always gets same hash
let hash1 = backend.put("model1.safetensors", &data)?;
let hash2 = backend.put("model2.safetensors", &data)?;
assert_eq!(hash1, hash2);  // Deduplication!

// Retrieve by hash
let retrieved = backend.get(&hash1)?;
assert_eq!(data, retrieved);
```

## Artifact Metadata

```rust
use entrenar::storage::ArtifactMetadata;

let metadata = backend.metadata(&hash)?;

println!("Size: {} bytes", metadata.size_bytes);
println!("Created: {}", metadata.created_at);
println!("Content-Type: {}", metadata.content_type);
```

## Batch Operations

```rust
// Upload multiple artifacts
let artifacts = vec![
    ("model.safetensors", model_bytes),
    ("config.json", config_bytes),
    ("tokenizer.json", tokenizer_bytes),
];

for (name, data) in artifacts {
    let hash = backend.put(name, &data)?;
    println!("{}: {}", name, hash);
}

// List all artifacts
let hashes = backend.list()?;
for hash in hashes {
    println!("  {}", hash);
}
```

## Integration with Experiment Tracking

```rust
use entrenar::storage::{SqliteBackend, LocalBackend, ExperimentStorage};

let mut storage = SqliteBackend::open("experiments.db")?;
let artifact_backend = LocalBackend::new("./artifacts");

// Log artifact with content-addressable hash
let model_bytes = save_model(&model);
let hash = artifact_backend.put("model.safetensors", &model_bytes)?;
storage.log_artifact(&run_id, "model.safetensors", &model_bytes)?;

// Reference artifact in experiment metadata
```

## In-Memory Backend (Testing)

```rust
use entrenar::storage::InMemoryBackend;

let backend = InMemoryBackend::new();

// Use for unit tests
backend.put("test.bin", &[1, 2, 3, 4])?;
let data = backend.get(&hash)?;
```

## Mock S3 Backend (Testing)

```rust
use entrenar::storage::MockS3Backend;

let mock = MockS3Backend::new();

// Simulates S3 behavior without network calls
mock.put("model.safetensors", &model_bytes)?;
```

## Cargo Run Example

```bash
# Store artifact locally
cargo run --example store_artifact -- --path ./model.safetensors

# Store to S3
cargo run --example store_artifact -- \
    --backend s3 \
    --bucket my-bucket \
    --path ./model.safetensors

# List artifacts
cargo run --example list_artifacts -- --backend local --path ./artifacts
```

## Configuration File

```yaml
# entrenar.yaml
storage:
  artifacts:
    backend: s3
    bucket: my-ml-artifacts
    region: us-east-1
    prefix: experiments/
```

## Best Practices

1. **Use content-addressable storage** - Automatic deduplication
2. **Store credentials in environment** - Never in code
3. **Use local backend for development** - Fast iteration
4. **Enable versioning on S3** - Protect against accidental deletion
5. **Set lifecycle policies** - Archive old artifacts to Glacier

## Error Handling

```rust
use entrenar::storage::CloudError;

match backend.get(&hash) {
    Ok(data) => process(data),
    Err(CloudError::NotFound(h)) => {
        eprintln!("Artifact not found: {}", h);
    }
    Err(CloudError::AccessDenied) => {
        eprintln!("Permission denied");
    }
    Err(e) => {
        eprintln!("Storage error: {}", e);
    }
}
```

## See Also

- [MLOps Overview](./overview.md)
- [Experiment Tracking](./experiment-tracking.md)
- [Model Registry](./model-registry.md)
