# Pacha Protocol

Pacha is the resource loading protocol for the Sovereign AI Stack, handling data sources, models, and remote API connections.

## Overview

| Feature | Description |
|---------|-------------|
| Protocol | `pacha://` URI scheme |
| Local | Filesystem-based resource loading |
| Remote | HTTP/HTTPS with retry and caching |
| Formats | `.ald` (data), `.apr` (models), JSON, CSV |

## URI Format

```
pacha://[host:port]/path/to/resource[?query]
```

### Examples

```
pacha://data/metrics           # Local data file
pacha://models/classifier      # Local model file
pacha://localhost:8080/api/v1  # Local server
pacha://server.example.com/api # Remote server
```

## Resource Types

| Type | Path Prefix | File Extensions |
|------|-------------|-----------------|
| Data | `/data/` | `.ald`, `.json`, `.csv` |
| Model | `/models/`, `/model/` | `.apr` |
| API | `/api/` | HTTP endpoints |

## PachaUri Parsing

```rust
use presentar_yaml::pacha::{PachaUri, ResourceType};

// Parse a URI
let uri = PachaUri::parse("pacha://data/metrics?limit=100")?;

assert_eq!(uri.resource_type, ResourceType::Data);
assert_eq!(uri.path, "/data/metrics");
assert_eq!(uri.query.get("limit"), Some(&"100".to_string()));
assert!(uri.is_local());

// Remote URI
let remote = PachaUri::parse("pacha://server.example.com:8080/api/v1")?;
assert!(remote.is_remote());
assert_eq!(remote.host, Some("server.example.com".to_string()));
assert_eq!(remote.port, Some(8080));
```

## Local Resource Loading

```rust
use presentar_yaml::pacha::{PachaLoader, ContentType};
use std::path::PathBuf;

// Create loader with base directory
let mut loader = PachaLoader::new(PathBuf::from("./data"));

// Or use current directory
let mut loader = PachaLoader::current_dir();

// Load a resource (with caching)
let resource = loader.load("pacha://data/metrics")?;

println!("URI: {}", resource.uri);
println!("Content type: {:?}", resource.content_type);
println!("Size: {} bytes", resource.data.len());

// Load without caching
let fresh = loader.load_fresh("pacha://data/metrics")?;

// Check cache status
if loader.is_cached("pacha://data/metrics") {
    loader.clear_cache();
}
```

## Content Types

```rust
use presentar_yaml::pacha::ContentType;

// Detect from extension
assert_eq!(ContentType::from_extension("ald"), ContentType::Ald);
assert_eq!(ContentType::from_extension("apr"), ContentType::Apr);
assert_eq!(ContentType::from_extension("json"), ContentType::Json);
assert_eq!(ContentType::from_extension("csv"), ContentType::Csv);

// Get extension
assert_eq!(ContentType::Ald.extension(), "ald");
assert_eq!(ContentType::Json.extension(), "json");
```

## Remote Resource Loading

```rust
use presentar_yaml::pacha::{RemotePachaLoader, RetryConfig, HttpClient};

// Create with custom HTTP client
let loader = RemotePachaLoader::new(my_http_client)
    .with_retry(RetryConfig {
        max_attempts: 3,
        initial_delay_ms: 500,
        max_delay_ms: 10_000,
        backoff_multiplier: 2.0,
    })
    .with_cache_ttl(60_000);  // 1 minute cache

// Load remote resource
let resource = loader.load("pacha://api.example.com/data/metrics")?;
```

## HTTP Client Interface

```rust
use presentar_yaml::pacha::{HttpClient, HttpRequest, HttpResponse, PachaError};

// Implement for your platform
impl HttpClient for MyClient {
    fn request(&self, req: HttpRequest) -> Result<HttpResponse, PachaError> {
        // Make HTTP request...
    }
}

// Build requests
let request = HttpRequest::get("https://api.example.com/data")
    .with_header("Authorization", "Bearer token123")
    .with_header("Accept", "application/json")
    .with_timeout(30_000);

// Check responses
let response = client.request(request)?;
if response.is_success() {
    let content_type = response.detect_content_type();
    // Process response.body...
}
```

## Retry Configuration

```rust
use presentar_yaml::pacha::RetryConfig;

let config = RetryConfig::default();  // 3 attempts, 500ms initial delay

// Check retry behavior
assert!(config.should_retry(0));   // Can retry
assert!(config.should_retry(2));   // Can retry
assert!(!config.should_retry(3));  // Max reached

// Calculate delays (exponential backoff)
assert_eq!(config.delay_for_attempt(0), 0);     // No delay for first
assert_eq!(config.delay_for_attempt(1), 500);   // 500ms
assert_eq!(config.delay_for_attempt(2), 1000);  // 1000ms
assert_eq!(config.delay_for_attempt(3), 2000);  // 2000ms
```

## Refresh Intervals

Parse human-readable refresh intervals:

```rust
use presentar_yaml::pacha::parse_refresh_interval;

assert_eq!(parse_refresh_interval("1s"), Some(1000));
assert_eq!(parse_refresh_interval("30s"), Some(30_000));
assert_eq!(parse_refresh_interval("5m"), Some(300_000));
assert_eq!(parse_refresh_interval("1h"), Some(3_600_000));
assert_eq!(parse_refresh_interval("100ms"), Some(100));
assert_eq!(parse_refresh_interval("1.5s"), Some(1500));
```

## YAML Data Sources

```yaml
# app.yaml
data:
  metrics:
    source: "pacha://data/metrics"
    refresh: "30s"

  users:
    source: "pacha://api.example.com/users"
    refresh: "5m"

  model:
    source: "pacha://models/classifier"

layout:
  sections:
    - id: dashboard
      widgets:
        - type: Chart
          data: "{{ metrics | filter(active=true) }}"
```

## Error Handling

```rust
use presentar_yaml::pacha::PachaError;

match loader.load("pacha://data/missing") {
    Ok(resource) => { /* use resource */ }
    Err(PachaError::NotFound(path)) => {
        eprintln!("Resource not found: {}", path);
    }
    Err(PachaError::InvalidProtocol(uri)) => {
        eprintln!("Invalid URI: {}", uri);
    }
    Err(PachaError::ConnectionError(msg)) => {
        eprintln!("Connection failed: {}", msg);
    }
    Err(PachaError::IoError(msg)) => {
        eprintln!("IO error: {}", msg);
    }
    Err(PachaError::ParseError(msg)) => {
        eprintln!("Parse error: {}", msg);
    }
    Err(PachaError::UnsupportedFormat(fmt)) => {
        eprintln!("Unsupported format: {}", fmt);
    }
}
```

## Local Path Resolution

```rust
use presentar_yaml::pacha::PachaUri;
use std::path::Path;

let uri = PachaUri::parse("pacha://data/metrics")?;

// Resolve to local filesystem path
let path = uri.to_local_path(Path::new("/app"));
assert_eq!(path, PathBuf::from("/app/data/metrics"));

// Loader tries multiple extensions if none specified:
// /app/data/metrics.ald
// /app/data/metrics.apr
// /app/data/metrics.json
// /app/data/metrics.csv
// /app/data/metrics
```

## WASM Integration

In WASM environments, use platform-specific HTTP clients:

```rust
#[cfg(target_arch = "wasm32")]
mod wasm {
    use presentar_yaml::pacha::{HttpClient, HttpRequest, HttpResponse, PachaError};

    pub struct WebFetchClient;

    impl HttpClient for WebFetchClient {
        fn request(&self, req: HttpRequest) -> Result<HttpResponse, PachaError> {
            // Use web_sys::fetch or similar
            todo!("Implement WASM fetch")
        }
    }
}
```

## Verified Test

```rust
#[test]
fn test_pacha_uri_parsing() {
    use presentar_yaml::pacha::{PachaUri, ResourceType};

    // Local data URI
    let uri = PachaUri::parse("pacha://data/metrics").unwrap();
    assert_eq!(uri.resource_type, ResourceType::Data);
    assert!(uri.is_local());
    assert_eq!(uri.path, "/data/metrics");

    // Model URI
    let model = PachaUri::parse("pacha://models/classifier").unwrap();
    assert_eq!(model.resource_type, ResourceType::Model);

    // Remote API
    let remote = PachaUri::parse("pacha://server.com:8080/api/v1?key=value").unwrap();
    assert!(remote.is_remote());
    assert_eq!(remote.host, Some("server.com".to_string()));
    assert_eq!(remote.port, Some(8080));
    assert_eq!(remote.query.get("key"), Some(&"value".to_string()));
}

#[test]
fn test_refresh_intervals() {
    use presentar_yaml::pacha::parse_refresh_interval;

    assert_eq!(parse_refresh_interval("1s"), Some(1000));
    assert_eq!(parse_refresh_interval("5m"), Some(300_000));
    assert_eq!(parse_refresh_interval("1h"), Some(3_600_000));
    assert_eq!(parse_refresh_interval("invalid"), None);
}
```
