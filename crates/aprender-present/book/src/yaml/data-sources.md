# Data Sources

Connecting to data in YAML manifests.

## Source Types

| Type | Scheme | Example |
|------|--------|---------|
| File | `.ald` | `users.ald` |
| HTTP | `http://` | `api/metrics` |
| WebSocket | `ws://` | `ws://live` |
| Inline | n/a | `[1, 2, 3]` |

## File Source

```yaml
data:
  users:
    source: "data/users.ald"
```

## HTTP Source

```yaml
data:
  metrics:
    source: "https://api.example.com/metrics"
    refresh: 30s
    headers:
      Authorization: "Bearer {{ token }}"
```

## WebSocket Source

```yaml
data:
  live_feed:
    source: "ws://localhost:8080/events"
    on_message:
      action: append
      target: events
```

## Inline Data

```yaml
data:
  options:
    inline:
      - { id: 1, label: "Option A" }
      - { id: 2, label: "Option B" }
```

## Refresh Options

| Option | Description |
|--------|-------------|
| `refresh: 30s` | Poll every 30 seconds |
| `refresh: 1m` | Poll every minute |
| `on_demand` | Manual refresh only |

## Transforms

```yaml
data:
  active_users:
    source: "users.ald"
    transform: "filter(active=true) | sort(name)"
```

## Caching

```yaml
data:
  large_dataset:
    source: "big_data.ald"
    cache: true
    cache_ttl: 5m
```

## Verified Test

```rust
#[test]
fn test_data_source_parsing() {
    // Parse data source configuration
    #[derive(Debug)]
    struct DataSource {
        source: String,
        refresh_seconds: Option<u64>,
    }

    fn parse_refresh(s: &str) -> Option<u64> {
        if s.ends_with('s') {
            s.trim_end_matches('s').parse().ok()
        } else if s.ends_with('m') {
            s.trim_end_matches('m').parse::<u64>().ok().map(|m| m * 60)
        } else {
            None
        }
    }

    assert_eq!(parse_refresh("30s"), Some(30));
    assert_eq!(parse_refresh("1m"), Some(60));
    assert_eq!(parse_refresh("5m"), Some(300));

    // Source type detection
    fn source_type(source: &str) -> &'static str {
        if source.starts_with("ws://") { "websocket" }
        else if source.starts_with("http") { "http" }
        else if source.ends_with(".ald") { "file" }
        else { "unknown" }
    }

    assert_eq!(source_type("users.ald"), "file");
    assert_eq!(source_type("https://api.example.com"), "http");
    assert_eq!(source_type("ws://live"), "websocket");
}
```
