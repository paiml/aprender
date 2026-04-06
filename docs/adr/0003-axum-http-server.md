# ADR-0003: Axum as Default HTTP Server with Swappable Backend

## Status

Accepted

## Date

2024-11-20

## Context

Realizar needs an HTTP server for its REST API. Requirements:

- High performance for inference workloads
- Async/await support for concurrent requests
- Easy to extend with middleware
- Optionally swappable for different deployment scenarios

Options considered:

1. **axum** - Tower-based, async, well-maintained
2. **actix-web** - Actor model, high performance
3. **hyper** - Low-level, maximum control
4. **warp** - Filter-based routing

## Decision

Use axum as the default HTTP server, with a trait-based interface allowing future backends.

## Rationale

1. **Tower ecosystem** - Middleware composability
2. **Type safety** - Compile-time route checking
3. **Performance** - Competitive with actix-web
4. **Active maintenance** - Tokio team involvement
5. **Swappable** - Trait interface allows alternatives

## Implementation

```rust
pub trait HttpServer: Send + Sync {
    fn serve(&self, addr: &str) -> Result<()>;
    fn routes(&self) -> Router;
}

pub struct AxumServer {
    model: Arc<Model>,
    config: ServerConfig,
}

impl HttpServer for AxumServer {
    fn serve(&self, addr: &str) -> Result<()> {
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async {
            let listener = TcpListener::bind(addr).await?;
            axum::serve(listener, self.routes()).await
        })
    }
}
```

## Consequences

### Positive
- Production-ready HTTP server out of the box
- Easy middleware integration (tracing, metrics)
- Can swap to hyper/actix-web if needed

### Negative
- axum + tokio add ~2MB to binary size
- Async runtime overhead for simple use cases
- Learning curve for Tower abstractions

## Alternatives

Future backends can implement the same trait:

```rust
pub struct HyperServer { /* ... */ }
impl HttpServer for HyperServer { /* ... */ }

pub struct ActixServer { /* ... */ }
impl HttpServer for ActixServer { /* ... */ }
```

## Validation

**Falsifiable claim**: axum-based server handles >10,000 requests/sec on commodity hardware (4-core, 8GB RAM).

**Test**: Load test with wrk or our Rust-based load test client.

## References

- [axum Documentation](https://docs.rs/axum)
- [Tower Service Trait](https://docs.rs/tower-service)
