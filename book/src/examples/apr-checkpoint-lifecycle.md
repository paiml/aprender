# APR Checkpoint Lifecycle

Demonstrates the full checkpoint save, load, filter, and validate cycle
per the APR Checkpoint Specification v1.3.0.

Exercises contracts:
- F-CKPT-002: Schema version present
- F-CKPT-003: Adapter has no `__training__.*` tensors
- F-CKPT-007: NaN/Inf rejected on checked read
- F-CKPT-009: Atomic writes (tmp+fsync+rename)
- F-CKPT-013: Post-load NaN scan
- F-CKPT-014: Shape-config validation
- F-CKPT-015: Canonical tensor ordering
- F-CKPT-016: Filtered reader skips training state
- F-CKPT-017: Provenance metadata present
- F-CKPT-018: Round-trip bit-identical

## Run

```bash
cargo run --example apr_checkpoint_lifecycle
```

## Source

```rust,ignore
// Run this example:
//   cargo run --example apr_checkpoint_lifecycle
//
// See the CLI reference and source code in crates/ for implementation details.
```
