# aprender-legacy-codec

A small serde codec that reads and writes the **bincode 1.x legacy layout** byte-for-byte
(fixed-width integers, little-endian, `u64` length prefixes, `u32` enum tags), without depending on
the unmaintained `bincode` crate (RUSTSEC-2025-0141).

It exists so `.apr` containers and RAG indexes written by older releases stay readable.

## Usage

```rust
use aprender_legacy_codec::{deserialize, serialize};

let bytes = serialize(&(1u8, "ab")).expect("encode");
let back: (u8, String) = deserialize(&bytes).expect("decode");
assert_eq!(back, (1, "ab".to_string()));
```

Entry points: `serialize`, `serialize_into`, `deserialize`, `deserialize_from`.

## Safety

Corrupt or hostile input returns an `Error`; an untrusted length prefix never drives allocation.
Golden fixtures written by `bincode = "=1.3.3"` pin the byte layout (`tests/byte_compat.rs`).

Part of the [Aprender monorepo](https://github.com/paiml/aprender) — a next-generation ML framework in pure Rust.

## License

MIT OR Apache-2.0
