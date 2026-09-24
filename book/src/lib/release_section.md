<!-- PCU: lib-release_section | contract: contracts/apr-page-lib-release_section-v1.yaml -->

# Module: `aprender::release_section`

Public module of the `aprender-core` crate.

The README's release-verification matrix, rendered from the model-ladder receipts
(`evidence/dogfood/models/<version>/<host>.json`) rather than typed by hand (#3769).
`render` turns a set of `Receipt`s (each carrying its `Rung`s) into the Markdown section;
the committed README section is gated to equal that render.

## Source

[`crates/aprender-core/src/release_section.rs`](https://github.com/paiml/aprender/blob/main/crates/aprender-core/src/release_section.rs) or directory.

## Example

<!-- example-cost: trivial -->
```rust
use aprender::release_section::{render, Receipt, Rung};
// See `cargo doc -p aprender-core --open` for full API reference.
```

## Full API

Run `cargo doc -p aprender-core --open` for the rendered rustdoc, or browse
[docs.rs/aprender](https://docs.rs/aprender) for the published version.
