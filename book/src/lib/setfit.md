<!-- PCU: lib-setfit | contract: contracts/apr-page-lib-setfit-v1.yaml -->

# Module: `aprender::setfit`

Public module of the `aprender-core` crate.

SetFit is few-shot text classification without prompt engineering: a sentence
encoder is fine-tuned on contrastive pairs built from a handful of labelled
examples, then a lightweight classification head is fitted over the resulting
embeddings.

## Source

[`crates/aprender-core/src/setfit.rs`](https://github.com/paiml/aprender/blob/main/crates/aprender-core/src/setfit.rs) or directory.

Pair construction lives in a separate crate,
[`aprender-contrastive-data`](https://github.com/paiml/aprender/tree/main/crates/aprender-contrastive-data),
which owns class buckets, balanced few-shot selection, bounded pair sampling and
the cross-split leakage checks. SetFit is that crate's first consumer, not its
owner.

## Example

<!-- example-cost: trivial -->
```rust
use aprender::setfit;
// See `cargo doc -p aprender-core --open` for full API reference.
```

## Full API

Run `cargo doc -p aprender-core --open` for the rendered rustdoc, or browse
[docs.rs/aprender](https://docs.rs/aprender) for the published version.
