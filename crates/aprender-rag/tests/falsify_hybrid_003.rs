//! FALSIFY-HYBRID-003 — `BM25Index` accepts an injected
//! `Tokenizer` trait object via `with_tokenizer()`. The trait is
//! public and reusable by future callers (notably a shared
//! inference path).
//!
//! Contract: `contracts/apr-hybrid-retrieval-v1.yaml` v1.3.0.
//!
//! Discharge strategy: build TWO `BM25Index` instances over the
//! exact same chunk content — one with the built-in tokenizer
//! (the default), one with a custom `MarkerTokenizer` that always
//! emits a fixed synthetic token. The two indexes are observably
//! different at the inverted-index-keys level (the load-bearing
//! evidence that the indexer consulted the tokenizer during
//! `add()`). Searching the index path is NOT useful evidence
//! because both `add()` and `search()` flow through the same
//! `tokenize()` method — so a regression that bypassed the
//! override on `add()` would also bypass it on `search()`, and
//! the round-trip would still appear consistent. Only inspecting
//! the indexed terms after `add()` separates the two paths
//! cleanly.

#![allow(clippy::unwrap_used)]

use aprender_rag::index::{BM25Index, SparseIndex};
use aprender_rag::tokenizer::Tokenizer;
use aprender_rag::{Chunk, DocumentId};
use std::sync::Arc;

/// Test-only tokenizer that ALWAYS emits a fixed token regardless of
/// input. Lets the falsifier prove the BM25 indexer is consulting
/// the override (the built-in tokenizer would emit content-derived
/// terms instead).
#[derive(Debug)]
struct MarkerTokenizer {
    marker: String,
}

impl Tokenizer for MarkerTokenizer {
    fn tokenize(&self, _text: &str) -> Vec<String> {
        vec![self.marker.clone()]
    }
}

#[test]
fn bm25_uses_injected_tokenizer() {
    let marker = "FALSIFY_HYBRID_003_MARKER";
    let tok: Arc<dyn Tokenizer> = Arc::new(MarkerTokenizer { marker: marker.to_string() });

    // Build two indexes over the same content; one with the
    // built-in tokenizer, one with the injected marker tokenizer.
    let mut default_index = BM25Index::new();
    let mut marker_index = BM25Index::new().with_tokenizer(Arc::clone(&tok));

    assert!(!default_index.has_custom_tokenizer());
    assert!(
        marker_index.has_custom_tokenizer(),
        "BM25Index::has_custom_tokenizer() must report true after \
         with_tokenizer() — the override flag is what \
         downstream consumers (and this gate) use to confirm the \
         injected path is active",
    );

    let chunk =
        Chunk::new(DocumentId::new(), "important content with searchable words".to_string(), 0, 38);
    default_index.add(&chunk);
    marker_index.add(&chunk);

    // The two indexes must hold OBSERVABLY DIFFERENT keys, proving
    // the marker_index's `add()` consulted the injected tokenizer
    // (which emits the marker) rather than the built-in (which
    // emits content-derived terms).
    let default_terms: Vec<&str> = default_index.indexed_terms();
    let marker_terms: Vec<&str> = marker_index.indexed_terms();

    assert!(
        default_terms.iter().any(|t| *t == "important")
            && default_terms.iter().any(|t| *t == "content"),
        "fixture sanity: built-in tokenizer should index 'important' \
         and 'content' from the chunk content; got {default_terms:?}",
    );

    assert_eq!(
        marker_terms,
        vec![marker],
        "FALSIFY-HYBRID-003: marker_index's inverted-index keys must \
         be exactly [{marker:?}] — proving `add()` consulted the \
         injected tokenizer, not the built-in. Got {marker_terms:?}.",
    );
}

#[test]
fn bm25_default_constructor_has_no_custom_tokenizer() {
    // Sanity: the override is opt-in. A fresh BM25Index uses the
    // built-in tokenizer (lowercase + word-boundary + stopwords).
    let index = BM25Index::new();
    assert!(!index.has_custom_tokenizer());
    let toks = index.tokenize("Hello, World!");
    // Built-in tokenizer lowercases and splits on non-alphanumeric.
    assert!(toks.iter().any(|t| t == "hello"));
    assert!(toks.iter().any(|t| t == "world"));
}

#[test]
fn tokenizer_trait_is_public_and_reusable() {
    // Structural assertion that the Tokenizer trait is part of the
    // crate's public API — a future inference tokenizer can implement
    // it from outside aprender-rag without forking the trait. This
    // is what the §2.5 sketch's "shared with inference path"
    // language requires; today the consumer is just our test
    // MarkerTokenizer, but the same surface takes a future Qwen /
    // Llama BPE tokenizer impl when one is wired.
    fn assert_object_safe<T: Tokenizer + ?Sized>(_: &T) {}
    let tok = MarkerTokenizer { marker: "x".to_string() };
    assert_object_safe(&tok);

    // Also: the type-id of MarkerTokenizer can be compared against
    // any other Tokenizer impl's type-id at runtime — exactly the
    // mechanism the §2.5 sketch's "type-id equals the inference
    // path's" language anticipates.
    let arc: Arc<dyn Tokenizer> = Arc::new(MarkerTokenizer { marker: "y".to_string() });
    let _ = arc.tokenize("test"); // proves trait-object dispatch works
}
