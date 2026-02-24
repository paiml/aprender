//! Entity Extraction Contract Falsification Tests
//!
//! Popperian falsification of NLP spec §2.1.7 claims:
//!   - Email extraction matches RFC 5322 basic patterns
//!   - URL extraction matches http/https URLs
//!   - Mention extraction matches @-prefixed tokens
//!   - Hashtag extraction matches #-prefixed tokens
//!   - Empty input produces empty entities
//!   - Extraction is deterministic
//!
//! Five-Whys (PMAT-352):
//!   Why #1: entities module has unit tests but zero FALSIFY-ENT-* tests
//!   Why #2: unit tests check example patterns, not RFC compliance contracts
//!   Why #3: no provable-contract YAML for entity extraction
//!   Why #4: entities module was built before DbC methodology
//!   Why #5: no systematic verification of pattern matching correctness
//!
//! References:
//!   - docs/specifications/nlp-models-techniques-spec.md §2.1.7
//!   - src/text/entities.rs

use super::*;

// ============================================================================
// FALSIFY-ENT-001: Email extraction
// Contract: well-formed emails are extracted, non-emails are not
// ============================================================================

#[test]
fn falsify_ent_001_email_extraction() {
    let extractor = EntityExtractor::new();

    let text = "Contact user@example.com or admin@test.org for info";
    let entities = extractor.extract(text).expect("extract");

    assert!(
        entities.emails.len() >= 2,
        "FALSIFIED ENT-001: expected >= 2 emails, got {:?}",
        entities.emails
    );
    assert!(
        entities
            .emails
            .iter()
            .any(|e| e.contains("user@example.com")),
        "FALSIFIED ENT-001: 'user@example.com' not extracted"
    );
}

#[test]
fn falsify_ent_001_no_false_email_positives() {
    let extractor = EntityExtractor::new();

    let text = "hello world no emails here at all";
    let entities = extractor.extract(text).expect("extract");

    assert!(
        entities.emails.is_empty(),
        "FALSIFIED ENT-001: false positive emails in '{}': {:?}",
        text,
        entities.emails
    );
}

// ============================================================================
// FALSIFY-ENT-002: URL extraction
// Contract: http/https URLs are extracted
// ============================================================================

#[test]
fn falsify_ent_002_url_extraction() {
    let extractor = EntityExtractor::new();

    let text = "Visit https://example.com or http://test.org for more";
    let entities = extractor.extract(text).expect("extract");

    assert!(
        !entities.urls.is_empty(),
        "FALSIFIED ENT-002: no URLs extracted from '{}'",
        text
    );
}

// ============================================================================
// FALSIFY-ENT-003: Mention extraction
// Contract: @-prefixed tokens are extracted as mentions
// ============================================================================

#[test]
fn falsify_ent_003_mention_extraction() {
    let extractor = EntityExtractor::new();

    let text = "Hey @alice and @bob check this out";
    let entities = extractor.extract(text).expect("extract");

    assert!(
        entities.mentions.len() >= 2,
        "FALSIFIED ENT-003: expected >= 2 mentions, got {:?}",
        entities.mentions
    );
}

// ============================================================================
// FALSIFY-ENT-004: Hashtag extraction
// Contract: #-prefixed tokens are extracted as hashtags
// ============================================================================

#[test]
fn falsify_ent_004_hashtag_extraction() {
    let extractor = EntityExtractor::new();

    let text = "Trending #rust #programming today";
    let entities = extractor.extract(text).expect("extract");

    assert!(
        entities.hashtags.len() >= 2,
        "FALSIFIED ENT-004: expected >= 2 hashtags, got {:?}",
        entities.hashtags
    );
}

// ============================================================================
// FALSIFY-ENT-005: Empty input
// Contract: extract("") returns empty entities
// ============================================================================

#[test]
fn falsify_ent_005_empty_input() {
    let extractor = EntityExtractor::new();

    let entities = extractor.extract("").expect("extract empty");

    assert!(
        entities.is_empty(),
        "FALSIFIED ENT-005: empty input produced entities: count={}",
        entities.total_count()
    );
}

// ============================================================================
// FALSIFY-ENT-006: Extraction determinism
// Contract: same input always produces same entities
// ============================================================================

#[test]
fn falsify_ent_006_extraction_determinism() {
    let extractor = EntityExtractor::new();
    let text = "Contact user@test.com, visit https://example.com, mention @alice, tag #rust";

    let e1 = extractor.extract(text).expect("first");
    let e2 = extractor.extract(text).expect("second");

    assert_eq!(e1, e2, "FALSIFIED ENT-006: extraction is non-deterministic");
}
