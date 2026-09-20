// A test asserts; `expect` here names the malformed fixture in the failure
// message, which is the diagnosis. The crate bans it in library code
// (Cargo.toml [lints.clippy]), and lib.rs already allows it under `cfg(test)`
// — an integration target is a separate crate, so it says so itself.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

//! `TRAITS` is upstream's table, or this test is RED.
//!
//! PMAT-3430, operator ruling 2026-09-20 clause 3: every live row's
//! `blck_size` / `type_size` equals the value extracted from ggml at the pinned
//! comparator commit, the extraction output is a checked-in fixture, and
//! mutating any one row turns this test RED.
//!
//! The fixture is `fixtures/ggml_traits.json`, written by
//! `scripts/extract_ggml_traits.sh` — which compiles a probe against the pinned
//! checkout's own libggml-base rather than parsing `sizeof(block_q4_1)` out of
//! a C initialiser by hand. Every in-tree table that had these numbers wrong
//! got them wrong by hand.
//!
//! WHAT THIS TEST DOES NOT CHECK: that the fixture's sha is still the pin. That
//! is a property of the repo, not of this crate (a packaged crate has no
//! `scripts/llama_pin.toml`), and it lives in
//! `crates/aprender-core/tests/monorepo_invariants.rs`.

use serde_json::Value;
use trueno_quant::{GgmlFamily, GgmlType, GgmlTypeError, ALL, GGML_TYPE_COUNT, TRAITS};

const FIXTURE: &str = include_str!("../fixtures/ggml_traits.json");

fn fixture() -> Value {
    serde_json::from_str(FIXTURE).expect("fixtures/ggml_traits.json is not valid JSON")
}

fn rows(fx: &Value) -> Vec<Value> {
    fx["types"]
        .as_array()
        .expect("fixture has no `types` array")
        .clone()
}

#[test]
fn traits_table_equals_the_upstream_fixture_row_for_row() {
    let fx = fixture();
    let rows = rows(&fx);
    assert_eq!(
        rows.len(),
        TRAITS.len(),
        "the fixture has {} ids and TRAITS has {}",
        rows.len(),
        TRAITS.len()
    );

    for row in &rows {
        let id = row["id"].as_u64().expect("row without an id") as usize;
        let traits = &TRAITS[id];
        let live = row["status"] == "live";

        assert_eq!(
            traits.name,
            row["enum_name"].as_str().expect("row without enum_name"),
            "id {id}: name disagrees with upstream"
        );
        assert_eq!(
            traits.ggml_name,
            row["ggml_name"].as_str().expect("row without ggml_name"),
            "id {id}: ggml_name disagrees with upstream"
        );
        assert_eq!(
            u64::from(traits.blck_size),
            row["blck_size"].as_u64().expect("row without blck_size"),
            "id {id} ({}): blck_size disagrees with upstream",
            traits.name
        );
        assert_eq!(
            u64::from(traits.type_size),
            row["type_size"].as_u64().expect("row without type_size"),
            "id {id} ({}): type_size disagrees with upstream",
            traits.name
        );
        assert_eq!(
            traits.family == GgmlFamily::Removed,
            !live,
            "id {id} ({}): family Removed disagrees with the fixture's status",
            traits.name
        );
    }
}

#[test]
fn the_family_classification_agrees_with_upstreams_is_quantized_flag() {
    // The one link between aprender's family column and upstream that can fail:
    // ggml has no families, but it does say which types are quantized.
    for row in rows(&fixture()) {
        let id = row["id"].as_u64().expect("row without an id") as usize;
        let quantized = row["is_quantized"]
            .as_bool()
            .expect("row without is_quantized");
        let family = TRAITS[id].family;
        let expected = !matches!(
            family,
            GgmlFamily::Float | GgmlFamily::Int | GgmlFamily::Removed
        );
        assert_eq!(
            expected, quantized,
            "id {id} ({}): family {family:?} implies is_quantized {expected}, upstream says {quantized}",
            TRAITS[id].name
        );
    }
}

#[test]
fn live_and_removed_counts_are_upstreams() {
    let fx = fixture();
    assert_eq!(
        fx["ggml_type_count"].as_u64(),
        Some(u64::from(GGML_TYPE_COUNT))
    );
    assert_eq!(fx["live_count"].as_u64(), Some(ALL.len() as u64));
    assert_eq!(
        fx["removed_count"].as_u64(),
        Some((TRAITS.len() - ALL.len()) as u64)
    );
}

#[test]
fn every_live_id_round_trips_and_no_removed_id_is_constructible() {
    let fx = fixture();
    for row in rows(&fx) {
        let id = row["id"].as_u64().expect("row without an id") as u32;
        let name = row["enum_name"].as_str().expect("row without enum_name");
        if row["status"] == "live" {
            let t = GgmlType::from_id(id)
                .unwrap_or_else(|| panic!("live id {id} ({name}) is not constructible"));
            assert_eq!(t.as_id(), id);
            assert_eq!(u32::from(t.as_byte()), id);
            assert_eq!(t.as_str(), name);
            assert!(GgmlType::try_from_id(id).is_ok());
        } else {
            assert!(
                GgmlType::from_id(id).is_none(),
                "removed id {id} ({name}) is constructible"
            );
            assert_eq!(
                GgmlType::try_from_id(id),
                Err(GgmlTypeError::Removed {
                    id,
                    name: TRAITS[id as usize].name
                }),
                "removed id {id} must be reported as removed, not unknown"
            );
        }
    }
    // Off the end of the table entirely.
    assert_eq!(
        GgmlType::try_from_id(GGML_TYPE_COUNT),
        Err(GgmlTypeError::Unknown {
            id: GGML_TYPE_COUNT
        })
    );
    assert_eq!(
        GgmlType::try_from_id(9999),
        Err(GgmlTypeError::Unknown { id: 9999 })
    );
}

#[test]
fn the_two_names_differ_in_case_only() {
    for t in ALL {
        assert!(
            t.as_str().eq_ignore_ascii_case(t.ggml_name()),
            "{}: GGUF name and ggml name are not the same string modulo case ({} vs {})",
            t.as_id(),
            t.as_str(),
            t.ggml_name()
        );
    }
}

#[test]
fn sizes_are_a_whole_number_of_bytes_per_block_and_tensor_bytes_agrees() {
    for t in ALL {
        assert!(t.block_size() > 0, "{t} has no block size");
        assert!(t.block_bytes() > 0, "{t} has no block bytes");
        // One full block, and a partial one.
        assert_eq!(t.tensor_bytes(t.block_size()), t.block_bytes());
        assert_eq!(
            t.checked_tensor_bytes(t.block_size()),
            Some(t.block_bytes())
        );
        assert_eq!(t.tensor_bytes(0), 0);
        if t.block_size() > 1 {
            // tensor_bytes ROUNDS UP; checked_tensor_bytes REFUSES. That
            // difference is the point of having both.
            assert_eq!(t.tensor_bytes(1), t.block_bytes());
            assert_eq!(t.checked_tensor_bytes(1), None);
        }
    }
}

/// The 16 ids and strings serve's `GgmlQuantType` carried before M1, asserted
/// here so the leaf cannot quietly change what they mean. Not the admitted set
/// of any parser — that stays each crate's own decision (#3430 Q1-c).
#[test]
fn serves_sixteen_strings_still_mean_what_they_meant() {
    let serve: [(u32, &str); 16] = [
        (0, "F32"),
        (1, "F16"),
        (2, "Q4_0"),
        (3, "Q4_1"),
        (6, "Q5_0"),
        (7, "Q5_1"),
        (8, "Q8_0"),
        (9, "Q8_1"),
        (10, "Q2_K"),
        (11, "Q3_K"),
        (12, "Q4_K"),
        (13, "Q5_K"),
        (14, "Q6_K"),
        (16, "IQ2_XXS"),
        (17, "IQ2_XS"),
        (30, "BF16"),
    ];
    for (id, name) in serve {
        let t = GgmlType::from_id(id).expect("serve's id must still be live");
        assert_eq!(t.as_str(), name, "id {id} renamed");
        assert_eq!(
            format!("{t}"),
            name,
            "id {id} Displays differently from as_str"
        );
        assert_eq!(
            GgmlType::from_str_lossy(name),
            Some(t),
            "{name} no longer parses"
        );
        assert_eq!(
            GgmlType::from_str_lossy(&name.to_ascii_lowercase()),
            Some(t),
            "{name} no longer parses in lower case"
        );
    }
    // Mixed case was rejected before and is still rejected — including ggml's
    // own spelling of the K-quants.
    assert_eq!(GgmlType::from_str_lossy("q4_K"), None);
    assert_eq!(GgmlType::from_str_lossy("Q4_k"), None);
    assert_eq!(GgmlType::from_str_lossy("nonsense"), None);
    assert_eq!(GgmlType::from_str_lossy(""), None);
}

/// compute's `block_bytes` / `block_size` for its 15 ids, from before M1.
#[test]
fn computes_fifteen_sizes_are_unchanged() {
    let compute: [(u32, usize, usize); 15] = [
        (0, 1, 4),
        (1, 1, 2),
        (2, 32, 18),
        (3, 32, 20),
        (6, 32, 22),
        (7, 32, 24),
        (8, 32, 34),
        (9, 32, 36),
        (10, 256, 84),
        (11, 256, 110),
        (12, 256, 144),
        (13, 256, 176),
        (14, 256, 210),
        (15, 256, 292),
        (30, 1, 2),
    ];
    for (id, block_size, block_bytes) in compute {
        let t = GgmlType::from_id(id).expect("compute's id must still be live");
        assert_eq!(t.block_size(), block_size, "{t} block_size changed");
        assert_eq!(t.block_bytes(), block_bytes, "{t} block_bytes changed");
    }
}
