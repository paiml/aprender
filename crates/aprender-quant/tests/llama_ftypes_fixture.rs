// A test asserts; `expect`/`panic!` here name the malformed fixture row in the failure
// message, which is the diagnosis. An integration target is a separate crate, so it says
// so itself (as tests/ggml_traits_fixture.rs does).
#![allow(clippy::expect_used, clippy::panic)]
//! #3762: `LLAMA_FTYPES` equals the upstream-extracted fixture, row for row.
//!
//! The fixture is `fixtures/llama_ftypes.json`, written by
//! `scripts/extract_llama_ftypes.sh` from llama.cpp's `include/llama.h` at the pinned
//! comparator commit. That the fixture's sha is still the pin is checked in
//! `crates/aprender-core/tests/monorepo_invariants.rs`, beside the ggml traits fixture.

use serde_json::Value;
use trueno_quant::{llama_ftype_name, LLAMA_FTYPES};

const FIXTURE: &str = include_str!("../fixtures/llama_ftypes.json");

fn rows() -> Vec<Value> {
    let fx: Value =
        serde_json::from_str(FIXTURE).expect("fixtures/llama_ftypes.json is not valid JSON");
    fx["ftypes"]
        .as_array()
        .expect("fixture has no `ftypes` array")
        .clone()
}

fn field<'a>(row: &'a Value, key: &str) -> &'a Value {
    row.get(key)
        .unwrap_or_else(|| panic!("fixture row {row} has no `{key}`"))
}

#[test]
fn llama_ftypes_equals_the_upstream_fixture_live_rows() {
    let live: Vec<(u32, String)> = rows()
        .iter()
        .filter(|r| field(r, "status") == "live")
        .map(|r| {
            let id = field(r, "id").as_u64().expect("numeric id") as u32;
            (
                id,
                field(r, "name").as_str().expect("string name").to_string(),
            )
        })
        .collect();
    let table: Vec<(u32, String)> = LLAMA_FTYPES
        .iter()
        .map(|&(id, n)| (id, n.to_string()))
        .collect();
    assert_eq!(
        table, live,
        "LLAMA_FTYPES must be the fixture's live rows, in order"
    );
}

#[test]
fn removed_ids_and_the_guessed_sentinel_name_no_scheme() {
    for r in rows().iter().filter(|r| field(r, "status") != "live") {
        let id = field(r, "id").as_u64().expect("numeric id") as u32;
        assert_eq!(
            llama_ftype_name(id),
            None,
            "id {id} ({}) is not a live scheme",
            field(r, "enum_name")
        );
    }
    assert_eq!(llama_ftype_name(15), Some("Q4_K_M"));
    assert_eq!(llama_ftype_name(30), Some("IQ4_XS"));
    assert_eq!(llama_ftype_name(999), None);
}
