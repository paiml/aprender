//! JSON object construction that does not go through `serde_json::json!`.
//!
//! `serde_json::json!` expands to `Result::unwrap` internally, and this repo
//! bans unwrap via `.clippy.toml` disallowed-methods (GH-41). The ban fired on
//! the `pv` command surface only after that surface moved from `main.rs` into
//! `lib.rs` — the diagnostics were real the whole time, just charged to a bin
//! target nothing linted.
//!
//! The usual fix elsewhere in this workspace is a local
//! `#[derive(serde::Serialize)]` struct (see
//! `aprender-qa-cli/src/main_tickets_and_parity.rs::save_tool_results_json_or_exit`).
//! `aprender-contracts-cli` depends on `serde_json` but not on `serde` itself,
//! so it builds the `serde_json::Map` directly instead. That is the same
//! representation `json!` produces — identical keys, identical key ordering
//! (`Map` is the one canonical map type either way), identical nesting — with
//! no unwrap anywhere.

use serde_json::{Map, Value};

/// Build a JSON object from an ordered list of `(key, value)` pairs.
///
/// Drop-in replacement for `serde_json::json!({ "k": v, ... })` where every
/// key is a literal.
pub fn obj<I>(pairs: I) -> Value
where
    I: IntoIterator<Item = (&'static str, Value)>,
{
    Value::Object(
        pairs
            .into_iter()
            .map(|(k, v)| (k.to_owned(), v))
            .collect::<Map<String, Value>>(),
    )
}

#[cfg(test)]
mod tests {
    use super::obj;
    use serde_json::Value;

    #[test]
    fn obj_builds_the_same_shape_as_the_json_macro() {
        let built = obj([
            ("name", Value::from("x")),
            ("count", Value::from(3usize)),
            ("ok", Value::from(true)),
        ]);
        let parsed: Value = serde_json::from_str(r#"{"name":"x","count":3,"ok":true}"#)
            .expect("literal is valid JSON");
        assert_eq!(built, parsed);
    }

    #[test]
    fn obj_nests() {
        let built = obj([("outer", obj([("inner", Value::from(1u64))]))]);
        assert_eq!(built["outer"]["inner"], Value::from(1u64));
    }

    #[test]
    fn obj_with_no_pairs_is_an_empty_object() {
        let built = obj([]);
        assert_eq!(built, Value::Object(serde_json::Map::new()));
    }
}
