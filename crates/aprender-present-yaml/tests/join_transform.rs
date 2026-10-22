#![allow(clippy::unwrap_used, clippy::disallowed_methods)]
//! `join(other, on=key)` transform: a left join over arrays of objects (#5566).
//! Driven through the public parser and executor.

use presentar_yaml::{DataContext, ExecutionError, ExpressionExecutor, ExpressionParser, Value};
use std::collections::HashMap;

fn s(v: &str) -> Value {
    Value::String(v.to_string())
}

fn obj(pairs: &[(&str, Value)]) -> Value {
    Value::Object(
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect::<HashMap<_, _>>(),
    )
}

fn arr(v: Vec<Value>) -> Value {
    Value::Array(v)
}

/// Run `{{ l | join(r, on=id) }}` with `l` and `r` in the context.
fn join(left: Value, right: Value) -> Result<Value, ExecutionError> {
    let mut ctx = DataContext::new();
    ctx.insert("l", left);
    ctx.insert("r", right);
    let expr = ExpressionParser::new()
        .parse("{{ l | join(r, on=id) }}")
        .unwrap();
    ExpressionExecutor::new().execute(&expr, &ctx)
}

#[test]
fn join_refuses_a_non_array_left_side() {
    assert!(matches!(
        join(obj(&[]), arr(vec![])),
        Err(ExecutionError::ExpectedArray)
    ));
}

#[test]
fn join_refuses_a_missing_source_by_name() {
    let mut ctx = DataContext::new();
    ctx.insert("l", arr(vec![]));
    let expr = ExpressionParser::new()
        .parse("{{ l | join(nope, on=id) }}")
        .unwrap();
    match ExpressionExecutor::new().execute(&expr, &ctx) {
        Err(ExecutionError::SourceNotFound(n)) => assert_eq!(n, "nope"),
        other => panic!("expected SourceNotFound, got {other:?}"),
    }
}

#[test]
fn join_refuses_a_non_array_right_side() {
    assert!(matches!(
        join(arr(vec![]), obj(&[("id", s("1"))])),
        Err(ExecutionError::ExpectedArray)
    ));
}

#[test]
fn join_merges_matching_fields_and_drops_the_right_key() {
    let left = arr(vec![obj(&[("id", s("1")), ("a", s("x"))])]);
    let right = arr(vec![obj(&[("id", s("1")), ("b", s("y"))])]);
    assert_eq!(
        join(left, right).unwrap(),
        arr(vec![obj(&[("id", s("1")), ("a", s("x")), ("b", s("y"))])])
    );
}

#[test]
fn join_prefixes_a_colliding_field_with_the_source_name_and_keeps_the_left() {
    let left = arr(vec![obj(&[("id", s("1")), ("v", s("left"))])]);
    let right = arr(vec![obj(&[("id", s("1")), ("v", s("right"))])]);
    assert_eq!(
        join(left, right).unwrap(),
        arr(vec![obj(&[
            ("id", s("1")),
            ("v", s("left")),
            ("r_v", s("right"))
        ])])
    );
}

#[test]
fn join_emits_one_row_per_matching_right_item_in_order() {
    let left = arr(vec![obj(&[("id", s("1"))])]);
    let right = arr(vec![
        obj(&[("id", s("1")), ("n", Value::Number(1.0))]),
        obj(&[("id", s("2")), ("n", Value::Number(9.0))]),
        obj(&[("id", s("1")), ("n", Value::Number(2.0))]),
    ]);
    let Value::Array(rows) = join(left, right).unwrap() else {
        panic!("array");
    };
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].get("n"), Some(&Value::Number(1.0)));
    assert_eq!(rows[1].get("n"), Some(&Value::Number(2.0)));
}

#[test]
fn join_keeps_an_unmatched_left_row_unchanged() {
    let l = obj(&[("id", s("7")), ("a", s("x"))]);
    let right = arr(vec![obj(&[("id", s("1")), ("b", s("y"))])]);
    assert_eq!(join(arr(vec![l.clone()]), right).unwrap(), arr(vec![l]));
}

#[test]
fn join_keeps_left_rows_without_the_key_and_non_objects() {
    let left = arr(vec![obj(&[("a", s("x"))]), Value::Number(3.0), s("plain")]);
    let right = arr(vec![obj(&[("id", s("1")), ("b", s("y"))])]);
    assert_eq!(join(left.clone(), right).unwrap(), left);
}

#[test]
fn join_ignores_right_rows_that_are_not_objects_or_lack_the_key() {
    let l = obj(&[("id", s("1"))]);
    let right = arr(vec![Value::Number(1.0), obj(&[("b", s("y"))])]);
    assert_eq!(join(arr(vec![l.clone()]), right).unwrap(), arr(vec![l]));
}

#[test]
fn join_compares_keys_by_their_string_form() {
    let left = arr(vec![obj(&[("id", Value::Number(1.0))])]);
    let right = arr(vec![obj(&[("id", s("1")), ("b", s("y"))])]);
    let Value::Array(rows) = join(left, right).unwrap() else {
        panic!("array");
    };
    assert_eq!(rows[0].get("b"), Some(&s("y")));
}

#[test]
fn join_resolves_a_dotted_source_name() {
    let mut ctx = DataContext::new();
    ctx.insert(
        "data",
        obj(&[("users", arr(vec![obj(&[("id", s("1")), ("n", s("ann"))])]))]),
    );
    ctx.insert("l", arr(vec![obj(&[("id", s("1"))])]));
    let expr = ExpressionParser::new()
        .parse("{{ l | join(data.users, on=id) }}")
        .unwrap();
    assert_eq!(
        ExpressionExecutor::new().execute(&expr, &ctx).unwrap(),
        arr(vec![obj(&[("id", s("1")), ("n", s("ann"))])])
    );
}
