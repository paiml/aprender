//! Tests for query engine

use trueno_db::query::{AggregateFunction, OrderDirection, QueryEngine};

#[test]
fn test_query_engine_parse() {
    // Test basic SQL parsing
    let engine = QueryEngine::new();
    let result = engine.parse("SELECT * FROM users");
    assert!(result.is_ok(), "Query parsing should succeed: {:?}", result.err());

    let plan = result.unwrap();
    assert_eq!(plan.table, "users");
    assert_eq!(plan.columns, vec!["*"]);
}

#[test]
fn test_query_engine_parse_complex() {
    // Test complex SQL
    let engine = QueryEngine::new();
    let queries = vec![
        "SELECT id, name FROM users WHERE age > 18",
        "SELECT COUNT(*) FROM events GROUP BY category",
        "SELECT * FROM data ORDER BY score DESC LIMIT 10",
    ];

    for sql in queries {
        let result = engine.parse(sql);
        assert!(result.is_ok(), "Query parsing should succeed for: {sql}");
    }
}

#[test]
fn test_query_engine_parse_empty() {
    // Test empty query
    let engine = QueryEngine::new();
    let result = engine.parse("");
    assert!(result.is_ok(), "Empty query should be allowed");
}

#[test]
fn test_where_clause() {
    let engine = QueryEngine::new();
    let plan = engine.parse("SELECT id, name FROM users WHERE age > 18").unwrap();

    assert_eq!(plan.table, "users");
    assert_eq!(plan.columns, vec!["id", "name"]);
    assert!(plan.filter.is_some());
    assert!(plan.filter.unwrap().contains("> 18"));
}

#[test]
fn test_group_by_with_aggregation() {
    let engine = QueryEngine::new();
    let plan =
        engine.parse("SELECT category, COUNT(*) as total FROM events GROUP BY category").unwrap();

    assert_eq!(plan.table, "events");
    assert_eq!(plan.group_by, vec!["category"]);
    assert_eq!(plan.aggregations.len(), 1);
    assert_eq!(plan.aggregations[0].0, AggregateFunction::Count);
}

#[test]
fn test_order_by_limit() {
    let engine = QueryEngine::new();
    let plan = engine.parse("SELECT * FROM data ORDER BY score DESC LIMIT 10").unwrap();

    assert_eq!(plan.table, "data");
    assert_eq!(plan.order_by.len(), 1);
    assert_eq!(plan.order_by[0].0, "score");
    assert_eq!(plan.order_by[0].1, OrderDirection::Desc);
    assert_eq!(plan.limit, Some(10));
}

#[test]
fn test_all_aggregations() {
    let engine = QueryEngine::new();
    let sql = "SELECT SUM(revenue) as total_revenue, AVG(price) as avg_price, COUNT(*) as count, MIN(age) as min_age, MAX(score) as max_score FROM sales";
    let plan = engine.parse(sql).unwrap();

    assert_eq!(plan.aggregations.len(), 5);
    assert_eq!(plan.aggregations[0].0, AggregateFunction::Sum);
    assert_eq!(plan.aggregations[1].0, AggregateFunction::Avg);
    assert_eq!(plan.aggregations[2].0, AggregateFunction::Count);
    assert_eq!(plan.aggregations[3].0, AggregateFunction::Min);
    assert_eq!(plan.aggregations[4].0, AggregateFunction::Max);
}

#[test]
fn test_reject_joins() {
    let engine = QueryEngine::new();
    let result = engine.parse("SELECT * FROM users JOIN orders ON users.id = orders.user_id");
    assert!(result.is_err(), "JOINs should be rejected");
}

#[test]
fn test_reject_multiple_statements() {
    let engine = QueryEngine::new();
    let result = engine.parse("SELECT * FROM users; SELECT * FROM orders;");
    assert!(result.is_err(), "Multiple statements should be rejected");
}

#[test]
fn test_reject_non_select() {
    let engine = QueryEngine::new();
    let result = engine.parse("INSERT INTO users (name) VALUES ('Alice')");
    assert!(result.is_err(), "Only SELECT supported");
}

#[test]
fn test_query_engine_default() {
    // Test default constructor
    let engine = QueryEngine::default();
    let plan = engine.parse("SELECT * FROM users").unwrap();
    assert_eq!(plan.table, "users");
}

#[test]
fn test_reject_multiple_from_tables() {
    // Test error when multiple tables in FROM clause (without JOIN)
    let engine = QueryEngine::new();
    let result = engine.parse("SELECT * FROM users, orders");
    assert!(result.is_err(), "Multiple FROM tables should be rejected");
    assert!(result.unwrap_err().to_string().contains("Multiple tables"));
}

#[test]
fn test_reject_qualified_wildcard() {
    // Test error when using table.* syntax
    let engine = QueryEngine::new();
    let result = engine.parse("SELECT users.* FROM users");
    assert!(result.is_err(), "Qualified wildcards should be rejected");
    assert!(result.unwrap_err().to_string().contains("Qualified wildcards"));
}

#[test]
fn test_order_by_ascending() {
    // Test ORDER BY ASC (default ascending)
    let engine = QueryEngine::new();

    // Explicit ASC
    let plan1 = engine.parse("SELECT * FROM data ORDER BY score ASC").unwrap();
    assert_eq!(plan1.order_by.len(), 1);
    assert_eq!(plan1.order_by[0].1, OrderDirection::Asc);

    // Implicit ASC (default)
    let plan2 = engine.parse("SELECT * FROM data ORDER BY score").unwrap();
    assert_eq!(plan2.order_by.len(), 1);
    assert_eq!(plan2.order_by[0].1, OrderDirection::Asc);
}

#[test]
fn test_column_alias_without_aggregation() {
    // Test aliasing regular columns (not aggregations)
    let engine = QueryEngine::new();
    let plan = engine.parse("SELECT id as user_id, name as username FROM users").unwrap();

    assert_eq!(plan.columns, vec!["user_id", "username"]);
    assert_eq!(plan.aggregations.len(), 0);
}

#[test]
fn test_invalid_sql_syntax() {
    // Test that invalid SQL returns ParseError
    let engine = QueryEngine::new();
    let result = engine.parse("SELECT FROM WHERE");
    assert!(result.is_err(), "Invalid SQL should fail");
    assert!(result.unwrap_err().to_string().contains("parse error"));
}
