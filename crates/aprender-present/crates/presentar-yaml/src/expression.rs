//! Expression language for data binding.
//!
//! Syntax: `{{ source | transform | transform }}`

use serde::{Deserialize, Serialize};

/// Parsed expression.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Expression {
    /// Source identifier (e.g., "data.transactions")
    pub source: String,
    /// Chain of transforms
    pub transforms: Vec<Transform>,
}

/// A transform operation in the expression pipeline.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Transform {
    /// Filter rows: `filter(field=value)`
    Filter {
        /// Field to filter on
        field: String,
        /// Value to match
        value: String,
    },
    /// Select columns: `select(field1, field2)`
    Select {
        /// Fields to select
        fields: Vec<String>,
    },
    /// Sort rows: `sort(field, desc=true)`
    Sort {
        /// Field to sort by
        field: String,
        /// Sort descending
        desc: bool,
    },
    /// Limit rows: `limit(n)`
    Limit {
        /// Maximum number of rows
        n: usize,
    },
    /// Count rows
    Count,
    /// Sum a field: `sum(field)`
    Sum {
        /// Field to sum
        field: String,
    },
    /// Average a field: `mean(field)`
    Mean {
        /// Field to average
        field: String,
    },
    /// Rate over window: `rate(window)`
    Rate {
        /// Time window (e.g., "1m", "5m")
        window: String,
    },
    /// Convert to percentage
    Percentage,
    /// Join with another dataset: `join(other, on=field)`
    Join {
        /// Other dataset
        other: String,
        /// Join field
        on: String,
    },
    /// Sample rows: `sample(n)`
    Sample {
        /// Number of rows to sample
        n: usize,
    },
    /// Group by field: `group_by(field)`
    GroupBy {
        /// Field to group by
        field: String,
    },
    /// Get distinct values: `distinct(field)`
    Distinct {
        /// Field to get distinct values from (optional)
        field: Option<String>,
    },
    /// Advanced filter with operators: `where(field, op, value)`
    Where {
        /// Field to filter on
        field: String,
        /// Comparison operator (eq, ne, gt, lt, gte, lte, contains)
        op: String,
        /// Value to compare
        value: String,
    },
    /// Offset/skip rows: `offset(n)`
    Offset {
        /// Number of rows to skip
        n: usize,
    },
    /// Minimum value: `min(field)`
    Min {
        /// Field to find minimum
        field: String,
    },
    /// Maximum value: `max(field)`
    Max {
        /// Field to find maximum
        field: String,
    },
    /// First n rows: `first(n)`
    First {
        /// Number of rows
        n: usize,
    },
    /// Last n rows: `last(n)`
    Last {
        /// Number of rows
        n: usize,
    },
    /// Flatten nested arrays: `flatten`
    Flatten,
    /// Reverse order: `reverse`
    Reverse,
    /// Map/transform each element: `map(expr)`
    Map {
        /// Expression to apply to each element
        expr: String,
    },
    /// Reduce/fold elements: `reduce(initial, accumulator_expr)`
    Reduce {
        /// Initial value
        initial: String,
        /// Accumulator expression (uses `acc` and `item` variables)
        expr: String,
    },
    /// Aggregate after group_by: `agg(field, op)`
    Aggregate {
        /// Field to aggregate
        field: String,
        /// Aggregation operation (sum, count, mean, min, max)
        op: AggregateOp,
    },
    /// Pivot table: `pivot(row_field, col_field, value_field)`
    Pivot {
        /// Field for row labels
        row_field: String,
        /// Field for column headers
        col_field: String,
        /// Field for values
        value_field: String,
    },
    /// Running total: `cumsum(field)`
    CumulativeSum {
        /// Field to sum
        field: String,
    },
    /// Rank within group: `rank(field, method)`
    Rank {
        /// Field to rank by
        field: String,
        /// Ranking method (dense, ordinal, average)
        method: RankMethod,
    },
    /// Moving average: `moving_avg(field, window)`
    MovingAverage {
        /// Field to average
        field: String,
        /// Window size
        window: usize,
    },
    /// Percent change: `pct_change(field)`
    PercentChange {
        /// Field to calculate percent change
        field: String,
    },
    /// Model suggestion: `suggest(prefix, count)` - for N-gram/autocomplete models
    Suggest {
        /// Input prefix to get suggestions for
        prefix: String,
        /// Maximum number of suggestions to return
        count: usize,
    },
}

/// Aggregation operations for group_by.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AggregateOp {
    /// Sum values
    Sum,
    /// Count values
    Count,
    /// Mean/average
    Mean,
    /// Minimum
    Min,
    /// Maximum
    Max,
    /// First value
    First,
    /// Last value
    Last,
}

impl AggregateOp {
    /// Parse from string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "sum" => Some(Self::Sum),
            "count" => Some(Self::Count),
            "mean" | "avg" | "average" => Some(Self::Mean),
            "min" => Some(Self::Min),
            "max" => Some(Self::Max),
            "first" => Some(Self::First),
            "last" => Some(Self::Last),
            _ => None,
        }
    }
}

/// Ranking methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum RankMethod {
    /// Dense ranking (1, 2, 2, 3)
    #[default]
    Dense,
    /// Ordinal ranking (1, 2, 3, 4)
    Ordinal,
    /// Average ranking (1, 2.5, 2.5, 4)
    Average,
}

impl RankMethod {
    /// Parse from string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "dense" => Some(Self::Dense),
            "ordinal" => Some(Self::Ordinal),
            "average" | "avg" => Some(Self::Average),
            _ => None,
        }
    }
}

/// Expression parser.
#[derive(Debug, Default)]
pub struct ExpressionParser;

impl ExpressionParser {
    /// Create a new parser.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Parse an expression string.
    ///
    /// # Errors
    ///
    /// Returns an error if the expression is invalid.
    pub fn parse(&self, input: &str) -> Result<Expression, ExpressionError> {
        let input = input.trim();

        // Check for {{ }} wrapper
        let inner = if input.starts_with("{{") && input.ends_with("}}") {
            input[2..input.len() - 2].trim()
        } else {
            input
        };

        // Split by pipe
        let parts: Vec<&str> = inner.split('|').map(str::trim).collect();

        if parts.is_empty() {
            return Err(ExpressionError::EmptyExpression);
        }

        let source = parts[0].to_string();
        let mut transforms = Vec::new();

        for part in &parts[1..] {
            let transform = self.parse_transform(part)?;
            transforms.push(transform);
        }

        Ok(Expression { source, transforms })
    }

    #[allow(clippy::too_many_lines)]
    fn parse_transform(&self, input: &str) -> Result<Transform, ExpressionError> {
        let input = input.trim();

        // Check for function call: name(args)
        if let Some(paren_pos) = input.find('(') {
            let name = &input[..paren_pos];
            let args_str = input[paren_pos + 1..].trim_end_matches(')').trim();

            match name {
                "filter" => {
                    let (field, value) = self.parse_key_value(args_str)?;
                    Ok(Transform::Filter { field, value })
                }
                "select" => {
                    let fields: Vec<String> = args_str
                        .split(',')
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect();
                    Ok(Transform::Select { fields })
                }
                "sort" => {
                    let parts: Vec<&str> = args_str.split(',').map(str::trim).collect();
                    let field = (*parts.first().unwrap_or(&"")).to_string();
                    let desc = parts.get(1).is_some_and(|s| s.contains("desc=true"));
                    Ok(Transform::Sort { field, desc })
                }
                "limit" => {
                    let n = args_str
                        .parse()
                        .map_err(|_| ExpressionError::InvalidArgument("limit".to_string()))?;
                    Ok(Transform::Limit { n })
                }
                "sum" => Ok(Transform::Sum {
                    field: args_str.to_string(),
                }),
                "mean" => Ok(Transform::Mean {
                    field: args_str.to_string(),
                }),
                "rate" => Ok(Transform::Rate {
                    window: args_str.to_string(),
                }),
                "join" => {
                    let parts: Vec<&str> = args_str.split(',').map(str::trim).collect();
                    let other = (*parts.first().unwrap_or(&"")).to_string();
                    let on = parts
                        .get(1)
                        .and_then(|s| s.strip_prefix("on="))
                        .unwrap_or("")
                        .to_string();
                    Ok(Transform::Join { other, on })
                }
                "sample" => {
                    let n = args_str
                        .parse()
                        .map_err(|_| ExpressionError::InvalidArgument("sample".to_string()))?;
                    Ok(Transform::Sample { n })
                }
                "group_by" => Ok(Transform::GroupBy {
                    field: args_str.to_string(),
                }),
                "distinct" => {
                    let field = if args_str.is_empty() {
                        None
                    } else {
                        Some(args_str.to_string())
                    };
                    Ok(Transform::Distinct { field })
                }
                "where" => {
                    let parts: Vec<&str> = args_str.split(',').map(str::trim).collect();
                    if parts.len() < 3 {
                        return Err(ExpressionError::InvalidArgument(
                            "where requires field, op, value".to_string(),
                        ));
                    }
                    Ok(Transform::Where {
                        field: parts[0].to_string(),
                        op: parts[1].to_string(),
                        value: parts[2].to_string(),
                    })
                }
                "offset" => {
                    let n = args_str
                        .parse()
                        .map_err(|_| ExpressionError::InvalidArgument("offset".to_string()))?;
                    Ok(Transform::Offset { n })
                }
                "min" => Ok(Transform::Min {
                    field: args_str.to_string(),
                }),
                "max" => Ok(Transform::Max {
                    field: args_str.to_string(),
                }),
                "first" => {
                    let n = args_str
                        .parse()
                        .map_err(|_| ExpressionError::InvalidArgument("first".to_string()))?;
                    Ok(Transform::First { n })
                }
                "last" => {
                    let n = args_str
                        .parse()
                        .map_err(|_| ExpressionError::InvalidArgument("last".to_string()))?;
                    Ok(Transform::Last { n })
                }
                "map" => Ok(Transform::Map {
                    expr: args_str.to_string(),
                }),
                "reduce" => {
                    let parts: Vec<&str> = args_str.splitn(2, ',').map(str::trim).collect();
                    if parts.len() < 2 {
                        return Err(ExpressionError::InvalidArgument(
                            "reduce requires initial, expr".to_string(),
                        ));
                    }
                    Ok(Transform::Reduce {
                        initial: parts[0].to_string(),
                        expr: parts[1].to_string(),
                    })
                }
                "agg" | "aggregate" => {
                    let parts: Vec<&str> = args_str.split(',').map(str::trim).collect();
                    if parts.len() < 2 {
                        return Err(ExpressionError::InvalidArgument(
                            "agg requires field, op".to_string(),
                        ));
                    }
                    let op = AggregateOp::from_str(parts[1]).ok_or_else(|| {
                        ExpressionError::InvalidArgument(format!(
                            "unknown aggregate op: {}",
                            parts[1]
                        ))
                    })?;
                    Ok(Transform::Aggregate {
                        field: parts[0].to_string(),
                        op,
                    })
                }
                "pivot" => {
                    let parts: Vec<&str> = args_str.split(',').map(str::trim).collect();
                    if parts.len() < 3 {
                        return Err(ExpressionError::InvalidArgument(
                            "pivot requires row_field, col_field, value_field".to_string(),
                        ));
                    }
                    Ok(Transform::Pivot {
                        row_field: parts[0].to_string(),
                        col_field: parts[1].to_string(),
                        value_field: parts[2].to_string(),
                    })
                }
                "cumsum" => Ok(Transform::CumulativeSum {
                    field: args_str.to_string(),
                }),
                "rank" => {
                    let parts: Vec<&str> = args_str.split(',').map(str::trim).collect();
                    let field = (*parts.first().unwrap_or(&"")).to_string();
                    let method = parts
                        .get(1)
                        .and_then(|s| RankMethod::from_str(s))
                        .unwrap_or_default();
                    Ok(Transform::Rank { field, method })
                }
                "moving_avg" | "ma" => {
                    let parts: Vec<&str> = args_str.split(',').map(str::trim).collect();
                    if parts.len() < 2 {
                        return Err(ExpressionError::InvalidArgument(
                            "moving_avg requires field, window".to_string(),
                        ));
                    }
                    let window = parts[1].parse().map_err(|_| {
                        ExpressionError::InvalidArgument("moving_avg window".to_string())
                    })?;
                    Ok(Transform::MovingAverage {
                        field: parts[0].to_string(),
                        window,
                    })
                }
                "pct_change" => Ok(Transform::PercentChange {
                    field: args_str.to_string(),
                }),
                "suggest" => {
                    let parts: Vec<&str> = args_str.split(',').map(str::trim).collect();
                    if parts.len() < 2 {
                        return Err(ExpressionError::InvalidArgument(
                            "suggest requires prefix, count".to_string(),
                        ));
                    }
                    let count = parts[1].parse().map_err(|_| {
                        ExpressionError::InvalidArgument("suggest count".to_string())
                    })?;
                    Ok(Transform::Suggest {
                        prefix: parts[0].to_string(),
                        count,
                    })
                }
                _ => Err(ExpressionError::UnknownTransform(name.to_string())),
            }
        } else {
            // Simple transform without args
            match input {
                "count" => Ok(Transform::Count),
                "percentage" => Ok(Transform::Percentage),
                "flatten" => Ok(Transform::Flatten),
                "reverse" => Ok(Transform::Reverse),
                "distinct" => Ok(Transform::Distinct { field: None }),
                _ => Err(ExpressionError::UnknownTransform(input.to_string())),
            }
        }
    }

    fn parse_key_value(&self, input: &str) -> Result<(String, String), ExpressionError> {
        let parts: Vec<&str> = input.splitn(2, '=').collect();
        if parts.len() != 2 {
            return Err(ExpressionError::InvalidArgument(input.to_string()));
        }
        Ok((parts[0].trim().to_string(), parts[1].trim().to_string()))
    }
}

/// Expression parsing error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExpressionError {
    /// Empty expression
    EmptyExpression,
    /// Unknown transform function
    UnknownTransform(String),
    /// Invalid argument
    InvalidArgument(String),
}

impl std::fmt::Display for ExpressionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyExpression => write!(f, "empty expression"),
            Self::UnknownTransform(name) => write!(f, "unknown transform: {name}"),
            Self::InvalidArgument(arg) => write!(f, "invalid argument: {arg}"),
        }
    }
}

impl std::error::Error for ExpressionError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_source() {
        let parser = ExpressionParser::new();
        let expr = parser.parse("data.transactions").unwrap();
        assert_eq!(expr.source, "data.transactions");
        assert!(expr.transforms.is_empty());
    }

    #[test]
    fn test_parse_with_braces() {
        let parser = ExpressionParser::new();
        let expr = parser.parse("{{ data.transactions }}").unwrap();
        assert_eq!(expr.source, "data.transactions");
    }

    #[test]
    fn test_parse_count() {
        let parser = ExpressionParser::new();
        let expr = parser.parse("{{ data.items | count }}").unwrap();
        assert_eq!(expr.transforms, vec![Transform::Count]);
    }

    #[test]
    fn test_parse_filter() {
        let parser = ExpressionParser::new();
        let expr = parser.parse("{{ data | filter(status=active) }}").unwrap();
        assert_eq!(
            expr.transforms,
            vec![Transform::Filter {
                field: "status".to_string(),
                value: "active".to_string(),
            }]
        );
    }

    #[test]
    fn test_parse_select() {
        let parser = ExpressionParser::new();
        let expr = parser
            .parse("{{ data | select(id, name, email) }}")
            .unwrap();
        assert_eq!(
            expr.transforms,
            vec![Transform::Select {
                fields: vec!["id".to_string(), "name".to_string(), "email".to_string()],
            }]
        );
    }

    #[test]
    fn test_parse_sort() {
        let parser = ExpressionParser::new();
        let expr = parser
            .parse("{{ data | sort(created_at, desc=true) }}")
            .unwrap();
        assert_eq!(
            expr.transforms,
            vec![Transform::Sort {
                field: "created_at".to_string(),
                desc: true,
            }]
        );
    }

    #[test]
    fn test_parse_limit() {
        let parser = ExpressionParser::new();
        let expr = parser.parse("{{ data | limit(10) }}").unwrap();
        assert_eq!(expr.transforms, vec![Transform::Limit { n: 10 }]);
    }

    #[test]
    fn test_parse_chain() {
        let parser = ExpressionParser::new();
        let expr = parser
            .parse("{{ data.transactions | filter(status=completed) | count }}")
            .unwrap();

        assert_eq!(expr.source, "data.transactions");
        assert_eq!(expr.transforms.len(), 2);
        assert_eq!(
            expr.transforms[0],
            Transform::Filter {
                field: "status".to_string(),
                value: "completed".to_string(),
            }
        );
        assert_eq!(expr.transforms[1], Transform::Count);
    }

    #[test]
    fn test_parse_join() {
        let parser = ExpressionParser::new();
        let expr = parser
            .parse("{{ data.orders | join(data.customers, on=customer_id) }}")
            .unwrap();

        assert_eq!(
            expr.transforms,
            vec![Transform::Join {
                other: "data.customers".to_string(),
                on: "customer_id".to_string(),
            }]
        );
    }

    #[test]
    fn test_parse_sample() {
        let parser = ExpressionParser::new();
        let expr = parser.parse("{{ data | sample(100) }}").unwrap();
        assert_eq!(expr.transforms, vec![Transform::Sample { n: 100 }]);
    }

    #[test]
    fn test_parse_sum() {
        let parser = ExpressionParser::new();
        let expr = parser.parse("{{ data | sum(amount) }}").unwrap();
        assert_eq!(
            expr.transforms,
            vec![Transform::Sum {
                field: "amount".to_string(),
            }]
        );
    }

    #[test]
    fn test_parse_error_unknown_transform() {
        let parser = ExpressionParser::new();
        let result = parser.parse("{{ data | unknown() }}");
        assert!(matches!(result, Err(ExpressionError::UnknownTransform(_))));
    }

    #[test]
    fn test_expression_error_display() {
        assert_eq!(
            ExpressionError::EmptyExpression.to_string(),
            "empty expression"
        );
        assert_eq!(
            ExpressionError::UnknownTransform("foo".to_string()).to_string(),
            "unknown transform: foo"
        );
    }

    // =========================================================================
    // Map/Reduce/GroupBy and Advanced Transform Tests
    // =========================================================================

    #[test]
    fn test_parse_map() {
        let parser = ExpressionParser::new();
        let expr = parser.parse("{{ data | map(item.value * 2) }}").unwrap();
        assert_eq!(
            expr.transforms,
            vec![Transform::Map {
                expr: "item.value * 2".to_string(),
            }]
        );
    }

    #[test]
    fn test_parse_reduce() {
        let parser = ExpressionParser::new();
        let expr = parser
            .parse("{{ data | reduce(0, acc + item.value) }}")
            .unwrap();
        assert_eq!(
            expr.transforms,
            vec![Transform::Reduce {
                initial: "0".to_string(),
                expr: "acc + item.value".to_string(),
            }]
        );
    }

    #[test]
    fn test_parse_reduce_missing_args() {
        let parser = ExpressionParser::new();
        let result = parser.parse("{{ data | reduce(0) }}");
        assert!(matches!(result, Err(ExpressionError::InvalidArgument(_))));
    }

    #[test]
    fn test_parse_aggregate_sum() {
        let parser = ExpressionParser::new();
        let expr = parser
            .parse("{{ data | group_by(category) | agg(amount, sum) }}")
            .unwrap();
        assert_eq!(
            expr.transforms,
            vec![
                Transform::GroupBy {
                    field: "category".to_string()
                },
                Transform::Aggregate {
                    field: "amount".to_string(),
                    op: AggregateOp::Sum,
                },
            ]
        );
    }

    #[test]
    fn test_parse_aggregate_mean() {
        let parser = ExpressionParser::new();
        let expr = parser.parse("{{ data | agg(price, mean) }}").unwrap();
        assert_eq!(
            expr.transforms,
            vec![Transform::Aggregate {
                field: "price".to_string(),
                op: AggregateOp::Mean,
            }]
        );
    }

    #[test]
    fn test_parse_aggregate_count() {
        let parser = ExpressionParser::new();
        let expr = parser.parse("{{ data | agg(id, count) }}").unwrap();
        assert_eq!(
            expr.transforms,
            vec![Transform::Aggregate {
                field: "id".to_string(),
                op: AggregateOp::Count,
            }]
        );
    }

    #[test]
    fn test_parse_aggregate_alias() {
        let parser = ExpressionParser::new();
        let expr = parser.parse("{{ data | aggregate(value, max) }}").unwrap();
        assert_eq!(
            expr.transforms,
            vec![Transform::Aggregate {
                field: "value".to_string(),
                op: AggregateOp::Max,
            }]
        );
    }

    #[test]
    fn test_parse_aggregate_invalid_op() {
        let parser = ExpressionParser::new();
        let result = parser.parse("{{ data | agg(field, unknown_op) }}");
        assert!(matches!(result, Err(ExpressionError::InvalidArgument(_))));
    }

    #[test]
    fn test_parse_pivot() {
        let parser = ExpressionParser::new();
        let expr = parser
            .parse("{{ data | pivot(date, product, sales) }}")
            .unwrap();
        assert_eq!(
            expr.transforms,
            vec![Transform::Pivot {
                row_field: "date".to_string(),
                col_field: "product".to_string(),
                value_field: "sales".to_string(),
            }]
        );
    }

    #[test]
    fn test_parse_pivot_missing_args() {
        let parser = ExpressionParser::new();
        let result = parser.parse("{{ data | pivot(date, product) }}");
        assert!(matches!(result, Err(ExpressionError::InvalidArgument(_))));
    }

    #[test]
    fn test_parse_cumsum() {
        let parser = ExpressionParser::new();
        let expr = parser.parse("{{ data | cumsum(balance) }}").unwrap();
        assert_eq!(
            expr.transforms,
            vec![Transform::CumulativeSum {
                field: "balance".to_string(),
            }]
        );
    }

    #[test]
    fn test_parse_rank_default() {
        let parser = ExpressionParser::new();
        let expr = parser.parse("{{ data | rank(score) }}").unwrap();
        assert_eq!(
            expr.transforms,
            vec![Transform::Rank {
                field: "score".to_string(),
                method: RankMethod::Dense,
            }]
        );
    }

    #[test]
    fn test_parse_rank_ordinal() {
        let parser = ExpressionParser::new();
        let expr = parser.parse("{{ data | rank(score, ordinal) }}").unwrap();
        assert_eq!(
            expr.transforms,
            vec![Transform::Rank {
                field: "score".to_string(),
                method: RankMethod::Ordinal,
            }]
        );
    }

    #[test]
    fn test_parse_rank_average() {
        let parser = ExpressionParser::new();
        let expr = parser.parse("{{ data | rank(score, average) }}").unwrap();
        assert_eq!(
            expr.transforms,
            vec![Transform::Rank {
                field: "score".to_string(),
                method: RankMethod::Average,
            }]
        );
    }

    #[test]
    fn test_parse_moving_average() {
        let parser = ExpressionParser::new();
        let expr = parser.parse("{{ data | moving_avg(price, 5) }}").unwrap();
        assert_eq!(
            expr.transforms,
            vec![Transform::MovingAverage {
                field: "price".to_string(),
                window: 5,
            }]
        );
    }

    #[test]
    fn test_parse_moving_average_alias() {
        let parser = ExpressionParser::new();
        let expr = parser.parse("{{ data | ma(price, 10) }}").unwrap();
        assert_eq!(
            expr.transforms,
            vec![Transform::MovingAverage {
                field: "price".to_string(),
                window: 10,
            }]
        );
    }

    #[test]
    fn test_parse_moving_average_missing_window() {
        let parser = ExpressionParser::new();
        let result = parser.parse("{{ data | moving_avg(price) }}");
        assert!(matches!(result, Err(ExpressionError::InvalidArgument(_))));
    }

    #[test]
    fn test_parse_pct_change() {
        let parser = ExpressionParser::new();
        let expr = parser.parse("{{ data | pct_change(value) }}").unwrap();
        assert_eq!(
            expr.transforms,
            vec![Transform::PercentChange {
                field: "value".to_string(),
            }]
        );
    }

    #[test]
    fn test_parse_complex_pipeline() {
        let parser = ExpressionParser::new();
        let expr = parser
            .parse("{{ data | filter(status=active) | group_by(category) | agg(amount, sum) | sort(amount, desc=true) | limit(10) }}")
            .unwrap();

        assert_eq!(expr.transforms.len(), 5);
        assert!(matches!(expr.transforms[0], Transform::Filter { .. }));
        assert!(matches!(expr.transforms[1], Transform::GroupBy { .. }));
        assert!(matches!(expr.transforms[2], Transform::Aggregate { .. }));
        assert!(matches!(
            expr.transforms[3],
            Transform::Sort { desc: true, .. }
        ));
        assert!(matches!(expr.transforms[4], Transform::Limit { n: 10 }));
    }

    #[test]
    fn test_parse_map_reduce_pipeline() {
        let parser = ExpressionParser::new();
        let expr = parser
            .parse("{{ data | map(item.value * 2) | reduce(0, acc + item) }}")
            .unwrap();

        assert_eq!(expr.transforms.len(), 2);
        assert!(matches!(expr.transforms[0], Transform::Map { .. }));
        assert!(matches!(expr.transforms[1], Transform::Reduce { .. }));
    }

    // =========================================================================
    // AggregateOp Tests
    // =========================================================================

    #[test]
    fn test_aggregate_op_from_str() {
        assert_eq!(AggregateOp::from_str("sum"), Some(AggregateOp::Sum));
        assert_eq!(AggregateOp::from_str("count"), Some(AggregateOp::Count));
        assert_eq!(AggregateOp::from_str("mean"), Some(AggregateOp::Mean));
        assert_eq!(AggregateOp::from_str("avg"), Some(AggregateOp::Mean));
        assert_eq!(AggregateOp::from_str("average"), Some(AggregateOp::Mean));
        assert_eq!(AggregateOp::from_str("min"), Some(AggregateOp::Min));
        assert_eq!(AggregateOp::from_str("max"), Some(AggregateOp::Max));
        assert_eq!(AggregateOp::from_str("first"), Some(AggregateOp::First));
        assert_eq!(AggregateOp::from_str("last"), Some(AggregateOp::Last));
        assert_eq!(AggregateOp::from_str("unknown"), None);
    }

    #[test]
    fn test_aggregate_op_case_insensitive() {
        assert_eq!(AggregateOp::from_str("SUM"), Some(AggregateOp::Sum));
        assert_eq!(AggregateOp::from_str("Sum"), Some(AggregateOp::Sum));
        assert_eq!(AggregateOp::from_str("MEAN"), Some(AggregateOp::Mean));
    }

    // =========================================================================
    // RankMethod Tests
    // =========================================================================

    #[test]
    fn test_rank_method_from_str() {
        assert_eq!(RankMethod::from_str("dense"), Some(RankMethod::Dense));
        assert_eq!(RankMethod::from_str("ordinal"), Some(RankMethod::Ordinal));
        assert_eq!(RankMethod::from_str("average"), Some(RankMethod::Average));
        assert_eq!(RankMethod::from_str("avg"), Some(RankMethod::Average));
        assert_eq!(RankMethod::from_str("unknown"), None);
    }

    #[test]
    fn test_rank_method_default() {
        assert_eq!(RankMethod::default(), RankMethod::Dense);
    }
}
