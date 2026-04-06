//! Query parsing and execution
//!
//! Toyota Way Principles:
//! - Kaizen: JIT compilation eliminates interpreter overhead (Neumann 2011)
//! - Muda elimination: Kernel fusion (Wu et al. 2012)
//!
//! ## Phase 1 SQL Subset
//!
//! Supports analytics workload (OLAP):
//! - SELECT with column list or *
//! - FROM single table (no JOINs in Phase 1)
//! - WHERE with simple predicates (>, <, =, >=, <=, !=)
//! - GROUP BY with aggregations (SUM, AVG, COUNT, MIN, MAX)
//! - ORDER BY (ASC/DESC)
//! - LIMIT
//!
//! References:
//! - sqlparser-rs: <https://docs.rs/sqlparser>
//! - TPC-H queries: Analytics benchmark patterns

pub mod executor;

pub use executor::QueryExecutor;

use sqlparser::ast::{Expr, Query, Select, SelectItem, SetExpr, Statement};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;

/// Type alias for aggregation tuple (function, column, optional alias)
pub type Aggregation = (AggregateFunction, String, Option<String>);

/// Parsed SQL query with extracted components
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryPlan {
    /// Selected columns (or * for all)
    pub columns: Vec<String>,
    /// Table name
    pub table: String,
    /// WHERE clause expression (optional)
    pub filter: Option<String>,
    /// GROUP BY columns (optional)
    pub group_by: Vec<String>,
    /// Aggregation functions: (function, column, alias)
    pub aggregations: Vec<Aggregation>,
    /// ORDER BY clauses
    pub order_by: Vec<(String, OrderDirection)>,
    /// LIMIT count (optional)
    pub limit: Option<usize>,
}

/// Supported aggregation functions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregateFunction {
    /// Sum of values
    Sum,
    /// Average of values
    Avg,
    /// Count of rows
    Count,
    /// Minimum value
    Min,
    /// Maximum value
    Max,
}

/// Sort order direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderDirection {
    /// Ascending order (smallest first)
    Asc,
    /// Descending order (largest first)
    Desc,
}

/// Query parser and executor
pub struct QueryEngine {
    dialect: GenericDialect,
}

impl Default for QueryEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl QueryEngine {
    /// Create a new query engine
    #[must_use]
    pub const fn new() -> Self {
        Self { dialect: GenericDialect {} }
    }

    /// Parse SQL query into query plan
    ///
    /// # Errors
    /// Returns error if:
    /// - SQL syntax is invalid
    /// - Query uses unsupported features (JOINs, subqueries, etc.)
    /// - Multiple statements provided
    ///
    /// # Example
    /// ```
    /// use trueno_db::query::QueryEngine;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let engine = QueryEngine::new();
    /// let plan = engine.parse("SELECT id, name FROM users WHERE age > 18")?;
    /// assert_eq!(plan.table, "users");
    /// # Ok(())
    /// # }
    pub fn parse(&self, sql: &str) -> crate::Result<QueryPlan> {
        // Handle empty query
        if sql.trim().is_empty() {
            return Ok(QueryPlan {
                columns: vec!["*".to_string()],
                table: String::new(),
                filter: None,
                group_by: Vec::new(),
                aggregations: Vec::new(),
                order_by: Vec::new(),
                limit: None,
            });
        }

        // Parse SQL
        let statements = Parser::parse_sql(&self.dialect, sql)
            .map_err(|e| crate::Error::ParseError(format!("SQL parse error: {e}")))?;

        // Validate single statement
        if statements.len() != 1 {
            return Err(crate::Error::ParseError("Only single statements supported".to_string()));
        }

        // Extract SELECT statement
        let stmt = &statements[0];
        let Statement::Query(query) = stmt else {
            return Err(crate::Error::ParseError("Only SELECT queries supported".to_string()));
        };

        Self::parse_select_query(query)
    }

    fn parse_select_query(query: &Query) -> crate::Result<QueryPlan> {
        // Extract SELECT body
        let SetExpr::Select(select) = query.body.as_ref() else {
            return Err(crate::Error::ParseError("Only SELECT queries supported".to_string()));
        };

        // Extract table name (FROM clause)
        let table = Self::extract_table_name(select)?;

        // Extract columns and aggregations
        let (columns, aggregations) = Self::extract_columns(&select.projection)?;

        // Extract WHERE clause
        let filter = select.selection.as_ref().map(ToString::to_string);

        // Extract GROUP BY
        let group_by = Self::extract_group_by(&select.group_by);

        // Extract ORDER BY
        let order_by = Self::extract_order_by(query.order_by.as_ref());

        // Extract LIMIT
        let limit = Self::extract_limit(query.limit.as_ref());

        Ok(QueryPlan { columns, table, filter, group_by, aggregations, order_by, limit })
    }

    fn extract_table_name(select: &Select) -> crate::Result<String> {
        if select.from.is_empty() {
            return Ok(String::new());
        }

        if select.from.len() > 1 {
            return Err(crate::Error::ParseError(
                "Multiple tables not supported in Phase 1".to_string(),
            ));
        }

        let table_with_joins = &select.from[0];
        if !table_with_joins.joins.is_empty() {
            return Err(crate::Error::ParseError("JOINs not supported in Phase 1".to_string()));
        }

        Ok(table_with_joins.relation.to_string())
    }

    fn extract_columns(
        projection: &[SelectItem],
    ) -> crate::Result<(Vec<String>, Vec<Aggregation>)> {
        let mut columns = Vec::new();
        let mut aggregations = Vec::new();

        for item in projection {
            match item {
                SelectItem::Wildcard(_) => {
                    columns.push("*".to_string());
                }
                SelectItem::UnnamedExpr(expr) => {
                    if let Some((func, col)) = Self::extract_aggregate(expr) {
                        aggregations.push((func, col, None));
                    } else {
                        columns.push(expr.to_string());
                    }
                }
                SelectItem::ExprWithAlias { expr, alias } => {
                    if let Some((func, col)) = Self::extract_aggregate(expr) {
                        aggregations.push((func, col, Some(alias.value.clone())));
                    } else {
                        columns.push(alias.value.clone());
                    }
                }
                SelectItem::QualifiedWildcard(..) => {
                    return Err(crate::Error::ParseError(
                        "Qualified wildcards not supported".to_string(),
                    ))
                }
            }
        }

        Ok((columns, aggregations))
    }

    fn extract_aggregate(expr: &Expr) -> Option<(AggregateFunction, String)> {
        if let Expr::Function(func) = expr {
            let func_name = func.name.to_string().to_uppercase();
            let agg_func = match func_name.as_str() {
                "SUM" => AggregateFunction::Sum,
                "AVG" => AggregateFunction::Avg,
                "COUNT" => AggregateFunction::Count,
                "MIN" => AggregateFunction::Min,
                "MAX" => AggregateFunction::Max,
                _ => return None,
            };

            // Extract column name from arguments
            let col = match &func.args {
                sqlparser::ast::FunctionArguments::List(func_arg_list) => {
                    func_arg_list.args.first().map_or_else(|| "*".to_string(), ToString::to_string)
                }
                _ => "*".to_string(),
            };
            return Some((agg_func, col));
        }
        None
    }

    fn extract_group_by(group_by: &sqlparser::ast::GroupByExpr) -> Vec<String> {
        match group_by {
            sqlparser::ast::GroupByExpr::All(_) => Vec::new(),
            sqlparser::ast::GroupByExpr::Expressions(exprs, _) => {
                exprs.iter().map(ToString::to_string).collect()
            }
        }
    }

    fn extract_order_by(
        order_by: Option<&sqlparser::ast::OrderBy>,
    ) -> Vec<(String, OrderDirection)> {
        order_by
            .map(|ob| {
                ob.exprs
                    .iter()
                    .map(|o| {
                        let col = o.expr.to_string();
                        let dir = if o.asc.unwrap_or(true) {
                            OrderDirection::Asc
                        } else {
                            OrderDirection::Desc
                        };
                        (col, dir)
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    fn extract_limit(limit: Option<&Expr>) -> Option<usize> {
        limit.and_then(|expr| {
            if let Expr::Value(sqlparser::ast::Value::Number(n, _)) = expr {
                n.parse().ok()
            } else {
                None
            }
        })
    }
}
