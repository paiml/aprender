//! Fluent builder API for JavaScript generation.
//!
//! Provides a type-safe, ergonomic API for constructing JavaScript code.
//!
//! # Example
//!
//! ```rust,no_run
//! use probar_js_gen::prelude::*;
//!
//! let js = JsModuleBuilder::new()
//!     .comment("Generated - DO NOT EDIT")
//!     .let_decl("x", Expr::num(42)).unwrap()
//!     .build();
//! ```
//!
//! # References
//! - Bloch (2008) "Effective Java" - Builder pattern
//! - Gamma et al. (1994) "Design Patterns" - Builder

use crate::hir::*;
use crate::Result;

/// Builder for JavaScript modules.
#[derive(Debug, Default)]
pub struct JsModuleBuilder {
    statements: Vec<Stmt>,
    metadata: Option<GenerationMetadata>,
}

impl JsModuleBuilder {
    /// Create a new empty module builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set generation metadata for immutability tracking.
    #[must_use]
    pub fn metadata(mut self, metadata: GenerationMetadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Add a statement.
    #[must_use]
    pub fn stmt(mut self, stmt: Stmt) -> Self {
        self.statements.push(stmt);
        self
    }

    /// Add multiple statements.
    #[must_use]
    pub fn stmts(mut self, stmts: impl IntoIterator<Item = Stmt>) -> Self {
        self.statements.extend(stmts);
        self
    }

    /// Add a comment.
    #[must_use]
    pub fn comment(self, text: impl Into<String>) -> Self {
        self.stmt(Stmt::Comment(text.into()))
    }

    /// Add a let declaration.
    pub fn let_decl(self, name: impl Into<String>, value: Expr) -> Result<Self> {
        let ident = Identifier::new(name)?;
        Ok(self.stmt(Stmt::Let { name: ident, value }))
    }

    /// Add a const declaration.
    pub fn const_decl(self, name: impl Into<String>, value: Expr) -> Result<Self> {
        let ident = Identifier::new(name)?;
        Ok(self.stmt(Stmt::Const { name: ident, value }))
    }

    /// Add an assignment.
    pub fn assign(self, name: impl Into<String>, value: Expr) -> Result<Self> {
        let ident = Identifier::new(name)?;
        Ok(self.stmt(Stmt::Assign { name: ident, value }))
    }

    /// Add an expression statement.
    #[must_use]
    pub fn expr(self, e: Expr) -> Self {
        self.stmt(Stmt::Expr(e))
    }

    /// Add a class definition.
    #[must_use]
    pub fn class(self, class: JsClass) -> Self {
        self.stmt(Stmt::Class(class))
    }

    /// Add a registerProcessor call.
    pub fn register_processor(
        self,
        name: impl Into<String>,
        class_name: impl Into<String>,
    ) -> Result<Self> {
        let class = Identifier::new(class_name)?;
        Ok(self.stmt(Stmt::RegisterProcessor {
            name: name.into(),
            class,
        }))
    }

    /// Build the module.
    #[must_use]
    pub fn build(self) -> JsModule {
        JsModule {
            statements: self.statements,
            metadata: self.metadata,
        }
    }
}

/// Builder for JavaScript classes.
#[derive(Debug)]
pub struct JsClassBuilder {
    name: Identifier,
    extends: Option<Identifier>,
    constructor: Option<Vec<Stmt>>,
    methods: Vec<JsMethod>,
}

impl JsClassBuilder {
    /// Create a new class builder.
    pub fn new(name: impl Into<String>) -> Result<Self> {
        Ok(Self {
            name: Identifier::new(name)?,
            extends: None,
            constructor: None,
            methods: Vec::new(),
        })
    }

    /// Set the parent class.
    pub fn extends(mut self, parent: impl Into<String>) -> Result<Self> {
        self.extends = Some(Identifier::new(parent)?);
        Ok(self)
    }

    /// Set the constructor body.
    #[must_use]
    pub fn constructor(mut self, body: Vec<Stmt>) -> Self {
        self.constructor = Some(body);
        self
    }

    /// Add a method.
    pub fn method(
        mut self,
        name: impl Into<String>,
        params: &[&str],
        body: Vec<Stmt>,
    ) -> Result<Self> {
        let name = Identifier::new(name)?;
        let params = params
            .iter()
            .map(|p| Identifier::new(*p))
            .collect::<Result<Vec<_>>>()?;

        self.methods.push(JsMethod { name, params, body });
        Ok(self)
    }

    /// Build the class.
    #[must_use]
    pub fn build(self) -> JsClass {
        JsClass {
            name: self.name,
            extends: self.extends,
            constructor: self.constructor,
            methods: self.methods,
        }
    }
}

/// Builder for JavaScript switch statements.
#[derive(Debug)]
pub struct JsSwitchBuilder {
    expr: Expr,
    cases: Vec<(Expr, Vec<Stmt>)>,
    default: Option<Vec<Stmt>>,
}

impl JsSwitchBuilder {
    /// Create a new switch builder.
    #[must_use]
    pub fn new(expr: Expr) -> Self {
        Self {
            expr,
            cases: Vec::new(),
            default: None,
        }
    }

    /// Add a case.
    #[must_use]
    pub fn case(mut self, value: Expr, body: Vec<Stmt>) -> Self {
        self.cases.push((value, body));
        self
    }

    /// Set the default case.
    #[must_use]
    pub fn default(mut self, body: Vec<Stmt>) -> Self {
        self.default = Some(body);
        self
    }

    /// Build the switch.
    #[must_use]
    pub fn build(self) -> JsSwitch {
        JsSwitch {
            expr: self.expr,
            cases: self.cases,
            default: self.default,
        }
    }
}

/// Expression builder helpers.
impl Expr {
    /// Create a null literal.
    #[must_use]
    pub const fn null() -> Self {
        Self::Null
    }

    /// Create a boolean literal.
    #[must_use]
    pub const fn bool(v: bool) -> Self {
        Self::Bool(v)
    }

    /// Create a number literal.
    #[must_use]
    pub fn num(v: impl Into<f64>) -> Self {
        Self::Num(v.into())
    }

    /// Create a string literal.
    #[must_use]
    pub fn str(s: impl Into<String>) -> Self {
        Self::Str(s.into())
    }

    /// Create an identifier reference.
    pub fn ident(name: impl Into<String>) -> Result<Self> {
        Ok(Self::Ident(Identifier::new(name)?))
    }

    /// Create `this` reference.
    #[must_use]
    pub const fn this() -> Self {
        Self::This
    }

    /// Member access: `self.prop`
    pub fn dot(self, prop: impl Into<String>) -> Result<Self> {
        Ok(Self::Member {
            object: Box::new(self),
            property: Identifier::new(prop)?,
        })
    }

    /// Computed member: `self[index]`
    #[must_use]
    pub fn index(self, idx: Expr) -> Self {
        Self::Index {
            object: Box::new(self),
            index: Box::new(idx),
        }
    }

    /// Function call.
    #[must_use]
    pub fn call(self, args: Vec<Expr>) -> Self {
        Self::Call {
            callee: Box::new(self),
            args,
        }
    }

    /// New expression.
    #[must_use]
    pub fn new_expr(self, args: Vec<Expr>) -> Self {
        Self::New {
            constructor: Box::new(self),
            args,
        }
    }

    /// Await expression.
    #[must_use]
    pub fn await_expr(self) -> Self {
        Self::Await(Box::new(self))
    }

    /// Dynamic import.
    #[must_use]
    pub fn import(path: Expr) -> Self {
        Self::Import(Box::new(path))
    }

    // Binary operations

    /// Addition: `self + other`
    #[must_use]
    pub fn add(self, other: Expr) -> Self {
        Self::Binary {
            left: Box::new(self),
            op: BinOp::Add,
            right: Box::new(other),
        }
    }

    /// Subtraction: `self - other`
    #[must_use]
    pub fn sub(self, other: Expr) -> Self {
        Self::Binary {
            left: Box::new(self),
            op: BinOp::Sub,
            right: Box::new(other),
        }
    }

    /// Multiplication: `self * other`
    #[must_use]
    pub fn mul(self, other: Expr) -> Self {
        Self::Binary {
            left: Box::new(self),
            op: BinOp::Mul,
            right: Box::new(other),
        }
    }

    /// Division: `self / other`
    #[must_use]
    pub fn div(self, other: Expr) -> Self {
        Self::Binary {
            left: Box::new(self),
            op: BinOp::Div,
            right: Box::new(other),
        }
    }

    /// Modulo: `self % other`
    #[must_use]
    pub fn modulo(self, other: Expr) -> Self {
        Self::Binary {
            left: Box::new(self),
            op: BinOp::Mod,
            right: Box::new(other),
        }
    }

    /// Strict equality: `self === other`
    #[must_use]
    pub fn eq(self, other: Expr) -> Self {
        Self::Binary {
            left: Box::new(self),
            op: BinOp::EqStrict,
            right: Box::new(other),
        }
    }

    /// Strict inequality: `self !== other`
    #[must_use]
    pub fn ne(self, other: Expr) -> Self {
        Self::Binary {
            left: Box::new(self),
            op: BinOp::NeStrict,
            right: Box::new(other),
        }
    }

    /// Less than: `self < other`
    #[must_use]
    pub fn lt(self, other: Expr) -> Self {
        Self::Binary {
            left: Box::new(self),
            op: BinOp::Lt,
            right: Box::new(other),
        }
    }

    /// Less than or equal: `self <= other`
    #[must_use]
    pub fn le(self, other: Expr) -> Self {
        Self::Binary {
            left: Box::new(self),
            op: BinOp::Le,
            right: Box::new(other),
        }
    }

    /// Greater than: `self > other`
    #[must_use]
    pub fn gt(self, other: Expr) -> Self {
        Self::Binary {
            left: Box::new(self),
            op: BinOp::Gt,
            right: Box::new(other),
        }
    }

    /// Greater than or equal: `self >= other`
    #[must_use]
    pub fn ge(self, other: Expr) -> Self {
        Self::Binary {
            left: Box::new(self),
            op: BinOp::Ge,
            right: Box::new(other),
        }
    }

    /// Logical and: `self && other`
    #[must_use]
    pub fn and(self, other: Expr) -> Self {
        Self::Binary {
            left: Box::new(self),
            op: BinOp::And,
            right: Box::new(other),
        }
    }

    /// Logical or: `self || other`
    #[must_use]
    pub fn or(self, other: Expr) -> Self {
        Self::Binary {
            left: Box::new(self),
            op: BinOp::Or,
            right: Box::new(other),
        }
    }

    // Unary operations

    /// Logical not: `!self`
    #[must_use]
    pub fn not(self) -> Self {
        Self::Unary {
            op: UnaryOp::Not,
            operand: Box::new(self),
        }
    }

    /// Negation: `-self`
    #[must_use]
    pub fn neg(self) -> Self {
        Self::Unary {
            op: UnaryOp::Neg,
            operand: Box::new(self),
        }
    }

    /// Typeof: `typeof self`
    #[must_use]
    pub fn type_of(self) -> Self {
        Self::Unary {
            op: UnaryOp::TypeOf,
            operand: Box::new(self),
        }
    }

    /// Ternary: `self ? then : else`
    #[must_use]
    pub fn ternary(self, then_expr: Expr, else_expr: Expr) -> Self {
        Self::Ternary {
            condition: Box::new(self),
            then_expr: Box::new(then_expr),
            else_expr: Box::new(else_expr),
        }
    }

    /// Object literal.
    #[must_use]
    pub fn object(pairs: Vec<(&str, Expr)>) -> Self {
        Self::Object(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
    }

    /// Array literal.
    #[must_use]
    pub fn array(items: Vec<Expr>) -> Self {
        Self::Array(items)
    }

    /// Arrow function.
    pub fn arrow(params: &[&str], body: Expr) -> Result<Self> {
        let params = params
            .iter()
            .map(|p| Identifier::new(*p))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self::Arrow {
            params,
            body: Box::new(body),
        })
    }

    /// Arrow function with block body.
    pub fn arrow_block(params: &[&str], body: Vec<Stmt>) -> Result<Self> {
        let params = params
            .iter()
            .map(|p| Identifier::new(*p))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self::ArrowBlock { params, body })
    }

    /// Assignment: `self = value`
    #[must_use]
    pub fn assign(self, value: Expr) -> Self {
        Self::Assign {
            target: Box::new(self),
            value: Box::new(value),
        }
    }
}

/// Statement builder helpers.
impl Stmt {
    /// Create a let declaration.
    pub fn let_decl(name: impl Into<String>, value: Expr) -> Result<Self> {
        Ok(Self::Let {
            name: Identifier::new(name)?,
            value,
        })
    }

    /// Create a const declaration.
    pub fn const_decl(name: impl Into<String>, value: Expr) -> Result<Self> {
        Ok(Self::Const {
            name: Identifier::new(name)?,
            value,
        })
    }

    /// Create an assignment.
    pub fn assign(name: impl Into<String>, value: Expr) -> Result<Self> {
        Ok(Self::Assign {
            name: Identifier::new(name)?,
            value,
        })
    }

    /// Create a member assignment.
    pub fn member_assign(obj: Expr, member: impl Into<String>, value: Expr) -> Result<Self> {
        Ok(Self::MemberAssign {
            object: obj,
            member: Identifier::new(member)?,
            value,
        })
    }

    /// Create an add-assign.
    #[must_use]
    pub fn add_assign(target: Expr, value: Expr) -> Self {
        Self::AddAssign { target, value }
    }

    /// Create a post-increment.
    #[must_use]
    pub fn post_increment(expr: Expr) -> Self {
        Self::PostIncrement(expr)
    }

    /// Create an expression statement.
    #[must_use]
    pub fn expr(e: Expr) -> Self {
        Self::Expr(e)
    }

    /// Create a return statement.
    #[must_use]
    pub fn ret() -> Self {
        Self::Return(None)
    }

    /// Create a return with value.
    #[must_use]
    pub fn ret_val(e: Expr) -> Self {
        Self::Return(Some(e))
    }

    /// Create an if statement.
    #[must_use]
    pub fn if_then(cond: Expr, then_body: Vec<Stmt>) -> Self {
        Self::If {
            condition: cond,
            then_branch: then_body,
            else_branch: None,
        }
    }

    /// Create an if-else statement.
    #[must_use]
    pub fn if_else(cond: Expr, then_body: Vec<Stmt>, else_body: Vec<Stmt>) -> Self {
        Self::If {
            condition: cond,
            then_branch: then_body,
            else_branch: Some(else_body),
        }
    }

    /// Create a for loop.
    pub fn for_loop(
        var: impl Into<String>,
        start: Expr,
        end: Expr,
        body: Vec<Stmt>,
    ) -> Result<Self> {
        Ok(Self::For {
            var: Identifier::new(var)?,
            start,
            end,
            body,
        })
    }

    /// Create a while loop.
    #[must_use]
    pub fn while_loop(cond: Expr, body: Vec<Stmt>) -> Self {
        Self::While {
            condition: cond,
            body,
        }
    }

    /// Create a try-catch.
    pub fn try_catch(
        body: Vec<Stmt>,
        catch_var: impl Into<String>,
        handler: Vec<Stmt>,
    ) -> Result<Self> {
        Ok(Self::TryCatch {
            body,
            catch_var: Identifier::new(catch_var)?,
            handler,
        })
    }

    /// Create a comment.
    #[must_use]
    pub fn comment(text: impl Into<String>) -> Self {
        Self::Comment(text.into())
    }

    /// Create an onmessage handler.
    #[must_use]
    pub fn on_message(body: Vec<Stmt>) -> Self {
        Self::OnMessage(body)
    }

    /// Create a registerProcessor call.
    pub fn register_processor(name: impl Into<String>, class: impl Into<String>) -> Result<Self> {
        Ok(Self::RegisterProcessor {
            name: name.into(),
            class: Identifier::new(class)?,
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::hir::GenerationMetadata;

    // ========================================================================
    // JsModuleBuilder Tests
    // ========================================================================

    #[test]
    fn module_builder_basic() {
        let module = JsModuleBuilder::new()
            .comment("test")
            .let_decl("x", Expr::num(42))
            .unwrap()
            .build();

        assert_eq!(module.statements.len(), 2);
    }

    #[test]
    fn module_builder_metadata() {
        let meta = GenerationMetadata {
            tool: "test".into(),
            version: "1.0".into(),
            input_hash: "abc123".into(),
            timestamp: "2024-01-01".into(),
            regenerate_cmd: "cargo run --bin gen".into(),
        };
        let module = JsModuleBuilder::new().metadata(meta).build();
        assert!(module.metadata.is_some());
        assert_eq!(module.metadata.unwrap().tool, "test");
    }

    #[test]
    fn module_builder_stmts() {
        let stmts = vec![Stmt::comment("a"), Stmt::comment("b")];
        let module = JsModuleBuilder::new().stmts(stmts).build();
        assert_eq!(module.statements.len(), 2);
    }

    #[test]
    fn module_builder_const_decl() {
        let module = JsModuleBuilder::new()
            .const_decl("GRAVITY", Expr::num(9.81))
            .unwrap()
            .build();
        assert_eq!(module.statements.len(), 1);
    }

    #[test]
    fn module_builder_assign() {
        let module = JsModuleBuilder::new()
            .assign("x", Expr::num(10))
            .unwrap()
            .build();
        assert_eq!(module.statements.len(), 1);
    }

    #[test]
    fn module_builder_expr() {
        let module = JsModuleBuilder::new()
            .expr(Expr::ident("foo").unwrap())
            .build();
        assert_eq!(module.statements.len(), 1);
    }

    #[test]
    fn module_builder_class() {
        let class = JsClassBuilder::new("Test").unwrap().build();
        let module = JsModuleBuilder::new().class(class).build();
        assert_eq!(module.statements.len(), 1);
    }

    #[test]
    fn module_builder_register_processor() {
        let module = JsModuleBuilder::new()
            .register_processor("my-processor", "MyProcessor")
            .unwrap()
            .build();
        assert_eq!(module.statements.len(), 1);
    }

    // ========================================================================
    // JsClassBuilder Tests
    // ========================================================================

    #[test]
    fn class_builder_basic() -> Result<()> {
        let class = JsClassBuilder::new("Foo")?
            .extends("Bar")?
            .constructor(vec![])
            .method("baz", &["x", "y"], vec![Stmt::ret()])?
            .build();

        assert_eq!(class.name.as_str(), "Foo");
        assert_eq!(
            class
                .extends
                .as_ref()
                .map(super::super::hir::Identifier::as_str),
            Some("Bar")
        );
        assert_eq!(class.methods.len(), 1);
        Ok(())
    }

    #[test]
    fn class_builder_no_extends() {
        let class = JsClassBuilder::new("Simple").unwrap().build();
        assert!(class.extends.is_none());
        assert!(class.constructor.is_none());
    }

    #[test]
    fn class_builder_invalid_name() {
        assert!(JsClassBuilder::new("class").is_err());
    }

    #[test]
    fn class_builder_invalid_extends() {
        let result = JsClassBuilder::new("Valid").unwrap().extends("123invalid");
        assert!(result.is_err());
    }

    #[test]
    fn class_builder_invalid_method_name() {
        let result = JsClassBuilder::new("Valid")
            .unwrap()
            .method("class", &[], vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn class_builder_invalid_method_param() {
        let result = JsClassBuilder::new("Valid")
            .unwrap()
            .method("test", &["class"], vec![]);
        assert!(result.is_err());
    }

    // ========================================================================
    // JsSwitchBuilder Tests
    // ========================================================================

    #[test]
    fn switch_builder_basic() {
        let sw = JsSwitchBuilder::new(Expr::ident("x").unwrap())
            .case(Expr::num(1), vec![Stmt::ret_val(Expr::str("one"))])
            .case(Expr::num(2), vec![Stmt::ret_val(Expr::str("two"))])
            .default(vec![Stmt::ret_val(Expr::str("other"))])
            .build();

        assert_eq!(sw.cases.len(), 2);
        assert!(sw.default.is_some());
    }

    #[test]
    fn switch_builder_no_default() {
        let sw = JsSwitchBuilder::new(Expr::num(0))
            .case(Expr::num(0), vec![])
            .build();

        assert_eq!(sw.cases.len(), 1);
        assert!(sw.default.is_none());
    }

    // ========================================================================
    // Expr Builder Tests
    // ========================================================================

    #[test]
    fn expr_builder_chain() -> Result<()> {
        let expr = Expr::ident("foo")?.dot("bar")?.call(vec![Expr::num(1)]);
        match expr {
            Expr::Call { callee, args } => {
                assert_eq!(args.len(), 1);
                match *callee {
                    Expr::Member { property, .. } => {
                        assert_eq!(property.as_str(), "bar");
                    }
                    _ => panic!("expected member"),
                }
            }
            _ => panic!("expected call"),
        }
        Ok(())
    }

    #[test]
    fn expr_literals() {
        assert!(matches!(Expr::null(), Expr::Null));
        assert!(matches!(Expr::bool(true), Expr::Bool(true)));
        assert!(matches!(Expr::num(42), Expr::Num(n) if (n - 42.0).abs() < f64::EPSILON));
        assert!(matches!(Expr::str("hi"), Expr::Str(s) if s == "hi"));
        assert!(matches!(Expr::this(), Expr::This));
    }

    #[test]
    fn expr_index() {
        let expr = Expr::ident("arr").unwrap().index(Expr::num(0));
        assert!(matches!(expr, Expr::Index { .. }));
    }

    #[test]
    fn expr_new() {
        let expr = Expr::ident("Date").unwrap().new_expr(vec![]);
        assert!(matches!(expr, Expr::New { .. }));
    }

    #[test]
    fn expr_await() {
        let expr = Expr::ident("promise").unwrap().await_expr();
        assert!(matches!(expr, Expr::Await(_)));
    }

    #[test]
    fn expr_import() {
        let expr = Expr::import(Expr::str("./module.js"));
        assert!(matches!(expr, Expr::Import(_)));
    }

    #[test]
    fn expr_binary_ops() {
        let a = Expr::num(1);
        let b = Expr::num(2);

        assert!(matches!(
            a.clone().add(b.clone()),
            Expr::Binary { op: BinOp::Add, .. }
        ));
        assert!(matches!(
            a.clone().sub(b.clone()),
            Expr::Binary { op: BinOp::Sub, .. }
        ));
        assert!(matches!(
            a.clone().mul(b.clone()),
            Expr::Binary { op: BinOp::Mul, .. }
        ));
        assert!(matches!(
            a.clone().div(b.clone()),
            Expr::Binary { op: BinOp::Div, .. }
        ));
        assert!(matches!(
            a.clone().modulo(b.clone()),
            Expr::Binary { op: BinOp::Mod, .. }
        ));
        assert!(matches!(
            a.clone().eq(b.clone()),
            Expr::Binary {
                op: BinOp::EqStrict,
                ..
            }
        ));
        assert!(matches!(
            a.clone().ne(b.clone()),
            Expr::Binary {
                op: BinOp::NeStrict,
                ..
            }
        ));
        assert!(matches!(
            a.clone().lt(b.clone()),
            Expr::Binary { op: BinOp::Lt, .. }
        ));
        assert!(matches!(
            a.clone().le(b.clone()),
            Expr::Binary { op: BinOp::Le, .. }
        ));
        assert!(matches!(
            a.clone().gt(b.clone()),
            Expr::Binary { op: BinOp::Gt, .. }
        ));
        assert!(matches!(
            a.clone().ge(b.clone()),
            Expr::Binary { op: BinOp::Ge, .. }
        ));
        assert!(matches!(
            a.clone().and(b.clone()),
            Expr::Binary { op: BinOp::And, .. }
        ));
        assert!(matches!(
            a.clone().or(b),
            Expr::Binary { op: BinOp::Or, .. }
        ));
    }

    #[test]
    fn expr_unary_ops() {
        let x = Expr::num(1);
        assert!(matches!(
            x.clone().not(),
            Expr::Unary {
                op: UnaryOp::Not,
                ..
            }
        ));
        assert!(matches!(
            x.clone().neg(),
            Expr::Unary {
                op: UnaryOp::Neg,
                ..
            }
        ));
        assert!(matches!(
            x.type_of(),
            Expr::Unary {
                op: UnaryOp::TypeOf,
                ..
            }
        ));
    }

    #[test]
    fn expr_ternary() {
        let expr = Expr::bool(true).ternary(Expr::num(1), Expr::num(0));
        assert!(matches!(expr, Expr::Ternary { .. }));
    }

    #[test]
    fn expr_object() {
        let expr = Expr::object(vec![("a", Expr::num(1)), ("b", Expr::num(2))]);
        match expr {
            Expr::Object(pairs) => assert_eq!(pairs.len(), 2),
            _ => panic!("expected object"),
        }
    }

    #[test]
    fn expr_array() {
        let expr = Expr::array(vec![Expr::num(1), Expr::num(2), Expr::num(3)]);
        match expr {
            Expr::Array(items) => assert_eq!(items.len(), 3),
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn expr_arrow() {
        let expr = Expr::arrow(&["x"], Expr::ident("x").unwrap()).unwrap();
        assert!(matches!(expr, Expr::Arrow { .. }));
    }

    #[test]
    fn expr_arrow_block() {
        let expr =
            Expr::arrow_block(&["x"], vec![Stmt::ret_val(Expr::ident("x").unwrap())]).unwrap();
        assert!(matches!(expr, Expr::ArrowBlock { .. }));
    }

    #[test]
    fn expr_arrow_invalid_param() {
        assert!(Expr::arrow(&["class"], Expr::null()).is_err());
        assert!(Expr::arrow_block(&["class"], vec![]).is_err());
    }

    #[test]
    fn expr_assign() {
        let expr = Expr::ident("x").unwrap().assign(Expr::num(10));
        assert!(matches!(expr, Expr::Assign { .. }));
    }

    #[test]
    fn expr_dot_invalid() {
        let result = Expr::ident("obj").unwrap().dot("class");
        assert!(result.is_err());
    }

    // ========================================================================
    // Stmt Builder Tests
    // ========================================================================

    #[test]
    fn stmt_let_const_assign() {
        assert!(Stmt::let_decl("x", Expr::num(1)).is_ok());
        assert!(Stmt::const_decl("Y", Expr::num(2)).is_ok());
        assert!(Stmt::assign("z", Expr::num(3)).is_ok());

        // Invalid identifiers
        assert!(Stmt::let_decl("class", Expr::num(1)).is_err());
        assert!(Stmt::const_decl("class", Expr::num(1)).is_err());
        assert!(Stmt::assign("class", Expr::num(1)).is_err());
    }

    #[test]
    fn stmt_member_assign() {
        let stmt = Stmt::member_assign(Expr::this(), "prop", Expr::num(42)).unwrap();
        assert!(matches!(stmt, Stmt::MemberAssign { .. }));

        // Invalid member
        assert!(Stmt::member_assign(Expr::this(), "class", Expr::null()).is_err());
    }

    #[test]
    fn stmt_add_assign() {
        let stmt = Stmt::add_assign(Expr::ident("x").unwrap(), Expr::num(1));
        assert!(matches!(stmt, Stmt::AddAssign { .. }));
    }

    #[test]
    fn stmt_post_increment() {
        let stmt = Stmt::post_increment(Expr::ident("i").unwrap());
        assert!(matches!(stmt, Stmt::PostIncrement(_)));
    }

    #[test]
    fn stmt_return() {
        assert!(matches!(Stmt::ret(), Stmt::Return(None)));
        assert!(matches!(Stmt::ret_val(Expr::num(0)), Stmt::Return(Some(_))));
    }

    #[test]
    fn stmt_if() {
        let stmt = Stmt::if_then(Expr::bool(true), vec![Stmt::ret()]);
        match stmt {
            Stmt::If { else_branch, .. } => assert!(else_branch.is_none()),
            _ => panic!("expected if"),
        }

        let stmt = Stmt::if_else(Expr::bool(true), vec![], vec![]);
        match stmt {
            Stmt::If { else_branch, .. } => assert!(else_branch.is_some()),
            _ => panic!("expected if"),
        }
    }

    #[test]
    fn stmt_for_loop() {
        let stmt = Stmt::for_loop("i", Expr::num(0), Expr::num(10), vec![]).unwrap();
        assert!(matches!(stmt, Stmt::For { .. }));

        // Invalid var
        assert!(Stmt::for_loop("class", Expr::num(0), Expr::num(10), vec![]).is_err());
    }

    #[test]
    fn stmt_while_loop() {
        let stmt = Stmt::while_loop(Expr::bool(true), vec![]);
        assert!(matches!(stmt, Stmt::While { .. }));
    }

    #[test]
    fn stmt_try_catch() {
        let stmt = Stmt::try_catch(vec![], "e", vec![]).unwrap();
        assert!(matches!(stmt, Stmt::TryCatch { .. }));

        // Invalid catch var
        assert!(Stmt::try_catch(vec![], "class", vec![]).is_err());
    }

    #[test]
    fn stmt_comment() {
        let stmt = Stmt::comment("test comment");
        assert!(matches!(stmt, Stmt::Comment(_)));
    }

    #[test]
    fn stmt_on_message() {
        let stmt = Stmt::on_message(vec![]);
        assert!(matches!(stmt, Stmt::OnMessage(_)));
    }

    #[test]
    fn stmt_register_processor() {
        let stmt = Stmt::register_processor("my-proc", "MyProc").unwrap();
        assert!(matches!(stmt, Stmt::RegisterProcessor { .. }));

        // Invalid class name
        assert!(Stmt::register_processor("proc", "class").is_err());
    }

    #[test]
    fn identifier_validation() {
        assert!(Identifier::new("validName").is_ok());
        assert!(Identifier::new("class").is_err());
        assert!(Identifier::new("123invalid").is_err());
    }
}
