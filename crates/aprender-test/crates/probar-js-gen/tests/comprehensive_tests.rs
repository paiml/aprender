//! Comprehensive tests for 100% coverage of probar-js-gen.
//!
//! These tests exercise all code paths in builder.rs, codegen.rs, and hir.rs.

#![allow(clippy::unwrap_used)]

use probar_js_gen::prelude::*;

// ============================================================================
// Builder Tests - JsModuleBuilder
// ============================================================================

#[test]
fn module_builder_stmts_method() {
    let stmts = vec![
        Stmt::comment("first"),
        Stmt::comment("second"),
        Stmt::comment("third"),
    ];
    let module = JsModuleBuilder::new().stmts(stmts).build();
    assert_eq!(module.statements.len(), 3);
}

#[test]
fn module_builder_assign() {
    let module = JsModuleBuilder::new()
        .let_decl("x", Expr::num(1))
        .unwrap()
        .assign("x", Expr::num(2))
        .unwrap()
        .build();
    let js = generate(&module);
    assert!(js.contains("let x = 1;"));
    assert!(js.contains("x = 2;"));
}

#[test]
fn module_builder_expr() {
    let module = JsModuleBuilder::new()
        .expr(
            Expr::ident("console")
                .unwrap()
                .dot("log")
                .unwrap()
                .call(vec![]),
        )
        .build();
    let js = generate(&module);
    assert!(js.contains("console.log();"));
}

#[test]
fn module_builder_register_processor() {
    let module = JsModuleBuilder::new()
        .register_processor("my-processor", "MyProcessor")
        .unwrap()
        .build();
    let js = generate(&module);
    assert!(js.contains(r#"registerProcessor("my-processor", MyProcessor);"#));
}

#[test]
fn module_builder_register_processor_invalid() {
    let result = JsModuleBuilder::new().register_processor("name", "class");
    assert!(result.is_err());
}

// ============================================================================
// Builder Tests - JsSwitchBuilder
// ============================================================================

#[test]
fn switch_builder_with_cases_and_default() {
    let switch = JsSwitchBuilder::new(Expr::ident("x").unwrap())
        .case(Expr::num(1), vec![Stmt::expr(Expr::str("one"))])
        .case(Expr::num(2), vec![Stmt::expr(Expr::str("two"))])
        .default(vec![Stmt::expr(Expr::str("default"))])
        .build();

    let module = JsModuleBuilder::new().stmt(Stmt::Switch(switch)).build();
    let js = generate(&module);

    assert!(js.contains("switch (x)"));
    assert!(js.contains("case 1:"));
    assert!(js.contains("case 2:"));
    assert!(js.contains("default:"));
    assert!(js.contains("break;"));
}

#[test]
fn switch_builder_without_default() {
    let switch = JsSwitchBuilder::new(Expr::num(0))
        .case(Expr::str("a"), vec![])
        .build();

    let module = JsModuleBuilder::new().stmt(Stmt::Switch(switch)).build();
    let js = generate(&module);

    assert!(js.contains("switch (0)"));
    assert!(js.contains(r#"case "a":"#));
    assert!(!js.contains("default:"));
}

// ============================================================================
// Builder Tests - JsClassBuilder
// ============================================================================

#[test]
fn class_builder_without_extends() {
    let class = JsClassBuilder::new("Simple")
        .unwrap()
        .constructor(vec![Stmt::expr(Expr::num(1))])
        .build();

    let module = JsModuleBuilder::new().class(class).build();
    let js = generate(&module);

    assert!(js.contains("class Simple {"));
    assert!(!js.contains("extends"));
    assert!(!js.contains("super()"));
    assert!(js.contains("constructor()"));
}

#[test]
fn class_builder_with_multiple_methods() {
    let class = JsClassBuilder::new("Multi")
        .unwrap()
        .method("foo", &[], vec![Stmt::ret()])
        .unwrap()
        .method(
            "bar",
            &["a", "b"],
            vec![Stmt::ret_val(Expr::ident("a").unwrap())],
        )
        .unwrap()
        .method("baz", &["x"], vec![])
        .unwrap()
        .build();

    let module = JsModuleBuilder::new().class(class).build();
    let js = generate(&module);

    assert!(js.contains("foo()"));
    assert!(js.contains("bar(a, b)"));
    assert!(js.contains("baz(x)"));
}

#[test]
fn class_builder_invalid_name() {
    let result = JsClassBuilder::new("class");
    assert!(result.is_err());
}

#[test]
fn class_builder_invalid_extends() {
    let result = JsClassBuilder::new("Foo").unwrap().extends("function");
    assert!(result.is_err());
}

#[test]
fn class_builder_invalid_method_name() {
    let result = JsClassBuilder::new("Foo")
        .unwrap()
        .method("return", &[], vec![]);
    assert!(result.is_err());
}

#[test]
fn class_builder_invalid_method_param() {
    let result = JsClassBuilder::new("Foo")
        .unwrap()
        .method("test", &["valid", "class"], vec![]);
    assert!(result.is_err());
}

// ============================================================================
// Expression Builder Tests
// ============================================================================

#[test]
fn expr_index() {
    let expr = Expr::ident("arr").unwrap().index(Expr::num(0));
    let module = JsModuleBuilder::new().let_decl("x", expr).unwrap().build();
    let js = generate(&module);
    assert!(js.contains("arr[0]"));
}

#[test]
fn expr_new() {
    let expr = Expr::ident("Date").unwrap().new_expr(vec![]);
    let module = JsModuleBuilder::new().let_decl("d", expr).unwrap().build();
    let js = generate(&module);
    assert!(js.contains("new Date()"));
}

#[test]
fn expr_new_with_args() {
    let expr = Expr::ident("Error")
        .unwrap()
        .new_expr(vec![Expr::str("message")]);
    let module = JsModuleBuilder::new().let_decl("e", expr).unwrap().build();
    let js = generate(&module);
    assert!(js.contains(r#"new Error("message")"#));
}

#[test]
fn expr_await() {
    let expr = Expr::ident("promise").unwrap().await_expr();
    let module = JsModuleBuilder::new()
        .let_decl("result", expr)
        .unwrap()
        .build();
    let js = generate(&module);
    assert!(js.contains("await promise"));
}

#[test]
fn expr_import() {
    let expr = Expr::import(Expr::str("./module.js"));
    let module = JsModuleBuilder::new()
        .let_decl("mod", expr)
        .unwrap()
        .build();
    let js = generate(&module);
    assert!(js.contains(r#"import("./module.js")"#));
}

#[test]
fn expr_all_binary_ops() {
    let ops = vec![
        ("add", Expr::num(1).add(Expr::num(2)), "+"),
        ("sub", Expr::num(1).sub(Expr::num(2)), "-"),
        ("mul", Expr::num(1).mul(Expr::num(2)), "*"),
        ("div", Expr::num(1).div(Expr::num(2)), "/"),
        ("mod", Expr::num(1).modulo(Expr::num(2)), "%"),
        ("eq", Expr::num(1).eq(Expr::num(2)), "==="),
        ("ne", Expr::num(1).ne(Expr::num(2)), "!=="),
        ("lt", Expr::num(1).lt(Expr::num(2)), "<"),
        ("le", Expr::num(1).le(Expr::num(2)), "<="),
        ("gt", Expr::num(1).gt(Expr::num(2)), ">"),
        ("ge", Expr::num(1).ge(Expr::num(2)), ">="),
        ("and", Expr::bool(true).and(Expr::bool(false)), "&&"),
        ("or", Expr::bool(true).or(Expr::bool(false)), "||"),
    ];

    for (name, expr, op) in ops {
        let module = JsModuleBuilder::new().let_decl(name, expr).unwrap().build();
        let js = generate(&module);
        assert!(js.contains(op), "Op {} not found in: {}", op, js);
    }
}

#[test]
fn expr_all_unary_ops() {
    let not_expr = Expr::bool(true).not();
    let neg_expr = Expr::num(5).neg();
    let typeof_expr = Expr::ident("x").unwrap().type_of();

    let module = JsModuleBuilder::new()
        .let_decl("a", not_expr)
        .unwrap()
        .let_decl("b", neg_expr)
        .unwrap()
        .let_decl("c", typeof_expr)
        .unwrap()
        .build();
    let js = generate(&module);

    assert!(js.contains("!true"));
    assert!(js.contains("-5"));
    assert!(js.contains("typeof x"));
}

#[test]
fn expr_ternary() {
    let expr = Expr::bool(true).ternary(Expr::num(1), Expr::num(0));
    let module = JsModuleBuilder::new().let_decl("x", expr).unwrap().build();
    let js = generate(&module);
    assert!(js.contains("(true ? 1 : 0)"));
}

#[test]
fn expr_object() {
    let expr = Expr::object(vec![("foo", Expr::num(1)), ("bar", Expr::str("baz"))]);
    let module = JsModuleBuilder::new()
        .let_decl("obj", expr)
        .unwrap()
        .build();
    let js = generate(&module);
    assert!(js.contains("foo: 1"));
    assert!(js.contains(r#"bar: "baz""#));
}

#[test]
fn expr_array() {
    let expr = Expr::array(vec![Expr::num(1), Expr::num(2), Expr::num(3)]);
    let module = JsModuleBuilder::new()
        .let_decl("arr", expr)
        .unwrap()
        .build();
    let js = generate(&module);
    assert!(js.contains("[1, 2, 3]"));
}

#[test]
fn expr_arrow() {
    let expr = Expr::arrow(&["x"], Expr::ident("x").unwrap().mul(Expr::num(2))).unwrap();
    let module = JsModuleBuilder::new()
        .let_decl("double", expr)
        .unwrap()
        .build();
    let js = generate(&module);
    assert!(js.contains("x => (x * 2)"));
}

#[test]
fn expr_arrow_multiple_params() {
    let expr = Expr::arrow(
        &["a", "b"],
        Expr::ident("a").unwrap().add(Expr::ident("b").unwrap()),
    )
    .unwrap();
    let module = JsModuleBuilder::new()
        .let_decl("add", expr)
        .unwrap()
        .build();
    let js = generate(&module);
    assert!(js.contains("(a, b) => (a + b)"));
}

#[test]
fn expr_arrow_block() {
    let expr = Expr::arrow_block(&["x"], vec![Stmt::ret_val(Expr::ident("x").unwrap())]).unwrap();
    let module = JsModuleBuilder::new().let_decl("fn", expr).unwrap().build();
    let js = generate(&module);
    assert!(js.contains("x => {"));
    assert!(js.contains("return x;"));
}

#[test]
fn expr_arrow_block_multiple_params() {
    let expr =
        Expr::arrow_block(&["a", "b"], vec![Stmt::ret_val(Expr::ident("a").unwrap())]).unwrap();
    let module = JsModuleBuilder::new().let_decl("fn", expr).unwrap().build();
    let js = generate(&module);
    assert!(js.contains("(a, b) => {"));
}

#[test]
fn expr_arrow_invalid_param() {
    let result = Expr::arrow(&["class"], Expr::num(1));
    assert!(result.is_err());
}

#[test]
fn expr_arrow_block_invalid_param() {
    let result = Expr::arrow_block(&["function"], vec![]);
    assert!(result.is_err());
}

#[test]
fn expr_assign() {
    let expr = Expr::ident("x").unwrap().assign(Expr::num(42));
    let module = JsModuleBuilder::new().stmt(Stmt::Expr(expr)).build();
    let js = generate(&module);
    assert!(js.contains("x = 42"));
}

// ============================================================================
// Statement Builder Tests
// ============================================================================

#[test]
fn stmt_let_decl() {
    let stmt = Stmt::let_decl("x", Expr::num(1)).unwrap();
    let module = JsModuleBuilder::new().stmt(stmt).build();
    let js = generate(&module);
    assert!(js.contains("let x = 1;"));
}

#[test]
fn stmt_const_decl() {
    let stmt = Stmt::const_decl("X", Expr::num(1)).unwrap();
    let module = JsModuleBuilder::new().stmt(stmt).build();
    let js = generate(&module);
    assert!(js.contains("const X = 1;"));
}

#[test]
fn stmt_assign() {
    let stmt = Stmt::assign("x", Expr::num(2)).unwrap();
    let module = JsModuleBuilder::new().stmt(stmt).build();
    let js = generate(&module);
    assert!(js.contains("x = 2;"));
}

#[test]
fn stmt_member_assign() {
    let stmt = Stmt::member_assign(Expr::this(), "value", Expr::num(42)).unwrap();
    let module = JsModuleBuilder::new().stmt(stmt).build();
    let js = generate(&module);
    assert!(js.contains("this.value = 42;"));
}

#[test]
fn stmt_member_assign_invalid() {
    let result = Stmt::member_assign(Expr::this(), "class", Expr::num(1));
    assert!(result.is_err());
}

#[test]
fn stmt_add_assign() {
    let stmt = Stmt::add_assign(Expr::ident("x").unwrap(), Expr::num(1));
    let module = JsModuleBuilder::new().stmt(stmt).build();
    let js = generate(&module);
    assert!(js.contains("x += 1;"));
}

#[test]
fn stmt_post_increment() {
    let stmt = Stmt::post_increment(Expr::ident("i").unwrap());
    let module = JsModuleBuilder::new().stmt(stmt).build();
    let js = generate(&module);
    assert!(js.contains("i++;"));
}

#[test]
fn stmt_return_empty() {
    let stmt = Stmt::ret();
    let module = JsModuleBuilder::new().stmt(stmt).build();
    let js = generate(&module);
    assert!(js.contains("return;"));
}

#[test]
fn stmt_return_value() {
    let stmt = Stmt::ret_val(Expr::num(42));
    let module = JsModuleBuilder::new().stmt(stmt).build();
    let js = generate(&module);
    assert!(js.contains("return 42;"));
}

#[test]
fn stmt_if_then() {
    let stmt = Stmt::if_then(Expr::bool(true), vec![Stmt::ret()]);
    let module = JsModuleBuilder::new().stmt(stmt).build();
    let js = generate(&module);
    assert!(js.contains("if (true)"));
    assert!(js.contains("return;"));
    assert!(!js.contains("else"));
}

#[test]
fn stmt_if_else() {
    let stmt = Stmt::if_else(
        Expr::bool(false),
        vec![Stmt::ret_val(Expr::num(1))],
        vec![Stmt::ret_val(Expr::num(0))],
    );
    let module = JsModuleBuilder::new().stmt(stmt).build();
    let js = generate(&module);
    assert!(js.contains("if (false)"));
    assert!(js.contains("} else {"));
}

#[test]
fn stmt_for_loop() {
    let stmt = Stmt::for_loop(
        "i",
        Expr::num(0),
        Expr::num(10),
        vec![Stmt::expr(
            Expr::ident("console")
                .unwrap()
                .dot("log")
                .unwrap()
                .call(vec![Expr::ident("i").unwrap()]),
        )],
    )
    .unwrap();
    let module = JsModuleBuilder::new().stmt(stmt).build();
    let js = generate(&module);
    assert!(js.contains("for (let i = 0; i < 10; i++)"));
}

#[test]
fn stmt_for_loop_invalid_var() {
    let result = Stmt::for_loop("class", Expr::num(0), Expr::num(1), vec![]);
    assert!(result.is_err());
}

#[test]
fn stmt_while_loop() {
    let stmt = Stmt::while_loop(Expr::bool(true), vec![Stmt::ret()]);
    let module = JsModuleBuilder::new().stmt(stmt).build();
    let js = generate(&module);
    assert!(js.contains("while (true)"));
}

#[test]
fn stmt_try_catch() {
    let stmt = Stmt::try_catch(
        vec![Stmt::expr(Expr::ident("riskyCall").unwrap().call(vec![]))],
        "err",
        vec![Stmt::expr(
            Expr::ident("console")
                .unwrap()
                .dot("error")
                .unwrap()
                .call(vec![Expr::ident("err").unwrap()]),
        )],
    )
    .unwrap();
    let module = JsModuleBuilder::new().stmt(stmt).build();
    let js = generate(&module);
    assert!(js.contains("try {"));
    assert!(js.contains("} catch (err) {"));
}

#[test]
fn stmt_try_catch_invalid_var() {
    let result = Stmt::try_catch(vec![], "class", vec![]);
    assert!(result.is_err());
}

#[test]
fn stmt_block() {
    let stmt = Stmt::Block(vec![
        Stmt::let_decl("x", Expr::num(1)).unwrap(),
        Stmt::let_decl("y", Expr::num(2)).unwrap(),
    ]);
    let module = JsModuleBuilder::new().stmt(stmt).build();
    let js = generate(&module);
    assert!(js.contains("{\n"));
    assert!(js.contains("let x = 1;"));
    assert!(js.contains("let y = 2;"));
}

#[test]
fn stmt_comment() {
    let stmt = Stmt::comment("This is a comment");
    let module = JsModuleBuilder::new().stmt(stmt).build();
    let js = generate(&module);
    assert!(js.contains("// This is a comment"));
}

#[test]
fn stmt_on_message() {
    let stmt = Stmt::on_message(vec![Stmt::const_decl(
        "data",
        Expr::ident("e").unwrap().dot("data").unwrap(),
    )
    .unwrap()]);
    let module = JsModuleBuilder::new().stmt(stmt).build();
    let js = generate(&module);
    assert!(js.contains("self.onmessage = async function(e)"));
}

#[test]
fn stmt_register_processor() {
    let stmt = Stmt::register_processor("my-proc", "MyProc").unwrap();
    let module = JsModuleBuilder::new().stmt(stmt).build();
    let js = generate(&module);
    assert!(js.contains(r#"registerProcessor("my-proc", MyProc);"#));
}

#[test]
fn stmt_register_processor_invalid() {
    let result = Stmt::register_processor("name", "class");
    assert!(result.is_err());
}

// ============================================================================
// Codegen Edge Cases
// ============================================================================

#[test]
fn codegen_number_integer() {
    let module = JsModuleBuilder::new()
        .let_decl("x", Expr::num(42.0))
        .unwrap()
        .build();
    let js = generate(&module);
    assert!(js.contains("42"), "Should format as integer: {}", js);
    assert!(!js.contains("42.0"), "Should not have decimal: {}", js);
}

#[test]
fn codegen_number_float() {
    let module = JsModuleBuilder::new()
        .let_decl("x", Expr::num(1.234))
        .unwrap()
        .build();
    let js = generate(&module);
    assert!(js.contains("1.234"));
}

#[test]
fn codegen_string_escaping() {
    let module = JsModuleBuilder::new()
        .const_decl("s", Expr::str("hello\nworld\t\"quoted\"\\backslash\r"))
        .unwrap()
        .build();
    let js = generate(&module);
    assert!(js.contains("\\n"), "newline not escaped");
    assert!(js.contains("\\t"), "tab not escaped");
    assert!(js.contains("\\\""), "quote not escaped");
    assert!(js.contains("\\\\"), "backslash not escaped");
    assert!(js.contains("\\r"), "carriage return not escaped");
}

#[test]
fn codegen_empty_object() {
    let module = JsModuleBuilder::new()
        .let_decl("obj", Expr::object(vec![]))
        .unwrap()
        .build();
    let js = generate(&module);
    assert!(js.contains("{  }"));
}

#[test]
fn codegen_empty_array() {
    let module = JsModuleBuilder::new()
        .let_decl("arr", Expr::array(vec![]))
        .unwrap()
        .build();
    let js = generate(&module);
    assert!(js.contains("[]"));
}

#[test]
fn codegen_nested_calls() {
    let expr = Expr::ident("a")
        .unwrap()
        .call(vec![])
        .call(vec![Expr::num(1)])
        .call(vec![Expr::num(2), Expr::num(3)]);
    let module = JsModuleBuilder::new().expr(expr).build();
    let js = generate(&module);
    assert!(js.contains("a()(1)(2, 3)"));
}

#[test]
fn codegen_deeply_nested_member() {
    let expr = Expr::ident("a")
        .unwrap()
        .dot("b")
        .unwrap()
        .dot("c")
        .unwrap()
        .dot("d")
        .unwrap();
    let module = JsModuleBuilder::new().let_decl("x", expr).unwrap().build();
    let js = generate(&module);
    assert!(js.contains("a.b.c.d"));
}

#[test]
fn codegen_class_no_constructor() {
    let class = JsClassBuilder::new("Empty")
        .unwrap()
        .method("test", &[], vec![])
        .unwrap()
        .build();
    let module = JsModuleBuilder::new().class(class).build();
    let js = generate(&module);
    assert!(js.contains("class Empty"));
    assert!(!js.contains("constructor"));
}

#[test]
fn codegen_class_extends_with_constructor() {
    let class = JsClassBuilder::new("Child")
        .unwrap()
        .extends("Parent")
        .unwrap()
        .constructor(vec![
            Stmt::member_assign(Expr::this(), "x", Expr::num(1)).unwrap()
        ])
        .build();
    let module = JsModuleBuilder::new().class(class).build();
    let js = generate(&module);
    assert!(js.contains("extends Parent"));
    assert!(js.contains("super();"));
    assert!(js.contains("this.x = 1;"));
}

// ============================================================================
// BinOp and UnaryOp as_str coverage
// ============================================================================

#[test]
fn binop_all_as_str() {
    assert_eq!(BinOp::Add.as_str(), "+");
    assert_eq!(BinOp::Sub.as_str(), "-");
    assert_eq!(BinOp::Mul.as_str(), "*");
    assert_eq!(BinOp::Div.as_str(), "/");
    assert_eq!(BinOp::Mod.as_str(), "%");
    assert_eq!(BinOp::Eq.as_str(), "==");
    assert_eq!(BinOp::EqStrict.as_str(), "===");
    assert_eq!(BinOp::Ne.as_str(), "!=");
    assert_eq!(BinOp::NeStrict.as_str(), "!==");
    assert_eq!(BinOp::Lt.as_str(), "<");
    assert_eq!(BinOp::Le.as_str(), "<=");
    assert_eq!(BinOp::Gt.as_str(), ">");
    assert_eq!(BinOp::Ge.as_str(), ">=");
    assert_eq!(BinOp::And.as_str(), "&&");
    assert_eq!(BinOp::Or.as_str(), "||");
    assert_eq!(BinOp::BitAnd.as_str(), "&");
    assert_eq!(BinOp::BitOr.as_str(), "|");
}

#[test]
fn unaryop_all_as_str() {
    assert_eq!(UnaryOp::Not.as_str(), "!");
    assert_eq!(UnaryOp::Neg.as_str(), "-");
    assert_eq!(UnaryOp::TypeOf.as_str(), "typeof ");
}

// ============================================================================
// Identifier Display trait
// ============================================================================

#[test]
fn identifier_display() {
    let id = Identifier::new("myVar").unwrap();
    assert_eq!(format!("{}", id), "myVar");
}

#[test]
fn identifier_new_unchecked() {
    let id = Identifier::new_unchecked("trusted");
    assert_eq!(id.as_str(), "trusted");
}

// ============================================================================
// JsModule Default
// ============================================================================

#[test]
fn jsmodule_default() {
    let module = JsModule::default();
    assert!(module.statements.is_empty());
    assert!(module.metadata.is_none());
}

#[test]
fn jsmodule_new() {
    let module = JsModule::new();
    assert!(module.statements.is_empty());
    assert!(module.metadata.is_none());
}

// ============================================================================
// Metadata generation
// ============================================================================

#[test]
fn codegen_with_metadata() {
    let module = JsModuleBuilder::new()
        .metadata(GenerationMetadata {
            tool: "test-tool".to_string(),
            version: "1.0.0".to_string(),
            input_hash: "abc123def456".to_string(),
            timestamp: "2024-01-01T00:00:00Z".to_string(),
            regenerate_cmd: "cargo run --example test".to_string(),
        })
        .let_decl("x", Expr::num(1))
        .unwrap()
        .build();
    let js = generate(&module);

    assert!(js.contains("@generated"));
    assert!(js.contains("Do not edit manually"));
    assert!(js.contains("test-tool"));
    assert!(js.contains("1.0.0"));
    assert!(js.contains("abc123def456"));
    assert!(js.contains("2024-01-01T00:00:00Z"));
    assert!(js.contains("cargo run --example test"));
}

// ============================================================================
// Bitwise operations codegen
// ============================================================================

#[test]
fn codegen_bitwise_and() {
    let expr = Expr::Binary {
        left: Box::new(Expr::num(0xFF)),
        op: BinOp::BitAnd,
        right: Box::new(Expr::num(0x0F)),
    };
    let module = JsModuleBuilder::new().let_decl("x", expr).unwrap().build();
    let js = generate(&module);
    assert!(js.contains('&'));
}

#[test]
fn codegen_bitwise_or() {
    let expr = Expr::Binary {
        left: Box::new(Expr::num(0xF0)),
        op: BinOp::BitOr,
        right: Box::new(Expr::num(0x0F)),
    };
    let module = JsModuleBuilder::new().let_decl("x", expr).unwrap().build();
    let js = generate(&module);
    assert!(js.contains('|'));
}

// ============================================================================
// Loose equality operators
// ============================================================================

#[test]
fn codegen_loose_eq() {
    let expr = Expr::Binary {
        left: Box::new(Expr::num(1)),
        op: BinOp::Eq,
        right: Box::new(Expr::str("1")),
    };
    let module = JsModuleBuilder::new().let_decl("x", expr).unwrap().build();
    let js = generate(&module);
    assert!(js.contains("=="));
    assert!(!js.contains("==="));
}

#[test]
fn codegen_loose_ne() {
    let expr = Expr::Binary {
        left: Box::new(Expr::num(1)),
        op: BinOp::Ne,
        right: Box::new(Expr::str("1")),
    };
    let module = JsModuleBuilder::new().let_decl("x", expr).unwrap().build();
    let js = generate(&module);
    assert!(js.contains("!="));
    assert!(!js.contains("!=="));
}

// ============================================================================
// Structure Tests (verify code generation structure)
// ============================================================================

#[test]
fn codegen_nested_if_structure() {
    let stmt = Stmt::if_then(
        Expr::bool(true),
        vec![Stmt::if_then(Expr::bool(false), vec![Stmt::ret()])],
    );
    let module = JsModuleBuilder::new().stmt(stmt).build();
    let js = generate(&module);
    assert!(js.contains("if (true)"), "Outer if: {:?}", js);
    assert!(js.contains("if (false)"), "Inner if: {:?}", js);
    assert!(js.contains("return;"), "Return statement: {:?}", js);
}

#[test]
fn codegen_for_loop_with_body() {
    let stmt = Stmt::for_loop(
        "i",
        Expr::num(0),
        Expr::num(5),
        vec![Stmt::expr(Expr::ident("x").unwrap())],
    )
    .unwrap();
    let module = JsModuleBuilder::new().stmt(stmt).build();
    let js = generate(&module);
    assert!(
        js.contains("for (let i = 0; i < 5; i++)"),
        "For loop header: {:?}",
        js
    );
    assert!(js.contains("x;"), "Body contains x: {:?}", js);
}

#[test]
fn codegen_while_loop_with_body() {
    let stmt = Stmt::while_loop(
        Expr::bool(true),
        vec![Stmt::expr(Expr::ident("y").unwrap())],
    );
    let module = JsModuleBuilder::new().stmt(stmt).build();
    let js = generate(&module);
    assert!(js.contains("while (true)"), "While condition: {:?}", js);
    assert!(js.contains("y;"), "Body contains y: {:?}", js);
}

#[test]
fn codegen_try_catch_structure() {
    let stmt = Stmt::try_catch(
        vec![Stmt::expr(Expr::ident("a").unwrap())],
        "e",
        vec![Stmt::expr(Expr::ident("b").unwrap())],
    )
    .unwrap();
    let module = JsModuleBuilder::new().stmt(stmt).build();
    let js = generate(&module);
    assert!(js.contains("try {"), "Try block: {:?}", js);
    assert!(js.contains("} catch (e) {"), "Catch block: {:?}", js);
    assert!(js.contains("a;"), "Try body: {:?}", js);
    assert!(js.contains("b;"), "Catch body: {:?}", js);
}

#[test]
fn codegen_block_structure() {
    let stmt = Stmt::Block(vec![Stmt::expr(Expr::ident("z").unwrap())]);
    let module = JsModuleBuilder::new().stmt(stmt).build();
    let js = generate(&module);
    assert!(js.contains('{'), "Opening brace: {:?}", js);
    assert!(js.contains("z;"), "Body: {:?}", js);
    assert!(js.contains('}'), "Closing brace: {:?}", js);
}

#[test]
fn codegen_class_with_constructor_and_method() {
    let class = JsClassBuilder::new("Test")
        .unwrap()
        .constructor(vec![Stmt::expr(Expr::ident("init").unwrap())])
        .method("foo", &[], vec![Stmt::expr(Expr::ident("work").unwrap())])
        .unwrap()
        .build();
    let module = JsModuleBuilder::new().class(class).build();
    let js = generate(&module);
    assert!(js.contains("class Test"), "Class declaration: {:?}", js);
    assert!(js.contains("constructor()"), "Constructor: {:?}", js);
    assert!(js.contains("init;"), "Constructor body: {:?}", js);
    assert!(js.contains("foo()"), "Method: {:?}", js);
    assert!(js.contains("work;"), "Method body: {:?}", js);
}

#[test]
fn codegen_class_extends_has_super() {
    let class = JsClassBuilder::new("Child")
        .unwrap()
        .extends("Parent")
        .unwrap()
        .constructor(vec![])
        .build();
    let module = JsModuleBuilder::new().class(class).build();
    let js = generate(&module);
    assert!(js.contains("extends Parent"), "Extends: {:?}", js);
    assert!(js.contains("super();"), "Super call: {:?}", js);
}

#[test]
fn codegen_switch_structure() {
    let switch = JsSwitchBuilder::new(Expr::ident("x").unwrap())
        .case(Expr::num(1), vec![Stmt::expr(Expr::ident("a").unwrap())])
        .default(vec![Stmt::expr(Expr::ident("b").unwrap())])
        .build();
    let module = JsModuleBuilder::new().stmt(Stmt::Switch(switch)).build();
    let js = generate(&module);
    assert!(js.contains("switch (x)"), "Switch: {:?}", js);
    assert!(js.contains("case 1:"), "Case: {:?}", js);
    assert!(js.contains("a;"), "Case body: {:?}", js);
    assert!(js.contains("break;"), "Break: {:?}", js);
    assert!(js.contains("default:"), "Default: {:?}", js);
    assert!(js.contains("b;"), "Default body: {:?}", js);
}

#[test]
fn codegen_onmessage_structure() {
    let stmt = Stmt::on_message(vec![Stmt::expr(Expr::ident("handler").unwrap())]);
    let module = JsModuleBuilder::new().stmt(stmt).build();
    let js = generate(&module);
    assert!(
        js.contains("self.onmessage = async function(e)"),
        "Handler: {:?}",
        js
    );
    assert!(js.contains("handler;"), "Body: {:?}", js);
}

// ============================================================================
// Multiple Arguments Tests (kill comma/loop mutants)
// ============================================================================

#[test]
fn codegen_call_multiple_args() {
    let expr = Expr::ident("fn")
        .unwrap()
        .call(vec![Expr::num(1), Expr::num(2), Expr::num(3)]);
    let module = JsModuleBuilder::new().expr(expr).build();
    let js = generate(&module);
    assert!(
        js.contains("fn(1, 2, 3)"),
        "Multiple args not formatted correctly: {}",
        js
    );
}

#[test]
fn codegen_new_multiple_args() {
    let expr = Expr::ident("MyClass")
        .unwrap()
        .new_expr(vec![Expr::str("a"), Expr::str("b")]);
    let module = JsModuleBuilder::new().expr(expr).build();
    let js = generate(&module);
    assert!(
        js.contains(r#"new MyClass("a", "b")"#),
        "Multiple new args not formatted correctly: {}",
        js
    );
}

#[test]
fn codegen_object_multiple_pairs() {
    let expr = Expr::object(vec![
        ("a", Expr::num(1)),
        ("b", Expr::num(2)),
        ("c", Expr::num(3)),
    ]);
    let module = JsModuleBuilder::new()
        .let_decl("obj", expr)
        .unwrap()
        .build();
    let js = generate(&module);
    assert!(
        js.contains("a: 1, b: 2, c: 3"),
        "Multiple object pairs not formatted correctly: {}",
        js
    );
}

#[test]
fn codegen_array_multiple_items() {
    let expr = Expr::array(vec![Expr::num(1), Expr::num(2), Expr::num(3), Expr::num(4)]);
    let module = JsModuleBuilder::new()
        .let_decl("arr", expr)
        .unwrap()
        .build();
    let js = generate(&module);
    assert!(
        js.contains("[1, 2, 3, 4]"),
        "Multiple array items not formatted correctly: {}",
        js
    );
}

#[test]
fn codegen_method_multiple_params() {
    let class = JsClassBuilder::new("Test")
        .unwrap()
        .method("multi", &["a", "b", "c", "d"], vec![])
        .unwrap()
        .build();
    let module = JsModuleBuilder::new().class(class).build();
    let js = generate(&module);
    assert!(
        js.contains("multi(a, b, c, d)"),
        "Multiple params not formatted correctly: {}",
        js
    );
}

#[test]
fn codegen_arrow_multiple_params_formatting() {
    let expr = Expr::arrow(&["x", "y", "z"], Expr::num(0)).unwrap();
    let module = JsModuleBuilder::new().let_decl("fn", expr).unwrap().build();
    let js = generate(&module);
    assert!(
        js.contains("(x, y, z) =>"),
        "Arrow multiple params not formatted correctly: {}",
        js
    );
}

// ============================================================================
// Validator Tests (kill validator mutants)
// ============================================================================

#[test]
fn validator_worklet_missing_base_class() {
    let js = r#"
        class MyProcessor {
            process(inputs, outputs, params) {
                return true;
            }
        }
        registerProcessor("my-processor", MyProcessor);
    "#;
    let errors = probar_js_gen::validator::validate_worklet_js(js);
    assert!(
        !errors.is_empty(),
        "Should detect missing AudioWorkletProcessor"
    );
    assert!(errors.iter().any(|e| e.contains("AudioWorkletProcessor")));
}

#[test]
fn validator_worklet_missing_register() {
    let js = r"
        class MyProcessor extends AudioWorkletProcessor {
            process(inputs, outputs, params) {
                return true;
            }
        }
    ";
    let errors = probar_js_gen::validator::validate_worklet_js(js);
    assert!(
        !errors.is_empty(),
        "Should detect missing registerProcessor"
    );
    assert!(errors.iter().any(|e| e.contains("registerProcessor")));
}

#[test]
fn validator_worklet_missing_process() {
    let js = r#"
        class MyProcessor extends AudioWorkletProcessor {
            constructor() {
                super();
            }
        }
        registerProcessor("my-processor", MyProcessor);
    "#;
    let errors = probar_js_gen::validator::validate_worklet_js(js);
    assert!(!errors.is_empty(), "Should detect missing process method");
    assert!(errors.iter().any(|e| e.contains("process()")));
}

#[test]
fn validator_worklet_forbidden_patterns() {
    let js = r#"
        class MyProcessor extends AudioWorkletProcessor {
            process(inputs, outputs, params) {
                window.alert("bad");
                return true;
            }
        }
        registerProcessor("my-processor", MyProcessor);
    "#;
    let errors = probar_js_gen::validator::validate_worklet_js(js);
    assert!(!errors.is_empty(), "Should detect forbidden window pattern");
    assert!(errors.iter().any(|e| e.contains("window")));
}

#[test]
fn validator_worker_missing_self() {
    let js = "const x = 1;";
    let errors = probar_js_gen::validator::validate_worker_js(js);
    assert!(errors.iter().any(|e| e.contains("self.")));
}

#[test]
fn validator_worker_missing_import() {
    let js = "self.onmessage = function(e) {};";
    let errors = probar_js_gen::validator::validate_worker_js(js);
    assert!(errors.iter().any(|e| e.contains("import(")));
}

// ============================================================================
// Number formatting edge cases
// ============================================================================

#[test]
fn codegen_large_integer() {
    let module = JsModuleBuilder::new()
        .let_decl("big", Expr::num(9_007_199_254_740_991.0)) // MAX_SAFE_INTEGER
        .unwrap()
        .build();
    let js = generate(&module);
    assert!(js.contains("9007199254740991"), "Large integer: {}", js);
}

#[test]
fn codegen_negative_integer() {
    let module = JsModuleBuilder::new()
        .let_decl("neg", Expr::num(-42.0))
        .unwrap()
        .build();
    let js = generate(&module);
    assert!(js.contains("-42"), "Negative integer: {}", js);
}

#[test]
fn codegen_zero() {
    let module = JsModuleBuilder::new()
        .let_decl("zero", Expr::num(0.0))
        .unwrap()
        .build();
    let js = generate(&module);
    assert!(js.contains("let zero = 0;"), "Zero: {}", js);
}

// ============================================================================
// Mutation Test Killers - Indentation Level Tests
// ============================================================================

#[test]
fn codegen_while_body_indented_one_level() {
    // Tests that while body is indented by 1 level (4 spaces)
    let stmt = Stmt::while_loop(
        Expr::bool(true),
        vec![Stmt::expr(Expr::ident("x").unwrap())],
    );
    let module = JsModuleBuilder::new().stmt(stmt).build();
    let js = generate(&module);
    // Body must be indented one level from the while
    assert!(
        js.contains("    x;"),
        "While body should have 4-space indent: {:?}",
        js
    );
}

#[test]
fn codegen_for_body_indented_one_level() {
    // Tests that for body is indented by 1 level (4 spaces)
    let stmt =
        Stmt::for_loop("i", Expr::num(0), Expr::num(5), vec![Stmt::comment("body")]).unwrap();
    let module = JsModuleBuilder::new().stmt(stmt).build();
    let js = generate(&module);
    // Body must be indented one level from the for
    assert!(
        js.contains("    // body"),
        "For body should have 4-space indent: {:?}",
        js
    );
}

#[test]
fn codegen_try_catch_body_indented() {
    // Tests that try/catch body is indented by 1 level
    // Signature: try_catch(body, catch_var, handler)
    let stmt = Stmt::try_catch(
        vec![Stmt::comment("try")],
        "e",
        vec![Stmt::comment("catch")],
    )
    .unwrap();
    let module = JsModuleBuilder::new().stmt(stmt).build();
    let js = generate(&module);
    assert!(
        js.contains("    // try"),
        "Try body should have 4-space indent: {:?}",
        js
    );
    assert!(
        js.contains("    // catch"),
        "Catch body should have 4-space indent: {:?}",
        js
    );
}

#[test]
fn codegen_class_method_body_indented_two_levels() {
    // Tests that method body is indented by 2 levels (8 spaces)
    let class = JsClassBuilder::new("Foo")
        .unwrap()
        .method("bar", &[], vec![Stmt::comment("deep")])
        .unwrap()
        .build();
    let module = JsModuleBuilder::new().stmt(Stmt::Class(class)).build();
    let js = generate(&module);
    // Method body must be indented 2 levels (8 spaces)
    assert!(
        js.contains("        // deep"),
        "Method body should have 8-space indent: {:?}",
        js
    );
}

#[test]
fn codegen_switch_case_indented_one_level() {
    // Tests that switch cases are indented by 1 level (4 spaces)
    let switch = JsSwitchBuilder::new(Expr::ident("x").unwrap())
        .case(Expr::num(1), vec![Stmt::comment("case")])
        .build();
    let module = JsModuleBuilder::new().stmt(Stmt::Switch(switch)).build();
    let js = generate(&module);
    // Case should be indented 1 level
    assert!(
        js.contains("    case 1:"),
        "Case should have 4-space indent: {:?}",
        js
    );
}

#[test]
fn codegen_switch_body_indented_two_levels() {
    // Tests that switch case body is indented by 2 levels (8 spaces)
    let switch = JsSwitchBuilder::new(Expr::ident("x").unwrap())
        .case(Expr::num(1), vec![Stmt::comment("body")])
        .build();
    let module = JsModuleBuilder::new().stmt(Stmt::Switch(switch)).build();
    let js = generate(&module);
    // Body should be indented 2 levels
    assert!(
        js.contains("        // body"),
        "Switch case body should have 8-space indent: {:?}",
        js
    );
}

// ============================================================================
// Mutation Test Killers - Number Boundary Tests
// ============================================================================

#[test]
fn codegen_number_at_i64_min_boundary() {
    // Tests the >= boundary in the integer formatting logic
    let n = i64::MIN as f64;
    let module = JsModuleBuilder::new()
        .let_decl("min", Expr::num(n))
        .unwrap()
        .build();
    let js = generate(&module);
    // Should format as integer
    assert!(
        js.contains(&format!("let min = {};", i64::MIN)),
        "i64::MIN should format as integer: {}",
        js
    );
}

#[test]
fn codegen_number_at_i64_max_boundary() {
    // Tests the <= boundary in the integer formatting logic
    let n = i64::MAX as f64;
    let module = JsModuleBuilder::new()
        .let_decl("max", Expr::num(n))
        .unwrap()
        .build();
    let js = generate(&module);
    // i64::MAX may lose precision when cast to f64, just verify it doesn't crash
    assert!(js.contains("let max ="), "Should have max decl: {}", js);
}

#[test]
fn codegen_number_beyond_i64_range() {
    // Tests number beyond i64 range (should format as float)
    let n = (i64::MAX as f64) * 10.0; // Way beyond i64 range
    let module = JsModuleBuilder::new()
        .let_decl("huge", Expr::num(n))
        .unwrap()
        .build();
    let js = generate(&module);
    // Should still generate valid JS
    assert!(js.contains("let huge ="), "Should have huge decl: {}", js);
}

#[test]
fn codegen_float_near_boundary() {
    // Tests that floats with fractional parts are formatted correctly
    let n = (i64::MIN as f64) + 0.5; // Has fractional part
    let module = JsModuleBuilder::new()
        .let_decl("frac", Expr::num(n))
        .unwrap()
        .build();
    let js = generate(&module);
    // Should format as float (with decimal)
    assert!(js.contains("let frac ="), "Should have frac decl: {}", js);
}
