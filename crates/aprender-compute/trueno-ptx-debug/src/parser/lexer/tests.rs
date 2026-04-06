use super::*;

#[test]
fn lex_directives() {
    let ptx = ".version 8.0\n.target sm_70\n.address_size 64";
    let mut lexer = Lexer::new(ptx);

    let tok = lexer.next_token().expect("lexer should produce token");
    assert_eq!(tok.kind, TokenKind::Directive);
    assert!(tok.text.contains(".version"), "Expected .version, got: {}", tok.text);
    assert!(tok.text.contains("8.0"), "Expected 8.0, got: {}", tok.text);

    let tok = lexer.next_token().expect("lexer should produce token");
    assert_eq!(tok.kind, TokenKind::Directive);
    assert!(tok.text.contains(".target"), "Expected .target, got: {}", tok.text);

    let tok = lexer.next_token().expect("lexer should produce token");
    assert_eq!(tok.kind, TokenKind::Directive);
    assert!(tok.text.contains(".address_size"), "Expected .address_size, got: {}", tok.text);
}

#[test]
fn lex_entry_keyword() {
    let ptx = ".entry test_kernel";
    let mut lexer = Lexer::new(ptx);

    let tok = lexer.next_token().expect("lexer should produce token");
    assert_eq!(tok.kind, TokenKind::Entry);
}

#[test]
fn lex_registers() {
    let ptx = "%r0 %rd1 %f2";
    let mut lexer = Lexer::new(ptx);

    let tok = lexer.next_token().expect("lexer should produce token");
    assert_eq!(tok.kind, TokenKind::Identifier);
    assert_eq!(tok.text, "%r0");

    let tok = lexer.next_token().expect("lexer should produce token");
    assert_eq!(tok.kind, TokenKind::Identifier);
    assert_eq!(tok.text, "%rd1");

    let tok = lexer.next_token().expect("lexer should produce token");
    assert_eq!(tok.kind, TokenKind::Identifier);
    assert_eq!(tok.text, "%f2");
}

#[test]
fn lex_instructions() {
    let ptx = "ld.shared.u32 %r0, [%r1]";
    let mut lexer = Lexer::new(ptx);

    let tok = lexer.next_token().expect("lexer should produce token");
    assert_eq!(tok.kind, TokenKind::Instruction);
    assert!(tok.text.starts_with("ld"));
}

#[test]
fn lex_numbers() {
    let ptx = "42 0x1234 3.14";
    let mut lexer = Lexer::new(ptx);

    let tok = lexer.next_token().expect("lexer should produce token");
    assert_eq!(tok.kind, TokenKind::Integer);
    assert_eq!(tok.text, "42");

    let tok = lexer.next_token().expect("lexer should produce token");
    assert_eq!(tok.kind, TokenKind::Integer);
    assert_eq!(tok.text, "0x1234");

    let tok = lexer.next_token().expect("lexer should produce token");
    assert_eq!(tok.kind, TokenKind::Float);
    assert_eq!(tok.text, "3.14");
}

#[test]
fn lex_comments() {
    let ptx = "mov.u32 %r0, 0 // comment\nmov.u32 %r1, 1";
    let mut lexer = Lexer::new(ptx);

    let tok = lexer.next_token().expect("lexer should produce token");
    assert_eq!(tok.kind, TokenKind::Instruction);

    // Comment should be skipped
    let tok = lexer.next_token().expect("lexer should produce token");
    assert_eq!(tok.kind, TokenKind::Instruction);
}

#[test]
fn lex_labels() {
    let ptx = "loop_start:";
    let mut lexer = Lexer::new(ptx);

    let tok = lexer.next_token().expect("lexer should produce token");
    assert_eq!(tok.kind, TokenKind::Label);
    assert!(tok.text.contains("loop_start"));
}
