//! Property-based tests for probar-js-gen.
//!
//! Uses proptest to verify invariants hold for arbitrary inputs.
//!
//! # References
//! - Claessen & Hughes (2000) "QuickCheck: A Lightweight Tool for Random Testing"
//! - McKeeman (1998) "Differential Testing for Software"

#![allow(clippy::unwrap_used)]

use probar_js_gen::prelude::*;
use proptest::prelude::*;

// === Identifier Property Tests ===

proptest! {
    /// Valid identifiers must be accepted.
    #[test]
    fn prop_valid_identifier_accepted(
        name in "[a-zA-Z_$][a-zA-Z0-9_$]{0,20}"
    ) {
        // Filter out reserved words
        if !Identifier::RESERVED_WORDS.contains(&name.as_str()) {
            let result = Identifier::new(&name);
            prop_assert!(result.is_ok(), "Valid identifier rejected: {}", name);
        }
    }

    /// Empty identifier must be rejected.
    #[test]
    fn prop_empty_identifier_rejected(
        // Any valid prefix
        _prefix in "[a-zA-Z]{0,5}"
    ) {
        let result = Identifier::new("");
        prop_assert!(result.is_err(), "Empty identifier should be rejected");
    }

    /// Identifiers starting with digits must be rejected.
    #[test]
    fn prop_digit_start_rejected(
        digit in "[0-9]",
        suffix in "[a-zA-Z0-9_$]{0,10}"
    ) {
        let name = format!("{}{}", digit, suffix);
        let result = Identifier::new(&name);
        prop_assert!(result.is_err(), "Digit-start should be rejected: {}", name);
    }

    /// Reserved words must be rejected.
    #[test]
    fn prop_reserved_word_rejected(
        idx in 0..Identifier::RESERVED_WORDS.len()
    ) {
        let word = Identifier::RESERVED_WORDS[idx];
        let result = Identifier::new(word);
        prop_assert!(result.is_err(), "Reserved word should be rejected: {}", word);
    }

    /// Identifiers with invalid chars must be rejected.
    #[test]
    fn prop_invalid_chars_rejected(
        valid_prefix in "[a-zA-Z_$]{1,5}",
        invalid_char in "[-!@#%^&*()+=\\[\\]{};':\"<>,./? ]",
        valid_suffix in "[a-zA-Z0-9_$]{0,5}"
    ) {
        let name = format!("{}{}{}", valid_prefix, invalid_char, valid_suffix);
        let result = Identifier::new(&name);
        prop_assert!(result.is_err(), "Invalid char should be rejected: {}", name);
    }
}

// === Expression Property Tests ===

proptest! {
    /// Numbers must round-trip correctly.
    #[test]
    fn prop_number_roundtrip(n in any::<i32>()) {
        let expr = Expr::num(f64::from(n));
        let module = JsModuleBuilder::new()
            .let_decl("x", expr).unwrap()
            .build();
        let js = generate(&module);
        prop_assert!(js.contains(&n.to_string()), "Number not in output: {}", n);
    }

    /// Strings must be properly quoted.
    #[test]
    fn prop_string_quoted(
        s in "[a-zA-Z0-9 ]{1,50}"  // Safe chars only
    ) {
        let expr = Expr::str(&s);
        let module = JsModuleBuilder::new()
            .const_decl("s", expr).unwrap()
            .build();
        let js = generate(&module);
        // Output must contain the string with quotes
        prop_assert!(
            js.contains(&format!(r#""{}""#, s)),
            "String not properly quoted in output"
        );
    }

    /// Boolean literals must be correct.
    #[test]
    fn prop_bool_literal(b in any::<bool>()) {
        let expr = Expr::bool(b);
        let module = JsModuleBuilder::new()
            .let_decl("b", expr).unwrap()
            .build();
        let js = generate(&module);
        let expected = if b { "true" } else { "false" };
        prop_assert!(js.contains(expected), "Bool not found: {}", b);
    }
}

// === Code Generation Property Tests ===

proptest! {
    /// Generated code must never contain forbidden patterns.
    #[test]
    fn prop_no_forbidden_patterns(
        var_name in "[a-zA-Z_][a-zA-Z0-9_]{0,10}",
        value in any::<i32>()
    ) {
        // Skip reserved words
        if Identifier::RESERVED_WORDS.contains(&var_name.as_str()) {
            return Ok(());
        }

        let module = JsModuleBuilder::new()
            .let_decl(&var_name, Expr::num(f64::from(value))).unwrap()
            .build();
        let js = generate(&module);

        for pattern in probar_js_gen::validator::FORBIDDEN_PATTERNS {
            prop_assert!(
                !js.contains(pattern),
                "Forbidden pattern '{}' found in output",
                pattern
            );
        }
    }

    /// Code generation must be deterministic.
    #[test]
    fn prop_deterministic_generation(
        var_name in "[a-zA-Z_][a-zA-Z0-9_]{0,10}",
        value in any::<i32>()
    ) {
        // Skip reserved words
        if Identifier::RESERVED_WORDS.contains(&var_name.as_str()) {
            return Ok(());
        }

        let build = || {
            JsModuleBuilder::new()
                .let_decl(&var_name, Expr::num(f64::from(value))).unwrap()
                .build()
        };

        let js1 = generate(&build());
        let js2 = generate(&build());

        prop_assert_eq!(js1, js2, "Generation not deterministic");
    }
}

// === Manifest Property Tests ===

proptest! {
    /// Hash must change when content changes.
    #[test]
    fn prop_hash_changes_with_content(
        content1 in "[a-zA-Z0-9 ]{1,100}",
        content2 in "[a-zA-Z0-9 ]{1,100}"
    ) {
        // Only test when contents differ
        if content1 != content2 {
            let hash1 = probar_js_gen::manifest::hash_file_contents(&content1);
            let hash2 = probar_js_gen::manifest::hash_file_contents(&content2);
            prop_assert_ne!(hash1, hash2, "Different content should have different hash");
        }
    }

    /// Hash must be deterministic.
    #[test]
    fn prop_hash_deterministic(
        content in "[a-zA-Z0-9 ]{1,100}"
    ) {
        let hash1 = probar_js_gen::manifest::hash_file_contents(&content);
        let hash2 = probar_js_gen::manifest::hash_file_contents(&content);
        prop_assert_eq!(hash1, hash2, "Hash not deterministic");
    }
}

// === Binary Operation Property Tests ===

proptest! {
    /// Binary operations must produce valid parenthesized output.
    #[test]
    fn prop_binop_parenthesized(
        a in -1000i32..1000i32,
        b in -1000i32..1000i32
    ) {
        let expr = Expr::num(f64::from(a)).add(Expr::num(f64::from(b)));
        let module = JsModuleBuilder::new()
            .let_decl("result", expr).unwrap()
            .build();
        let js = generate(&module);

        // Must have parentheses around binary op
        prop_assert!(js.contains('('), "Missing opening paren");
        prop_assert!(js.contains(')'), "Missing closing paren");
        prop_assert!(js.contains('+'), "Missing operator");
    }
}

// === Class Generation Property Tests ===

proptest! {
    /// Classes with extends must have super() call.
    #[test]
    fn prop_extends_has_super(
        class_name in "[A-Z][a-zA-Z0-9]{0,10}",
        parent_name in "[A-Z][a-zA-Z0-9]{0,10}"
    ) {
        // Skip reserved words
        if Identifier::RESERVED_WORDS.contains(&class_name.as_str())
            || Identifier::RESERVED_WORDS.contains(&parent_name.as_str())
        {
            return Ok(());
        }

        let class = JsClassBuilder::new(&class_name).unwrap()
            .extends(&parent_name).unwrap()
            .constructor(vec![])
            .build();

        let module = JsModuleBuilder::new().class(class).build();
        let js = generate(&module);

        prop_assert!(
            js.contains("super()"),
            "Class with extends must have super() in constructor"
        );
    }

    /// Classes without extends must NOT have super() call.
    #[test]
    fn prop_no_extends_no_super(
        class_name in "[A-Z][a-zA-Z0-9]{0,10}"
    ) {
        // Skip reserved words
        if Identifier::RESERVED_WORDS.contains(&class_name.as_str()) {
            return Ok(());
        }

        let class = JsClassBuilder::new(&class_name).unwrap()
            .constructor(vec![])
            .build();

        let module = JsModuleBuilder::new().class(class).build();
        let js = generate(&module);

        prop_assert!(
            !js.contains("super()"),
            "Class without extends must NOT have super()"
        );
    }
}

// === Special Character Escaping Tests ===

proptest! {
    /// Special characters in strings must be escaped.
    #[test]
    fn prop_special_chars_escaped(
        prefix in "[a-zA-Z]{0,10}",
        suffix in "[a-zA-Z]{0,10}"
    ) {
        // Test each special char
        for (char, escaped) in &[
            ('"', "\\\""),
            ('\\', "\\\\"),
            ('\n', "\\n"),
            ('\r', "\\r"),
            ('\t', "\\t"),
        ] {
            let s = format!("{}{}{}", prefix, char, suffix);
            let expr = Expr::str(&s);
            let module = JsModuleBuilder::new()
                .const_decl("s", expr).unwrap()
                .build();
            let js = generate(&module);

            prop_assert!(
                js.contains(escaped),
                "Character {:?} should be escaped to '{}'",
                char,
                escaped
            );
        }
    }
}
