//! Property-based tests for Keypad (PMAT-CALC-007)
//!
//! Probar: Error prevention - Property tests catch edge cases that humans miss

#![allow(clippy::unwrap_used)]

use proptest::prelude::*;
use showcase_calculator::wasm::{KeypadAction, WasmKeypad};

// ===== Strategy definitions =====

/// Generate any valid digit (0-9)
fn digit_strategy() -> impl Strategy<Value = u8> {
    0u8..=9u8
}

/// Generate any valid operator
fn operator_strategy() -> impl Strategy<Value = char> {
    prop_oneof![
        Just('+'),
        Just('-'),
        Just('*'),
        Just('/'),
        Just('^'),
        Just('%')
    ]
}

/// Generate any valid keypad action
fn keypad_action_strategy() -> impl Strategy<Value = KeypadAction> {
    prop_oneof![
        digit_strategy().prop_map(KeypadAction::Digit),
        Just(KeypadAction::Decimal),
        operator_strategy().prop_map(KeypadAction::Operator),
        Just(KeypadAction::Equals),
        Just(KeypadAction::Clear),
        Just(KeypadAction::OpenParen),
        Just(KeypadAction::CloseParen),
    ]
}

/// Generate valid grid positions
fn grid_position_strategy() -> impl Strategy<Value = (usize, usize)> {
    (0usize..5usize, 0usize..4usize)
}

// ===== Property tests for KeypadAction =====

proptest! {
    /// All digits should convert to their character representation
    #[test]
    fn prop_digit_action_to_char(d in digit_strategy()) {
        let action = KeypadAction::Digit(d);
        let ch = action.to_char();
        prop_assert!(ch.is_some());
        prop_assert_eq!(ch.unwrap().to_digit(10), Some(d as u32));
    }

    /// All operators should convert to their character
    #[test]
    fn prop_operator_action_to_char(op in operator_strategy()) {
        let action = KeypadAction::Operator(op);
        prop_assert_eq!(action.to_char(), Some(op));
    }

    /// Every action should have a non-empty label
    #[test]
    fn prop_action_has_label(action in keypad_action_strategy()) {
        let label = action.label();
        prop_assert!(!label.is_empty());
    }

    /// Actions are copy-able
    #[test]
    fn prop_action_copy_identity(action in keypad_action_strategy()) {
        let copied = action;
        prop_assert_eq!(action, copied);
    }
}

// ===== Property tests for WasmKeypad =====

proptest! {
    /// Button at valid position should exist
    #[test]
    fn prop_button_at_valid_position_exists((row, col) in grid_position_strategy()) {
        let keypad = WasmKeypad::new();
        let button = keypad.get_button_at(row, col);
        prop_assert!(button.is_some());
    }

    /// Button at invalid row should not exist
    #[test]
    fn prop_button_at_invalid_row_missing(row in 5usize..100usize, col in 0usize..4usize) {
        let keypad = WasmKeypad::new();
        prop_assert!(keypad.get_button_at(row, col).is_none());
    }

    /// Button at invalid col should not exist
    #[test]
    fn prop_button_at_invalid_col_missing(row in 0usize..5usize, col in 4usize..100usize) {
        let keypad = WasmKeypad::new();
        prop_assert!(keypad.get_button_at(row, col).is_none());
    }

    /// Every button should have a unique ID
    #[test]
    fn prop_all_buttons_have_unique_ids(_seed in any::<u32>()) {
        let keypad = WasmKeypad::new();
        let mut ids = std::collections::HashSet::new();
        for btn in keypad.buttons() {
            prop_assert!(ids.insert(btn.id.clone()), "Duplicate ID: {}", btn.id);
        }
    }

    /// Every button should have a unique position
    #[test]
    fn prop_all_buttons_have_unique_positions(_seed in any::<u32>()) {
        let keypad = WasmKeypad::new();
        let mut positions = std::collections::HashSet::new();
        for btn in keypad.buttons() {
            let pos = (btn.row, btn.col);
            prop_assert!(positions.insert(pos), "Duplicate position: {:?}", pos);
        }
    }

    /// Finding button by ID then looking up by position should be consistent
    #[test]
    fn prop_button_id_position_consistency((row, col) in grid_position_strategy()) {
        let keypad = WasmKeypad::new();
        if let Some(btn) = keypad.get_button_at(row, col) {
            let found = keypad.find_button_by_id(&btn.id);
            prop_assert!(found.is_some());
            prop_assert_eq!(found.unwrap().row, row);
            prop_assert_eq!(found.unwrap().col, col);
        }
    }

    /// All digits should have a findable button
    #[test]
    fn prop_all_digits_findable(d in digit_strategy()) {
        let keypad = WasmKeypad::new();
        let ch = char::from_digit(d as u32, 10).unwrap();
        prop_assert!(keypad.find_button_by_char(ch).is_some());
    }

    /// All operators should have a findable button
    #[test]
    fn prop_all_operators_findable(op in operator_strategy()) {
        let keypad = WasmKeypad::new();
        // Note: % is not in the standard keypad, filter it out
        if op != '%' {
            prop_assert!(keypad.find_button_by_char(op).is_some());
        }
    }
}

// ===== Property tests for keyboard mapping =====

proptest! {
    /// All digit keys should map to digit actions
    #[test]
    fn prop_digit_keys_map_to_digit_actions(d in digit_strategy()) {
        let key = d.to_string();
        let action = WasmKeypad::key_to_action(&key);
        prop_assert_eq!(action, Some(KeypadAction::Digit(d)));
    }

    /// All operator keys should map to operator actions
    #[test]
    fn prop_operator_keys_map_to_operator_actions(op in operator_strategy()) {
        let key = op.to_string();
        let action = WasmKeypad::key_to_action(&key);
        prop_assert_eq!(action, Some(KeypadAction::Operator(op)));
    }

    /// Unknown keys should map to None
    #[test]
    fn prop_unknown_keys_map_to_none(key in "[a-zA-Z]{2,10}") {
        // Multi-character keys that aren't special should return None
        let action = WasmKeypad::key_to_action(&key);
        if key != "Enter" && key != "Escape" {
            prop_assert!(action.is_none() || matches!(action, Some(KeypadAction::Clear)));
        }
    }
}

// ===== Property tests for DOM element creation =====

proptest! {
    /// All created DOM elements should have the keypad-btn class
    #[test]
    fn prop_all_dom_elements_have_class(_seed in any::<u32>()) {
        let keypad = WasmKeypad::new();
        let elements = keypad.create_dom_elements();
        for elem in elements {
            prop_assert!(elem.has_class("keypad-btn"));
        }
    }

    /// All created DOM elements should be buttons
    #[test]
    fn prop_all_dom_elements_are_buttons(_seed in any::<u32>()) {
        let keypad = WasmKeypad::new();
        let elements = keypad.create_dom_elements();
        for elem in elements {
            prop_assert_eq!(elem.tag, "button");
        }
    }

    /// Keypad element should have all 20 children
    #[test]
    fn prop_keypad_element_has_20_children(_seed in any::<u32>()) {
        let keypad = WasmKeypad::new();
        let elem = keypad.create_keypad_element();
        prop_assert_eq!(elem.children.len(), 20);
    }
}

// ===== Invariant tests =====

#[test]
fn invariant_keypad_always_has_20_buttons() {
    let keypad = WasmKeypad::new();
    assert_eq!(keypad.button_count(), 20);
}

#[test]
fn invariant_keypad_always_5_by_4() {
    let keypad = WasmKeypad::new();
    assert_eq!(keypad.dimensions(), (5, 4));
}

#[test]
fn invariant_keypad_has_all_digits() {
    let keypad = WasmKeypad::new();
    for d in 0..=9 {
        let ch = char::from_digit(d, 10).unwrap();
        assert!(
            keypad.find_button_by_char(ch).is_some(),
            "Missing digit {d}"
        );
    }
}

#[test]
fn invariant_keypad_has_required_operators() {
    let keypad = WasmKeypad::new();
    for op in ['+', '-', '*', '/', '^'] {
        assert!(
            keypad.find_button_by_char(op).is_some(),
            "Missing operator {op}"
        );
    }
}

#[test]
fn invariant_keypad_has_special_buttons() {
    let keypad = WasmKeypad::new();
    assert!(keypad.find_button_by_id("btn-equals").is_some());
    assert!(keypad.find_button_by_id("btn-clear").is_some());
    assert!(keypad.find_button_by_id("btn-decimal").is_some());
    assert!(keypad.find_button_by_id("btn-open-paren").is_some());
    assert!(keypad.find_button_by_id("btn-close-paren").is_some());
}
