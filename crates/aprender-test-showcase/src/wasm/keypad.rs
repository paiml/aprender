//! WASM Keypad for Calculator
//!
//! Probar: Visual feedback - Visual buttons for browser-based interaction
//!
//! This module provides keypad functionality for the WASM calculator,
//! mirroring the TUI keypad for cross-platform consistency.

use super::dom::{DomElement, MockDom};

/// Actions that keypad buttons can perform (shared with TUI)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeypadAction {
    /// Insert a digit (0-9)
    Digit(u8),
    /// Insert a decimal point
    Decimal,
    /// Insert an operator
    Operator(char),
    /// Evaluate the expression
    Equals,
    /// Clear the input
    Clear,
    /// Open parenthesis
    OpenParen,
    /// Close parenthesis
    CloseParen,
}

impl KeypadAction {
    /// Returns the character to insert for this action
    #[must_use]
    pub fn to_char(&self) -> Option<char> {
        match self {
            KeypadAction::Digit(d) => char::from_digit(*d as u32, 10),
            KeypadAction::Decimal => Some('.'),
            KeypadAction::Operator(op) => Some(*op),
            KeypadAction::OpenParen => Some('('),
            KeypadAction::CloseParen => Some(')'),
            KeypadAction::Equals | KeypadAction::Clear => None,
        }
    }

    /// Returns the button label for this action
    #[must_use]
    pub fn label(&self) -> String {
        match self {
            KeypadAction::Digit(d) => d.to_string(),
            KeypadAction::Decimal => ".".to_string(),
            KeypadAction::Operator(op) => op.to_string(),
            KeypadAction::Equals => "=".to_string(),
            KeypadAction::Clear => "C".to_string(),
            KeypadAction::OpenParen => "(".to_string(),
            KeypadAction::CloseParen => ")".to_string(),
        }
    }
}

/// A single keypad button definition
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeypadButtonDef {
    /// The action this button performs
    pub action: KeypadAction,
    /// The DOM element ID for this button
    pub id: String,
    /// Grid row (0-indexed)
    pub row: usize,
    /// Grid column (0-indexed)
    pub col: usize,
}

impl KeypadButtonDef {
    /// Creates a new button definition
    #[must_use]
    pub fn new(action: KeypadAction, row: usize, col: usize) -> Self {
        let id = match action {
            KeypadAction::Digit(d) => format!("btn-{}", d),
            KeypadAction::Decimal => "btn-decimal".to_string(),
            KeypadAction::Operator(op) => format!("btn-{}", op_name(op)),
            KeypadAction::Equals => "btn-equals".to_string(),
            KeypadAction::Clear => "btn-clear".to_string(),
            KeypadAction::OpenParen => "btn-open-paren".to_string(),
            KeypadAction::CloseParen => "btn-close-paren".to_string(),
        };
        Self {
            action,
            id,
            row,
            col,
        }
    }
}

/// Returns a name for an operator (for element IDs)
fn op_name(op: char) -> &'static str {
    match op {
        '+' => "plus",
        '-' => "minus",
        '*' => "times",
        '/' => "divide",
        '^' => "power",
        '%' => "mod",
        _ => "op",
    }
}

/// WASM Keypad layout definition
/// Layout:
/// ```text
/// [ 7 ] [ 8 ] [ 9 ] [ / ]
/// [ 4 ] [ 5 ] [ 6 ] [ * ]
/// [ 1 ] [ 2 ] [ 3 ] [ - ]
/// [ 0 ] [ . ] [ = ] [ + ]
/// [ C ] [ ( ] [ ) ] [ ^ ]
/// ```
#[derive(Debug, Clone)]
pub struct WasmKeypad {
    /// Button definitions
    buttons: Vec<KeypadButtonDef>,
    /// Number of columns
    cols: usize,
    /// Number of rows
    rows: usize,
}

impl Default for WasmKeypad {
    fn default() -> Self {
        Self::new()
    }
}

impl WasmKeypad {
    /// Creates a new standard calculator keypad
    #[must_use]
    pub fn new() -> Self {
        let buttons = vec![
            // Row 0: 7 8 9 /
            KeypadButtonDef::new(KeypadAction::Digit(7), 0, 0),
            KeypadButtonDef::new(KeypadAction::Digit(8), 0, 1),
            KeypadButtonDef::new(KeypadAction::Digit(9), 0, 2),
            KeypadButtonDef::new(KeypadAction::Operator('/'), 0, 3),
            // Row 1: 4 5 6 *
            KeypadButtonDef::new(KeypadAction::Digit(4), 1, 0),
            KeypadButtonDef::new(KeypadAction::Digit(5), 1, 1),
            KeypadButtonDef::new(KeypadAction::Digit(6), 1, 2),
            KeypadButtonDef::new(KeypadAction::Operator('*'), 1, 3),
            // Row 2: 1 2 3 -
            KeypadButtonDef::new(KeypadAction::Digit(1), 2, 0),
            KeypadButtonDef::new(KeypadAction::Digit(2), 2, 1),
            KeypadButtonDef::new(KeypadAction::Digit(3), 2, 2),
            KeypadButtonDef::new(KeypadAction::Operator('-'), 2, 3),
            // Row 3: 0 . = +
            KeypadButtonDef::new(KeypadAction::Digit(0), 3, 0),
            KeypadButtonDef::new(KeypadAction::Decimal, 3, 1),
            KeypadButtonDef::new(KeypadAction::Equals, 3, 2),
            KeypadButtonDef::new(KeypadAction::Operator('+'), 3, 3),
            // Row 4: C ( ) ^
            KeypadButtonDef::new(KeypadAction::Clear, 4, 0),
            KeypadButtonDef::new(KeypadAction::OpenParen, 4, 1),
            KeypadButtonDef::new(KeypadAction::CloseParen, 4, 2),
            KeypadButtonDef::new(KeypadAction::Operator('^'), 4, 3),
        ];

        Self {
            buttons,
            cols: 4,
            rows: 5,
        }
    }

    /// Returns the number of buttons
    #[must_use]
    pub fn button_count(&self) -> usize {
        self.buttons.len()
    }

    /// Returns the grid dimensions (rows, cols)
    #[must_use]
    pub fn dimensions(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Gets all button definitions
    #[must_use]
    pub fn buttons(&self) -> &[KeypadButtonDef] {
        &self.buttons
    }

    /// Gets a button by index
    #[must_use]
    pub fn get_button(&self, index: usize) -> Option<&KeypadButtonDef> {
        self.buttons.get(index)
    }

    /// Gets a button by row and column
    #[must_use]
    pub fn get_button_at(&self, row: usize, col: usize) -> Option<&KeypadButtonDef> {
        if row < self.rows && col < self.cols {
            self.buttons.get(row * self.cols + col)
        } else {
            None
        }
    }

    /// Finds a button by element ID
    #[must_use]
    pub fn find_button_by_id(&self, id: &str) -> Option<&KeypadButtonDef> {
        self.buttons.iter().find(|b| b.id == id)
    }

    /// Finds a button by the character it inserts
    #[must_use]
    pub fn find_button_by_char(&self, ch: char) -> Option<&KeypadButtonDef> {
        self.buttons.iter().find(|b| b.action.to_char() == Some(ch))
    }

    /// Creates DOM elements for all keypad buttons
    pub fn create_dom_elements(&self) -> Vec<DomElement> {
        self.buttons
            .iter()
            .map(|btn| {
                DomElement::new("button")
                    .with_id(&btn.id)
                    .with_text(&btn.action.label())
                    .with_class("keypad-btn")
                    .with_class(&format!("keypad-row-{}", btn.row))
                    .with_class(&format!("keypad-col-{}", btn.col))
                    .with_attr("data-action", &format!("{:?}", btn.action))
            })
            .collect()
    }

    /// Creates a keypad container element with all buttons
    #[must_use]
    pub fn create_keypad_element(&self) -> DomElement {
        let mut keypad = DomElement::new("div")
            .with_id("calc-keypad")
            .with_class("keypad");

        for btn_elem in self.create_dom_elements() {
            keypad = keypad.with_child(btn_elem);
        }

        keypad
    }

    /// Processes a button click event and returns the action
    #[must_use]
    pub fn handle_click(&self, element_id: &str) -> Option<KeypadAction> {
        self.find_button_by_id(element_id).map(|btn| btn.action)
    }

    /// Maps a keyboard key to a keypad action
    #[must_use]
    pub fn key_to_action(key: &str) -> Option<KeypadAction> {
        match key {
            "0" => Some(KeypadAction::Digit(0)),
            "1" => Some(KeypadAction::Digit(1)),
            "2" => Some(KeypadAction::Digit(2)),
            "3" => Some(KeypadAction::Digit(3)),
            "4" => Some(KeypadAction::Digit(4)),
            "5" => Some(KeypadAction::Digit(5)),
            "6" => Some(KeypadAction::Digit(6)),
            "7" => Some(KeypadAction::Digit(7)),
            "8" => Some(KeypadAction::Digit(8)),
            "9" => Some(KeypadAction::Digit(9)),
            "." => Some(KeypadAction::Decimal),
            "+" => Some(KeypadAction::Operator('+')),
            "-" => Some(KeypadAction::Operator('-')),
            "*" => Some(KeypadAction::Operator('*')),
            "/" => Some(KeypadAction::Operator('/')),
            "^" => Some(KeypadAction::Operator('^')),
            "%" => Some(KeypadAction::Operator('%')),
            "(" => Some(KeypadAction::OpenParen),
            ")" => Some(KeypadAction::CloseParen),
            "Enter" | "=" => Some(KeypadAction::Equals),
            "Escape" | "c" | "C" => Some(KeypadAction::Clear),
            _ => None,
        }
    }
}

/// Extension trait for MockDom to add keypad
pub trait MockDomKeypadExt {
    /// Adds keypad to an existing calculator DOM
    fn add_keypad(&mut self, keypad: &WasmKeypad);
}

impl MockDomKeypadExt for MockDom {
    fn add_keypad(&mut self, keypad: &WasmKeypad) {
        let keypad_elem = keypad.create_keypad_element();

        // Register keypad container
        self.register_element(keypad_elem);

        // Register individual buttons
        for btn_def in keypad.buttons() {
            let btn_elem = DomElement::new("button")
                .with_id(&btn_def.id)
                .with_text(&btn_def.action.label())
                .with_class("keypad-btn");
            self.register_element(btn_elem);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===== KeypadAction tests =====

    #[test]
    fn test_keypad_action_digit_to_char() {
        for d in 0..=9 {
            let action = KeypadAction::Digit(d);
            assert_eq!(action.to_char(), char::from_digit(d as u32, 10));
        }
    }

    #[test]
    fn test_keypad_action_operator_to_char() {
        for op in ['+', '-', '*', '/', '^'] {
            let action = KeypadAction::Operator(op);
            assert_eq!(action.to_char(), Some(op));
        }
    }

    #[test]
    fn test_keypad_action_decimal_to_char() {
        assert_eq!(KeypadAction::Decimal.to_char(), Some('.'));
    }

    #[test]
    fn test_keypad_action_parens_to_char() {
        assert_eq!(KeypadAction::OpenParen.to_char(), Some('('));
        assert_eq!(KeypadAction::CloseParen.to_char(), Some(')'));
    }

    #[test]
    fn test_keypad_action_equals_clear_no_char() {
        assert_eq!(KeypadAction::Equals.to_char(), None);
        assert_eq!(KeypadAction::Clear.to_char(), None);
    }

    #[test]
    fn test_keypad_action_label() {
        assert_eq!(KeypadAction::Digit(5).label(), "5");
        assert_eq!(KeypadAction::Decimal.label(), ".");
        assert_eq!(KeypadAction::Operator('+').label(), "+");
        assert_eq!(KeypadAction::Equals.label(), "=");
        assert_eq!(KeypadAction::Clear.label(), "C");
        assert_eq!(KeypadAction::OpenParen.label(), "(");
        assert_eq!(KeypadAction::CloseParen.label(), ")");
    }

    #[test]
    fn test_keypad_action_copy() {
        let action = KeypadAction::Digit(5);
        let copied = action;
        assert_eq!(action, copied);
    }

    #[test]
    fn test_keypad_action_debug() {
        let action = KeypadAction::Operator('+');
        let debug = format!("{:?}", action);
        assert!(debug.contains("Operator"));
    }

    // ===== KeypadButtonDef tests =====

    #[test]
    fn test_button_def_new_digit() {
        let btn = KeypadButtonDef::new(KeypadAction::Digit(5), 1, 1);
        assert_eq!(btn.id, "btn-5");
        assert_eq!(btn.row, 1);
        assert_eq!(btn.col, 1);
    }

    #[test]
    fn test_button_def_new_operators() {
        let btn_plus = KeypadButtonDef::new(KeypadAction::Operator('+'), 3, 3);
        assert_eq!(btn_plus.id, "btn-plus");

        let btn_minus = KeypadButtonDef::new(KeypadAction::Operator('-'), 2, 3);
        assert_eq!(btn_minus.id, "btn-minus");

        let btn_times = KeypadButtonDef::new(KeypadAction::Operator('*'), 1, 3);
        assert_eq!(btn_times.id, "btn-times");

        let btn_divide = KeypadButtonDef::new(KeypadAction::Operator('/'), 0, 3);
        assert_eq!(btn_divide.id, "btn-divide");

        let btn_power = KeypadButtonDef::new(KeypadAction::Operator('^'), 4, 3);
        assert_eq!(btn_power.id, "btn-power");
    }

    #[test]
    fn test_button_def_new_special() {
        let decimal = KeypadButtonDef::new(KeypadAction::Decimal, 3, 1);
        assert_eq!(decimal.id, "btn-decimal");

        let equals = KeypadButtonDef::new(KeypadAction::Equals, 3, 2);
        assert_eq!(equals.id, "btn-equals");

        let clear = KeypadButtonDef::new(KeypadAction::Clear, 4, 0);
        assert_eq!(clear.id, "btn-clear");

        let open = KeypadButtonDef::new(KeypadAction::OpenParen, 4, 1);
        assert_eq!(open.id, "btn-open-paren");

        let close = KeypadButtonDef::new(KeypadAction::CloseParen, 4, 2);
        assert_eq!(close.id, "btn-close-paren");
    }

    #[test]
    fn test_button_def_clone() {
        let btn = KeypadButtonDef::new(KeypadAction::Digit(7), 0, 0);
        let cloned = btn.clone();
        assert_eq!(btn, cloned);
    }

    #[test]
    fn test_button_def_debug() {
        let btn = KeypadButtonDef::new(KeypadAction::Digit(9), 0, 2);
        let debug = format!("{:?}", btn);
        assert!(debug.contains("KeypadButtonDef"));
    }

    // ===== op_name tests =====

    #[test]
    fn test_op_name_all() {
        assert_eq!(op_name('+'), "plus");
        assert_eq!(op_name('-'), "minus");
        assert_eq!(op_name('*'), "times");
        assert_eq!(op_name('/'), "divide");
        assert_eq!(op_name('^'), "power");
        assert_eq!(op_name('%'), "mod");
        assert_eq!(op_name('?'), "op"); // unknown
    }

    // ===== WasmKeypad tests =====

    #[test]
    fn test_wasm_keypad_new() {
        let keypad = WasmKeypad::new();
        assert_eq!(keypad.button_count(), 20);
    }

    #[test]
    fn test_wasm_keypad_default() {
        let keypad = WasmKeypad::default();
        assert_eq!(keypad.button_count(), 20);
    }

    #[test]
    fn test_wasm_keypad_dimensions() {
        let keypad = WasmKeypad::new();
        assert_eq!(keypad.dimensions(), (5, 4));
    }

    #[test]
    fn test_wasm_keypad_get_button() {
        let keypad = WasmKeypad::new();
        let btn = keypad.get_button(0).unwrap();
        assert_eq!(btn.action, KeypadAction::Digit(7));
    }

    #[test]
    fn test_wasm_keypad_get_button_out_of_bounds() {
        let keypad = WasmKeypad::new();
        assert!(keypad.get_button(100).is_none());
    }

    #[test]
    fn test_wasm_keypad_get_button_at() {
        let keypad = WasmKeypad::new();
        // Row 0, Col 0 = 7
        assert_eq!(
            keypad.get_button_at(0, 0).unwrap().action,
            KeypadAction::Digit(7)
        );
        // Row 0, Col 3 = /
        assert_eq!(
            keypad.get_button_at(0, 3).unwrap().action,
            KeypadAction::Operator('/')
        );
        // Row 4, Col 0 = C
        assert_eq!(
            keypad.get_button_at(4, 0).unwrap().action,
            KeypadAction::Clear
        );
    }

    #[test]
    fn test_wasm_keypad_get_button_at_out_of_bounds() {
        let keypad = WasmKeypad::new();
        assert!(keypad.get_button_at(10, 10).is_none());
    }

    #[test]
    fn test_wasm_keypad_get_button_at_boundary_row() {
        let keypad = WasmKeypad::new();
        // Row 4 is valid (last row), row 5 is not
        assert!(keypad.get_button_at(4, 0).is_some());
        assert!(keypad.get_button_at(5, 0).is_none());
    }

    #[test]
    fn test_wasm_keypad_get_button_at_boundary_col() {
        let keypad = WasmKeypad::new();
        // Col 3 is valid (last col), col 4 is not
        assert!(keypad.get_button_at(0, 3).is_some());
        assert!(keypad.get_button_at(0, 4).is_none());
    }

    #[test]
    fn test_wasm_keypad_get_button_at_exact_boundary() {
        let keypad = WasmKeypad::new();
        // Test exact boundary: row == rows should fail (5 == 5)
        // but col is valid (< 4)
        assert!(keypad.get_button_at(5, 0).is_none());
        assert!(keypad.get_button_at(5, 1).is_none());
        assert!(keypad.get_button_at(5, 2).is_none());
        assert!(keypad.get_button_at(5, 3).is_none());
    }

    #[test]
    fn test_wasm_keypad_find_by_id() {
        let keypad = WasmKeypad::new();
        let btn = keypad.find_button_by_id("btn-5").unwrap();
        assert_eq!(btn.action, KeypadAction::Digit(5));
    }

    #[test]
    fn test_wasm_keypad_find_by_id_not_found() {
        let keypad = WasmKeypad::new();
        assert!(keypad.find_button_by_id("nonexistent").is_none());
    }

    #[test]
    fn test_wasm_keypad_find_by_char() {
        let keypad = WasmKeypad::new();
        let btn = keypad.find_button_by_char('+').unwrap();
        assert_eq!(btn.action, KeypadAction::Operator('+'));
    }

    #[test]
    fn test_wasm_keypad_find_by_char_not_found() {
        let keypad = WasmKeypad::new();
        assert!(keypad.find_button_by_char('?').is_none());
    }

    // ===== Keypad layout verification =====

    #[test]
    fn test_keypad_row_0() {
        let keypad = WasmKeypad::new();
        assert_eq!(
            keypad.get_button_at(0, 0).unwrap().action,
            KeypadAction::Digit(7)
        );
        assert_eq!(
            keypad.get_button_at(0, 1).unwrap().action,
            KeypadAction::Digit(8)
        );
        assert_eq!(
            keypad.get_button_at(0, 2).unwrap().action,
            KeypadAction::Digit(9)
        );
        assert_eq!(
            keypad.get_button_at(0, 3).unwrap().action,
            KeypadAction::Operator('/')
        );
    }

    #[test]
    fn test_keypad_row_1() {
        let keypad = WasmKeypad::new();
        assert_eq!(
            keypad.get_button_at(1, 0).unwrap().action,
            KeypadAction::Digit(4)
        );
        assert_eq!(
            keypad.get_button_at(1, 1).unwrap().action,
            KeypadAction::Digit(5)
        );
        assert_eq!(
            keypad.get_button_at(1, 2).unwrap().action,
            KeypadAction::Digit(6)
        );
        assert_eq!(
            keypad.get_button_at(1, 3).unwrap().action,
            KeypadAction::Operator('*')
        );
    }

    #[test]
    fn test_keypad_row_2() {
        let keypad = WasmKeypad::new();
        assert_eq!(
            keypad.get_button_at(2, 0).unwrap().action,
            KeypadAction::Digit(1)
        );
        assert_eq!(
            keypad.get_button_at(2, 1).unwrap().action,
            KeypadAction::Digit(2)
        );
        assert_eq!(
            keypad.get_button_at(2, 2).unwrap().action,
            KeypadAction::Digit(3)
        );
        assert_eq!(
            keypad.get_button_at(2, 3).unwrap().action,
            KeypadAction::Operator('-')
        );
    }

    #[test]
    fn test_keypad_row_3() {
        let keypad = WasmKeypad::new();
        assert_eq!(
            keypad.get_button_at(3, 0).unwrap().action,
            KeypadAction::Digit(0)
        );
        assert_eq!(
            keypad.get_button_at(3, 1).unwrap().action,
            KeypadAction::Decimal
        );
        assert_eq!(
            keypad.get_button_at(3, 2).unwrap().action,
            KeypadAction::Equals
        );
        assert_eq!(
            keypad.get_button_at(3, 3).unwrap().action,
            KeypadAction::Operator('+')
        );
    }

    #[test]
    fn test_keypad_row_4() {
        let keypad = WasmKeypad::new();
        assert_eq!(
            keypad.get_button_at(4, 0).unwrap().action,
            KeypadAction::Clear
        );
        assert_eq!(
            keypad.get_button_at(4, 1).unwrap().action,
            KeypadAction::OpenParen
        );
        assert_eq!(
            keypad.get_button_at(4, 2).unwrap().action,
            KeypadAction::CloseParen
        );
        assert_eq!(
            keypad.get_button_at(4, 3).unwrap().action,
            KeypadAction::Operator('^')
        );
    }

    // ===== DOM integration tests =====

    #[test]
    fn test_create_dom_elements() {
        let keypad = WasmKeypad::new();
        let elements = keypad.create_dom_elements();
        assert_eq!(elements.len(), 20);

        // Check first element
        assert_eq!(elements[0].id, "btn-7");
        assert_eq!(elements[0].tag, "button");
        assert!(elements[0].has_class("keypad-btn"));
    }

    #[test]
    fn test_create_keypad_element() {
        let keypad = WasmKeypad::new();
        let elem = keypad.create_keypad_element();
        assert_eq!(elem.id, "calc-keypad");
        assert!(elem.has_class("keypad"));
        assert_eq!(elem.children.len(), 20);
    }

    #[test]
    fn test_handle_click_digit() {
        let keypad = WasmKeypad::new();
        let action = keypad.handle_click("btn-5");
        assert_eq!(action, Some(KeypadAction::Digit(5)));
    }

    #[test]
    fn test_handle_click_operator() {
        let keypad = WasmKeypad::new();
        let action = keypad.handle_click("btn-plus");
        assert_eq!(action, Some(KeypadAction::Operator('+')));
    }

    #[test]
    fn test_handle_click_equals() {
        let keypad = WasmKeypad::new();
        let action = keypad.handle_click("btn-equals");
        assert_eq!(action, Some(KeypadAction::Equals));
    }

    #[test]
    fn test_handle_click_clear() {
        let keypad = WasmKeypad::new();
        let action = keypad.handle_click("btn-clear");
        assert_eq!(action, Some(KeypadAction::Clear));
    }

    #[test]
    fn test_handle_click_unknown() {
        let keypad = WasmKeypad::new();
        let action = keypad.handle_click("nonexistent");
        assert_eq!(action, None);
    }

    // ===== Keyboard mapping tests =====

    #[test]
    fn test_key_to_action_digits() {
        for d in '0'..='9' {
            let action = WasmKeypad::key_to_action(&d.to_string());
            let expected = d.to_digit(10).unwrap() as u8;
            assert_eq!(action, Some(KeypadAction::Digit(expected)));
        }
    }

    #[test]
    fn test_key_to_action_operators() {
        assert_eq!(
            WasmKeypad::key_to_action("+"),
            Some(KeypadAction::Operator('+'))
        );
        assert_eq!(
            WasmKeypad::key_to_action("-"),
            Some(KeypadAction::Operator('-'))
        );
        assert_eq!(
            WasmKeypad::key_to_action("*"),
            Some(KeypadAction::Operator('*'))
        );
        assert_eq!(
            WasmKeypad::key_to_action("/"),
            Some(KeypadAction::Operator('/'))
        );
        assert_eq!(
            WasmKeypad::key_to_action("^"),
            Some(KeypadAction::Operator('^'))
        );
        assert_eq!(
            WasmKeypad::key_to_action("%"),
            Some(KeypadAction::Operator('%'))
        );
    }

    #[test]
    fn test_key_to_action_special() {
        assert_eq!(WasmKeypad::key_to_action("."), Some(KeypadAction::Decimal));
        assert_eq!(
            WasmKeypad::key_to_action("("),
            Some(KeypadAction::OpenParen)
        );
        assert_eq!(
            WasmKeypad::key_to_action(")"),
            Some(KeypadAction::CloseParen)
        );
    }

    #[test]
    fn test_key_to_action_equals() {
        assert_eq!(
            WasmKeypad::key_to_action("Enter"),
            Some(KeypadAction::Equals)
        );
        assert_eq!(WasmKeypad::key_to_action("="), Some(KeypadAction::Equals));
    }

    #[test]
    fn test_key_to_action_clear() {
        assert_eq!(
            WasmKeypad::key_to_action("Escape"),
            Some(KeypadAction::Clear)
        );
        assert_eq!(WasmKeypad::key_to_action("c"), Some(KeypadAction::Clear));
        assert_eq!(WasmKeypad::key_to_action("C"), Some(KeypadAction::Clear));
    }

    #[test]
    fn test_key_to_action_unknown() {
        assert_eq!(WasmKeypad::key_to_action("x"), None);
        assert_eq!(WasmKeypad::key_to_action("Shift"), None);
    }

    // ===== MockDom extension tests =====

    #[test]
    fn test_mock_dom_add_keypad() {
        let mut dom = MockDom::calculator();
        let keypad = WasmKeypad::new();
        dom.add_keypad(&keypad);

        // Verify keypad container exists
        assert!(dom.get_element("calc-keypad").is_some());

        // Verify individual buttons exist
        assert!(dom.get_element("btn-5").is_some());
        assert!(dom.get_element("btn-plus").is_some());
        assert!(dom.get_element("btn-equals").is_some());
        assert!(dom.get_element("btn-clear").is_some());
    }

    #[test]
    fn test_keypad_clone() {
        let keypad = WasmKeypad::new();
        let cloned = keypad.clone();
        assert_eq!(keypad.button_count(), cloned.button_count());
    }

    #[test]
    fn test_keypad_debug() {
        let keypad = WasmKeypad::new();
        let debug = format!("{:?}", keypad);
        assert!(debug.contains("WasmKeypad"));
    }

    // ===== Property-based tests =====

    #[test]
    fn prop_all_digits_have_buttons() {
        let keypad = WasmKeypad::new();
        for d in 0..=9 {
            let ch = char::from_digit(d, 10).unwrap();
            assert!(
                keypad.find_button_by_char(ch).is_some(),
                "Missing button for digit {d}"
            );
        }
    }

    #[test]
    fn prop_all_operators_have_buttons() {
        let keypad = WasmKeypad::new();
        for op in ['+', '-', '*', '/', '^'] {
            assert!(
                keypad.find_button_by_char(op).is_some(),
                "Missing button for operator {op}"
            );
        }
    }

    #[test]
    fn prop_button_positions_unique() {
        let keypad = WasmKeypad::new();
        let mut positions = std::collections::HashSet::new();
        for btn in keypad.buttons() {
            let pos = (btn.row, btn.col);
            assert!(positions.insert(pos), "Duplicate position {:?}", pos);
        }
    }

    #[test]
    fn prop_button_ids_unique() {
        let keypad = WasmKeypad::new();
        let mut ids = std::collections::HashSet::new();
        for btn in keypad.buttons() {
            assert!(ids.insert(btn.id.clone()), "Duplicate ID {}", btn.id);
        }
    }
}
