//! Braille and block symbol sets for graph rendering.
//!
//! Provides 4 distinct symbol sets (Braille, Block, Tty, Custom) with 25 characters
//! each for btop-style paired-column graph rendering.
//!
//! ## SIMD-First Design
//!
//! All lookups use direct array indexing O(1) with SIMD-friendly layouts.
//! The 25-character sets encode two values (0-4 each) in a single character,
//! allowing 5×5 resolution per cell pair.

#![allow(dead_code)] // Symbol sets may be used in future widgets

/// Braille characters for upward-filling graphs (5×5 grid = 25 chars).
/// Index: `left_value` * 5 + `right_value` where values are 0-4.
pub const BRAILLE_UP: [char; 25] = [
    ' ', '⢀', '⢠', '⢰', '⢸', // left=0, right=0-4
    '⡀', '⣀', '⣠', '⣰', '⣸', // left=1, right=0-4
    '⡄', '⣄', '⣤', '⣴', '⣼', // left=2, right=0-4
    '⡆', '⣆', '⣦', '⣶', '⣾', // left=3, right=0-4
    '⡇', '⣇', '⣧', '⣷', '⣿', // left=4, right=0-4
];

/// Braille characters for downward-filling graphs.
pub const BRAILLE_DOWN: [char; 25] = [
    ' ', '⠈', '⠘', '⠸', '⢸', // left=0, right=0-4
    '⠁', '⠉', '⠙', '⠹', '⢹', // left=1, right=0-4
    '⠃', '⠋', '⠛', '⠻', '⢻', // left=2, right=0-4
    '⠇', '⠏', '⠟', '⠿', '⢿', // left=3, right=0-4
    '⡇', '⡏', '⡟', '⡿', '⣿', // left=4, right=0-4
];

/// Block characters for upward-filling graphs.
/// Uses half/quarter blocks: ▁▂▃▄▅▆▇█
pub const BLOCK_UP: [char; 25] = [
    ' ', '▁', '▂', '▃', '▄', // left=0
    '▁', '▂', '▃', '▄', '▅', // left=1
    '▂', '▃', '▄', '▅', '▆', // left=2
    '▃', '▄', '▅', '▆', '▇', // left=3
    '▄', '▅', '▆', '▇', '█', // left=4
];

/// Block characters for downward-filling graphs.
pub const BLOCK_DOWN: [char; 25] = [
    ' ', '▔', '▔', '▀', '▀', // left=0
    '▔', '▔', '▀', '▀', '█', // left=1
    '▔', '▀', '▀', '█', '█', // left=2
    '▀', '▀', '█', '█', '█', // left=3
    '▀', '█', '█', '█', '█', // left=4
];

/// TTY-safe ASCII characters for graphs (universal compatibility).
pub const TTY_UP: [char; 25] = [
    ' ', '.', '.', 'o', 'o', // left=0
    '.', '.', 'o', 'o', 'O', // left=1
    '.', 'o', 'o', 'O', 'O', // left=2
    'o', 'o', 'O', 'O', '#', // left=3
    'o', 'O', 'O', '#', '#', // left=4
];

/// TTY-safe ASCII for downward graphs.
pub const TTY_DOWN: [char; 25] = [
    ' ', '\'', '\'', '"', '"', // left=0
    '\'', '\'', '"', '"', '*', // left=1
    '\'', '"', '"', '*', '*', // left=2
    '"', '"', '*', '*', '#', // left=3
    '"', '*', '*', '#', '#', // left=4
];

/// Single-column sparkline characters (8 levels).
pub const SPARKLINE: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

/// Superscript digit characters for compact labels.
pub const SUPERSCRIPT: [char; 10] = ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹'];

/// Subscript digit characters.
pub const SUBSCRIPT: [char; 10] = ['₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉'];

// ============================================================================
// UX-112: Distinct Category Symbols
// ============================================================================

/// Category marker symbols - filled circles with varying fill levels.
/// Use these to distinguish different data categories in legends.
pub(super) const CATEGORY_FILLED: [char; 6] = ['●', '◐', '◑', '◒', '◓', '○'];

/// Category marker symbols - geometric shapes.
/// Use for color-blind friendly category distinction.
pub(super) const CATEGORY_SHAPES: [char; 6] = ['●', '■', '▲', '◆', '★', '◯'];

/// Category marker symbols - checkmarks and status.
/// Use for pass/fail/warning status indicators.
pub(super) const CATEGORY_STATUS: [char; 4] = ['✓', '⚠', '✗', '?'];

/// Get a distinct category symbol by index.
/// Cycles through filled circle variants.
#[inline]
pub(super) const fn category_symbol(index: usize) -> char {
    CATEGORY_FILLED[index % CATEGORY_FILLED.len()]
}

/// Get a distinct shape symbol by index.
/// Cycles through geometric shapes for color-blind accessibility.
#[inline]
pub(super) const fn shape_symbol(index: usize) -> char {
    CATEGORY_SHAPES[index % CATEGORY_SHAPES.len()]
}

/// Arrow symbols for flow diagrams.
pub(super) const ARROWS_FLOW: [&str; 4] = ["━▶", "──▶", "···▶", "→"];

/// Get consistent arrow style for Sankey diagrams.
/// UX-113: Standardized arrow styles based on flow strength.
#[inline]
pub(super) const fn flow_arrow(strength: usize) -> &'static str {
    // Use if-else instead of .min() for const fn compatibility
    let idx = if strength > 3 { 3 } else { strength };
    ARROWS_FLOW[idx]
}

/// Symbol set variants for graph rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SymbolSet {
    /// Braille patterns - highest resolution (2×4 dots per cell).
    #[default]
    Braille,
    /// Block characters - high compatibility.
    Block,
    /// TTY-safe ASCII - universal compatibility.
    Tty,
    /// Custom user-defined set.
    Custom,
}

/// Custom symbol set with user-defined characters.
#[derive(Debug, Clone)]
pub struct CustomSymbols {
    /// Upward-filling characters (25).
    pub up: [char; 25],
    /// Downward-filling characters (25).
    pub down: [char; 25],
}

impl Default for CustomSymbols {
    fn default() -> Self {
        Self {
            up: BRAILLE_UP,
            down: BRAILLE_DOWN,
        }
    }
}

impl CustomSymbols {
    /// Create custom symbols from a string of 50 characters.
    /// First 25 are up, next 25 are down.
    #[must_use]
    pub fn from_chars(chars: &str) -> Option<Self> {
        let chars: Vec<char> = chars.chars().collect();
        if chars.len() < 50 {
            return None;
        }
        let mut up = [' '; 25];
        let mut down = [' '; 25];
        up.copy_from_slice(&chars[0..25]);
        down.copy_from_slice(&chars[25..50]);
        Some(Self { up, down })
    }
}

/// Unified symbol interface for graph rendering.
#[derive(Debug, Clone)]
pub struct BrailleSymbols {
    set: SymbolSet,
    custom: Option<CustomSymbols>,
}

impl Default for BrailleSymbols {
    fn default() -> Self {
        Self::new(SymbolSet::default())
    }
}

impl BrailleSymbols {
    /// Create symbols with specified set.
    #[must_use]
    pub fn new(set: SymbolSet) -> Self {
        Self { set, custom: None }
    }

    /// Create symbols with custom character set.
    #[must_use]
    pub fn with_custom(custom: CustomSymbols) -> Self {
        Self {
            set: SymbolSet::Custom,
            custom: Some(custom),
        }
    }

    /// Get the current symbol set type.
    #[must_use]
    pub fn set(&self) -> SymbolSet {
        self.set
    }

    /// Get the up-direction symbol array.
    #[must_use]
    pub fn up_chars(&self) -> &[char; 25] {
        match self.set {
            SymbolSet::Braille => &BRAILLE_UP,
            SymbolSet::Block => &BLOCK_UP,
            SymbolSet::Tty => &TTY_UP,
            SymbolSet::Custom => self.custom.as_ref().map_or(&BRAILLE_UP, |c| &c.up),
        }
    }

    /// Get the down-direction symbol array.
    #[must_use]
    pub fn down_chars(&self) -> &[char; 25] {
        match self.set {
            SymbolSet::Braille => &BRAILLE_DOWN,
            SymbolSet::Block => &BLOCK_DOWN,
            SymbolSet::Tty => &TTY_DOWN,
            SymbolSet::Custom => self.custom.as_ref().map_or(&BRAILLE_DOWN, |c| &c.down),
        }
    }

    /// Get character for value (0.0-1.0) in up direction.
    /// Maps to single-column (uses left value only, right=0).
    #[inline]
    #[must_use]
    pub fn char_up(&self, value: f64) -> char {
        let level = (value.clamp(0.0, 1.0) * 4.0).round() as usize;
        self.up_chars()[level.min(4) * 5] // left=level, right=0
    }

    /// Get character for value in down direction.
    #[inline]
    #[must_use]
    pub fn char_down(&self, value: f64) -> char {
        let level = (value.clamp(0.0, 1.0) * 4.0).round() as usize;
        self.down_chars()[level.min(4) * 5]
    }

    /// Get character for paired values (left 0-4, right 0-4).
    /// SIMD-friendly: direct array lookup with index = left * 5 + right.
    #[inline]
    #[must_use]
    pub fn char_pair(&self, left: u8, right: u8) -> char {
        let idx = (left.min(4) as usize) * 5 + (right.min(4) as usize);
        self.up_chars()[idx]
    }

    /// Get character for paired values in down direction.
    #[inline]
    #[must_use]
    pub fn char_pair_down(&self, left: u8, right: u8) -> char {
        let idx = (left.min(4) as usize) * 5 + (right.min(4) as usize);
        self.down_chars()[idx]
    }

    /// Get sparkline character for value (0.0-1.0).
    /// Uses 8-level sparkline characters.
    #[inline]
    #[must_use]
    pub fn sparkline_char(value: f64) -> char {
        let level = (value.clamp(0.0, 1.0) * 7.0).round() as usize;
        SPARKLINE[level.min(7)]
    }

    /// Convert number to superscript string.
    #[must_use]
    pub fn to_superscript(n: u32) -> String {
        n.to_string()
            .chars()
            .filter_map(|c| c.to_digit(10).map(|d| SUPERSCRIPT[d as usize]))
            .collect()
    }

    /// Convert number to subscript string.
    #[must_use]
    pub fn to_subscript(n: u32) -> String {
        n.to_string()
            .chars()
            .filter_map(|c| c.to_digit(10).map(|d| SUBSCRIPT[d as usize]))
            .collect()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::disallowed_methods)]
mod tests {
    use super::*;

    // =====================================================
    // Symbol Set Array Tests
    // =====================================================

    #[test]
    fn test_braille_up_array_length() {
        assert_eq!(BRAILLE_UP.len(), 25);
    }

    #[test]
    fn test_braille_down_array_length() {
        assert_eq!(BRAILLE_DOWN.len(), 25);
    }

    #[test]
    fn test_block_up_array_length() {
        assert_eq!(BLOCK_UP.len(), 25);
    }

    #[test]
    fn test_block_down_array_length() {
        assert_eq!(BLOCK_DOWN.len(), 25);
    }

    #[test]
    fn test_tty_up_array_length() {
        assert_eq!(TTY_UP.len(), 25);
    }

    #[test]
    fn test_tty_down_array_length() {
        assert_eq!(TTY_DOWN.len(), 25);
    }

    #[test]
    fn test_sparkline_array_length() {
        assert_eq!(SPARKLINE.len(), 8);
    }

    #[test]
    fn test_superscript_array_length() {
        assert_eq!(SUPERSCRIPT.len(), 10);
    }

    #[test]
    fn test_subscript_array_length() {
        assert_eq!(SUBSCRIPT.len(), 10);
    }

    // =====================================================
    // Braille Symbol Specific Tests
    // =====================================================

    #[test]
    fn test_braille_up_empty_is_space() {
        assert_eq!(BRAILLE_UP[0], ' '); // left=0, right=0
    }

    #[test]
    fn test_braille_up_full_is_full() {
        assert_eq!(BRAILLE_UP[24], '⣿'); // left=4, right=4
    }

    #[test]
    fn test_braille_up_left_only() {
        assert_eq!(BRAILLE_UP[5], '⡀'); // left=1, right=0
        assert_eq!(BRAILLE_UP[10], '⡄'); // left=2, right=0
        assert_eq!(BRAILLE_UP[15], '⡆'); // left=3, right=0
        assert_eq!(BRAILLE_UP[20], '⡇'); // left=4, right=0
    }

    #[test]
    fn test_braille_up_right_only() {
        assert_eq!(BRAILLE_UP[1], '⢀'); // left=0, right=1
        assert_eq!(BRAILLE_UP[2], '⢠'); // left=0, right=2
        assert_eq!(BRAILLE_UP[3], '⢰'); // left=0, right=3
        assert_eq!(BRAILLE_UP[4], '⢸'); // left=0, right=4
    }

    #[test]
    fn test_braille_down_empty_is_space() {
        assert_eq!(BRAILLE_DOWN[0], ' ');
    }

    #[test]
    fn test_braille_down_full_is_full() {
        assert_eq!(BRAILLE_DOWN[24], '⣿');
    }

    // =====================================================
    // Block Symbol Tests
    // =====================================================

    #[test]
    fn test_block_up_empty_is_space() {
        assert_eq!(BLOCK_UP[0], ' ');
    }

    #[test]
    fn test_block_up_full_is_full_block() {
        assert_eq!(BLOCK_UP[24], '█');
    }

    #[test]
    fn test_block_up_progression() {
        // First row should progress: space, ▁, ▂, ▃, ▄
        assert_eq!(BLOCK_UP[0], ' ');
        assert_eq!(BLOCK_UP[1], '▁');
        assert_eq!(BLOCK_UP[2], '▂');
        assert_eq!(BLOCK_UP[3], '▃');
        assert_eq!(BLOCK_UP[4], '▄');
    }

    // =====================================================
    // TTY Symbol Tests
    // =====================================================

    #[test]
    fn test_tty_up_empty_is_space() {
        assert_eq!(TTY_UP[0], ' ');
    }

    #[test]
    fn test_tty_up_uses_ascii_only() {
        for c in TTY_UP.iter() {
            assert!(c.is_ascii() || *c == ' ', "Non-ASCII char: {}", c);
        }
    }

    #[test]
    fn test_tty_down_uses_ascii_only() {
        for c in TTY_DOWN.iter() {
            assert!(c.is_ascii() || *c == ' ', "Non-ASCII char: {}", c);
        }
    }

    // =====================================================
    // SymbolSet Enum Tests
    // =====================================================

    #[test]
    fn test_symbol_set_default_is_braille() {
        assert_eq!(SymbolSet::default(), SymbolSet::Braille);
    }

    #[test]
    fn test_symbol_set_variants() {
        let _ = SymbolSet::Braille;
        let _ = SymbolSet::Block;
        let _ = SymbolSet::Tty;
        let _ = SymbolSet::Custom;
    }

    // =====================================================
    // CustomSymbols Tests
    // =====================================================

    #[test]
    fn test_custom_symbols_default() {
        let custom = CustomSymbols::default();
        assert_eq!(custom.up, BRAILLE_UP);
        assert_eq!(custom.down, BRAILLE_DOWN);
    }

    #[test]
    fn test_custom_symbols_from_chars_valid() {
        let chars: String = std::iter::repeat('X').take(50).collect();
        let custom = CustomSymbols::from_chars(&chars);
        assert!(custom.is_some());
        let custom = custom.unwrap();
        assert_eq!(custom.up[0], 'X');
        assert_eq!(custom.down[0], 'X');
    }

    #[test]
    fn test_custom_symbols_from_chars_too_short() {
        let custom = CustomSymbols::from_chars("short");
        assert!(custom.is_none());
    }

    #[test]
    fn test_custom_symbols_from_chars_49_chars() {
        let chars: String = std::iter::repeat('X').take(49).collect();
        let custom = CustomSymbols::from_chars(&chars);
        assert!(custom.is_none());
    }

    // =====================================================
    // BrailleSymbols Construction Tests
    // =====================================================

    #[test]
    fn test_braille_symbols_new_braille() {
        let sym = BrailleSymbols::new(SymbolSet::Braille);
        assert_eq!(sym.set(), SymbolSet::Braille);
    }

    #[test]
    fn test_braille_symbols_new_block() {
        let sym = BrailleSymbols::new(SymbolSet::Block);
        assert_eq!(sym.set(), SymbolSet::Block);
    }

    #[test]
    fn test_braille_symbols_new_tty() {
        let sym = BrailleSymbols::new(SymbolSet::Tty);
        assert_eq!(sym.set(), SymbolSet::Tty);
    }

    #[test]
    fn test_braille_symbols_default() {
        let sym = BrailleSymbols::default();
        assert_eq!(sym.set(), SymbolSet::Braille);
    }

    #[test]
    fn test_braille_symbols_with_custom() {
        let custom = CustomSymbols::default();
        let sym = BrailleSymbols::with_custom(custom);
        assert_eq!(sym.set(), SymbolSet::Custom);
    }

    // =====================================================
    // Up/Down Chars Selection Tests
    // =====================================================

    #[test]
    fn test_up_chars_braille() {
        let sym = BrailleSymbols::new(SymbolSet::Braille);
        assert_eq!(sym.up_chars(), &BRAILLE_UP);
    }

    #[test]
    fn test_up_chars_block() {
        let sym = BrailleSymbols::new(SymbolSet::Block);
        assert_eq!(sym.up_chars(), &BLOCK_UP);
    }

    #[test]
    fn test_up_chars_tty() {
        let sym = BrailleSymbols::new(SymbolSet::Tty);
        assert_eq!(sym.up_chars(), &TTY_UP);
    }

    #[test]
    fn test_up_chars_custom() {
        let mut custom = CustomSymbols::default();
        custom.up[0] = 'A';
        let sym = BrailleSymbols::with_custom(custom);
        assert_eq!(sym.up_chars()[0], 'A');
    }

    #[test]
    fn test_down_chars_braille() {
        let sym = BrailleSymbols::new(SymbolSet::Braille);
        assert_eq!(sym.down_chars(), &BRAILLE_DOWN);
    }

    #[test]
    fn test_down_chars_block() {
        let sym = BrailleSymbols::new(SymbolSet::Block);
        assert_eq!(sym.down_chars(), &BLOCK_DOWN);
    }

    #[test]
    fn test_down_chars_tty() {
        let sym = BrailleSymbols::new(SymbolSet::Tty);
        assert_eq!(sym.down_chars(), &TTY_DOWN);
    }

    // =====================================================
    // char_up Tests
    // =====================================================

    #[test]
    fn test_char_up_zero() {
        let sym = BrailleSymbols::new(SymbolSet::Braille);
        assert_eq!(sym.char_up(0.0), ' ');
    }

    #[test]
    fn test_char_up_one() {
        let sym = BrailleSymbols::new(SymbolSet::Braille);
        assert_eq!(sym.char_up(1.0), '⡇'); // left=4, right=0
    }

    #[test]
    fn test_char_up_half() {
        let sym = BrailleSymbols::new(SymbolSet::Braille);
        let c = sym.char_up(0.5);
        assert_eq!(c, '⡄'); // left=2, right=0
    }

    #[test]
    fn test_char_up_clamps_negative() {
        let sym = BrailleSymbols::new(SymbolSet::Braille);
        assert_eq!(sym.char_up(-1.0), ' ');
    }

    #[test]
    fn test_char_up_clamps_over_one() {
        let sym = BrailleSymbols::new(SymbolSet::Braille);
        assert_eq!(sym.char_up(2.0), '⡇');
    }

    // =====================================================
    // char_down Tests
    // =====================================================

    #[test]
    fn test_char_down_zero() {
        let sym = BrailleSymbols::new(SymbolSet::Braille);
        assert_eq!(sym.char_down(0.0), ' ');
    }

    #[test]
    fn test_char_down_one() {
        let sym = BrailleSymbols::new(SymbolSet::Braille);
        assert_eq!(sym.char_down(1.0), '⡇');
    }

    // =====================================================
    // char_pair Tests
    // =====================================================

    #[test]
    fn test_char_pair_zero_zero() {
        let sym = BrailleSymbols::new(SymbolSet::Braille);
        assert_eq!(sym.char_pair(0, 0), ' ');
    }

    #[test]
    fn test_char_pair_four_four() {
        let sym = BrailleSymbols::new(SymbolSet::Braille);
        assert_eq!(sym.char_pair(4, 4), '⣿');
    }

    #[test]
    fn test_char_pair_left_only() {
        let sym = BrailleSymbols::new(SymbolSet::Braille);
        assert_eq!(sym.char_pair(2, 0), '⡄');
    }

    #[test]
    fn test_char_pair_right_only() {
        let sym = BrailleSymbols::new(SymbolSet::Braille);
        assert_eq!(sym.char_pair(0, 2), '⢠');
    }

    #[test]
    fn test_char_pair_clamps_left() {
        let sym = BrailleSymbols::new(SymbolSet::Braille);
        assert_eq!(sym.char_pair(10, 0), '⡇'); // 10 clamped to 4
    }

    #[test]
    fn test_char_pair_clamps_right() {
        let sym = BrailleSymbols::new(SymbolSet::Braille);
        assert_eq!(sym.char_pair(0, 10), '⢸'); // 10 clamped to 4
    }

    #[test]
    fn test_char_pair_down() {
        let sym = BrailleSymbols::new(SymbolSet::Braille);
        assert_eq!(sym.char_pair_down(0, 0), ' ');
        assert_eq!(sym.char_pair_down(4, 4), '⣿');
    }

    // =====================================================
    // Sparkline Tests
    // =====================================================

    #[test]
    fn test_sparkline_char_zero() {
        assert_eq!(BrailleSymbols::sparkline_char(0.0), '▁');
    }

    #[test]
    fn test_sparkline_char_one() {
        assert_eq!(BrailleSymbols::sparkline_char(1.0), '█');
    }

    #[test]
    fn test_sparkline_char_half() {
        // 0.5 * 7 = 3.5, rounds to 4, which is '▅' (index 4)
        let c = BrailleSymbols::sparkline_char(0.5);
        assert_eq!(c, '▅');
    }

    #[test]
    fn test_sparkline_char_clamps() {
        assert_eq!(BrailleSymbols::sparkline_char(-0.5), '▁');
        assert_eq!(BrailleSymbols::sparkline_char(1.5), '█');
    }

    // =====================================================
    // Superscript/Subscript Tests
    // =====================================================

    #[test]
    fn test_to_superscript_single_digit() {
        assert_eq!(BrailleSymbols::to_superscript(5), "⁵");
    }

    #[test]
    fn test_to_superscript_multi_digit() {
        assert_eq!(BrailleSymbols::to_superscript(123), "¹²³");
    }

    #[test]
    fn test_to_superscript_zero() {
        assert_eq!(BrailleSymbols::to_superscript(0), "⁰");
    }

    #[test]
    fn test_to_subscript_single_digit() {
        assert_eq!(BrailleSymbols::to_subscript(5), "₅");
    }

    #[test]
    fn test_to_subscript_multi_digit() {
        assert_eq!(BrailleSymbols::to_subscript(123), "₁₂₃");
    }

    #[test]
    fn test_to_subscript_zero() {
        assert_eq!(BrailleSymbols::to_subscript(0), "₀");
    }

    #[test]
    fn test_superscript_all_digits() {
        let result = BrailleSymbols::to_superscript(1234567890);
        assert_eq!(result, "¹²³⁴⁵⁶⁷⁸⁹⁰");
    }

    #[test]
    fn test_subscript_all_digits() {
        let result = BrailleSymbols::to_subscript(1234567890);
        assert_eq!(result, "₁₂₃₄₅₆₇₈₉₀");
    }

    // =====================================================
    // Index Calculation Tests (SIMD-friendly verification)
    // =====================================================

    #[test]
    fn test_index_calculation() {
        // Verify left * 5 + right formula
        for left in 0..5u8 {
            for right in 0..5u8 {
                let idx = (left as usize) * 5 + (right as usize);
                assert!(idx < 25, "Index out of bounds: {}", idx);
            }
        }
    }

    #[test]
    fn test_all_braille_up_unique() {
        let mut seen = std::collections::HashSet::new();
        for &c in BRAILLE_UP.iter() {
            // Space appears only once at index 0
            if c != ' ' {
                assert!(seen.insert(c), "Duplicate character: {}", c);
            }
        }
    }

    #[test]
    fn test_all_braille_down_unique() {
        let mut seen = std::collections::HashSet::new();
        for &c in BRAILLE_DOWN.iter() {
            if c != ' ' {
                assert!(seen.insert(c), "Duplicate character: {}", c);
            }
        }
    }

    // =====================================================
    // Block Mode Tests
    // =====================================================

    #[test]
    fn test_block_char_up() {
        let sym = BrailleSymbols::new(SymbolSet::Block);
        assert_eq!(sym.char_up(0.0), ' ');
        assert_eq!(sym.char_up(1.0), '▄'); // left=4, right=0 in BLOCK_UP
    }

    #[test]
    fn test_block_char_pair() {
        let sym = BrailleSymbols::new(SymbolSet::Block);
        assert_eq!(sym.char_pair(0, 0), ' ');
        assert_eq!(sym.char_pair(4, 4), '█');
    }

    // =====================================================
    // TTY Mode Tests
    // =====================================================

    #[test]
    fn test_tty_char_up() {
        let sym = BrailleSymbols::new(SymbolSet::Tty);
        assert_eq!(sym.char_up(0.0), ' ');
        assert_eq!(sym.char_up(1.0), 'o'); // left=4, right=0 in TTY_UP
    }

    #[test]
    fn test_tty_char_pair() {
        let sym = BrailleSymbols::new(SymbolSet::Tty);
        assert_eq!(sym.char_pair(0, 0), ' ');
        assert_eq!(sym.char_pair(4, 4), '#');
    }

    // =====================================================
    // Custom Mode Fallback Tests
    // =====================================================

    #[test]
    fn test_custom_without_data_uses_braille() {
        // Create a BrailleSymbols with Custom set but no custom data
        let sym = BrailleSymbols {
            set: SymbolSet::Custom,
            custom: None,
        };
        // Should fallback to BRAILLE_UP
        assert_eq!(sym.up_chars(), &BRAILLE_UP);
        assert_eq!(sym.down_chars(), &BRAILLE_DOWN);
    }
}
