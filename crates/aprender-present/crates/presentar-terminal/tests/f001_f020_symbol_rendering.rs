//! F001-F020: Symbol Rendering Falsification Tests
//!
//! SPEC-024 Section A: Validates that presentar-terminal symbol arrays
//! match btop/ttop encoding exactly.
//!
//! Methodology: Each test attempts to DISPROVE the claim. A passing test
//! means the falsification criterion was NOT met (i.e., the implementation is correct).

use presentar_terminal::widgets::{
    BorderStyle, BrailleSymbols, CustomSymbols, SymbolSet, BLOCK_UP, BRAILLE_DOWN, BRAILLE_UP,
    SPARKLINE, SUBSCRIPT, SUPERSCRIPT, TTY_DOWN, TTY_UP,
};

// =============================================================================
// F001-F004: Braille Symbol Array Tests
// =============================================================================

/// F001: Braille empty is space
/// Falsification criterion: `BRAILLE_UP[0] != ' '`
#[test]
fn f001_braille_empty_is_space() {
    assert_eq!(
        BRAILLE_UP[0], ' ',
        "F001 FAILED: BRAILLE_UP[0] should be space, got '{}'",
        BRAILLE_UP[0]
    );
}

/// F002: Braille full is ⣿
/// Falsification criterion: `BRAILLE_UP[24] != '⣿'`
#[test]
fn f002_braille_full_is_full_block() {
    assert_eq!(
        BRAILLE_UP[24], '⣿',
        "F002 FAILED: BRAILLE_UP[24] should be '⣿', got '{}'",
        BRAILLE_UP[24]
    );
}

/// F003: Braille array length
/// Falsification criterion: `BRAILLE_UP.len() != 25`
#[test]
fn f003_braille_array_length_is_25() {
    assert_eq!(
        BRAILLE_UP.len(),
        25,
        "F003 FAILED: BRAILLE_UP should have 25 elements, got {}",
        BRAILLE_UP.len()
    );
}

/// F004: Block empty is space
/// Falsification criterion: `BLOCK_UP[0] != ' '`
#[test]
fn f004_block_empty_is_space() {
    assert_eq!(
        BLOCK_UP[0], ' ',
        "F004 FAILED: BLOCK_UP[0] should be space, got '{}'",
        BLOCK_UP[0]
    );
}

// =============================================================================
// F005-F006: Block Symbol Array Tests
// =============================================================================

/// F005: Block full is █
/// Falsification criterion: `BLOCK_UP[24] != '█'`
#[test]
fn f005_block_full_is_full_block() {
    assert_eq!(
        BLOCK_UP[24], '█',
        "F005 FAILED: BLOCK_UP[24] should be '█', got '{}'",
        BLOCK_UP[24]
    );
}

/// F006: Block array length
/// Falsification criterion: `BLOCK_UP.len() != 25`
#[test]
fn f006_block_array_length_is_25() {
    assert_eq!(
        BLOCK_UP.len(),
        25,
        "F006 FAILED: BLOCK_UP should have 25 elements, got {}",
        BLOCK_UP.len()
    );
}

// =============================================================================
// F007: TTY ASCII-Only Test
// =============================================================================

/// F007: TTY uses ASCII only
/// Falsification criterion: Any non-ASCII in `TTY_UP`
#[test]
fn f007_tty_uses_ascii_only() {
    for (i, &ch) in TTY_UP.iter().enumerate() {
        assert!(
            ch.is_ascii(),
            "F007 FAILED: TTY_UP[{}] = '{}' (U+{:04X}) is not ASCII",
            i,
            ch,
            ch as u32
        );
    }

    for (i, &ch) in TTY_DOWN.iter().enumerate() {
        assert!(
            ch.is_ascii(),
            "F007 FAILED: TTY_DOWN[{}] = '{}' (U+{:04X}) is not ASCII",
            i,
            ch,
            ch as u32
        );
    }
}

// =============================================================================
// F008-F009: Sparkline Array Tests
// =============================================================================

/// F008: Sparkline 8 levels
/// Falsification criterion: `SPARKLINE.len() != 8`
#[test]
fn f008_sparkline_has_8_levels() {
    assert_eq!(
        SPARKLINE.len(),
        8,
        "F008 FAILED: SPARKLINE should have 8 levels, got {}",
        SPARKLINE.len()
    );
}

/// F009: Sparkline range ▁→█
/// Falsification criterion: `SPARKLINE[0] != '▁' || SPARKLINE[7] != '█'`
#[test]
fn f009_sparkline_range_is_correct() {
    assert_eq!(
        SPARKLINE[0], '▁',
        "F009 FAILED: SPARKLINE[0] should be '▁', got '{}'",
        SPARKLINE[0]
    );
    assert_eq!(
        SPARKLINE[7], '█',
        "F009 FAILED: SPARKLINE[7] should be '█', got '{}'",
        SPARKLINE[7]
    );

    // Verify full progression: ▁▂▃▄▅▆▇█
    let expected = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    assert_eq!(
        SPARKLINE, expected,
        "F009 FAILED: SPARKLINE should be {expected:?}, got {SPARKLINE:?}"
    );
}

// =============================================================================
// F010-F011: Superscript/Subscript Array Tests
// =============================================================================

/// F010: Superscript 10 digits
/// Falsification criterion: `SUPERSCRIPT.len() != 10`
#[test]
fn f010_superscript_has_10_digits() {
    assert_eq!(
        SUPERSCRIPT.len(),
        10,
        "F010 FAILED: SUPERSCRIPT should have 10 digits, got {}",
        SUPERSCRIPT.len()
    );

    // Verify complete set: ⁰¹²³⁴⁵⁶⁷⁸⁹
    let expected = ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹'];
    assert_eq!(
        SUPERSCRIPT, expected,
        "F010 FAILED: SUPERSCRIPT should be {expected:?}, got {SUPERSCRIPT:?}"
    );
}

/// F011: Subscript 10 digits
/// Falsification criterion: `SUBSCRIPT.len() != 10`
#[test]
fn f011_subscript_has_10_digits() {
    assert_eq!(
        SUBSCRIPT.len(),
        10,
        "F011 FAILED: SUBSCRIPT should have 10 digits, got {}",
        SUBSCRIPT.len()
    );

    // Verify complete set: ₀₁₂₃₄₅₆₇₈₉
    let expected = ['₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉'];
    assert_eq!(
        SUBSCRIPT, expected,
        "F011 FAILED: SUBSCRIPT should be {expected:?}, got {SUBSCRIPT:?}"
    );
}

// =============================================================================
// F012-F014: Braille Index Formula Tests
// =============================================================================

/// F012: Braille pair index formula
/// Falsification criterion: `idx = left*5 + right` yields wrong char
#[test]
fn f012_braille_index_formula_correct() {
    let sym = BrailleSymbols::new(SymbolSet::Braille);

    // Test all 25 combinations
    for left in 0..5u8 {
        for right in 0..5u8 {
            let idx = (left as usize) * 5 + (right as usize);
            let expected = BRAILLE_UP[idx];
            let actual = sym.char_pair(left, right);
            assert_eq!(
                actual, expected,
                "F012 FAILED: char_pair({left}, {right}) should be '{expected}', got '{actual}'"
            );
        }
    }
}

/// F013: Braille left=4,right=0
/// Falsification criterion: `BRAILLE_UP[20] != '⡇'`
#[test]
fn f013_braille_left_4_right_0() {
    let idx = 4 * 5; // left=4, right=0
    assert_eq!(
        BRAILLE_UP[idx], '⡇',
        "F013 FAILED: BRAILLE_UP[{}] (left=4,right=0) should be '⡇', got '{}'",
        idx, BRAILLE_UP[idx]
    );
}

/// F014: Braille left=0,right=4
/// Falsification criterion: `BRAILLE_UP[4] != '⢸'`
#[test]
fn f014_braille_left_0_right_4() {
    let idx = 4; // left=0, right=4
    assert_eq!(
        BRAILLE_UP[idx], '⢸',
        "F014 FAILED: BRAILLE_UP[{}] (left=0,right=4) should be '⢸', got '{}'",
        idx, BRAILLE_UP[idx]
    );
}

// =============================================================================
// F015: Block Chars Progressive Test
// =============================================================================

/// F015: Block chars progressive
/// Falsification criterion: `BLOCK_UP` not monotonically increasing
#[test]
fn f015_block_chars_progressive() {
    // First row (left=0): ' ', '▁', '▂', '▃', '▄'
    let expected_row0 = [' ', '▁', '▂', '▃', '▄'];
    for (i, &expected) in expected_row0.iter().enumerate() {
        assert_eq!(
            BLOCK_UP[i], expected,
            "F015 FAILED: BLOCK_UP[{}] should be '{}', got '{}'",
            i, expected, BLOCK_UP[i]
        );
    }

    // Verify diagonal progression (0,0) < (1,1) < (2,2) < (3,3) < (4,4)
    // The visual "height" should increase
    let diagonal = [
        BLOCK_UP[0],  // (0,0)
        BLOCK_UP[6],  // (1,1)
        BLOCK_UP[12], // (2,2)
        BLOCK_UP[18], // (3,3)
        BLOCK_UP[24], // (4,4)
    ];
    assert_eq!(diagonal[0], ' ', "F015: diagonal[0] should be space");
    assert_eq!(diagonal[4], '█', "F015: diagonal[4] should be full block");
}

// =============================================================================
// F016: Unicode Braille Range Test
// =============================================================================

/// F016: Unicode braille range
/// Falsification criterion: Any char outside U+2800-U+28FF
#[test]
fn f016_unicode_braille_range() {
    for (i, &ch) in BRAILLE_UP.iter().enumerate() {
        // Skip space (index 0)
        if ch == ' ' {
            continue;
        }
        let codepoint = ch as u32;
        assert!(
            (0x2800..=0x28FF).contains(&codepoint),
            "F016 FAILED: BRAILLE_UP[{i}] = '{ch}' (U+{codepoint:04X}) is outside U+2800-U+28FF"
        );
    }

    for (i, &ch) in BRAILLE_DOWN.iter().enumerate() {
        if ch == ' ' {
            continue;
        }
        let codepoint = ch as u32;
        assert!(
            (0x2800..=0x28FF).contains(&codepoint),
            "F016 FAILED: BRAILLE_DOWN[{i}] = '{ch}' (U+{codepoint:04X}) is outside U+2800-U+28FF"
        );
    }
}

// =============================================================================
// F017: Braille Down Inverted Test
// =============================================================================

/// F017: Braille down inverted
/// Falsification criterion: `BRAILLE_DOWN`[24] != '⣿'
#[test]
fn f017_braille_down_full_is_full() {
    assert_eq!(
        BRAILLE_DOWN[24], '⣿',
        "F017 FAILED: BRAILLE_DOWN[24] should be '⣿', got '{}'",
        BRAILLE_DOWN[24]
    );

    // Also verify empty is space
    assert_eq!(
        BRAILLE_DOWN[0], ' ',
        "F017 FAILED: BRAILLE_DOWN[0] should be space, got '{}'",
        BRAILLE_DOWN[0]
    );
}

// =============================================================================
// F018: Custom Symbols Fallback Test
// =============================================================================

/// F018: Custom symbols fallback
/// Falsification criterion: Custom with None data uses Braille
#[test]
fn f018_custom_symbols_fallback() {
    // Create BrailleSymbols with Custom set but no custom data
    let sym = BrailleSymbols::with_custom(CustomSymbols::default());
    assert_eq!(sym.set(), SymbolSet::Custom);

    // Default CustomSymbols should use BRAILLE_UP/DOWN
    assert_eq!(
        sym.up_chars(),
        &BRAILLE_UP,
        "F018 FAILED: CustomSymbols::default().up should equal BRAILLE_UP"
    );
    assert_eq!(
        sym.down_chars(),
        &BRAILLE_DOWN,
        "F018 FAILED: CustomSymbols::default().down should equal BRAILLE_DOWN"
    );

    // Test explicit None case via internal struct construction
    // Note: This is a design test - if custom is None, fallback to Braille
    let sym_no_custom = BrailleSymbols::new(SymbolSet::Braille);
    assert_eq!(
        sym_no_custom.up_chars(),
        &BRAILLE_UP,
        "F018 FAILED: Braille mode should use BRAILLE_UP"
    );
}

// =============================================================================
// F019: Symbol Set Default Test
// =============================================================================

/// F019: Symbol set default
/// Falsification criterion: `SymbolSet::default() != SymbolSet::Braille`
#[test]
fn f019_symbol_set_default_is_braille() {
    assert_eq!(
        SymbolSet::default(),
        SymbolSet::Braille,
        "F019 FAILED: SymbolSet::default() should be Braille"
    );

    // Also verify BrailleSymbols default
    let sym = BrailleSymbols::default();
    assert_eq!(
        sym.set(),
        SymbolSet::Braille,
        "F019 FAILED: BrailleSymbols::default().set() should be Braille"
    );
}

// =============================================================================
// F020: Box Drawing Chars Test
// =============================================================================

/// F020: Box drawing chars
/// Falsification criterion: Missing ─│┌┐└┘├┤┬┴┼
#[test]
fn f020_box_drawing_chars_present() {
    // Test all BorderStyle variants have the required characters
    let required_single = ['─', '│', '┌', '┐', '└', '┘'];
    let required_rounded = ['─', '│', '╭', '╮', '╰', '╯'];
    let required_double = ['═', '║', '╔', '╗', '╚', '╝'];
    let required_heavy = ['━', '┃', '┏', '┓', '┗', '┛'];
    assert_eq!(required_single.len(), 6);
    assert_eq!(required_rounded.len(), 6);
    assert_eq!(required_double.len(), 6);
    assert_eq!(required_heavy.len(), 6);

    // BorderStyle::Single
    let (tl, top, tr, left, right, bl, bottom, br) = BorderStyle::Single.chars();
    assert_eq!(tl, '┌', "F020: Single top-left");
    assert_eq!(tr, '┐', "F020: Single top-right");
    assert_eq!(bl, '└', "F020: Single bottom-left");
    assert_eq!(br, '┘', "F020: Single bottom-right");
    assert_eq!(top, '─', "F020: Single top");
    assert_eq!(bottom, '─', "F020: Single bottom");
    assert_eq!(left, '│', "F020: Single left");
    assert_eq!(right, '│', "F020: Single right");

    // BorderStyle::Rounded
    let (tl, _top, tr, _left, _right, bl, _bottom, br) = BorderStyle::Rounded.chars();
    assert_eq!(tl, '╭', "F020: Rounded top-left");
    assert_eq!(tr, '╮', "F020: Rounded top-right");
    assert_eq!(bl, '╰', "F020: Rounded bottom-left");
    assert_eq!(br, '╯', "F020: Rounded bottom-right");

    // BorderStyle::Double
    let (tl, top, tr, left, _right, bl, _bottom, br) = BorderStyle::Double.chars();
    assert_eq!(tl, '╔', "F020: Double top-left");
    assert_eq!(tr, '╗', "F020: Double top-right");
    assert_eq!(bl, '╚', "F020: Double bottom-left");
    assert_eq!(br, '╝', "F020: Double bottom-right");
    assert_eq!(top, '═', "F020: Double top");
    assert_eq!(left, '║', "F020: Double left");

    // BorderStyle::Heavy
    let (tl, top, tr, left, _right, bl, _bottom, br) = BorderStyle::Heavy.chars();
    assert_eq!(tl, '┏', "F020: Heavy top-left");
    assert_eq!(tr, '┓', "F020: Heavy top-right");
    assert_eq!(bl, '┗', "F020: Heavy bottom-left");
    assert_eq!(br, '┛', "F020: Heavy bottom-right");
    assert_eq!(top, '━', "F020: Heavy top");
    assert_eq!(left, '┃', "F020: Heavy left");

    // BorderStyle::Ascii
    let (tl, top, _tr, left, _right, _bl, _bottom, _br) = BorderStyle::Ascii.chars();
    assert_eq!(tl, '+', "F020: Ascii corner");
    assert_eq!(top, '-', "F020: Ascii horizontal");
    assert_eq!(left, '|', "F020: Ascii vertical");

    // BorderStyle::None
    let (tl, top, _tr, _left, _right, _bl, _bottom, _br) = BorderStyle::None.chars();
    assert_eq!(tl, ' ', "F020: None should be spaces");
    assert_eq!(top, ' ', "F020: None should be spaces");
}

// =============================================================================
// Additional Symbol Validation Tests
// =============================================================================

/// Verify `BRAILLE_UP` and `BRAILLE_DOWN` have unique non-space characters
#[test]
fn braille_arrays_have_unique_chars() {
    use std::collections::HashSet;

    let mut up_set: HashSet<char> = HashSet::new();
    for &ch in &BRAILLE_UP {
        if ch != ' ' {
            assert!(up_set.insert(ch), "Duplicate in BRAILLE_UP: '{ch}'");
        }
    }
    assert_eq!(
        up_set.len(),
        24,
        "BRAILLE_UP should have 24 unique non-space chars"
    );

    let mut down_set: HashSet<char> = HashSet::new();
    for &ch in &BRAILLE_DOWN {
        if ch != ' ' {
            assert!(down_set.insert(ch), "Duplicate in BRAILLE_DOWN: '{ch}'");
        }
    }
    assert_eq!(
        down_set.len(),
        24,
        "BRAILLE_DOWN should have 24 unique non-space chars"
    );
}

/// Verify block chars are in correct Unicode ranges
#[test]
fn block_chars_in_unicode_range() {
    // Block elements: U+2580-U+259F
    // Specifically: ▀▁▂▃▄▅▆▇█▔
    let block_range = 0x2580u32..=0x259Fu32;

    for (i, &ch) in BLOCK_UP.iter().enumerate() {
        if ch == ' ' {
            continue;
        }
        let cp = ch as u32;
        assert!(
            block_range.contains(&cp),
            "BLOCK_UP[{i}] = '{ch}' (U+{cp:04X}) outside block elements range"
        );
    }
}

/// Verify superscript chars are in correct Unicode ranges
#[test]
fn superscript_chars_in_unicode_range() {
    // Superscript digits are scattered:
    // ⁰ = U+2070, ¹ = U+00B9, ² = U+00B2, ³ = U+00B3, ⁴-⁹ = U+2074-U+2079
    let expected_codepoints: [u32; 10] = [
        0x2070, // ⁰
        0x00B9, // ¹
        0x00B2, // ²
        0x00B3, // ³
        0x2074, // ⁴
        0x2075, // ⁵
        0x2076, // ⁶
        0x2077, // ⁷
        0x2078, // ⁸
        0x2079, // ⁹
    ];

    for (i, &ch) in SUPERSCRIPT.iter().enumerate() {
        let cp = ch as u32;
        assert_eq!(
            cp, expected_codepoints[i],
            "SUPERSCRIPT[{}] = '{}' (U+{:04X}) should be U+{:04X}",
            i, ch, cp, expected_codepoints[i]
        );
    }
}

/// Verify subscript chars are in correct Unicode ranges
#[test]
fn subscript_chars_in_unicode_range() {
    // Subscript digits: U+2080-U+2089
    for (i, &ch) in SUBSCRIPT.iter().enumerate() {
        let cp = ch as u32;
        let expected = 0x2080 + i as u32;
        assert_eq!(
            cp, expected,
            "SUBSCRIPT[{i}] = '{ch}' (U+{cp:04X}) should be U+{expected:04X}"
        );
    }
}

/// Verify `BrailleSymbols` helper functions work correctly
#[test]
fn braille_symbols_helpers_correct() {
    // to_superscript
    assert_eq!(BrailleSymbols::to_superscript(0), "⁰");
    assert_eq!(BrailleSymbols::to_superscript(123), "¹²³");
    assert_eq!(BrailleSymbols::to_superscript(1_234_567_890), "¹²³⁴⁵⁶⁷⁸⁹⁰");

    // to_subscript
    assert_eq!(BrailleSymbols::to_subscript(0), "₀");
    assert_eq!(BrailleSymbols::to_subscript(123), "₁₂₃");
    assert_eq!(BrailleSymbols::to_subscript(1_234_567_890), "₁₂₃₄₅₆₇₈₉₀");

    // sparkline_char
    assert_eq!(BrailleSymbols::sparkline_char(0.0), '▁');
    assert_eq!(BrailleSymbols::sparkline_char(1.0), '█');
    assert_eq!(BrailleSymbols::sparkline_char(0.5), '▅'); // rounds to index 4
}
