//! LaTeX conversion utilities for contract math notation.
//!
//! Converts Unicode math notation found in YAML contracts into
//! LaTeX math mode suitable for rendering with `KaTeX` or full LaTeX.

/// Escape special LaTeX characters in plain text.
pub fn latex_escape(s: &str) -> String {
    s.replace('\\', "\\textbackslash{}")
        .replace('&', "\\&")
        .replace('%', "\\%")
        .replace('$', "\\$")
        .replace('#', "\\#")
        .replace('_', "\\_")
        .replace('{', "\\{")
        .replace('}', "\\}")
        .replace('~', "\\textasciitilde{}")
        .replace('^', "\\textasciicircum{}")
}

/// Convert contract math notation to LaTeX math mode.
///
/// Handles common patterns found in our YAML contracts:
/// - Greek letters (ε, σ, α, etc.)
/// - Subscripts (`x_i`, `A_{ij}`)
/// - Superscripts (x^T, ℝ^n)
/// - Operators (Σ, ∈, ≈, ≤, ≥, ∀, →)
/// - Special sets (ℝ, ℤ)
/// - Functions (sqrt, exp, log, softmax, etc.)
pub fn math_to_latex(s: &str) -> String {
    let mut out = s.to_string();

    // Unicode → LaTeX replacements. Each entry: (unicode, latex_cmd, is_command).
    // When is_command is true, a trailing space is inserted before the next
    // alphabetic character to prevent `\foralli` instead of `\forall i`.
    let replacements: &[(&str, &str, bool)] = &[
        // Greek letters
        ("α", "\\alpha", true),
        ("β", "\\beta", true),
        ("γ", "\\gamma", true),
        ("δ", "\\delta", true),
        ("ε", "\\varepsilon", true),
        ("θ", "\\theta", true),
        ("λ", "\\lambda", true),
        ("σ", "\\sigma", true),
        ("τ", "\\tau", true),
        ("Σ", "\\sum", true),
        ("Φ", "\\Phi", true),
        ("π", "\\pi", true),
        // Operators
        ("∈", "\\in", true),
        ("∉", "\\notin", true),
        ("≈", "\\approx", true),
        ("≤", "\\leq", true),
        ("≥", "\\geq", true),
        ("≠", "\\neq", true),
        ("∀", "\\forall", true),
        ("∃", "\\exists", true),
        ("→", "\\to", true),
        ("←", "\\leftarrow", true),
        ("⊗", "\\otimes", true),
        ("⁺", "^{+}", false),
        // Special sets
        ("ℝ", "\\mathbb{R}", false),
        ("ℤ", "\\mathbb{Z}", false),
    ];
    for &(uni, tex, is_cmd) in replacements {
        if is_cmd {
            out = replace_unicode_cmd(&out, uni, tex);
        } else {
            out = out.replace(uni, tex);
        }
    }

    // sqrt(...) → \sqrt{...}
    out = replace_func(&out, "sqrt", "\\sqrt");

    // exp(...) → \exp(...)
    out = out.replace("exp(", "\\exp(");

    // log(...) → \log(...)
    out = out.replace("log(", "\\log(");

    // Escape % and # in formulas
    out = out.replace('%', "\\%");
    out = out.replace('#', "\\#");

    // Fix double superscripts: ^{+}^X → ^{+X}
    // This arises when ⁺ (→ ^{+}) is followed by ^N in the source.
    while let Some(pos) = out.find("^{+}^") {
        let after = &out[pos + 5..]; // after "^{+}^"
        // Collect the next superscript content: either {braced} or a single char
        if after.starts_with('{') {
            if let Some(close) = after.find('}') {
                let inner = &after[1..close];
                let replacement = format!("^{{+{inner}}}");
                out = format!("{}{}{}", &out[..pos], replacement, &after[close + 1..]);
            }
        } else if let Some(ch) = after.chars().next() {
            let replacement = format!("^{{+{ch}}}");
            out = format!("{}{}{}", &out[..pos], replacement, &after[ch.len_utf8()..]);
        }
    }

    out
}

/// Replace a Unicode symbol with a LaTeX command, inserting a trailing
/// space when the next character is alphabetic (prevents `\foralli`).
pub fn replace_unicode_cmd(s: &str, uni: &str, tex: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(pos) = rest.find(uni) {
        result.push_str(&rest[..pos]);
        result.push_str(tex);
        let after = &rest[pos + uni.len()..];
        // Insert space before next alphabetic char to keep LaTeX happy
        if after.starts_with(|c: char| c.is_ascii_alphabetic()) {
            result.push(' ');
        }
        rest = after;
    }
    result.push_str(rest);
    result
}

/// Replace `func(...)` with `\cmd{...}` handling nested parens.
/// Applies recursively so `sqrt(a + sqrt(b))` becomes `\sqrt{a + \sqrt{b}}`.
pub fn replace_func(s: &str, func: &str, cmd: &str) -> String {
    let pattern = format!("{func}(");
    let mut result = String::with_capacity(s.len());
    let mut rest = s;

    while let Some(pos) = rest.find(&pattern) {
        result.push_str(&rest[..pos]);
        let after = &rest[pos + pattern.len()..];

        // Find matching closing paren
        let mut depth = 1;
        let mut end = 0;
        for (i, ch) in after.char_indices() {
            match ch {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        end = i;
                        break;
                    }
                }
                _ => {}
            }
        }

        if depth == 0 {
            let inner = &after[..end];
            // Recurse to handle nested calls like sqrt(a + sqrt(b))
            let inner_replaced = replace_func(inner, func, cmd);
            result.push_str(&format!("{cmd}{{{inner_replaced}}}"));
            rest = &after[end + 1..];
        } else {
            // Unmatched paren — emit as-is
            result.push_str(&pattern);
            rest = after;
        }
    }

    result.push_str(rest);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_math_to_latex_greek() {
        assert_eq!(math_to_latex("ε > 0"), "\\varepsilon > 0");
        assert_eq!(math_to_latex("α_t"), "\\alpha_t");
    }

    #[test]
    fn test_math_to_latex_operators() {
        assert_eq!(math_to_latex("x ∈ ℝ^n"), "x \\in \\mathbb{R}^n");
        assert_eq!(math_to_latex("a ≈ b"), "a \\approx b");
        assert_eq!(math_to_latex("∀i: x_i ≥ 0"), "\\forall i: x_i \\geq 0");
    }

    #[test]
    fn test_math_to_latex_sqrt() {
        assert_eq!(
            math_to_latex("Q / sqrt(mean(Q²) + ε)"),
            "Q / \\sqrt{mean(Q²) + \\varepsilon}"
        );
    }

    #[test]
    fn test_math_to_latex_exp() {
        assert_eq!(math_to_latex("exp(x_i - max(x))"), "\\exp(x_i - max(x))");
    }

    #[test]
    fn test_replace_func_nested() {
        assert_eq!(
            replace_func("sqrt(a + sqrt(b))", "sqrt", "\\sqrt"),
            "\\sqrt{a + \\sqrt{b}}"
        );
    }

    #[test]
    fn test_latex_escape() {
        assert_eq!(latex_escape("a_b"), "a\\_b");
        assert_eq!(latex_escape("100%"), "100\\%");
    }
}
