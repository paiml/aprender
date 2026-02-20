
// ==================== Common Fix Templates ====================

/// Pre-defined fix templates for common Rust errors.
///
/// These templates are used by the fix generator to suggest fixes.
#[allow(dead_code)]
pub(super) mod templates {
    use super::{FixTemplate, Placeholder};

    /// Template: Add .`to_string()` for String/&str conversion
    #[must_use]
    pub(crate) fn to_string_conversion() -> FixTemplate {
        FixTemplate::new("$expr.to_string()", "Convert &str to String")
            .with_placeholder(Placeholder::expression("expr"))
            .with_code("E0308")
            .with_confidence(0.9)
    }

    /// Template: Add .`as_str()` for String/&str conversion
    #[must_use]
    pub(crate) fn as_str_conversion() -> FixTemplate {
        FixTemplate::new("$expr.as_str()", "Convert String to &str")
            .with_placeholder(Placeholder::expression("expr"))
            .with_code("E0308")
            .with_confidence(0.85)
    }

    /// Template: Add .`clone()` for ownership
    #[must_use]
    pub(crate) fn clone_value() -> FixTemplate {
        FixTemplate::new("$expr.clone()", "Clone the value to avoid move")
            .with_placeholder(Placeholder::expression("expr"))
            .with_code("E0382")
            .with_confidence(0.8)
    }

    /// Template: Add & for reference
    #[must_use]
    pub(crate) fn add_reference() -> FixTemplate {
        FixTemplate::new("&$expr", "Add reference")
            .with_placeholder(Placeholder::expression("expr"))
            .with_code("E0308")
            .with_confidence(0.7)
    }

    /// Template: Add &mut for mutable reference
    #[must_use]
    pub(crate) fn add_mut_reference() -> FixTemplate {
        FixTemplate::new("&mut $expr", "Add mutable reference")
            .with_placeholder(Placeholder::expression("expr"))
            .with_code("E0308")
            .with_confidence(0.7)
    }

    /// Template: Dereference
    #[must_use]
    pub(crate) fn dereference() -> FixTemplate {
        FixTemplate::new("*$expr", "Dereference")
            .with_placeholder(Placeholder::expression("expr"))
            .with_code("E0308")
            .with_confidence(0.6)
    }

    /// Template: Add .`into()` for type conversion
    #[must_use]
    pub(crate) fn into_conversion() -> FixTemplate {
        FixTemplate::new("$expr.into()", "Use Into trait for conversion")
            .with_placeholder(Placeholder::expression("expr"))
            .with_code("E0308")
            .with_confidence(0.75)
    }

    /// Template: `Vec::new()` for empty vector
    #[must_use]
    pub(crate) fn vec_new() -> FixTemplate {
        FixTemplate::new("Vec::new()", "Create empty Vec")
            .with_code("E0308")
            .with_confidence(0.8)
    }

    /// Template: `String::new()` for empty string
    #[must_use]
    pub(crate) fn string_new() -> FixTemplate {
        FixTemplate::new("String::new()", "Create empty String")
            .with_code("E0308")
            .with_confidence(0.8)
    }

    /// Template: `Option::unwrap_or_default()`
    #[must_use]
    pub(crate) fn unwrap_or_default() -> FixTemplate {
        FixTemplate::new(
            "$expr.unwrap_or_default()",
            "Unwrap Option with default value",
        )
        .with_placeholder(Placeholder::expression("expr"))
        .with_code("E0308")
        .with_confidence(0.7)
    }

    // ==================== E0382 Templates (Use of Moved Value) ====================

    /// Template: Borrow instead of move
    #[must_use]
    pub(crate) fn borrow_instead_of_move() -> FixTemplate {
        FixTemplate::new("&$expr", "Borrow instead of moving the value")
            .with_placeholder(Placeholder::expression("expr"))
            .with_code("E0382")
            .with_confidence(0.85)
    }

    /// Template: Wrap in Rc for shared ownership
    #[must_use]
    pub(crate) fn rc_wrap() -> FixTemplate {
        FixTemplate::new("Rc::new($expr)", "Wrap in Rc for shared ownership")
            .with_placeholder(Placeholder::expression("expr"))
            .with_code("E0382")
            .with_confidence(0.7)
    }

    /// Template: Wrap in Arc for thread-safe shared ownership
    #[must_use]
    pub(crate) fn arc_wrap() -> FixTemplate {
        FixTemplate::new(
            "Arc::new($expr)",
            "Wrap in Arc for thread-safe shared ownership",
        )
        .with_placeholder(Placeholder::expression("expr"))
        .with_code("E0382")
        .with_confidence(0.65)
    }

    // ==================== E0277 Templates (Trait Bound Not Satisfied) ====================

    /// Template: Derive Debug trait
    #[must_use]
    pub(crate) fn derive_debug() -> FixTemplate {
        FixTemplate::new("#[derive(Debug)]", "Add Debug derive to type")
            .with_code("E0277")
            .with_confidence(0.9)
    }

    /// Template: Derive Clone trait
    #[must_use]
    pub(crate) fn derive_clone_trait() -> FixTemplate {
        FixTemplate::new("#[derive(Clone)]", "Add Clone derive to type")
            .with_code("E0277")
            .with_confidence(0.85)
    }

    /// Template: Implement Display trait
    #[must_use]
    pub(crate) fn impl_display() -> FixTemplate {
        FixTemplate::new(
            "impl std::fmt::Display for $type { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, \"{}\", self.$field) } }",
            "Implement Display trait"
        )
        .with_placeholder(Placeholder::type_name("type"))
        .with_placeholder(Placeholder::identifier("field"))
        .with_code("E0277")
        .with_confidence(0.7)
    }

    /// Template: Implement From trait
    #[must_use]
    pub(crate) fn impl_from() -> FixTemplate {
        FixTemplate::new(
            "impl From<$source> for $target { fn from(value: $source) -> Self { Self($value) } }",
            "Implement From trait for type conversion",
        )
        .with_placeholder(Placeholder::type_name("source"))
        .with_placeholder(Placeholder::type_name("target"))
        .with_placeholder(Placeholder::expression("value"))
        .with_code("E0277")
        .with_confidence(0.65)
    }

    // ==================== E0515 Templates (Cannot Return Reference to Local) ====================

    /// Template: Return owned value instead of reference
    #[must_use]
    pub(crate) fn return_owned() -> FixTemplate {
        FixTemplate::new("$expr", "Return owned value instead of reference")
            .with_placeholder(Placeholder::expression("expr"))
            .with_code("E0515")
            .with_confidence(0.8)
    }

    /// Template: Return a clone
    #[must_use]
    pub(crate) fn return_cloned() -> FixTemplate {
        FixTemplate::new("$expr.clone()", "Return a clone instead of reference")
            .with_placeholder(Placeholder::expression("expr"))
            .with_code("E0515")
            .with_confidence(0.75)
    }

    /// Template: Use Cow for efficient return
    #[must_use]
    pub(crate) fn use_cow() -> FixTemplate {
        FixTemplate::new(
            "Cow::Owned($expr)",
            "Use Cow for efficient owned/borrowed return",
        )
        .with_placeholder(Placeholder::expression("expr"))
        .with_code("E0515")
        .with_confidence(0.7)
    }

    /// Get all standard templates.
    #[must_use]
    pub(crate) fn all_templates() -> Vec<FixTemplate> {
        vec![
            // E0308 (Type Mismatch)
            to_string_conversion(),
            as_str_conversion(),
            add_reference(),
            add_mut_reference(),
            dereference(),
            into_conversion(),
            vec_new(),
            string_new(),
            unwrap_or_default(),
            // E0382 (Use of Moved Value)
            clone_value(),
            borrow_instead_of_move(),
            rc_wrap(),
            arc_wrap(),
            // E0277 (Trait Bound Not Satisfied)
            derive_debug(),
            derive_clone_trait(),
            impl_display(),
            impl_from(),
            // E0515 (Cannot Return Reference to Local)
            return_owned(),
            return_cloned(),
            use_cow(),
        ]
    }
}

#[cfg(test)]
mod tests;
