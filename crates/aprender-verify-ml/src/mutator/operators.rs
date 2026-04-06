//! Mutation operator definitions
//!
//! From Jia & Harman (2011) "An Analysis and Survey of the Development of Mutation Testing"

/// Mutation operators from the standard catalog
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MutationOperator {
    /// Arithmetic Operator Replacement
    /// `a + b` → `a - b`, `a * b`, `a / b`, `a % b`
    Aor,

    /// Relational Operator Replacement
    /// `a < b` → `a <= b`, `a > b`, `a >= b`, `a == b`, `a != b`
    Ror,

    /// Logical Operator Replacement
    /// `a and b` → `a or b`
    Lor,

    /// Unary Operator Insertion
    /// `x` → `-x`, `not x`
    Uoi,

    /// Absolute Value Insertion
    /// `x` → `abs(x)`
    Abs,

    /// Statement Deletion
    /// Delete a statement from the program
    Sdl,

    /// Scalar Variable Replacement
    /// `x` → `y` (where y has the same type)
    Svr,

    /// Boundary Substitution Replacement
    /// `0` → `-1`, `""` → `" "`, `[]` → `[None]`
    Bsr,
}

impl MutationOperator {
    /// Get all mutation operators
    #[must_use]
    pub fn all() -> Vec<Self> {
        vec![
            Self::Aor,
            Self::Ror,
            Self::Lor,
            Self::Uoi,
            Self::Abs,
            Self::Sdl,
            Self::Svr,
            Self::Bsr,
        ]
    }

    /// Get operators recommended for ASTTransform bugs (P0 priority)
    ///
    /// From spec: ASTTransform bugs are 40-62% of defects
    #[must_use]
    pub fn ast_transform_operators() -> Vec<Self> {
        vec![Self::Aor, Self::Ror, Self::Lor]
    }

    /// Get operators recommended for OwnershipBorrow bugs (P1 priority)
    ///
    /// From spec: OwnershipBorrow bugs are 15-20% of defects
    #[must_use]
    pub fn ownership_operators() -> Vec<Self> {
        vec![Self::Bsr, Self::Svr]
    }

    /// Get description of the operator
    #[must_use]
    pub fn description(&self) -> &'static str {
        match self {
            Self::Aor => "Arithmetic Operator Replacement",
            Self::Ror => "Relational Operator Replacement",
            Self::Lor => "Logical Operator Replacement",
            Self::Uoi => "Unary Operator Insertion",
            Self::Abs => "Absolute Value Insertion",
            Self::Sdl => "Statement Deletion",
            Self::Svr => "Scalar Variable Replacement",
            Self::Bsr => "Boundary Substitution Replacement",
        }
    }
}

impl std::fmt::Display for MutationOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Aor => write!(f, "AOR"),
            Self::Ror => write!(f, "ROR"),
            Self::Lor => write!(f, "LOR"),
            Self::Uoi => write!(f, "UOI"),
            Self::Abs => write!(f, "ABS"),
            Self::Sdl => write!(f, "SDL"),
            Self::Svr => write!(f, "SVR"),
            Self::Bsr => write!(f, "BSR"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_operators() {
        let ops = MutationOperator::all();
        assert_eq!(ops.len(), 8);
    }

    #[test]
    fn test_ast_transform_operators() {
        let ops = MutationOperator::ast_transform_operators();
        assert_eq!(ops.len(), 3);
        assert!(ops.contains(&MutationOperator::Aor));
        assert!(ops.contains(&MutationOperator::Ror));
        assert!(ops.contains(&MutationOperator::Lor));
    }

    #[test]
    fn test_ownership_operators() {
        let ops = MutationOperator::ownership_operators();
        assert_eq!(ops.len(), 2);
        assert!(ops.contains(&MutationOperator::Bsr));
        assert!(ops.contains(&MutationOperator::Svr));
    }

    #[test]
    fn test_operator_display_all() {
        assert_eq!(format!("{}", MutationOperator::Aor), "AOR");
        assert_eq!(format!("{}", MutationOperator::Ror), "ROR");
        assert_eq!(format!("{}", MutationOperator::Lor), "LOR");
        assert_eq!(format!("{}", MutationOperator::Uoi), "UOI");
        assert_eq!(format!("{}", MutationOperator::Abs), "ABS");
        assert_eq!(format!("{}", MutationOperator::Sdl), "SDL");
        assert_eq!(format!("{}", MutationOperator::Svr), "SVR");
        assert_eq!(format!("{}", MutationOperator::Bsr), "BSR");
    }

    #[test]
    fn test_operator_description_all() {
        assert_eq!(
            MutationOperator::Aor.description(),
            "Arithmetic Operator Replacement"
        );
        assert_eq!(
            MutationOperator::Ror.description(),
            "Relational Operator Replacement"
        );
        assert_eq!(
            MutationOperator::Lor.description(),
            "Logical Operator Replacement"
        );
        assert_eq!(
            MutationOperator::Uoi.description(),
            "Unary Operator Insertion"
        );
        assert_eq!(
            MutationOperator::Abs.description(),
            "Absolute Value Insertion"
        );
        assert_eq!(MutationOperator::Sdl.description(), "Statement Deletion");
        assert_eq!(
            MutationOperator::Svr.description(),
            "Scalar Variable Replacement"
        );
        assert_eq!(
            MutationOperator::Bsr.description(),
            "Boundary Substitution Replacement"
        );
    }

    #[test]
    fn test_operator_clone_copy() {
        let op = MutationOperator::Aor;
        let cloned = op.clone();
        let copied = op;
        assert_eq!(op, cloned);
        assert_eq!(op, copied);
    }

    #[test]
    fn test_operator_eq() {
        assert_eq!(MutationOperator::Aor, MutationOperator::Aor);
        assert_ne!(MutationOperator::Aor, MutationOperator::Ror);
    }

    #[test]
    fn test_operator_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(MutationOperator::Aor);
        set.insert(MutationOperator::Ror);
        set.insert(MutationOperator::Aor); // duplicate
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_operator_debug() {
        let debug_str = format!("{:?}", MutationOperator::Aor);
        assert!(debug_str.contains("Aor"));
    }
}
