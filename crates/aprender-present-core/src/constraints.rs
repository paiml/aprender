//! Layout constraints for widgets.
//!
//! # Examples
//!
//! ```
//! use presentar_core::{Constraints, Size};
//!
//! // Create constraints with min/max bounds
//! let constraints = Constraints::new(0.0, 200.0, 0.0, 100.0);
//!
//! // Constrain a size to fit
//! let size = Size::new(300.0, 50.0);  // Too wide
//! let bounded = constraints.constrain(size);
//! assert_eq!(bounded.width, 200.0);   // Clamped to max
//! assert_eq!(bounded.height, 50.0);   // Within bounds
//! ```

use crate::geometry::Size;
use provable_contracts_macros::contract;
use serde::{Deserialize, Serialize};

/// Layout constraints that specify minimum and maximum sizes.
///
/// # Examples
///
/// ```
/// use presentar_core::{Constraints, Size};
///
/// // Tight constraints allow only one size
/// let tight = Constraints::tight(Size::new(100.0, 50.0));
/// assert_eq!(tight.min_width, 100.0);
/// assert_eq!(tight.max_width, 100.0);
///
/// // Loose constraints allow any size up to maximum
/// let loose = Constraints::loose(Size::new(400.0, 300.0));
/// assert_eq!(loose.min_width, 0.0);
/// assert_eq!(loose.max_width, 400.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Constraints {
    /// Minimum width
    pub min_width: f32,
    /// Maximum width
    pub max_width: f32,
    /// Minimum height
    pub min_height: f32,
    /// Maximum height
    pub max_height: f32,
}

impl Constraints {
    /// Create new constraints.
    #[must_use]
    pub const fn new(min_width: f32, max_width: f32, min_height: f32, max_height: f32) -> Self {
        Self {
            min_width,
            max_width,
            min_height,
            max_height,
        }
    }

    /// Create tight constraints that allow only the exact size.
    #[must_use]
    pub const fn tight(size: Size) -> Self {
        Self::new(size.width, size.width, size.height, size.height)
    }

    /// Create loose constraints that allow any size up to the given maximum.
    #[must_use]
    pub const fn loose(size: Size) -> Self {
        Self::new(0.0, size.width, 0.0, size.height)
    }

    /// Create unbounded constraints.
    #[must_use]
    pub const fn unbounded() -> Self {
        Self::new(0.0, f32::INFINITY, 0.0, f32::INFINITY)
    }

    /// Constrain a size to fit within these constraints.
    #[must_use]
    #[contract("constraints-layout-v1", equation = "constrain")]
    pub fn constrain(&self, size: Size) -> Size {
        Size::new(
            size.width.clamp(self.min_width, self.max_width),
            size.height.clamp(self.min_height, self.max_height),
        )
    }

    /// Check if constraints specify an exact size.
    #[must_use]
    pub fn is_tight(&self) -> bool {
        self.min_width == self.max_width && self.min_height == self.max_height
    }

    /// Check if width is bounded (not infinite).
    #[must_use]
    pub fn has_bounded_width(&self) -> bool {
        self.max_width.is_finite()
    }

    /// Check if height is bounded (not infinite).
    #[must_use]
    pub fn has_bounded_height(&self) -> bool {
        self.max_height.is_finite()
    }

    /// Check if both dimensions are bounded.
    #[must_use]
    pub fn is_bounded(&self) -> bool {
        self.has_bounded_width() && self.has_bounded_height()
    }

    /// Get the biggest size that satisfies these constraints.
    #[must_use]
    pub fn biggest(&self) -> Size {
        Size::new(
            if self.max_width.is_finite() {
                self.max_width
            } else {
                self.min_width
            },
            if self.max_height.is_finite() {
                self.max_height
            } else {
                self.min_height
            },
        )
    }

    /// Get the smallest size that satisfies these constraints.
    #[must_use]
    pub const fn smallest(&self) -> Size {
        Size::new(self.min_width, self.min_height)
    }

    /// Create constraints with a different minimum width.
    #[must_use]
    pub const fn with_min_width(&self, min_width: f32) -> Self {
        Self::new(min_width, self.max_width, self.min_height, self.max_height)
    }

    /// Create constraints with a different maximum width.
    #[must_use]
    pub const fn with_max_width(&self, max_width: f32) -> Self {
        Self::new(self.min_width, max_width, self.min_height, self.max_height)
    }

    /// Create constraints with a different minimum height.
    #[must_use]
    pub const fn with_min_height(&self, min_height: f32) -> Self {
        Self::new(self.min_width, self.max_width, min_height, self.max_height)
    }

    /// Create constraints with a different maximum height.
    #[must_use]
    pub const fn with_max_height(&self, max_height: f32) -> Self {
        Self::new(self.min_width, self.max_width, self.min_height, max_height)
    }

    /// Deflate constraints by padding.
    #[must_use]
    pub fn deflate(&self, horizontal: f32, vertical: f32) -> Self {
        Self::new(
            (self.min_width - horizontal).max(0.0),
            (self.max_width - horizontal).max(0.0),
            (self.min_height - vertical).max(0.0),
            (self.max_height - vertical).max(0.0),
        )
    }
}

impl Default for Constraints {
    fn default() -> Self {
        Self::unbounded()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::disallowed_methods)]
mod tests {
    use super::*;

    #[test]
    fn test_constraints_default() {
        let c = Constraints::default();
        assert_eq!(c.min_width, 0.0);
        assert_eq!(c.max_width, f32::INFINITY);
    }

    #[test]
    fn test_constraints_biggest() {
        let c = Constraints::new(10.0, 100.0, 20.0, 200.0);
        assert_eq!(c.biggest(), Size::new(100.0, 200.0));
    }

    #[test]
    fn test_constraints_smallest() {
        let c = Constraints::new(10.0, 100.0, 20.0, 200.0);
        assert_eq!(c.smallest(), Size::new(10.0, 20.0));
    }

    #[test]
    fn test_constraints_deflate() {
        let c = Constraints::new(100.0, 200.0, 100.0, 200.0);
        let deflated = c.deflate(20.0, 20.0);
        assert_eq!(deflated.min_width, 80.0);
        assert_eq!(deflated.max_width, 180.0);
    }

    #[test]
    fn test_constraints_with_methods() {
        let c = Constraints::new(0.0, 100.0, 0.0, 100.0);
        assert_eq!(c.with_min_width(10.0).min_width, 10.0);
        assert_eq!(c.with_max_width(200.0).max_width, 200.0);
        assert_eq!(c.with_min_height(10.0).min_height, 10.0);
        assert_eq!(c.with_max_height(200.0).max_height, 200.0);
    }

    #[test]
    fn test_constraints_tight() {
        let c = Constraints::tight(Size::new(100.0, 50.0));
        assert_eq!(c.min_width, 100.0);
        assert_eq!(c.max_width, 100.0);
        assert_eq!(c.min_height, 50.0);
        assert_eq!(c.max_height, 50.0);
        assert!(c.is_tight());
    }

    #[test]
    fn test_constraints_loose() {
        let c = Constraints::loose(Size::new(100.0, 50.0));
        assert_eq!(c.min_width, 0.0);
        assert_eq!(c.max_width, 100.0);
        assert_eq!(c.min_height, 0.0);
        assert_eq!(c.max_height, 50.0);
        assert!(!c.is_tight());
    }

    #[test]
    fn test_constraints_unbounded() {
        let c = Constraints::unbounded();
        assert_eq!(c.min_width, 0.0);
        assert!(c.max_width.is_infinite());
        assert!(!c.is_bounded());
    }

    #[test]
    fn test_constraints_constrain() {
        let c = Constraints::new(10.0, 100.0, 20.0, 80.0);
        assert_eq!(c.constrain(Size::new(50.0, 50.0)), Size::new(50.0, 50.0));
        assert_eq!(c.constrain(Size::new(5.0, 5.0)), Size::new(10.0, 20.0));
        assert_eq!(c.constrain(Size::new(200.0, 200.0)), Size::new(100.0, 80.0));
    }

    #[test]
    fn test_constraints_is_tight_false() {
        let c = Constraints::new(0.0, 100.0, 0.0, 100.0);
        assert!(!c.is_tight());
    }

    #[test]
    fn test_constraints_has_bounded_width() {
        let c = Constraints::new(0.0, 100.0, 0.0, f32::INFINITY);
        assert!(c.has_bounded_width());
        assert!(!c.has_bounded_height());
    }

    #[test]
    fn test_constraints_is_bounded() {
        let bounded = Constraints::new(0.0, 100.0, 0.0, 100.0);
        assert!(bounded.is_bounded());

        let unbounded = Constraints::unbounded();
        assert!(!unbounded.is_bounded());
    }

    #[test]
    fn test_constraints_biggest_unbounded() {
        let c = Constraints::unbounded();
        assert_eq!(c.biggest(), Size::new(0.0, 0.0));
    }

    #[test]
    fn test_constraints_deflate_to_zero() {
        let c = Constraints::new(10.0, 20.0, 10.0, 20.0);
        let deflated = c.deflate(50.0, 50.0);
        assert_eq!(deflated.min_width, 0.0);
        assert_eq!(deflated.max_width, 0.0);
    }

    // =========================================================================
    // Clone and Copy Trait Tests
    // =========================================================================

    #[test]
    fn test_constraints_clone() {
        let c = Constraints::new(10.0, 100.0, 20.0, 200.0);
        let cloned = c;
        assert_eq!(c, cloned);
    }

    #[test]
    fn test_constraints_copy() {
        let c = Constraints::new(10.0, 100.0, 20.0, 200.0);
        let copied = c;
        // Both should be valid and equal
        assert_eq!(c.min_width, copied.min_width);
        assert_eq!(c.max_width, copied.max_width);
    }

    // =========================================================================
    // Debug Trait Tests
    // =========================================================================

    #[test]
    fn test_constraints_debug() {
        let c = Constraints::new(10.0, 100.0, 20.0, 200.0);
        let debug = format!("{c:?}");
        assert!(debug.contains("Constraints"));
        assert!(debug.contains("min_width"));
        assert!(debug.contains("max_width"));
    }

    // =========================================================================
    // PartialEq Tests
    // =========================================================================

    #[test]
    fn test_constraints_equality() {
        let c1 = Constraints::new(10.0, 100.0, 20.0, 200.0);
        let c2 = Constraints::new(10.0, 100.0, 20.0, 200.0);
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_constraints_inequality_min_width() {
        let c1 = Constraints::new(10.0, 100.0, 20.0, 200.0);
        let c2 = Constraints::new(15.0, 100.0, 20.0, 200.0);
        assert_ne!(c1, c2);
    }

    #[test]
    fn test_constraints_inequality_max_width() {
        let c1 = Constraints::new(10.0, 100.0, 20.0, 200.0);
        let c2 = Constraints::new(10.0, 150.0, 20.0, 200.0);
        assert_ne!(c1, c2);
    }

    #[test]
    fn test_constraints_inequality_min_height() {
        let c1 = Constraints::new(10.0, 100.0, 20.0, 200.0);
        let c2 = Constraints::new(10.0, 100.0, 25.0, 200.0);
        assert_ne!(c1, c2);
    }

    #[test]
    fn test_constraints_inequality_max_height() {
        let c1 = Constraints::new(10.0, 100.0, 20.0, 200.0);
        let c2 = Constraints::new(10.0, 100.0, 20.0, 250.0);
        assert_ne!(c1, c2);
    }

    // =========================================================================
    // Serialization Tests
    // =========================================================================

    #[test]
    fn test_constraints_serialize() {
        let c = Constraints::new(10.0, 100.0, 20.0, 200.0);
        let json = serde_json::to_string(&c).unwrap();
        assert!(json.contains("min_width"));
        assert!(json.contains("10"));
    }

    #[test]
    fn test_constraints_deserialize() {
        let json = r#"{"min_width":10.0,"max_width":100.0,"min_height":20.0,"max_height":200.0}"#;
        let c: Constraints = serde_json::from_str(json).unwrap();
        assert_eq!(c.min_width, 10.0);
        assert_eq!(c.max_width, 100.0);
        assert_eq!(c.min_height, 20.0);
        assert_eq!(c.max_height, 200.0);
    }

    #[test]
    fn test_constraints_roundtrip_serialization() {
        let original = Constraints::new(15.5, 150.5, 25.5, 250.5);
        let json = serde_json::to_string(&original).unwrap();
        let deserialized: Constraints = serde_json::from_str(&json).unwrap();
        assert_eq!(original, deserialized);
    }

    // =========================================================================
    // Constrain Edge Cases
    // =========================================================================

    #[test]
    fn test_constrain_at_minimum() {
        let c = Constraints::new(10.0, 100.0, 20.0, 200.0);
        let size = Size::new(10.0, 20.0);
        assert_eq!(c.constrain(size), size);
    }

    #[test]
    fn test_constrain_at_maximum() {
        let c = Constraints::new(10.0, 100.0, 20.0, 200.0);
        let size = Size::new(100.0, 200.0);
        assert_eq!(c.constrain(size), size);
    }

    #[test]
    fn test_constrain_zero_size() {
        let c = Constraints::new(10.0, 100.0, 20.0, 200.0);
        let size = Size::new(0.0, 0.0);
        assert_eq!(c.constrain(size), Size::new(10.0, 20.0));
    }

    #[test]
    fn test_constrain_negative_clamped() {
        let c = Constraints::new(0.0, 100.0, 0.0, 100.0);
        let size = Size::new(-10.0, -20.0);
        assert_eq!(c.constrain(size), Size::new(0.0, 0.0));
    }

    #[test]
    fn test_constrain_with_zero_constraints() {
        let c = Constraints::new(0.0, 0.0, 0.0, 0.0);
        let size = Size::new(100.0, 100.0);
        assert_eq!(c.constrain(size), Size::new(0.0, 0.0));
    }

    // =========================================================================
    // is_tight Edge Cases
    // =========================================================================

    #[test]
    fn test_is_tight_width_only() {
        let c = Constraints::new(50.0, 50.0, 0.0, 100.0);
        assert!(!c.is_tight()); // Height is not tight
    }

    #[test]
    fn test_is_tight_height_only() {
        let c = Constraints::new(0.0, 100.0, 50.0, 50.0);
        assert!(!c.is_tight()); // Width is not tight
    }

    #[test]
    fn test_is_tight_zero_size() {
        let c = Constraints::tight(Size::new(0.0, 0.0));
        assert!(c.is_tight());
    }

    // =========================================================================
    // Bounded Tests
    // =========================================================================

    #[test]
    fn test_has_bounded_height_only() {
        let c = Constraints::new(0.0, f32::INFINITY, 0.0, 100.0);
        assert!(!c.has_bounded_width());
        assert!(c.has_bounded_height());
        assert!(!c.is_bounded());
    }

    #[test]
    fn test_has_bounded_width_only() {
        let c = Constraints::new(0.0, 100.0, 0.0, f32::INFINITY);
        assert!(c.has_bounded_width());
        assert!(!c.has_bounded_height());
        assert!(!c.is_bounded());
    }

    // =========================================================================
    // biggest() Edge Cases
    // =========================================================================

    #[test]
    fn test_biggest_with_infinity_width_only() {
        let c = Constraints::new(50.0, f32::INFINITY, 0.0, 100.0);
        let biggest = c.biggest();
        assert_eq!(biggest.width, 50.0); // Falls back to min
        assert_eq!(biggest.height, 100.0);
    }

    #[test]
    fn test_biggest_with_infinity_height_only() {
        let c = Constraints::new(0.0, 100.0, 50.0, f32::INFINITY);
        let biggest = c.biggest();
        assert_eq!(biggest.width, 100.0);
        assert_eq!(biggest.height, 50.0); // Falls back to min
    }

    #[test]
    fn test_biggest_tight_constraints() {
        let c = Constraints::tight(Size::new(42.0, 24.0));
        assert_eq!(c.biggest(), Size::new(42.0, 24.0));
    }

    // =========================================================================
    // smallest() Tests
    // =========================================================================

    #[test]
    fn test_smallest_unbounded() {
        let c = Constraints::unbounded();
        assert_eq!(c.smallest(), Size::new(0.0, 0.0));
    }

    #[test]
    fn test_smallest_tight() {
        let c = Constraints::tight(Size::new(42.0, 24.0));
        assert_eq!(c.smallest(), Size::new(42.0, 24.0));
    }

    #[test]
    fn test_smallest_loose() {
        let c = Constraints::loose(Size::new(100.0, 200.0));
        assert_eq!(c.smallest(), Size::new(0.0, 0.0));
    }

    // =========================================================================
    // with_* Methods Chain Tests
    // =========================================================================

    #[test]
    fn test_with_methods_chained() {
        let c = Constraints::unbounded()
            .with_min_width(10.0)
            .with_max_width(100.0)
            .with_min_height(20.0)
            .with_max_height(200.0);

        assert_eq!(c.min_width, 10.0);
        assert_eq!(c.max_width, 100.0);
        assert_eq!(c.min_height, 20.0);
        assert_eq!(c.max_height, 200.0);
    }

    #[test]
    fn test_with_methods_preserve_other_values() {
        let c = Constraints::new(10.0, 100.0, 20.0, 200.0);

        let c2 = c.with_min_width(15.0);
        assert_eq!(c2.max_width, 100.0);
        assert_eq!(c2.min_height, 20.0);
        assert_eq!(c2.max_height, 200.0);

        let c3 = c.with_max_width(150.0);
        assert_eq!(c3.min_width, 10.0);
        assert_eq!(c3.min_height, 20.0);
        assert_eq!(c3.max_height, 200.0);
    }

    // =========================================================================
    // deflate() Edge Cases
    // =========================================================================

    #[test]
    fn test_deflate_asymmetric() {
        let c = Constraints::new(20.0, 100.0, 30.0, 150.0);
        let deflated = c.deflate(10.0, 20.0);
        assert_eq!(deflated.min_width, 10.0);
        assert_eq!(deflated.max_width, 90.0);
        assert_eq!(deflated.min_height, 10.0);
        assert_eq!(deflated.max_height, 130.0);
    }

    #[test]
    fn test_deflate_zero() {
        let c = Constraints::new(10.0, 100.0, 20.0, 200.0);
        let deflated = c.deflate(0.0, 0.0);
        assert_eq!(c, deflated);
    }

    #[test]
    fn test_deflate_exact_match() {
        let c = Constraints::new(10.0, 100.0, 20.0, 200.0);
        let deflated = c.deflate(10.0, 20.0);
        assert_eq!(deflated.min_width, 0.0);
        assert_eq!(deflated.max_width, 90.0);
        assert_eq!(deflated.min_height, 0.0);
        assert_eq!(deflated.max_height, 180.0);
    }

    #[test]
    fn test_deflate_negative_becomes_zero() {
        let c = Constraints::new(5.0, 10.0, 5.0, 10.0);
        let deflated = c.deflate(15.0, 15.0);
        assert_eq!(deflated.min_width, 0.0);
        assert_eq!(deflated.max_width, 0.0);
        assert_eq!(deflated.min_height, 0.0);
        assert_eq!(deflated.max_height, 0.0);
    }

    // =========================================================================
    // Constructor Edge Cases
    // =========================================================================

    #[test]
    fn test_new_with_zero_values() {
        let c = Constraints::new(0.0, 0.0, 0.0, 0.0);
        assert_eq!(c.min_width, 0.0);
        assert_eq!(c.max_width, 0.0);
        assert!(c.is_tight());
    }

    #[test]
    fn test_tight_with_large_values() {
        let c = Constraints::tight(Size::new(10000.0, 10000.0));
        assert!(c.is_tight());
        assert_eq!(c.biggest(), Size::new(10000.0, 10000.0));
    }

    #[test]
    fn test_loose_with_zero() {
        let c = Constraints::loose(Size::new(0.0, 0.0));
        assert!(c.is_tight()); // min and max are both 0
        assert_eq!(c.biggest(), Size::new(0.0, 0.0));
    }

    // =========================================================================
    // Default Trait Tests
    // =========================================================================

    #[test]
    fn test_default_is_unbounded() {
        let default = Constraints::default();
        let unbounded = Constraints::unbounded();
        assert_eq!(default, unbounded);
    }

    #[test]
    fn test_default_not_bounded() {
        let c = Constraints::default();
        assert!(!c.is_bounded());
        assert!(!c.has_bounded_width());
        assert!(!c.has_bounded_height());
    }
}
