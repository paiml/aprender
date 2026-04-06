//! Flexbox layout types.

use serde::{Deserialize, Serialize};

/// Direction for flex layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum FlexDirection {
    /// Horizontal (left to right)
    #[default]
    Row,
    /// Horizontal (right to left)
    RowReverse,
    /// Vertical (top to bottom)
    Column,
    /// Vertical (bottom to top)
    ColumnReverse,
}

/// Main axis alignment for flex layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum FlexJustify {
    /// Pack items at the start
    #[default]
    Start,
    /// Pack items at the end
    End,
    /// Center items
    Center,
    /// Distribute space evenly between items
    SpaceBetween,
    /// Distribute space evenly around items
    SpaceAround,
    /// Distribute space evenly, including edges
    SpaceEvenly,
}

/// Cross axis alignment for flex layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum FlexAlign {
    /// Align to the start
    Start,
    /// Align to the end
    End,
    /// Center items
    #[default]
    Center,
    /// Stretch to fill
    Stretch,
    /// Align to baseline
    Baseline,
}

/// Flex item properties.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct FlexItem {
    /// Flex grow factor
    pub grow: f32,
    /// Flex shrink factor
    pub shrink: f32,
    /// Flex basis (initial size)
    pub basis: Option<f32>,
    /// Self alignment override
    pub align_self: Option<FlexAlign>,
    /// UX-107: Collapse to zero size when content is empty.
    /// When true, items with no content will have 0 size in layout.
    pub collapse_if_empty: bool,
}

impl FlexItem {
    /// Create a new flex item with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the grow factor.
    #[must_use]
    pub const fn grow(mut self, grow: f32) -> Self {
        self.grow = grow;
        self
    }

    /// Set the shrink factor.
    #[must_use]
    pub const fn shrink(mut self, shrink: f32) -> Self {
        self.shrink = shrink;
        self
    }

    /// Set the basis.
    #[must_use]
    pub const fn basis(mut self, basis: f32) -> Self {
        self.basis = Some(basis);
        self
    }

    /// Set self alignment.
    #[must_use]
    pub const fn align_self(mut self, align: FlexAlign) -> Self {
        self.align_self = Some(align);
        self
    }

    /// UX-107: Enable auto-collapse when content is empty.
    #[must_use]
    pub const fn collapse_if_empty(mut self) -> Self {
        self.collapse_if_empty = true;
        self
    }
}

/// Distribute available space among flex items.
/// UX-107: Items with `collapse_if_empty=true` and size=0 are excluded from distribution.
/// Compute which items are collapsed (empty and collapse_if_empty=true).
fn compute_collapsed(items: &[FlexItem], sizes: &[f32]) -> Vec<bool> {
    items
        .iter()
        .zip(sizes.iter())
        .map(|(item, &size)| item.collapse_if_empty && size == 0.0)
        .collect()
}

/// Sum a flex factor (grow or shrink) for non-collapsed items.
fn sum_flex_factor(
    items: &[FlexItem],
    collapsed: &[bool],
    get_factor: fn(&FlexItem) -> f32,
) -> f32 {
    items
        .iter()
        .zip(collapsed.iter())
        .filter(|(_, &is_collapsed)| !is_collapsed)
        .map(|(item, _)| get_factor(item))
        .sum()
}

/// Apply flex adjustment to sizes.
fn apply_flex_adjustment(
    sizes: &[f32],
    items: &[FlexItem],
    collapsed: &[bool],
    remaining: f32,
    total_factor: f32,
    get_factor: fn(&FlexItem) -> f32,
    clamp: bool,
) -> Vec<f32> {
    sizes
        .iter()
        .zip(items.iter())
        .zip(collapsed.iter())
        .map(|((&size, item), &is_collapsed)| {
            if is_collapsed {
                0.0
            } else {
                let adjusted = size + (remaining * get_factor(item) / total_factor);
                if clamp {
                    adjusted.max(0.0)
                } else {
                    adjusted
                }
            }
        })
        .collect()
}

#[must_use]
#[allow(dead_code)]
pub(crate) fn distribute_flex(items: &[FlexItem], sizes: &[f32], available: f32) -> Vec<f32> {
    if items.is_empty() {
        return Vec::new();
    }

    // UX-107: Collapsed items keep size 0 and don't participate in flex distribution
    let collapsed = compute_collapsed(items, sizes);

    let total_size: f32 = sizes
        .iter()
        .zip(collapsed.iter())
        .filter(|(_, &is_collapsed)| !is_collapsed)
        .map(|(&s, _)| s)
        .sum();

    let remaining = available - total_size;

    if remaining.abs() < 0.001 {
        return sizes.to_vec();
    }

    let get_grow: fn(&FlexItem) -> f32 = |i| i.grow;
    let get_shrink: fn(&FlexItem) -> f32 = |i| i.shrink;

    if remaining > 0.0 {
        let total_grow = sum_flex_factor(items, &collapsed, get_grow);
        if total_grow > 0.0 {
            return apply_flex_adjustment(
                sizes, items, &collapsed, remaining, total_grow, get_grow, false,
            );
        }
    } else {
        let total_shrink = sum_flex_factor(items, &collapsed, get_shrink);
        if total_shrink > 0.0 {
            return apply_flex_adjustment(
                sizes,
                items,
                &collapsed,
                remaining,
                total_shrink,
                get_shrink,
                true,
            );
        }
    }

    // Keep collapsed items at 0
    sizes
        .iter()
        .zip(collapsed.iter())
        .map(|(&size, &is_collapsed)| if is_collapsed { 0.0 } else { size })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flex_direction_default() {
        assert_eq!(FlexDirection::default(), FlexDirection::Row);
    }

    #[test]
    fn test_flex_justify_default() {
        assert_eq!(FlexJustify::default(), FlexJustify::Start);
    }

    #[test]
    fn test_flex_align_default() {
        assert_eq!(FlexAlign::default(), FlexAlign::Center);
    }

    #[test]
    fn test_flex_item_builder() {
        let item = FlexItem::new()
            .grow(1.0)
            .shrink(0.0)
            .basis(100.0)
            .align_self(FlexAlign::Start);

        assert_eq!(item.grow, 1.0);
        assert_eq!(item.shrink, 0.0);
        assert_eq!(item.basis, Some(100.0));
        assert_eq!(item.align_self, Some(FlexAlign::Start));
    }

    #[test]
    fn test_distribute_flex_empty() {
        let result = distribute_flex(&[], &[], 100.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_distribute_flex_exact_fit() {
        let items = vec![FlexItem::new(), FlexItem::new()];
        let sizes = vec![50.0, 50.0];
        let result = distribute_flex(&items, &sizes, 100.0);
        assert_eq!(result, vec![50.0, 50.0]);
    }

    #[test]
    fn test_distribute_flex_grow() {
        let items = vec![FlexItem::new().grow(1.0), FlexItem::new().grow(1.0)];
        let sizes = vec![25.0, 25.0];
        let result = distribute_flex(&items, &sizes, 100.0);
        assert_eq!(result, vec![50.0, 50.0]);
    }

    #[test]
    fn test_distribute_flex_grow_uneven() {
        let items = vec![FlexItem::new().grow(1.0), FlexItem::new().grow(3.0)];
        let sizes = vec![0.0, 0.0];
        let result = distribute_flex(&items, &sizes, 100.0);
        assert_eq!(result, vec![25.0, 75.0]);
    }

    #[test]
    fn test_distribute_flex_shrink() {
        let items = vec![FlexItem::new().shrink(1.0), FlexItem::new().shrink(1.0)];
        let sizes = vec![75.0, 75.0];
        let result = distribute_flex(&items, &sizes, 100.0);
        assert_eq!(result, vec![50.0, 50.0]);
    }

    // =========================================================================
    // FlexDirection Tests
    // =========================================================================

    #[test]
    fn test_flex_direction_clone() {
        let dir = FlexDirection::Column;
        let cloned = dir;
        assert_eq!(dir, cloned);
    }

    #[test]
    fn test_flex_direction_all_variants() {
        assert_eq!(FlexDirection::Row, FlexDirection::Row);
        assert_eq!(FlexDirection::RowReverse, FlexDirection::RowReverse);
        assert_eq!(FlexDirection::Column, FlexDirection::Column);
        assert_eq!(FlexDirection::ColumnReverse, FlexDirection::ColumnReverse);
    }

    #[test]
    fn test_flex_direction_debug() {
        let dir = FlexDirection::Row;
        let debug = format!("{:?}", dir);
        assert!(debug.contains("Row"));
    }

    // =========================================================================
    // FlexJustify Tests
    // =========================================================================

    #[test]
    fn test_flex_justify_all_variants() {
        assert_eq!(FlexJustify::Start, FlexJustify::Start);
        assert_eq!(FlexJustify::End, FlexJustify::End);
        assert_eq!(FlexJustify::Center, FlexJustify::Center);
        assert_eq!(FlexJustify::SpaceBetween, FlexJustify::SpaceBetween);
        assert_eq!(FlexJustify::SpaceAround, FlexJustify::SpaceAround);
        assert_eq!(FlexJustify::SpaceEvenly, FlexJustify::SpaceEvenly);
    }

    #[test]
    fn test_flex_justify_clone() {
        let justify = FlexJustify::SpaceBetween;
        let cloned = justify;
        assert_eq!(justify, cloned);
    }

    #[test]
    fn test_flex_justify_debug() {
        let justify = FlexJustify::Center;
        let debug = format!("{:?}", justify);
        assert!(debug.contains("Center"));
    }

    // =========================================================================
    // FlexAlign Tests
    // =========================================================================

    #[test]
    fn test_flex_align_all_variants() {
        assert_eq!(FlexAlign::Start, FlexAlign::Start);
        assert_eq!(FlexAlign::End, FlexAlign::End);
        assert_eq!(FlexAlign::Center, FlexAlign::Center);
        assert_eq!(FlexAlign::Stretch, FlexAlign::Stretch);
        assert_eq!(FlexAlign::Baseline, FlexAlign::Baseline);
    }

    #[test]
    fn test_flex_align_clone() {
        let align = FlexAlign::Stretch;
        let cloned = align;
        assert_eq!(align, cloned);
    }

    #[test]
    fn test_flex_align_debug() {
        let align = FlexAlign::Baseline;
        let debug = format!("{:?}", align);
        assert!(debug.contains("Baseline"));
    }

    // =========================================================================
    // FlexItem Tests
    // =========================================================================

    #[test]
    fn test_flex_item_default() {
        let item = FlexItem::default();
        assert_eq!(item.grow, 0.0);
        assert_eq!(item.shrink, 0.0);
        assert_eq!(item.basis, None);
        assert_eq!(item.align_self, None);
    }

    #[test]
    fn test_flex_item_new() {
        let item = FlexItem::new();
        assert_eq!(item.grow, 0.0);
        assert_eq!(item.shrink, 0.0);
    }

    #[test]
    fn test_flex_item_grow_only() {
        let item = FlexItem::new().grow(2.5);
        assert_eq!(item.grow, 2.5);
        assert_eq!(item.shrink, 0.0);
    }

    #[test]
    fn test_flex_item_shrink_only() {
        let item = FlexItem::new().shrink(0.5);
        assert_eq!(item.shrink, 0.5);
        assert_eq!(item.grow, 0.0);
    }

    #[test]
    fn test_flex_item_basis_only() {
        let item = FlexItem::new().basis(200.0);
        assert_eq!(item.basis, Some(200.0));
    }

    #[test]
    fn test_flex_item_align_self_only() {
        let item = FlexItem::new().align_self(FlexAlign::End);
        assert_eq!(item.align_self, Some(FlexAlign::End));
    }

    #[test]
    fn test_flex_item_clone() {
        let item = FlexItem::new().grow(1.0).shrink(0.5);
        let cloned = item;
        assert_eq!(item.grow, cloned.grow);
        assert_eq!(item.shrink, cloned.shrink);
    }

    #[test]
    fn test_flex_item_debug() {
        let item = FlexItem::new().grow(1.0);
        let debug = format!("{:?}", item);
        assert!(debug.contains("FlexItem"));
    }

    // =========================================================================
    // distribute_flex Tests
    // =========================================================================

    #[test]
    fn test_distribute_flex_no_grow_no_shrink() {
        let items = vec![FlexItem::new(), FlexItem::new()];
        let sizes = vec![30.0, 30.0];
        let result = distribute_flex(&items, &sizes, 100.0);
        // No grow factor, so sizes remain unchanged
        assert_eq!(result, vec![30.0, 30.0]);
    }

    #[test]
    fn test_distribute_flex_single_item_grow() {
        let items = vec![FlexItem::new().grow(1.0)];
        let sizes = vec![50.0];
        let result = distribute_flex(&items, &sizes, 100.0);
        assert_eq!(result, vec![100.0]);
    }

    #[test]
    fn test_distribute_flex_single_item_shrink() {
        let items = vec![FlexItem::new().shrink(1.0)];
        let sizes = vec![150.0];
        let result = distribute_flex(&items, &sizes, 100.0);
        assert_eq!(result, vec![100.0]);
    }

    #[test]
    fn test_distribute_flex_shrink_uneven() {
        let items = vec![FlexItem::new().shrink(1.0), FlexItem::new().shrink(3.0)];
        let sizes = vec![100.0, 100.0];
        let result = distribute_flex(&items, &sizes, 100.0);
        // Total: 200, need to shrink by 100
        // item1: 100 - 100 * 1/4 = 75
        // item2: 100 - 100 * 3/4 = 25
        assert_eq!(result, vec![75.0, 25.0]);
    }

    #[test]
    fn test_distribute_flex_shrink_to_zero() {
        let items = vec![FlexItem::new().shrink(1.0)];
        let sizes = vec![50.0];
        // Need to shrink more than available
        let result = distribute_flex(&items, &sizes, 0.0);
        assert_eq!(result, vec![0.0]); // Can't go below 0
    }

    #[test]
    fn test_distribute_flex_mixed_grow() {
        let items = vec![
            FlexItem::new().grow(0.0), // Won't grow
            FlexItem::new().grow(1.0), // Will take all remaining
        ];
        let sizes = vec![50.0, 0.0];
        let result = distribute_flex(&items, &sizes, 100.0);
        assert_eq!(result, vec![50.0, 50.0]);
    }

    #[test]
    fn test_distribute_flex_three_items() {
        let items = vec![
            FlexItem::new().grow(1.0),
            FlexItem::new().grow(2.0),
            FlexItem::new().grow(1.0),
        ];
        let sizes = vec![0.0, 0.0, 0.0];
        let result = distribute_flex(&items, &sizes, 100.0);
        assert_eq!(result, vec![25.0, 50.0, 25.0]);
    }

    #[test]
    fn test_distribute_flex_near_exact_fit() {
        let items = vec![FlexItem::new().grow(1.0), FlexItem::new().grow(1.0)];
        let sizes = vec![49.9995, 50.0005];
        let result = distribute_flex(&items, &sizes, 100.0);
        // Should be treated as exact fit (within 0.001 tolerance)
        assert_eq!(result, vec![49.9995, 50.0005]);
    }

    // =========================================================================
    // UX-107: collapse_if_empty Tests
    // =========================================================================

    #[test]
    fn test_flex_item_collapse_if_empty() {
        let item = FlexItem::new().collapse_if_empty();
        assert!(item.collapse_if_empty);
    }

    #[test]
    fn test_flex_item_collapse_if_empty_default_false() {
        let item = FlexItem::new();
        assert!(!item.collapse_if_empty);
    }

    #[test]
    fn test_distribute_flex_collapsed_item_stays_zero() {
        let items = vec![
            FlexItem::new().grow(1.0).collapse_if_empty(),
            FlexItem::new().grow(1.0),
        ];
        let sizes = vec![0.0, 50.0]; // First item empty (collapsed), second has content
        let result = distribute_flex(&items, &sizes, 100.0);
        // Collapsed item stays 0, second item gets all the extra space
        assert_eq!(result, vec![0.0, 100.0]);
    }

    #[test]
    fn test_distribute_flex_collapsed_doesnt_participate_in_grow() {
        let items = vec![
            FlexItem::new().grow(1.0).collapse_if_empty(),
            FlexItem::new().grow(1.0),
            FlexItem::new().grow(1.0),
        ];
        let sizes = vec![0.0, 25.0, 25.0]; // First collapsed, others have content
        let result = distribute_flex(&items, &sizes, 100.0);
        // Collapsed stays 0, remaining 50 is split evenly between items 2 and 3
        assert_eq!(result, vec![0.0, 50.0, 50.0]);
    }

    #[test]
    fn test_distribute_flex_collapsed_with_size_not_collapsed() {
        // collapse_if_empty only collapses if size is 0
        let items = vec![
            FlexItem::new().grow(1.0).collapse_if_empty(),
            FlexItem::new().grow(1.0),
        ];
        let sizes = vec![30.0, 30.0]; // Both have content, collapse flag doesn't apply
        let result = distribute_flex(&items, &sizes, 100.0);
        // Both participate in grow
        assert_eq!(result, vec![50.0, 50.0]);
    }

    #[test]
    fn test_distribute_flex_all_collapsed() {
        let items = vec![
            FlexItem::new().grow(1.0).collapse_if_empty(),
            FlexItem::new().grow(1.0).collapse_if_empty(),
        ];
        let sizes = vec![0.0, 0.0]; // Both empty
        let result = distribute_flex(&items, &sizes, 100.0);
        // Both stay at 0
        assert_eq!(result, vec![0.0, 0.0]);
    }

    #[test]
    fn test_distribute_flex_collapsed_in_shrink() {
        let items = vec![
            FlexItem::new().shrink(1.0).collapse_if_empty(),
            FlexItem::new().shrink(1.0),
        ];
        let sizes = vec![0.0, 120.0]; // First empty, second needs shrinking
        let result = distribute_flex(&items, &sizes, 100.0);
        // Collapsed stays 0, second shrinks to fit
        assert_eq!(result, vec![0.0, 100.0]);
    }
}
