# Flexbox Model

Presentar uses a CSS Flexbox-inspired layout model.

## Core Concepts

### Main Axis

Direction children are laid out:
- **Row**: Horizontal (left to right)
- **Column**: Vertical (top to bottom)

### Cross Axis

Perpendicular to main axis.

## MainAxisAlignment

Controls spacing along main axis:

```rust
use presentar_widgets::row::MainAxisAlignment;

// Start (default)
Row::new().main_axis_alignment(MainAxisAlignment::Start)
//  [A][B][C]____________

// Center
Row::new().main_axis_alignment(MainAxisAlignment::Center)
//  ____[A][B][C]____

// End
Row::new().main_axis_alignment(MainAxisAlignment::End)
//  ____________[A][B][C]

// SpaceBetween
Row::new().main_axis_alignment(MainAxisAlignment::SpaceBetween)
//  [A]______[B]______[C]

// SpaceAround
Row::new().main_axis_alignment(MainAxisAlignment::SpaceAround)
//  __[A]____[B]____[C]__

// SpaceEvenly
Row::new().main_axis_alignment(MainAxisAlignment::SpaceEvenly)
//  ___[A]___[B]___[C]___
```

## CrossAxisAlignment

Controls alignment on cross axis:

```rust
use presentar_widgets::row::CrossAxisAlignment;

// Start
Row::new().cross_axis_alignment(CrossAxisAlignment::Start)
//  [A]
//  [B]  <- aligned to top

// Center
Row::new().cross_axis_alignment(CrossAxisAlignment::Center)
//      [A]
//  [B] <- centered vertically

// End
Row::new().cross_axis_alignment(CrossAxisAlignment::End)
//      [A]
//  [B] <- aligned to bottom

// Stretch
Row::new().cross_axis_alignment(CrossAxisAlignment::Stretch)
//  [A====]
//  [B====] <- fills available space
```

## Gap

Space between children:

```rust
Row::new().gap(16.0)
//  [A]--16px--[B]--16px--[C]
```

## Nesting

Combine Row and Column:

```rust
let layout = Column::new()
    .gap(16.0)
    .child(
        Row::new()
            .main_axis_alignment(MainAxisAlignment::SpaceBetween)
            .child(logo)
            .child(nav)
    )
    .child(content)
    .child(footer);
```

## Verified Test

```rust
#[test]
fn test_flexbox_alignment() {
    use presentar_widgets::Row;
    use presentar_widgets::row::{MainAxisAlignment, CrossAxisAlignment};

    let row = Row::new()
        .main_axis_alignment(MainAxisAlignment::Center)
        .cross_axis_alignment(CrossAxisAlignment::Stretch)
        .gap(10.0);

    assert_eq!(row.children().len(), 0);
}
```
