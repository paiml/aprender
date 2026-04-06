# Precision-Recall Curves

Precision-Recall (PR) curves are particularly useful for imbalanced
classification problems where the positive class is rare.

## Basic PR Curve

```rust
use trueno_viz::prelude::*;
use trueno_viz::plots::PrCurve;

let y_true = vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0];
let y_scores = vec![0.1, 0.3, 0.8, 0.9, 0.4, 0.7, 0.2, 0.85];

let pr = PrCurve::new(&y_true, &y_scores).build();

// Average precision
let ap = pr.average_precision();
println!("Average Precision: {:.4}", ap);
```

**Test Reference**: `src/plots/pr.rs::test_pr_basic`

## Average Precision

```rust
use trueno_viz::plots::PrCurve;

let pr = PrCurve::new(&y_true, &y_scores).build();

// AP = Area under PR curve
let ap = pr.average_precision();
```

## Customization

```rust
use trueno_viz::prelude::*;
use trueno_viz::plots::PrCurve;

let pr = PrCurve::new(&y_true, &y_scores)
    .color(Rgba::new(52, 168, 83, 255))
    .line_width(2.0)
    .show_baseline(true)  // Random classifier baseline
    .show_ap(true)
    .build();
```

## Multiple Models

```rust
use trueno_viz::prelude::*;
use trueno_viz::plots::PrCurve;

let pr = PrCurve::new(&[], &[])
    .add_model("Model A", &y_true, &scores_a, Rgba::BLUE)
    .add_model("Model B", &y_true, &scores_b, Rgba::RED)
    .show_legend(true)
    .build();
```

## F1 Iso-Lines

Show F1 score contours:

```rust
use trueno_viz::plots::PrCurve;

let pr = PrCurve::new(&y_true, &y_scores)
    .show_f1_iso(true)
    .f1_levels(&[0.2, 0.4, 0.6, 0.8])
    .build();
```

## Complete Example

```rust
use trueno_viz::prelude::*;
use trueno_viz::plots::PrCurve;

fn main() -> Result<()> {
    let y_true = vec![
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
    ];
    let y_scores = vec![
        0.1, 0.2, 0.15, 0.25, 0.3, 0.35, 0.4, 0.8, 0.75, 0.9,
    ];

    let pr = PrCurve::new(&y_true, &y_scores)
        .color(Rgba::new(102, 178, 102, 255))
        .show_ap(true)
        .show_baseline(true)
        .title("Precision-Recall Curve")
        .xlabel("Recall")
        .ylabel("Precision")
        .build();

    pr.render_to_file("pr_curve.png")?;

    Ok(())
}
```

## When to Use PR vs ROC

| Scenario | Recommended |
|----------|-------------|
| Balanced classes | ROC |
| Imbalanced classes | PR |
| Focus on positive class | PR |
| Overall discrimination | ROC |

## Next Chapter

Continue to [Loss Curves](./loss.md) for training progress visualization.
