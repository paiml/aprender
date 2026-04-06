# Visual Regression

Pixel-perfect snapshot testing with pure Rust.

## Basic Usage

```rust
use presentar_test::Snapshot;

Snapshot::assert_match("button-default", &screenshot, 0.001);
```

## Threshold

The third parameter is the maximum allowed difference ratio:

| Value | Meaning |
|-------|---------|
| `0.0` | Pixel-perfect match |
| `0.001` | 0.1% tolerance (recommended) |
| `0.01` | 1% tolerance |

## Creating Baselines

```bash
SNAPSHOT_UPDATE=1 cargo test
```

Creates new baselines in `tests/snapshots/`.

## File Structure

```
tests/
  snapshots/
    button-default.png        # Baseline
    button-default.actual.png # Actual (on failure)
    button-default.diff.png   # Visual diff (on failure)
```

## Image API

```rust
use presentar_test::snapshot::Image;

// Create image
let mut img = Image::new(100, 100);

// Fill with color
let red = Image::filled(100, 100, 255, 0, 0, 255);

// Pixel access
img.set_pixel(50, 50, [255, 255, 255, 255]);
let pixel = img.get_pixel(50, 50);
```

## Diff Calculation

```rust
let diff = Snapshot::diff(&baseline, &actual);
// Returns 0.0 (identical) to 1.0 (completely different)
```

## Determinism

Guaranteed via:
- Fixed DPI: `1.0`
- Grayscale antialiasing only
- Fixed viewport: `1280x720` default
- Embedded test font

## Verified Test

```rust
#[test]
fn test_snapshot_diff() {
    use presentar_test::snapshot::{Image, Snapshot};

    let a = Image::filled(10, 10, 255, 0, 0, 255);
    let b = Image::filled(10, 10, 255, 0, 0, 255);

    assert_eq!(Snapshot::diff(&a, &b), 0.0);
}
```
