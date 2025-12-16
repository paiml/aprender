# Case Study: APR CLI Tool Demo

This example demonstrates using the `apr` command-line tool to inspect, validate, debug, and compare APR model files.

## Creating a Test Model

First, let's create a model to work with:

```rust
use aprender::linear_model::LinearRegression;
use aprender::traits::Estimator;
use aprender::format::SaveOptions;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create and train a simple model
    let x = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]];
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];

    let mut model = LinearRegression::new();
    model.fit(&x, &y)?;

    // Save with metadata
    let options = SaveOptions::new()
        .with_name("demo-linear-regression")
        .with_description("Demo model for apr CLI tutorial")
        .with_compression(true);

    model.save_with_options("demo_model.apr", options)?;

    println!("Model saved to demo_model.apr");
    Ok(())
}
```

## Inspecting the Model

Use `apr inspect` to view model metadata:

```bash
$ apr inspect demo_model.apr

=== demo_model.apr ===

  Type: LinearRegression
  Version: 1.0
  Size: 512 B
  Flags: COMPRESSED
  Created: 2025-01-15T12:00:00Z
  Framework: aprender 0.18.2
  Name: demo-linear-regression
  Description: Demo model for apr CLI tutorial
```

### JSON Output for Automation

```bash
$ apr inspect demo_model.apr --json
{
  "file": "demo_model.apr",
  "valid": true,
  "model_type": "LinearRegression",
  "version": "1.0",
  "size_bytes": 512,
  "compressed_size": 256,
  "uncompressed_size": 512,
  "flags": {
    "encrypted": false,
    "signed": false,
    "compressed": true,
    "streaming": false,
    "quantized": false
  },
  "metadata": {
    "created_at": "2025-01-15T12:00:00Z",
    "aprender_version": "0.18.2",
    "model_name": "demo-linear-regression",
    "description": "Demo model for apr CLI tutorial"
  }
}
```

## Debugging the Model

### Basic Debug Output

```bash
$ apr debug demo_model.apr
demo_model.apr: APR v1.0 LinearRegression (512 B)
  magic: APRN (valid)
  flags: compressed
  health: OK
```

### Drama Mode

For theatrical debugging (useful for presentations and demos):

```bash
$ apr debug demo_model.apr --drama

====[ DRAMA: demo_model.apr ]====

ACT I: THE HEADER
  Scene 1: Magic bytes... APRN (applause!)
  Scene 2: Version check... 1.0 (standing ovation!)
  Scene 3: Model type... LinearRegression (the protagonist!)

ACT II: THE METADATA
  Scene 1: File size... 512 B
  Scene 2: Flags... COMPRESSED

ACT III: THE VERDICT
  CURTAIN CALL: Model is READY!

====[ END DRAMA ]====
```

### Hex Dump

```bash
$ apr debug demo_model.apr --hex --limit 64
Hex dump of demo_model.apr (first 64 bytes):

00000000: 41 50 52 4e 01 00 01 00  40 00 00 00 00 02 00 00  |APRN....@.......|
00000010: 00 02 00 00 01 00 00 00  00 00 00 00 00 00 00 00  |................|
00000020: 82 a9 63 72 65 61 74 65  64 5f 61 74 b4 32 30 32  |..created_at.202|
00000030: 35 2d 30 31 2d 31 35 54  31 32 3a 30 30 3a 30 30  |5-01-15T12:00:00|
```

## Validating the Model

### Basic Validation

```bash
$ apr validate demo_model.apr
Validating demo_model.apr...

[PASS] Header complete (32 bytes)
[PASS] Magic bytes: APRN
[PASS] Version: 1.0 (supported)
[WARN] No digital signature
[PASS] Metadata readable

Result: VALID (with 1 warnings)
```

### Quality Assessment

```bash
$ apr validate demo_model.apr --quality
Validating demo_model.apr...

[PASS] Header complete (32 bytes)
[PASS] Magic bytes: APRN
[PASS] Version: 1.0 (supported)
[WARN] No digital signature
[PASS] Metadata readable

Result: VALID (with 1 warnings)

=== 100-Point Quality Assessment ===

Structure: 25/25
  - Header valid:        5/5
  - Metadata complete:   5/5
  - Checksum valid:      5/5
  - Magic valid:         5/5
  - Version supported:   5/5

Security: 20/25
  - No pickle code:      5/5
  - No eval/exec:        5/5
  - Signed:              0/5
  - Safe format:         5/5
  - Safe tensors:        5/5

Weights: 25/25
  - No NaN values:       5/5
  - No Inf values:       5/5
  - Reasonable range:    5/5
  - Low sparsity:        5/5
  - Healthy distribution: 5/5

Metadata: 25/25
  - Training info:       5/5
  - Hyperparameters:     5/5
  - Metrics recorded:    5/5
  - Provenance:          5/5
  - Description:         5/5

TOTAL: 95/100 (EXCELLENT)
```

## Comparing Models

Create a second model for comparison:

```rust
// Train with different data
let x2 = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]];
let y2 = vec![3.0, 5.0, 7.0, 9.0, 11.0];

let mut model2 = LinearRegression::new();
model2.fit(&x2, &y2)?;

let options2 = SaveOptions::new()
    .with_name("demo-linear-regression-v2")
    .with_description("Updated model with new data");

model2.save_with_options("demo_model_v2.apr", options2)?;
```

Then compare:

```bash
$ apr diff demo_model.apr demo_model_v2.apr

Comparing demo_model.apr vs demo_model_v2.apr

DIFF: 2 differences found:

  model_name: demo-linear-regression → demo-linear-regression-v2
  description: Demo model for apr CLI tutorial → Updated model with new data
```

## Inspecting Tensors

List tensor names, shapes, and statistics:

```bash
$ apr tensors demo_model.apr

=== Tensors: demo_model.apr ===

  Total tensors: 2
  Total size: 24 B

  weights [f32] [1, 1]
    Size: 4 B
  bias [f32] [1]
    Size: 4 B
```

### Filter Tensors by Name

```bash
$ apr tensors model.apr --filter encoder

=== Tensors: model.apr ===

  encoder.conv1.weight [f32] [384, 80, 3]
    Size: 360.0 KiB
  encoder.conv1.bias [f32] [384]
    Size: 1.5 KiB
```

### Tensor Statistics

```bash
$ apr tensors model.apr --stats

=== Tensors: model.apr ===

  encoder.conv1.weight [f32] [384, 80, 3]
    Size: 360.0 KiB
    Stats: mean=0.0012, std=0.0534
    Range: [-0.1823, 0.1756]
```

### JSON Output for Automation

```bash
$ apr tensors model.apr --json
{
  "file": "model.apr",
  "tensor_count": 4,
  "total_size_bytes": 83569920,
  "tensors": [
    {
      "name": "encoder.conv1.weight",
      "shape": [384, 80, 3],
      "dtype": "f32",
      "size_bytes": 368640
    }
  ]
}
```

## CI/CD Integration

Add to your GitHub Actions workflow:

```yaml
- name: Validate Models
  run: |
    for model in models/*.apr; do
      apr validate "$model" --strict || exit 1
    done

- name: Quality Check
  run: |
    apr validate models/production.apr --quality
    # Fail if score < 90
```

## Key Takeaways

1. **Genchi Genbutsu**: `apr inspect` lets you see actual model data
2. **Genchi Genbutsu**: `apr tensors` reveals actual tensor structure and statistics
3. **Jidoka**: `apr validate --strict` enforces quality gates
4. **Visualization**: `apr debug --drama` makes debugging memorable
5. **Kaizen**: `apr diff` enables tracking model evolution

## See Also

- [apr CLI Tool Reference](../tools/apr-cli.md)
- [APR Model Format](./model-format.md)
- [APR 100-Point Quality Scoring](./apr-scoring.md)
