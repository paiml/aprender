# Benchmark Commands

The `entrenar-bench` CLI provides tools for distillation benchmarking, hyperparameter sweeps, and cost-performance
analysis.

## Commands Overview

```bash
entrenar-bench <COMMAND>

Commands:
  temperature       Sweep temperature hyperparameter
  alpha             Sweep alpha hyperparameter
  compare           Compare distillation strategies
  ablation          Run ablation study
  cost-performance  Analyze cost vs performance trade-offs
  recommend         Recommend configurations based on constraints
```

## temperature

Run a temperature hyperparameter sweep for knowledge distillation.

```bash
entrenar-bench temperature [OPTIONS]
```

### Options

| Option | Description |
|--------|-------------|
| `--start <VALUE>` | Starting temperature (default: 1.0) |
| `--end <VALUE>` | Ending temperature (default: 8.0) |
| `--step <VALUE>` | Temperature step size (default: 0.5) |
| `--runs <N>` | Runs per temperature point (default: 3) |
| `--format <FORMAT>` | Output format: text, json (default: text) |

### Example

```bash
# Default temperature sweep
entrenar-bench temperature

# Custom range with more granularity
entrenar-bench temperature --start 2.0 --end 6.0 --step 0.25

# More runs for statistical significance
entrenar-bench temperature --runs 5 --format json
```

### Output

```
Temperature Sweep Results
========================

Temp  | Accuracy | Loss   | Std Dev
------|----------|--------|--------
1.0   | 0.823    | 0.412  | ±0.008
1.5   | 0.841    | 0.387  | ±0.006
2.0   | 0.856    | 0.358  | ±0.005
...

Best temperature: 4.0 (accuracy: 0.872)
```

## alpha

Run an alpha (interpolation weight) hyperparameter sweep.

```bash
entrenar-bench alpha [OPTIONS]
```

### Options

| Option | Description |
|--------|-------------|
| `--start <VALUE>` | Starting alpha (default: 0.0) |
| `--end <VALUE>` | Ending alpha (default: 1.0) |
| `--step <VALUE>` | Alpha step size (default: 0.1) |
| `--runs <N>` | Runs per alpha point (default: 3) |
| `--format <FORMAT>` | Output format: text, json (default: text) |

### Example

```bash
# Default alpha sweep
entrenar-bench alpha

# Fine-grained sweep around expected optimum
entrenar-bench alpha --start 0.3 --end 0.7 --step 0.05
```

## compare

Compare multiple distillation strategies head-to-head.

```bash
entrenar-bench compare [OPTIONS]
```

### Options

| Option | Description |
|--------|-------------|
| `--strategies <LIST>` | Comma-separated strategies: kd, progressive, attention, mse, combined |
| `--runs <N>` | Runs per strategy (default: 5) |
| `--format <FORMAT>` | Output format: text, json (default: text) |

### Example

```bash
# Compare all strategies
entrenar-bench compare --strategies kd,progressive,attention,mse,combined

# Compare specific strategies
entrenar-bench compare --strategies kd,progressive --runs 10
```

### Output

```
Strategy Comparison Results
===========================

Strategy     | Accuracy | Loss   | Time (s) | Memory (GB)
-------------|----------|--------|----------|------------
kd           | 0.872    | 0.298  | 145.2    | 12.4
progressive  | 0.881    | 0.287  | 312.8    | 14.2
attention    | 0.878    | 0.291  | 198.4    | 16.8
mse          | 0.845    | 0.342  | 98.6     | 10.2
combined     | 0.889    | 0.276  | 425.1    | 18.4

Statistical significance (p < 0.05):
- progressive > kd
- combined > progressive
```

## ablation

Run an ablation study to understand component contributions.

```bash
entrenar-bench ablation [OPTIONS]
```

### Options

| Option | Description |
|--------|-------------|
| `--base <CONFIG>` | Base configuration file |
| `--components <LIST>` | Components to ablate |
| `--runs <N>` | Runs per configuration (default: 3) |
| `--format <FORMAT>` | Output format: text, json (default: text) |

### Example

```bash
# Ablation study on distillation components
entrenar-bench ablation \
  --base config.yaml \
  --components "temperature,attention_loss,layer_matching"
```

### Output

```
Ablation Study Results
======================

Configuration              | Accuracy | Δ Accuracy
---------------------------|----------|----------
Full model                 | 0.889    | baseline
- temperature scaling      | 0.856    | -0.033
- attention loss           | 0.871    | -0.018
- layer matching           | 0.882    | -0.007
- temp - attn              | 0.843    | -0.046
```

## cost-performance

Analyze cost vs performance trade-offs for different configurations.

```bash
entrenar-bench cost-performance [OPTIONS]
```

### Options

| Option | Description |
|--------|-------------|
| `--gpu <TYPE>` | GPU type: a100-80gb, v100, t4 (default: a100-80gb) |
| `--configs <PATH>` | Path to configurations file |
| `--format <FORMAT>` | Output format: text, json (default: text) |

### GPU Cost Models

| GPU | Cost/Hour | Memory | Performance Factor |
|-----|-----------|--------|-------------------|
| A100-80GB | $3.00 | 80 GB | 1.0x |
| V100 | $2.00 | 16 GB | 0.6x |
| T4 | $0.50 | 16 GB | 0.3x |

### Example

```bash
# Analyze with A100 pricing
entrenar-bench cost-performance --gpu a100-80gb

# Analyze with budget GPU
entrenar-bench cost-performance --gpu t4 --format json
```

### Output

```
Cost-Performance Analysis (A100-80GB @ $3.00/hr)
================================================

Config           | Hours | Cost   | Accuracy | Pareto
-----------------|-------|--------|----------|-------
LoRA r=8         | 2.1   | $6.30  | 0.845    | Yes
LoRA r=16        | 3.2   | $9.60  | 0.862    | Yes
LoRA r=32        | 5.8   | $17.40 | 0.871    | No
LoRA r=64        | 10.4  | $31.20 | 0.878    | Yes
Full fine-tune   | 48.2  | $144.60| 0.882    | Yes

Pareto Frontier: 4 configurations
Cost efficiency winner: LoRA r=8 (0.134 acc/$)
```

## recommend

Get configuration recommendations based on constraints.

```bash
entrenar-bench recommend [OPTIONS]
```

### Options

| Option | Description |
|--------|-------------|
| `--max-cost <USD>` | Maximum budget in USD |
| `--min-accuracy <VALUE>` | Minimum required accuracy (0.0-1.0) |
| `--max-time <HOURS>` | Maximum training time in hours |
| `--max-memory <GB>` | Maximum GPU memory in GB |
| `--gpu <TYPE>` | GPU type for cost calculation |
| `--format <FORMAT>` | Output format: text, json (default: text) |

### Example

```bash
# Budget-constrained recommendation
entrenar-bench recommend --max-cost 50

# Accuracy-constrained recommendation
entrenar-bench recommend --min-accuracy 0.85

# Multiple constraints
entrenar-bench recommend \
  --max-cost 100 \
  --min-accuracy 0.87 \
  --max-memory 16 \
  --gpu v100
```

### Output

```
Recommendations (Budget: $50, Min Accuracy: 0.85)
=================================================

Recommended Configuration:
  Method: LoRA
  Rank: 32
  Learning Rate: 1e-4
  Batch Size: 8

Expected Results:
  Accuracy: 0.871
  Training Time: 5.8 hours
  Cost: $17.40
  Memory: 14.2 GB

Rationale:
  - Best accuracy within budget
  - 2.9x cost savings vs full fine-tuning
  - Pareto optimal configuration

Alternative Options:
  1. LoRA r=16: $9.60, 0.862 acc (budget-friendly)
  2. LoRA r=64: $31.20, 0.878 acc (higher accuracy)
```

## Output Formats

All commands support multiple output formats:

### Text Format (default)

Human-readable tables and summaries for terminal display.

```bash
entrenar-bench temperature --format text
```

### JSON Format

Machine-readable JSON for programmatic processing.

```bash
entrenar-bench temperature --format json | jq '.best_temperature'
```

Example JSON output:

```json
{
  "sweep_type": "temperature",
  "range": {"start": 1.0, "end": 8.0, "step": 0.5},
  "results": [
    {"temperature": 1.0, "accuracy": 0.823, "loss": 0.412, "std_dev": 0.008},
    {"temperature": 1.5, "accuracy": 0.841, "loss": 0.387, "std_dev": 0.006}
  ],
  "best": {"temperature": 4.0, "accuracy": 0.872},
  "statistical_analysis": {
    "mean_accuracy": 0.856,
    "variance": 0.00034
  }
}
```

## Integration with Research Workflow

The benchmark CLI integrates with the research artifact system:

```bash
# 1. Initialize research artifact
entrenar research init \
  --id distillation-benchmark \
  --title "Knowledge Distillation Benchmark Study" \
  --type dataset

# 2. Pre-register experiment
entrenar research preregister artifact.yaml \
  --hypothesis "Temperature T=4 is optimal" \
  --methods "Grid search T in [1,8], step 0.5, 5 runs each"

# 3. Run benchmark
entrenar-bench temperature --start 1 --end 8 --step 0.5 --runs 5 \
  --format json > results.json

# 4. Analyze cost-performance
entrenar-bench cost-performance --gpu a100-80gb

# 5. Get recommendations
entrenar-bench recommend --max-cost 100 --min-accuracy 0.85

# 6. Bundle results
entrenar research bundle artifact.yaml --zip
```

## Programmatic API

The benchmark functionality is also available as a Rust library:

```rust
use entrenar_bench::{
    temperature_sweep, compare_strategies,
    CostModel, CostPerformanceAnalysis, Constraints,
};

// Temperature sweep
let result = temperature_sweep(1.0..8.0, 0.5, 3)?;
println!("Best temperature: {}", result.best_param);

// Cost-performance analysis
let cost_model = CostModel::a100_80gb();
let analysis = CostPerformanceAnalysis::new(cost_model);
let pareto = analysis.compute_pareto_frontier(&points);

// Get recommendations
let constraints = Constraints {
    max_cost: Some(50.0),
    min_accuracy: Some(0.85),
    ..Default::default()
};
let recommendations = analysis.recommend(&points, &constraints);
```

## See Also

- [CLI Overview](./overview.md) - General CLI reference
- [Research Commands](./research.md) - Research artifact CLI
- [Knowledge Distillation](../distillation/what-is-distillation.md) - Distillation concepts
