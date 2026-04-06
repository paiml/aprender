# Fraud Detection

ML-powered fraud detection dashboard.

## Architecture

```
Transactions → Aprender Model → Risk Score → Dashboard
```

## YAML Configuration

```yaml
app:
  name: "Fraud Detection"

data:
  transactions:
    source: "transactions.ald"
    refresh: 5s

  model:
    source: "fraud_detector.apr"

widgets:
  root:
    type: Column
    children:
      - type: Row
        children:
          - type: DataCard
            title: "Flagged Today"
            value: "{{ transactions | filter(flagged=true) | count }}"
            color: "red"
          - type: DataCard
            title: "Total Processed"
            value: "{{ transactions | count }}"
          - type: DataCard
            title: "Avg Risk Score"
            value: "{{ transactions | mean('risk_score') | percentage }}"
      - type: DataTable
        data: "{{ transactions | filter(risk_score > 0.7) | limit(50) }}"
        columns:
          - { key: "id", label: "TX ID" }
          - { key: "amount", label: "Amount", format: "currency" }
          - { key: "risk_score", label: "Risk", render: "risk_badge" }
          - { key: "timestamp", label: "Time", format: "datetime" }
```

## Risk Score Display

| Score | Color | Label |
|-------|-------|-------|
| < 0.3 | Green | Low |
| 0.3-0.7 | Yellow | Medium |
| > 0.7 | Red | High |

## Model Integration

```rust
// Run inference on transaction
let features = extract_features(&transaction);
let risk_score = model.predict(&features);
```

## Real-time Updates

```yaml
data:
  live_feed:
    source: "ws://transactions"
    on_message:
      action: prepend
      target: transactions
```

## Verified Test

```rust
#[test]
fn test_fraud_risk_classification() {
    // Risk score classification
    fn classify_risk(score: f32) -> &'static str {
        match score {
            s if s < 0.3 => "low",
            s if s < 0.7 => "medium",
            _ => "high",
        }
    }

    assert_eq!(classify_risk(0.1), "low");
    assert_eq!(classify_risk(0.5), "medium");
    assert_eq!(classify_risk(0.9), "high");

    // Edge cases
    assert_eq!(classify_risk(0.0), "low");
    assert_eq!(classify_risk(0.3), "medium");
    assert_eq!(classify_risk(0.7), "high");
}
```
