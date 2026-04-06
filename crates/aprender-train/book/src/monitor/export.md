# Export Formats

Export training metrics to various formats for external tools.

## Prometheus Format

```rust
use entrenar::monitor::{MetricsExporter, ExportFormat, MetricsCollector};

let collector = MetricsCollector::new();
// ... record metrics ...

let exporter = MetricsExporter::new();
let prometheus = exporter.export(&collector, ExportFormat::Prometheus);
println!("{}", prometheus);
```

Output:
```
# HELP training_loss Training loss
# TYPE training_loss gauge
training_loss{run="default"} 0.1234

# HELP training_accuracy Model accuracy
# TYPE training_accuracy gauge
training_accuracy{run="default"} 0.9567
```

## JSON Format

```rust
let json = exporter.export(&collector, ExportFormat::Json);
```

Output:
```json
{
  "timestamp": "2024-11-28T12:00:00Z",
  "metrics": {
    "loss": {"mean": 0.1234, "std": 0.05, "min": 0.08, "max": 0.25},
    "accuracy": {"mean": 0.95, "std": 0.02, "min": 0.90, "max": 0.98}
  }
}
```

## CSV Format

```rust
let csv = exporter.export(&collector, ExportFormat::Csv);
```

Output:
```csv
timestamp,metric,mean,std,min,max
2024-11-28T12:00:00Z,loss,0.1234,0.05,0.08,0.25
2024-11-28T12:00:00Z,accuracy,0.95,0.02,0.90,0.98
```

## Integration with Grafana

1. Export Prometheus metrics to file or HTTP endpoint
2. Configure Prometheus to scrape the endpoint
3. Add Grafana dashboard for visualization

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'entrenar'
    static_configs:
      - targets: ['localhost:9090']
```
