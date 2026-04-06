# Batuta Integration

Batuta provides GPU pricing and queue management services. The
ecosystem module integrates with Batuta for cost estimation and
ETA adjustments.

## BatutaClient

The client for interacting with Batuta pricing and queue services:

```rust
use entrenar::ecosystem::BatutaClient;

// Create client with fallback pricing
let client = BatutaClient::new();

// Or connect to a Batuta instance
let client = BatutaClient::with_url("http://batuta.local:8080")
    .with_timeout(Duration::from_secs(10));
```

## GPU Pricing

Get hourly rates for GPU types:

```rust
let pricing = client.get_hourly_rate("a100-80gb")?;

println!("GPU: {}", pricing.gpu_type);
println!("Rate: ${}/hr", pricing.hourly_rate);
println!("Memory: {} GB", pricing.memory_gb);
println!("Spot: {}", pricing.is_spot);
println!("Provider: {}", pricing.provider);
println!("Region: {}", pricing.region);
```

### Available GPUs

| GPU Type | Hourly Rate | Memory |
|----------|-------------|--------|
| `a100-80gb` | $3.00 | 80 GB |
| `a100-40gb` | $2.50 | 40 GB |
| `h100-80gb` | $4.50 | 80 GB |
| `v100` | $2.00 | 16 GB |
| `t4` | $0.50 | 16 GB |
| `l4` | $0.75 | 24 GB |
| `a10g` | $1.00 | 24 GB |

### Cost Estimation

```rust
// Estimate training cost
let hours = 10.0;
let cost = client.estimate_cost("a100-80gb", hours)?;
println!("Estimated cost: ${:.2}", cost);

// Find cheapest GPU meeting requirements
if let Some(gpu) = client.cheapest_gpu(24) { // 24GB minimum
    println!("Recommended: {} @ ${}/hr", gpu.gpu_type, gpu.hourly_rate);
}
```

## Queue Management

Monitor queue state for GPU availability:

```rust
let queue = client.get_queue_depth("a100-80gb")?;

println!("Queue depth: {}", queue.queue_depth);
println!("Available GPUs: {}/{}", queue.available_gpus, queue.total_gpus);
println!("Avg wait: {}s", queue.avg_wait_seconds);
println!("Utilization: {:.1}%", queue.utilization() * 100.0);

if queue.is_available() {
    println!("GPUs available now!");
}
```

## ETA Adjustment

Adjust estimated completion time based on queue state:

```rust
use entrenar::ecosystem::adjust_eta;

let base_eta_seconds = 3600; // 1 hour training time
let queue = client.get_queue_depth("a100-80gb")?;

let adjusted = adjust_eta(base_eta_seconds, &queue);
println!("Adjusted ETA: {:?}", adjusted);
```

### Adjustment Factors

The ETA is adjusted based on:

1. **Queue wait time** - If GPUs not immediately available
2. **Average wait time** - Historical wait times per queued job
3. **Utilization** - High utilization (>80%) increases estimates
4. **Queue ETA** - Uses queue-provided ETA if higher

```rust
// Example adjustments:
// - No queue: ETA unchanged
// - 3 jobs queued, 5min avg wait: +15 minutes
// - 90% utilization: +20% to ETA
```

## Fallback Pricing

When Batuta is unavailable, the client uses fallback pricing:

```rust
use entrenar::ecosystem::FallbackPricing;

let mut fallback = FallbackPricing::new();

// Get fallback rate
if let Some(pricing) = fallback.get_rate("v100") {
    println!("Fallback rate: ${}/hr", pricing.hourly_rate);
}

// Add custom GPU pricing
fallback.set_rate(GpuPricing::new("rtx-4090", 0.80, 24));
```

### Custom Fallback

```rust
let client = BatutaClient::new()
    .with_fallback(FallbackPricing::new());
```

## Combined Status

Get pricing and queue state together:

```rust
let (pricing, queue) = client.get_status("a100-80gb")?;

println!("GPU: {} @ ${}/hr", pricing.gpu_type, pricing.hourly_rate);
println!("Available: {}", queue.is_available());
```

## Error Handling

```rust
use entrenar::ecosystem::BatutaError;

match client.get_hourly_rate("unknown-gpu") {
    Ok(pricing) => println!("Rate: ${}", pricing.hourly_rate),
    Err(BatutaError::UnknownGpuType(gpu)) => {
        println!("Unknown GPU type: {}", gpu);
    }
    Err(BatutaError::ServiceUnavailable(msg)) => {
        println!("Batuta unavailable: {}", msg);
    }
    Err(e) => println!("Error: {}", e),
}
```

## See Also

- [Ecosystem Overview](./overview.md)
- [Benchmark Commands](../cli/benchmark.md) - Cost-performance analysis
