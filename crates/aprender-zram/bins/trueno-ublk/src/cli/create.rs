//! Create command - creates a new trueno-ublk device

use super::{parse_size, CreateArgs};
use crate::backend::BackendType;
use crate::device::UblkDevice;
use crate::perf::{PerfConfig, PollingConfig};
use anyhow::Result;
use std::path::PathBuf;

pub fn run(args: CreateArgs) -> Result<()> {
    let size = parse_size(&args.size)?;
    let mem_limit = args.mem_limit.as_ref().map(|s| parse_size(s)).transpose()?;
    let writeback_limit = args.writeback_limit.as_ref().map(|s| parse_size(s)).transpose()?;

    let streams = if args.streams == 0 { num_cpus() } else { args.streams };

    // PERF-001: Build performance configuration from CLI args
    let perf = build_perf_config(&args);

    tracing::info!(
        size = %super::format_size(size),
        algorithm = ?args.algorithm,
        streams = streams,
        gpu = args.gpu,
        perf = perf.is_some(),
        "Creating trueno-ublk device"
    );

    // KERN-001/002/003: Parse backend type
    let backend: BackendType = args.backend.parse().map_err(|e: String| anyhow::anyhow!(e))?;
    let zram_device = if matches!(backend, BackendType::KernelZram | BackendType::Tiered) {
        Some(PathBuf::from(&args.zram_device))
    } else {
        None
    };

    // KERN-003: Parse cold tier path for high-entropy pages
    let cold_tier = args.cold_tier.as_ref().map(PathBuf::from);

    let config = crate::device::DeviceConfig {
        dev_id: args.dev_id,
        size,
        algorithm: args.algorithm.to_trueno(),
        streams,
        gpu_enabled: args.gpu,
        mem_limit,
        backing_dev: args.backing_dev,
        writeback_limit,
        entropy_skip_threshold: args.entropy_skip,
        gpu_batch_size: args.gpu_batch,
        foreground: args.foreground,
        // --no-batched overrides --batched (default: enabled)
        batched: args.batched && !args.no_batched,
        batch_threshold: args.batch_threshold,
        flush_timeout_ms: args.flush_timeout_ms,
        perf,
        // PERF-003: Multi-queue parallelism
        nr_hw_queues: args.queues,
        // PERF-006: Zero-copy mode (EXPERIMENTAL)
        zero_copy: args.zero_copy,
        // KERN-001/002/003: Kernel-Cooperative Tiered Storage
        backend,
        entropy_routing: args.entropy_routing,
        zram_device,
        cold_tier,
        entropy_kernel_threshold: args.entropy_kernel_threshold,
        // VIZ-002: Renacer Visualization Integration
        visualize: args.visualize,
        // VIZ-004: OTLP Integration
        otlp_endpoint: args.otlp_endpoint.clone(),
        otlp_service_name: args.otlp_service_name.clone(),
    };

    if config.batched {
        tracing::info!(
            batch_threshold = args.batch_threshold,
            flush_timeout_ms = args.flush_timeout_ms,
            "Batched compression mode enabled (target: >10 GB/s)"
        );
    }

    if let Some(ref perf) = config.perf {
        tracing::info!(
            polling = perf.polling_enabled,
            batch_size = perf.batch_size,
            affinity = ?perf.cpu_cores,
            numa = perf.numa_node,
            "PERF-001: High-performance mode enabled"
        );
    }

    if config.nr_hw_queues > 1 {
        tracing::info!(queues = config.nr_hw_queues, "PERF-003: Multi-queue mode enabled");
    }

    // KERN-001/002/003: Log tiered storage configuration
    if !matches!(config.backend, BackendType::Memory) {
        tracing::info!(
            backend = %config.backend,
            entropy_routing = config.entropy_routing,
            zram_device = ?config.zram_device,
            cold_tier = ?config.cold_tier,
            kernel_threshold = config.entropy_kernel_threshold,
            skip_threshold = config.entropy_skip_threshold,
            "KERN-001: Kernel-cooperative tiered storage enabled"
        );
        if config.cold_tier.is_some() {
            tracing::info!(
                "KERN-003: NVMe cold tier enabled for high-entropy pages (H(X) > {})",
                config.entropy_skip_threshold
            );
        }
    }

    // VIZ-002: Log visualization configuration
    if config.visualize {
        tracing::info!("VIZ-002: Renacer TUI visualization enabled (requires foreground mode)");
        if !config.foreground {
            tracing::warn!("Visualization requires foreground mode. Add -f/--foreground flag.");
        }
    }

    // VIZ-004: Log OTLP configuration
    if let Some(ref endpoint) = config.otlp_endpoint {
        tracing::info!(
            endpoint = %endpoint,
            service_name = %config.otlp_service_name,
            "VIZ-004: OTLP tracing enabled"
        );
    }

    let device = UblkDevice::create(config)?;
    println!("{}", device.path().display());

    Ok(())
}

/// Build PerfConfig from CLI arguments (PERF-001)
fn build_perf_config(args: &CreateArgs) -> Option<PerfConfig> {
    // Check if any perf option is enabled
    let has_perf = args.high_perf
        || args.max_perf
        || args.polling
        || args.cpu_affinity.is_some()
        || args.numa_node >= 0;

    if !has_perf {
        return None;
    }

    // Start with preset or default
    let mut config = if args.max_perf {
        PerfConfig::maximum()
    } else if args.high_perf {
        PerfConfig::high_performance()
    } else {
        PerfConfig::default()
    };

    // Override with explicit CLI args
    if args.polling {
        config.polling_enabled = true;
        config.polling = PollingConfig {
            enabled: true,
            spin_cycles: args.poll_spin_cycles,
            adaptive: args.poll_adaptive,
            ..Default::default()
        };
    }

    // Override batch settings if explicitly set
    if args.batch_pages != 64 {
        config.batch_size = args.batch_pages;
    }
    if args.batch_timeout_us != 100 {
        config.batch_timeout_us = args.batch_timeout_us;
    }

    // CPU affinity
    if let Some(ref affinity) = args.cpu_affinity {
        config.cpu_cores = affinity.split(',').filter_map(|c| c.trim().parse().ok()).collect();
    }

    // NUMA node
    if args.numa_node >= 0 {
        config.numa_node = args.numa_node;
    }

    Some(config)
}

fn num_cpus() -> usize {
    std::thread::available_parallelism().map(|p| p.get()).unwrap_or(1)
}
