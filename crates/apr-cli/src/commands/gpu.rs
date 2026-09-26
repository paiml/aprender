//! GPU status and VRAM reservation management (GPU-SHARE-001, GH-152).
//!
//! Displays GPU detection info, VRAM capacity, active reservations,
//! and available budget from the entrenar VRAM ledger.

use crate::error::Result;
use crate::CliError;

/// What `apr gpu` suggests when the ledger finds no GPU (#2661).
///
/// The ledger only sees NVIDIA devices, so "no GPU" here does not mean CPU-only:
/// wgpu reaches a Metal or Vulkan GPU the ledger cannot see. The old hint named
/// `apr run --device cpu`, a flag `apr run` does not have; every command quoted
/// here is parsed by `apr`'s own CLI in the tests.
pub(crate) const NO_LEDGER_GPU_HINT: [&str; 2] = [
    "Hint: `apr run --backend wgpu <model>` uses a Metal/Vulkan GPU if present;",
    "      `apr run --backend cpu <model>` forces CPU inference.",
];

#[provable_contracts_macros::contract(
    "apr-cli-operations-v1",
    equation = "side_effect_classification"
)]
pub fn run(json: bool) -> Result<()> {
    contract_pre_json_output_consistency!();
    let uuid = entrenar::gpu::ledger::detect_gpu_uuid();
    let total_mb = entrenar::gpu::ledger::detect_total_memory_mb();
    let mem_type = entrenar::gpu::ledger::detect_memory_type();

    // Contract: apr-gpu-presence-v1 F-GPU-PRESENCE-001 (paiml/aprender#624).
    // The entrenar API returns sentinel values ("GPU-unknown", 0) on hosts with no
    // discrete GPU. Disambiguate so consumers can tell "no GPU" from "GPU with 0 MB used".
    let gpu_present = uuid != "GPU-unknown" && total_mb > 0;

    let ledger =
        entrenar::gpu::ledger::VramLedger::new(uuid.clone(), total_mb, mem_type.reserve_factor());

    if json {
        let reservations = ledger
            .read_reservations()
            .map_err(|e| CliError::Aprender(format!("ledger read: {e}")))?;
        let reserved: usize = reservations
            .iter()
            .map(|r| r.actual_mb.unwrap_or(r.budget_mb))
            .sum();

        let json_val = serde_json::json!({
            "gpu_present": gpu_present,
            "gpu_uuid": uuid,
            "total_mb": total_mb,
            "memory_type": format!("{mem_type:?}"),
            "reserve_factor": (f64::from(mem_type.reserve_factor()) * 100.0).round() / 100.0,
            "capacity_mb": ledger.capacity_mb(),
            "reserved_mb": reserved,
            "available_mb": ledger.capacity_mb().saturating_sub(reserved),
            "reservations": reservations.iter().map(|r| serde_json::json!({
                "id": r.id,
                "pid": r.pid,
                "budget_mb": r.budget_mb,
                "actual_mb": r.actual_mb,
                "task": r.task,
                "started": r.started.to_rfc3339(),
                "lease_expires": r.lease_expires.to_rfc3339(),
            })).collect::<Vec<_>>(),
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&json_val).unwrap_or_default()
        );
    } else if !gpu_present {
        // Contract: apr-gpu-presence-v1 F-GPU-PRESENCE-001 (paiml/aprender#624).
        println!("No discrete GPU detected on this host.");
        println!("  (entrenar ledger returned uuid={uuid}, total_mb={total_mb})");
        for line in NO_LEDGER_GPU_HINT {
            println!("  {line}");
        }
    } else {
        println!("GPU: {uuid}");
        println!("Total: {total_mb} MB");
        println!(
            "Type: {mem_type:?} (reserve factor: {:.0}%)",
            mem_type.reserve_factor() * 100.0
        );
        println!();

        match entrenar::gpu::ledger::gpu_status_display(&ledger) {
            Ok(status) => print!("{status}"),
            Err(e) => eprintln!("Ledger error: {e}"),
        }
    }

    contract_post_json_output_consistency!(&());
    Ok(())
}

#[cfg(test)]
mod no_ledger_gpu_hint_tests {
    use super::NO_LEDGER_GPU_HINT;

    /// Parse argv on a 16 MB stack: clap's recursive destructuring of the full
    /// `Commands` enum overflows the default 2 MiB test-thread stack in debug builds.
    fn parse_error(argv: &[&str]) -> Option<String> {
        let argv: Vec<String> = argv.iter().map(|s| (*s).to_string()).collect();
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(move || {
                use clap::Parser;
                crate::Cli::try_parse_from(argv)
                    .err()
                    .map(|e| e.to_string())
            })
            .expect("spawn parse thread")
            .join()
            .expect("join parse thread")
    }

    /// Every `apr ...` command quoted in the hint must parse (#2661: the old
    /// hint quoted `--device cpu`, which `apr run` rejects).
    #[test]
    fn every_command_in_the_hint_parses() {
        let mut checked = 0;
        for line in NO_LEDGER_GPU_HINT {
            let quoted = line.split('`').nth(1).expect("hint quotes a command");
            let argv: Vec<&str> = quoted
                .split_whitespace()
                .map(|w| if w == "<model>" { "m.gguf" } else { w })
                .collect();
            assert_eq!(argv[0], "apr", "{line}");
            if let Some(e) = parse_error(&argv) {
                panic!("`{quoted}` does not parse: {e}");
            }
            checked += 1;
        }
        assert_eq!(checked, 2);
    }

    /// The pre-#2661 hint must fail the same check (proves the test can go RED).
    #[test]
    fn the_old_hint_does_not_parse() {
        assert!(parse_error(&["apr", "run", "--device", "cpu", "m.gguf"]).is_some());
    }

    #[test]
    fn the_hint_offers_the_gpu_backend_the_ledger_cannot_see() {
        assert!(NO_LEDGER_GPU_HINT
            .iter()
            .any(|l| l.contains("--backend wgpu")));
        assert!(!NO_LEDGER_GPU_HINT.iter().any(|l| l.contains("--device")));
    }
}
