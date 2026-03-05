//! GPU status and VRAM reservation management (GPU-SHARE-001, GH-152).
//!
//! Displays GPU detection info, VRAM capacity, active reservations,
//! and available budget from the entrenar VRAM ledger.

use crate::error::Result;
use crate::CliError;

pub fn run(json: bool) -> Result<()> {
    let uuid = entrenar::gpu::ledger::detect_gpu_uuid();
    let total_mb = entrenar::gpu::ledger::detect_total_memory_mb();
    let mem_type = entrenar::gpu::ledger::detect_memory_type();

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
            "gpu_uuid": uuid,
            "total_mb": total_mb,
            "memory_type": format!("{mem_type:?}"),
            "reserve_factor": mem_type.reserve_factor(),
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

    Ok(())
}
