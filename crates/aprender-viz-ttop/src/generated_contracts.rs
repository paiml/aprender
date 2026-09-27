// Auto-generated contract assertions from YAML — DO NOT EDIT.
// Zero cost in release builds (debug_assert!).
// Regenerate: pv codegen contracts/ -o src/generated_contracts.rs
// Include:   #[macro_use] #[allow(unused_macros)] mod generated_contracts;

// Auto-generated from contracts/panel-render-v1.yaml — DO NOT EDIT
// Contract: ttop-panel-v2

/// Preconditions for buffer bounds.
macro_rules! contract_pre_buffer_bounds {
    () => {{
        let _ = std::mem::size_of_val(&());
    }};
    ($width:expr, $height:expr) => {{
        debug_assert!(
            $width > 0,
            "Contract buffer_bounds: precondition violated — width > 0"
        );
        debug_assert!(
            $height > 0,
            "Contract buffer_bounds: precondition violated — height > 0"
        );
    }};
}

/// Postconditions for buffer bounds.
macro_rules! contract_post_buffer_bounds {
    () => {{
        let _ = std::mem::size_of_val(&());
    }};
    ($cells_len:expr, $width:expr, $height:expr) => {{
        debug_assert_eq!(
            $cells_len,
            $width as usize * $height as usize,
            "Contract buffer_bounds: postcondition violated — cells.len() == width * height"
        );
    }};
}

/// Preconditions for cpu_percent.
macro_rules! contract_pre_cpu_percent {
    () => {{
        let _ = std::mem::size_of_val(&());
    }};
    ($per_core:expr) => {{
        debug_assert!(
            $per_core.iter().all(|p| *p >= 0.0),
            "Contract cpu_percent: precondition violated — all values >= 0.0"
        );
    }};
}

/// Postconditions for cpu_percent.
macro_rules! contract_post_cpu_percent {
    () => {{
        let _ = std::mem::size_of_val(&());
    }};
    ($per_core:expr) => {{
        debug_assert!(
            $per_core.iter().all(|p| *p <= 100.0),
            "Contract cpu_percent: postcondition violated — all values <= 100.0"
        );
    }};
}

/// Preconditions for memory_consistency.
macro_rules! contract_pre_memory_consistency {
    () => {{
        let _ = std::mem::size_of_val(&());
    }};
    ($mem_total:expr) => {{
        debug_assert!(
            $mem_total > 0,
            "Contract memory_consistency: precondition violated — mem_total > 0"
        );
    }};
}

/// Postconditions for memory_consistency.
macro_rules! contract_post_memory_consistency {
    () => {{
        let _ = std::mem::size_of_val(&());
    }};
    ($mem_used:expr, $mem_total:expr) => {{
        debug_assert!(
            $mem_used <= $mem_total,
            "Contract memory_consistency: postcondition violated — mem_used <= mem_total"
        );
    }};
}

/// Postconditions for display_format.
macro_rules! contract_post_display_format {
    () => {{
        let _ = std::mem::size_of_val(&());
    }};
    ($result:expr) => {{
        debug_assert!(
            !$result.is_empty(),
            "Contract display_format: postcondition violated — result is non-empty"
        );
    }};
}

// Total: 5 preconditions, 5 postconditions from 1 contract
