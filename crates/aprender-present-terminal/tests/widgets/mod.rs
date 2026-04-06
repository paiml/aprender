//! SPEC-024: Widget Interface Tests
//!
//! **TESTS DEFINE INTERFACE. IMPLEMENTATION FOLLOWS.**
//!
//! Each widget MUST have tests that verify:
//! 1. Widget creates without panic
//! 2. Widget renders within bounds
//! 3. Widget handles edge cases (zero size, minimum size)

mod braille_graph_test;
mod cpu_grid_test;
mod gauge_test;
mod gpu_panel_test;
mod memory_bar_test;
mod meter_test;
mod network_panel_test;
mod process_table_test;
mod title_bar_test;
