//! Presentar: WASM-first visualization and rapid application framework.
//!
//! Built on the Sovereign AI Stack (Trueno, Aprender, Realizar, Pacha).
//!
//! # Browser Usage (WASM)
//!
//! ```javascript
//! import init, { App, log } from './presentar.js';
//!
//! async function main() {
//!     await init();
//!     const app = new App('canvas');
//!     app.render_json('[{"Rect": {...}}]');
//! }
//! ```

#![allow(
    dead_code,
    unused_imports,
    clippy::doc_markdown,
    clippy::missing_const_for_fn,
    clippy::use_self,
    clippy::pub_underscore_fields,
    clippy::match_same_arms,
    clippy::unwrap_used,
    clippy::disallowed_methods,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::needless_pass_by_value,
    clippy::too_many_lines,
    clippy::module_name_repetitions,
    clippy::unnecessary_unwrap,
    clippy::struct_excessive_bools,
    clippy::type_complexity,
    clippy::too_many_arguments,
    clippy::similar_names,
    clippy::map_unwrap_or,
    clippy::redundant_else,
    clippy::collapsible_if,
    clippy::manual_let_else,
    clippy::if_not_else,
    clippy::uninlined_format_args,
    clippy::suboptimal_flops,
    clippy::unnecessary_wraps,
    clippy::float_cmp,
    clippy::clone_on_copy,
    clippy::single_match,
    clippy::trivially_copy_pass_by_ref,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::let_and_return,
    clippy::items_after_statements,
    clippy::ptr_arg,
    clippy::cast_lossless,
    clippy::struct_field_names,
    clippy::unused_self,
    clippy::fn_params_excessive_bools,
    clippy::many_single_char_names,
    clippy::match_like_matches_macro,
    clippy::assigning_clones,
    clippy::wrong_self_convention,
    clippy::derive_partial_eq_without_eq,
    clippy::needless_raw_string_hashes,
    unreachable_pub
)]

#[macro_use]
#[allow(unused_macros)]
mod generated_contracts;

pub use presentar_core::*;
pub use presentar_layout as layout;
pub use presentar_widgets as widgets;
pub use presentar_yaml as yaml;

pub mod browser;

#[cfg(target_arch = "wasm32")]
pub use browser::{App, Canvas2DRenderer};

pub use browser::{BrowserRouter, RouteMatch, RouteMatcher};

// WebGPU types available when gpu feature is enabled
#[cfg(feature = "gpu")]
mod webgpu;
#[cfg(feature = "gpu")]
pub use webgpu::{
    commands_to_instances, Instance as GpuInstance, Uniforms as GpuUniforms, Vertex as GpuVertex,
};
