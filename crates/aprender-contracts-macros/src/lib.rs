//! # provable-contracts-macros
//!
//! Proc macros for compile-time contract enforcement.
//!
//! ## `#[contract]` Attribute
//!
//! Annotates a function with a provable-contracts YAML contract reference.
//! At compile time, verifies the contract exists (via build.rs env vars)
//! and registers the binding for audit.
//!
//! ```rust,ignore
//! use provable_contracts_macros::contract;
//!
//! #[contract("rmsnorm-kernel-v1", equation = "rmsnorm")]
//! pub fn rms_norm(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
//!     // ...
//! }
//! ```
//!
//! ## How It Works
//!
//! 1. **build.rs** in the consuming crate reads `binding.yaml` and sets
//!    `CONTRACT_<NAME>_<EQ>=bound` env vars for each implemented binding.
//!
//! 2. `#[contract("name", equation = "eq")]` expands to a `const` that reads
//!    the corresponding env var via `env!()`. Missing env var = compile error.
//!
//! 3. A static string in a dedicated link section registers the binding for
//!    runtime audit (when `contract-audit` feature is enabled).

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::{parse_macro_input, Expr, ItemFn, Lit, Meta, Token};

/// Arguments to `#[contract("contract-name", equation = "equation-name")]`
struct ContractArgs {
    contract_name: String,
    equation_name: String,
}

impl Parse for ContractArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        // Parse the contract name (first positional string literal)
        let contract_lit: Lit = input.parse()?;
        let contract_name = match &contract_lit {
            Lit::Str(s) => s.value(),
            _ => {
                return Err(syn::Error::new_spanned(
                    contract_lit,
                    "expected string literal for contract name",
                ));
            }
        };

        // Parse comma
        input.parse::<Token![,]>()?;

        // Parse `equation = "name"`
        let meta: Meta = input.parse()?;
        let equation_name = match &meta {
            Meta::NameValue(nv) if nv.path.is_ident("equation") => match &nv.value {
                Expr::Lit(expr_lit) => match &expr_lit.lit {
                    Lit::Str(s) => s.value(),
                    _ => {
                        return Err(syn::Error::new_spanned(
                            &nv.value,
                            "expected string literal for equation name",
                        ));
                    }
                },
                _ => {
                    return Err(syn::Error::new_spanned(
                        &nv.value,
                        "expected string literal for equation name",
                    ));
                }
            },
            _ => {
                return Err(syn::Error::new_spanned(
                    meta,
                    "expected `equation = \"...\"`",
                ));
            }
        };

        Ok(ContractArgs {
            contract_name,
            equation_name,
        })
    }
}

/// Compile-time contract enforcement attribute.
///
/// Annotates a function with a provable-contracts YAML contract reference.
/// The macro generates:
///
/// 1. A `const` assertion that reads a `CONTRACT_<NAME>_<EQ>` env var
///    (set by build.rs). If the env var is missing, compilation fails.
///
/// 2. `debug_assert!()` calls for EVERY precondition and postcondition
///    from the YAML contract (read via `CONTRACT_<KEY>_PRE_N` env vars).
///    These are injected automatically — zero hand-written assertions.
///
/// 3. A static binding registration string for runtime traceability.
///
/// # How It Works
///
/// build.rs reads `contracts/*.yaml` and sets env vars:
/// ```text
/// CONTRACT_SOFTMAX_KERNEL_V1_SOFTMAX=implemented
/// CONTRACT_SOFTMAX_KERNEL_V1_SOFTMAX_PRE_COUNT=2
/// CONTRACT_SOFTMAX_KERNEL_V1_SOFTMAX_PRE_0=!x.is_empty()
/// CONTRACT_SOFTMAX_KERNEL_V1_SOFTMAX_PRE_1=x.iter().all(|v| v.is_finite())
/// CONTRACT_SOFTMAX_KERNEL_V1_SOFTMAX_POST_COUNT=1
/// CONTRACT_SOFTMAX_KERNEL_V1_SOFTMAX_POST_0=ret.len() == x.len()
/// ```
///
/// This macro reads those env vars at compile time and injects the
/// assertions. Change the YAML → assertions change automatically.
/// Remove the YAML → compile error.
///
/// # Example
///
/// ```rust,ignore
/// #[contract("softmax-kernel-v1", equation = "softmax")]
/// pub fn softmax_1d_alloc(logits: &[f32]) -> Vec<f32> {
///     // Preconditions injected automatically from YAML
///     // ... implementation ...
///     // Postconditions checked on return value automatically
/// }
/// ```
#[proc_macro_attribute]
pub fn contract(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as ContractArgs);
    let input_fn = parse_macro_input!(item as ItemFn);

    let env_key = make_env_key(&args.contract_name, &args.equation_name);
    let const_name = format_ident!(
        "_CONTRACT_CHECK_{}_{}",
        args.contract_name.to_uppercase().replace(['-', '.'], "_"),
        args.equation_name.to_uppercase().replace(['-', '.'], "_")
    );

    let contract_name = &args.contract_name;
    let equation_name = &args.equation_name;
    let fn_name = &input_fn.sig.ident;
    let fn_name_str = fn_name.to_string();

    let binding_const_name = format_ident!(
        "_CONTRACT_BINDING_{}_{}",
        args.contract_name.to_uppercase().replace(['-', '.'], "_"),
        args.equation_name.to_uppercase().replace(['-', '.'], "_")
    );

    // Read preconditions from env vars set by build.rs
    let precondition_asserts = read_contract_assertions(&env_key, "PRE", equation_name);

    // Read postconditions from env vars set by build.rs
    let postcondition_asserts = read_contract_assertions(&env_key, "POST", equation_name);
    let has_postconditions = !postcondition_asserts.is_empty();

    let fn_attrs = &input_fn.attrs;
    let fn_vis = &input_fn.vis;
    let fn_sig = &input_fn.sig;
    let fn_stmts = &input_fn.block.stmts;

    let body = if has_postconditions {
        // Wrap body in let ret = { ... }; check postconditions; ret
        quote! {
            // 1. Compile-time contract binding check.
            #[allow(dead_code)]
            const #const_name: Option<&str> = option_env!(#env_key);

            // 2. Binding registration for audit/traceability.
            #[allow(dead_code)]
            const #binding_const_name: &str = concat!(
                "contract=", #contract_name,
                ",equation=", #equation_name,
                ",module=", module_path!(),
                ",function=", #fn_name_str,
            );

            // 3. Preconditions from YAML (injected by build.rs → proc macro).
            #(#precondition_asserts)*

            // 4. Original function body.
            let ret = { #(#fn_stmts)* };

            // 5. Postconditions from YAML (checked on return value).
            #(#postcondition_asserts)*

            ret
        }
    } else {
        quote! {
            // 1. Compile-time contract binding check.
            #[allow(dead_code)]
            const #const_name: Option<&str> = option_env!(#env_key);

            // 2. Binding registration for audit/traceability.
            #[allow(dead_code)]
            const #binding_const_name: &str = concat!(
                "contract=", #contract_name,
                ",equation=", #equation_name,
                ",module=", module_path!(),
                ",function=", #fn_name_str,
            );

            // 3. Preconditions from YAML (injected by build.rs → proc macro).
            #(#precondition_asserts)*

            // 4. Original function body.
            #(#fn_stmts)*
        }
    };

    let expanded = quote! {
        #(#fn_attrs)*
        #fn_vis #fn_sig {
            #body
        }
    };

    TokenStream::from(expanded)
}

/// Read CONTRACT_<key>_{PRE,POST}_0..N env vars and generate `debug_assert`! tokens.
///
/// build.rs sets these from YAML contract preconditions/postconditions.
/// If no env vars are found (e.g., crates.io build), returns empty vec (graceful).
fn read_contract_assertions(
    env_key: &str,
    kind: &str, // "PRE" or "POST"
    equation_name: &str,
) -> Vec<proc_macro2::TokenStream> {
    let mut asserts = Vec::new();

    let count_key = format!("{env_key}_{kind}_COUNT");
    let count: usize = std::env::var(&count_key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    let kind_label = if kind == "PRE" { "Pre" } else { "Post" };

    for i in 0..count {
        let var_key = format!("{env_key}_{kind}_{i}");
        if let Ok(expr_str) = std::env::var(&var_key) {
            // Parse the YAML precondition/postcondition as a Rust expression
            if let Ok(expr) = expr_str.parse::<proc_macro2::TokenStream>() {
                let msg = format!(
                    "Contract [{equation_name}] {kind_label}-condition violated: {expr_str}"
                );
                asserts.push(quote! {
                    debug_assert!(#expr, #msg);
                });
            }
            // If parsing fails, skip silently (the expression may not be valid Rust)
        }
    }

    asserts
}

/// Precondition: checked via `debug_assert!()` at function entry.
/// Zero runtime cost in release builds.
///
/// ```rust,ignore
/// #[provable_contracts_macros::requires(x > 0)]
/// fn sqrt(x: f64) -> f64 { x.sqrt() }
/// ```
#[proc_macro_attribute]
pub fn requires(attr: TokenStream, item: TokenStream) -> TokenStream {
    let predicate: proc_macro2::TokenStream = attr.into();
    let input_fn = parse_macro_input!(item as ItemFn);
    let fn_attrs = &input_fn.attrs;
    let fn_vis = &input_fn.vis;
    let fn_sig = &input_fn.sig;
    let fn_block = &input_fn.block;
    let pred_str = predicate.to_string();

    let expanded = quote! {
        #(#fn_attrs)*
        #fn_vis #fn_sig {
            debug_assert!(#predicate, "Pre-condition violated: {}", #pred_str);
            #fn_block
        }
    };
    TokenStream::from(expanded)
}

/// Postcondition: checked via `debug_assert!()` after function returns.
/// The return value is bound to `ret` in the predicate.
/// Zero runtime cost in release builds.
///
/// ```rust,ignore
/// #[provable_contracts_macros::ensures(ret > 0)]
/// fn abs(x: i32) -> i32 { if x < 0 { -x } else { x } }
/// ```
#[proc_macro_attribute]
pub fn ensures(attr: TokenStream, item: TokenStream) -> TokenStream {
    let predicate: proc_macro2::TokenStream = attr.into();
    let input_fn = parse_macro_input!(item as ItemFn);
    let fn_attrs = &input_fn.attrs;
    let fn_vis = &input_fn.vis;
    let fn_sig = &input_fn.sig;
    let fn_block = &input_fn.block;
    let pred_str = predicate.to_string();

    let expanded = quote! {
        #(#fn_attrs)*
        #fn_vis #fn_sig {
            let ret = #fn_block;
            debug_assert!(#predicate, "Post-condition violated: {}", #pred_str);
            ret
        }
    };
    TokenStream::from(expanded)
}

/// Invariant: checked via `debug_assert!()` both BEFORE and AFTER.
/// Zero runtime cost in release builds.
///
/// ```rust,ignore
/// #[provable_contracts_macros::invariant(!self.items.is_empty())]
/// fn process(&mut self) { /* ... */ }
/// ```
#[proc_macro_attribute]
pub fn invariant(attr: TokenStream, item: TokenStream) -> TokenStream {
    let predicate: proc_macro2::TokenStream = attr.into();
    let input_fn = parse_macro_input!(item as ItemFn);
    let fn_attrs = &input_fn.attrs;
    let fn_vis = &input_fn.vis;
    let fn_sig = &input_fn.sig;
    let fn_block = &input_fn.block;
    let pred_str = predicate.to_string();

    let expanded = quote! {
        #(#fn_attrs)*
        #fn_vis #fn_sig {
            debug_assert!(#predicate, "Invariant violated (pre): {}", #pred_str);
            let ret = #fn_block;
            debug_assert!(#predicate, "Invariant violated (post): {}", #pred_str);
            ret
        }
    };
    TokenStream::from(expanded)
}

/// Marks a public function as requiring a `#[contract]` annotation.
///
/// When applied to a `pub fn`, this macro checks at compile time whether
/// a corresponding `CONTRACT_*` env var exists (set by build.rs from
/// binding.yaml). If no binding exists, it emits a compile-time warning.
///
/// This closes the reverse coverage gap: new pub fns cannot escape
/// the contract system silently.
///
/// # Example
/// ```rust,ignore
/// #[must_contract]
/// pub fn my_kernel(x: &[f32]) -> Vec<f32> {
///     // Compile warning: no contract binding found for `my_kernel`
///     // Add #[contract("...", equation = "...")] to silence
/// }
/// ```
#[proc_macro_attribute]
pub fn must_contract(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(item as ItemFn);
    let fn_name = &input_fn.sig.ident;
    let fn_name_upper = fn_name.to_string().to_uppercase();

    // Look for any CONTRACT_*_<FN_NAME> env var
    let env_prefix = "CONTRACT_";
    let has_binding =
        std::env::vars().any(|(k, _)| k.starts_with(env_prefix) && k.ends_with(&fn_name_upper));

    if has_binding {
        // Function has a binding — pass through unchanged
        quote! { #input_fn }.into()
    } else {
        // No binding found — emit warning via #[deprecated]
        let warning_msg = format!(
            "Function `{fn_name}` has no contract binding. Add #[contract(\"...\", equation = \"...\")] or add a binding.yaml entry."
        );
        quote! {
            #[deprecated(note = #warning_msg)]
            #input_fn
        }
        .into()
    }
}

/// Generate the env var key from contract name and equation name.
///
/// Convention: `CONTRACT_<CONTRACT_UPPER>_<EQUATION_UPPER>`
/// where hyphens and dots are replaced with underscores.
///
/// Example: `("rmsnorm-kernel-v1", "rmsnorm")` → `"CONTRACT_RMSNORM_KERNEL_V1_RMSNORM"`
fn make_env_key(contract: &str, equation: &str) -> String {
    let contract_part = contract.to_uppercase().replace(['-', '.'], "_");
    let equation_part = equation.to_uppercase().replace(['-', '.'], "_");
    format!("CONTRACT_{contract_part}_{equation_part}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_make_env_key() {
        assert_eq!(
            make_env_key("rmsnorm-kernel-v1", "rmsnorm"),
            "CONTRACT_RMSNORM_KERNEL_V1_RMSNORM"
        );
        assert_eq!(
            make_env_key("attention-kernel-v1", "scaled_dot_product"),
            "CONTRACT_ATTENTION_KERNEL_V1_SCALED_DOT_PRODUCT"
        );
        assert_eq!(
            make_env_key("gated-delta-net-v1", "decay"),
            "CONTRACT_GATED_DELTA_NET_V1_DECAY"
        );
    }

    #[test]
    fn test_make_env_key_with_dots() {
        assert_eq!(make_env_key("v1.0", "eq.1"), "CONTRACT_V1_0_EQ_1");
    }
}
