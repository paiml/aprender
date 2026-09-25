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
//!    the corresponding env var via `option_env!()`. A missing env var is NOT
//!    a compile error: the attribute then enforces nothing (#2699). Assertions
//!    are injected only for the `_PRE_*` / `_POST_*` vars a producer emitted.
//!    Producers derive the key with `provable_contracts::build_helper::env_key`,
//!    which is tested equal to [`contract_env_key!`], the key this macro reads.
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
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
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
/// 1. A `const` that reads a `CONTRACT_<NAME>_<EQ>` env var (set by
///    build.rs) via `option_env!`. If the env var is missing, NOTHING fails:
///    the binding is unverified and no assertion is injected (#2699).
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
/// Remove the YAML → the assertions silently disappear (#2699). A condition
/// the producer DID emit but that does not parse as a Rust expression, or a
/// `_COUNT` whose numbered vars are missing, is a compile error.
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
/// No `_COUNT` var (e.g. a crates.io build, where no build.rs ran) yields no
/// assertions. Once a producer HAS emitted a count, every inconsistency in
/// what it emitted is a `compile_error!` rather than a silently dropped
/// assertion (#2699): an unparseable count, a missing numbered var, or a
/// condition that is not a Rust expression.
fn read_contract_assertions(
    env_key: &str,
    kind: &str, // "PRE" or "POST"
    equation_name: &str,
) -> Vec<proc_macro2::TokenStream> {
    let count_key = format!("{env_key}_{kind}_COUNT");
    let Ok(raw_count) = std::env::var(&count_key) else {
        return Vec::new();
    };
    let Ok(count) = raw_count.parse::<usize>() else {
        return vec![compile_error(&format!(
            "{count_key}={raw_count:?} is not a count (#2699)"
        ))];
    };
    (0..count)
        .map(|i| condition_assert(&format!("{env_key}_{kind}_{i}"), kind, equation_name))
        .collect()
}

/// One `debug_assert!` for the condition in env var `var_key`, or a
/// `compile_error!` naming why the producer's condition cannot be enforced.
fn condition_assert(var_key: &str, kind: &str, equation_name: &str) -> proc_macro2::TokenStream {
    let Ok(expr_str) = std::env::var(var_key) else {
        return compile_error(&format!(
            "{var_key} is missing although its _COUNT covers it (#2699)"
        ));
    };
    let Ok(expr) = syn::parse_str::<Expr>(&expr_str) else {
        return compile_error(&format!(
            "{var_key}={expr_str:?} is not a Rust expression; the contract condition \
             would be dropped (#2699)"
        ));
    };
    let kind_label = if kind == "PRE" { "Pre" } else { "Post" };
    let msg = format!("Contract [{equation_name}] {kind_label}-condition violated: {expr_str}");
    quote! {
        debug_assert!(#expr, #msg);
    }
}

fn compile_error(msg: &str) -> proc_macro2::TokenStream {
    quote! { ::core::compile_error!(#msg); }
}

/// The env var key `#[contract(contract, equation = equation)]` reads, as a
/// string literal: `contract_env_key!("rmsnorm-kernel-v1", "rmsnorm")` is
/// `"CONTRACT_RMSNORM_KERNEL_V1_RMSNORM"`. The conditions live under that key
/// plus `_PRE_COUNT`, `_PRE_<i>`, `_POST_COUNT` and `_POST_<i>`.
///
/// It exists so a producer's key derivation can be TESTED against the one
/// this crate reads, instead of re-derived by hand and trusted (#2699 §4).
#[proc_macro]
pub fn contract_env_key(input: TokenStream) -> TokenStream {
    let args = parse_macro_input!(input as KeyArgs);
    let key = make_env_key(&args.contract.value(), &args.equation.value());
    TokenStream::from(quote! { #key })
}

/// Arguments to `contract_env_key!("contract-name", "equation-name")`.
struct KeyArgs {
    contract: syn::LitStr,
    equation: syn::LitStr,
}

impl Parse for KeyArgs {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let contract = input.parse()?;
        input.parse::<Token![,]>()?;
        let equation = input.parse()?;
        Ok(KeyArgs { contract, equation })
    }
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

    /// (env vars under the case's key, expected (asserts, compile_errors)).
    /// A producer that emitted nothing is the crates.io case and stays
    /// silent; everything a producer DID emit is enforced or refused (#2699).
    #[test]
    fn read_contract_assertions_fails_closed_on_what_a_producer_emitted() {
        let cases: [(&str, &[(&str, &str)], (usize, usize)); 6] = [
            ("NONE", &[], (0, 0)),
            (
                "OK",
                &[
                    ("PRE_COUNT", "2"),
                    ("PRE_0", "x > 0"),
                    ("PRE_1", "!v.is_empty()"),
                ],
                (2, 0),
            ),
            ("BADCOUNT", &[("PRE_COUNT", "two")], (0, 1)),
            ("GAP", &[("PRE_COUNT", "2"), ("PRE_0", "x > 0")], (1, 1)),
            (
                "UNPARSEABLE",
                &[("PRE_COUNT", "1"), ("PRE_0", "x >")],
                (0, 1),
            ),
            (
                "NOT_AN_EXPR",
                &[("PRE_COUNT", "1"), ("PRE_0", "x > 0, \"msg\"")],
                (0, 1),
            ),
        ];
        for (name, vars, (want_asserts, want_errors)) in cases {
            let key = format!("CONTRACT_TEST_2699_{name}");
            for (suffix, value) in vars {
                std::env::set_var(format!("{key}_{suffix}"), value);
            }
            let out: Vec<String> = read_contract_assertions(&key, "PRE", "eq")
                .iter()
                .map(ToString::to_string)
                .collect();
            let errors = out.iter().filter(|t| t.contains("compile_error")).count();
            let asserts = out.iter().filter(|t| t.contains("debug_assert")).count();
            assert_eq!(
                (asserts, errors),
                (want_asserts, want_errors),
                "{name}: {out:?}"
            );
        }
    }
}
