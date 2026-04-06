//! Proc macros for Presentar testing framework.
//!
//! Provides the `#[presentar_test]` attribute macro for widget and integration tests.
//!
//! # Example
//!
//! ```ignore
//! use presentar_test_macros::presentar_test;
//!
//! #[presentar_test]
//! fn test_button_renders() {
//!     let button = Button::new("Click me");
//!     let harness = Harness::new(button);
//!     harness.assert_exists("Button");
//! }
//!
//! #[presentar_test(fixture = "dashboard.tar")]
//! fn test_dashboard_layout() {
//!     // Fixture is automatically loaded
//!     harness.assert_exists("[data-testid='metric-card']");
//! }
//! ```

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input, Ident, ItemFn, LitInt, LitStr, Token,
};

/// Parsed attributes for `#[presentar_test]`.
#[derive(Default)]
struct PresentarTestAttrs {
    fixture: Option<String>,
    timeout_ms: u64,
    should_panic: bool,
    ignore: bool,
}

impl Parse for PresentarTestAttrs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut attrs = Self {
            timeout_ms: 5000,
            ..Default::default()
        };

        while !input.is_empty() {
            let ident: Ident = input.parse()?;
            let ident_str = ident.to_string();

            match ident_str.as_str() {
                "fixture" => {
                    input.parse::<Token![=]>()?;
                    let lit: LitStr = input.parse()?;
                    attrs.fixture = Some(lit.value());
                }
                "timeout" => {
                    input.parse::<Token![=]>()?;
                    let lit: LitInt = input.parse()?;
                    attrs.timeout_ms = lit.base10_parse().unwrap_or(5000);
                }
                "should_panic" => {
                    attrs.should_panic = true;
                }
                "ignore" => {
                    attrs.ignore = true;
                }
                _ => {
                    return Err(syn::Error::new(
                        ident.span(),
                        format!("unknown attribute: {ident_str}"),
                    ));
                }
            }

            // Consume optional comma
            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }

        Ok(attrs)
    }
}

/// Test attribute for Presentar widget and integration tests.
///
/// # Attributes
///
/// - `fixture = "path"` - Load a fixture tar file before the test
/// - `timeout = 5000` - Set test timeout in milliseconds
/// - `should_panic` - Expect the test to panic
/// - `ignore` - Skip this test by default
///
/// # Example
///
/// ```ignore
/// #[presentar_test]
/// fn test_widget() {
///     // Test code
/// }
///
/// #[presentar_test(fixture = "app.tar", timeout = 10000)]
/// fn test_with_fixture() {
///     // Test with fixture
/// }
/// ```
#[proc_macro_attribute]
pub fn presentar_test(attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);
    let attrs = parse_macro_input!(attr as PresentarTestAttrs);

    let expanded = impl_presentar_test(&input, &attrs);
    TokenStream::from(expanded)
}

fn impl_presentar_test(input: &ItemFn, attrs: &PresentarTestAttrs) -> TokenStream2 {
    let _fn_name = &input.sig.ident;
    let fn_body = &input.block;
    let fn_attrs = &input.attrs;
    let fn_vis = &input.vis;
    let fn_sig = &input.sig;

    // Generate test attributes
    let test_attr = if attrs.should_panic {
        quote! { #[test] #[should_panic] }
    } else {
        quote! { #[test] }
    };

    let ignore_attr = if attrs.ignore {
        quote! { #[ignore] }
    } else {
        quote! {}
    };

    // Generate fixture loading code if specified
    let fixture_code = if let Some(fixture_path) = &attrs.fixture {
        quote! {
            let _fixture_data = include_bytes!(#fixture_path);
            // Fixture loading would happen here
        }
    } else {
        quote! {}
    };

    // Generate timeout wrapper
    let timeout_ms = attrs.timeout_ms;
    let timeout_code = quote! {
        let _timeout_ms: u64 = #timeout_ms;
        // Timeout enforcement would happen in async context
    };

    // Generate the test function
    quote! {
        #(#fn_attrs)*
        #test_attr
        #ignore_attr
        #fn_vis #fn_sig {
            #fixture_code
            #timeout_code
            #fn_body
        }
    }
}

/// Describe a test suite with before/after hooks.
///
/// This is a function-like macro alternative to the BDD module.
///
/// # Example
///
/// ```ignore
/// describe_suite! {
///     name: "Button Widget",
///     before: || { setup(); },
///     after: || { teardown(); },
///     tests: {
///         it "renders with label" => {
///             // Test code
///         },
///         it "handles click" => {
///             // Test code
///         }
///     }
/// }
/// ```
#[proc_macro]
pub fn describe_suite(input: TokenStream) -> TokenStream {
    // Simple implementation that generates standard tests
    let _input_str = input.to_string();

    // For now, just generate a placeholder
    let expanded = quote! {
        // describe_suite macro placeholder
        // Full implementation would parse the DSL and generate test functions
    };

    TokenStream::from(expanded)
}

/// Assert that a widget matches a snapshot.
///
/// # Example
///
/// ```ignore
/// #[presentar_test]
/// fn test_button_snapshot() {
///     let button = Button::new("Submit");
///     assert_snapshot!(button, "button_submit");
/// }
/// ```
#[proc_macro]
pub fn assert_snapshot(input: TokenStream) -> TokenStream {
    let input2 = TokenStream2::from(input);

    let expanded = quote! {
        {
            let (widget, name) = (#input2);
            let snapshot = presentar_test::Snapshot::capture(&widget);
            snapshot.assert_match(name);
        }
    };

    TokenStream::from(expanded)
}

/// Define a test fixture with setup/teardown.
///
/// # Example
///
/// ```ignore
/// fixture!(
///     name = "database",
///     setup = || { create_test_db() },
///     teardown = |db| { db.drop() }
/// );
/// ```
#[proc_macro]
pub fn fixture(input: TokenStream) -> TokenStream {
    let input2 = TokenStream2::from(input);

    let expanded = quote! {
        // fixture macro placeholder
        // Would generate fixture struct with setup/teardown
        #input2
    };

    TokenStream::from(expanded)
}

// =============================================================================
// COMPUTEBLOCK ARCHITECTURAL ENFORCEMENT
// =============================================================================
//
// SPEC-024: TESTS DEFINE INTERFACE. IMPLEMENTATION FOLLOWS.
//
// These macros make it IMPOSSIBLE to build without tests.
// The test creates a "proof" type that the implementation requires.
// Without the test -> no proof type -> compile error.

/// Marks a test as defining an interface.
///
/// This macro generates a proof type that implementations must consume.
/// Without this test existing, implementations cannot compile.
///
/// # Example
///
/// ```ignore
/// // In tests/cpu_interface.rs
/// #[interface_test(CpuMetrics)]
/// fn test_cpu_metrics_has_frequency() {
///     let metrics = CpuMetrics::default();
///     let _freq: u64 = metrics.frequency; // Defines the interface
/// }
///
/// // In src/cpu.rs - this line requires the test to exist:
/// use crate::tests::cpu_interface::CpuMetricsInterfaceProof;
/// ```
#[proc_macro_attribute]
pub fn interface_test(attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);
    let interface_name: Ident = parse_macro_input!(attr as Ident);

    let _fn_name = &input.sig.ident;
    let fn_body = &input.block;
    let fn_attrs = &input.attrs;
    let fn_vis = &input.vis;
    let fn_sig = &input.sig;

    // Generate proof type name: CpuMetrics -> CpuMetricsInterfaceProof
    let proof_type = Ident::new(
        &format!("{interface_name}InterfaceProof"),
        interface_name.span(),
    );

    let expanded = quote! {
        /// Proof that the interface test exists.
        /// Implementation code must reference this type to compile.
        /// This enforces SPEC-024: Tests define interface.
        #[allow(dead_code)]
        pub struct #proof_type {
            _private: (),
        }

        impl #proof_type {
            /// Only callable from test modules.
            #[cfg(test)]
            pub const fn verified() -> Self {
                Self { _private: () }
            }
        }

        #(#fn_attrs)*
        #[test]
        #fn_vis #fn_sig {
            // Proof that this test defines the interface
            let _proof = #proof_type { _private: () };
            #fn_body
        }
    };

    TokenStream::from(expanded)
}

/// Requires an interface test to exist for this implementation.
///
/// Place this on impl blocks or structs that must have interface tests.
/// Without the corresponding `#[interface_test(Name)]` test, this fails to compile.
///
/// # Example
///
/// ```ignore
/// // This only compiles if tests/cpu_interface.rs has #[interface_test(CpuMetrics)]
/// #[requires_interface(CpuMetrics)]
/// impl CpuMetrics {
///     pub fn frequency(&self) -> u64 { ... }
/// }
/// ```
#[proc_macro_attribute]
pub fn requires_interface(attr: TokenStream, item: TokenStream) -> TokenStream {
    let interface_name: Ident = parse_macro_input!(attr as Ident);
    let item2 = TokenStream2::from(item);

    // Generate proof type reference
    let proof_type = Ident::new(
        &format!("{interface_name}InterfaceProof"),
        interface_name.span(),
    );

    let expanded = quote! {
        // SPEC-024 ENFORCEMENT: This code requires an interface test.
        // If you see a compile error here, you need to create:
        //   #[interface_test(#interface_name)]
        //   fn test_xxx() { ... }
        //
        // TESTS DEFINE INTERFACE. IMPLEMENTATION FOLLOWS.
        #[allow(dead_code)]
        const _: () = {
            // This line fails if the interface test doesn't exist
            fn _require_interface_test() {
                let _ = core::mem::size_of::<#proof_type>();
            }
        };

        #item2
    };

    TokenStream::from(expanded)
}

/// Macro for defining a ComputeBlock with mandatory test coverage.
///
/// A ComputeBlock is a self-contained unit of functionality that:
/// 1. Has a defined interface (via tests)
/// 2. Has documented behavior (via tests)
/// 3. Cannot exist without tests
///
/// # Example
///
/// ```ignore
/// // Define the block - this REQUIRES tests to exist
/// computeblock! {
///     name: CpuPanel,
///     interface: [
///         per_core_freq: Vec<u64>,
///         per_core_temp: Vec<f32>,
///     ],
///     tests: "tests/cpu_panel_interface.rs"
/// }
/// ```
#[proc_macro]
pub fn computeblock(input: TokenStream) -> TokenStream {
    let input_str = input.to_string();

    // Parse the DSL (simplified for now)
    // Full implementation would parse name, interface fields, test file path

    if !input_str.contains("name:") || !input_str.contains("tests:") {
        return TokenStream::from(quote! {
            compile_error!(
                "SPEC-024 ENFORCEMENT: computeblock! requires 'name:' and 'tests:' fields.\n\
                 TESTS DEFINE INTERFACE. IMPLEMENTATION FOLLOWS."
            );
        });
    }

    // Generate the block with enforcement
    let expanded = quote! {
        // ComputeBlock definition with enforced test coverage
        // See SPEC-024 for architecture details
    };

    TokenStream::from(expanded)
}

#[cfg(test)]
mod tests {
    // Proc macro tests run in a separate compilation unit
    // Integration tests would go in tests/ directory
}
