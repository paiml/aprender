//! trueno-gemm-codegen: Compile-time GEMM microkernel code generation.
//!
//! Contract: cgp-gemm-codegen-v1.yaml (C-CODEGEN-001 through C-CODEGEN-004)
//!
//! Generates shape-specialized AVX-512 microkernels at compile time via proc macros.
//! Sovereign implementation — no external BLAS dependencies.
//!
//! # Usage
//! ```ignore
//! use trueno_gemm_codegen::avx512_microkernel;
//!
//! avx512_microkernel!(mr = 8, nr = 32);
//! // Generates: pub unsafe fn microkernel_8x32_avx512_gen(k, a, b, c, ldc)
//! ```

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::{LitInt, Token};

/// Parameters for microkernel generation.
struct MicrokernelParams {
    mr: usize,
    nr: usize,
}

impl Parse for MicrokernelParams {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        // Parse: mr = N, nr = M
        let _mr_ident: syn::Ident = input.parse()?;
        let _eq: Token![=] = input.parse()?;
        let mr_lit: LitInt = input.parse()?;
        let _comma: Token![,] = input.parse()?;
        let _nr_ident: syn::Ident = input.parse()?;
        let _eq2: Token![=] = input.parse()?;
        let nr_lit: LitInt = input.parse()?;

        Ok(MicrokernelParams { mr: mr_lit.base10_parse()?, nr: nr_lit.base10_parse()? })
    }
}

/// Generate an AVX-512 row-major C microkernel.
///
/// Layout: A is MR×K packed column-major, B is K×NR packed row-major,
/// C is MR×NR row-major with stride `ldc`.
///
/// Strategy for row-major C (MR rows, NR columns):
/// - Each C row spans ceil(NR/16) zmm registers
/// - Total accumulators = MR * ceil(NR/16)
/// - Per K step: load ceil(NR/16) B zmm, broadcast MR A scalars, MR*ceil(NR/16) FMAs
///
/// Register budget check (C-CODEGEN-004):
///   accumulators + B loads + headroom <= 32 zmm
#[proc_macro]
pub fn avx512_microkernel(input: TokenStream) -> TokenStream {
    let params = syn::parse_macro_input!(input as MicrokernelParams);
    let mr = params.mr;
    let nr = params.nr;

    let b_regs = nr.div_ceil(16); // ceil(NR/16) zmm registers for B
    let acc_count = mr * b_regs; // Total accumulator registers
    let total_regs = acc_count + b_regs + 4; // +4 headroom for A broadcasts

    if total_regs > 32 {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            format!(
                "C-CODEGEN-004: {mr}x{nr} needs {total_regs} zmm registers (max 32). \
                 Reduce MR or NR. accumulators={acc_count}, B_loads={b_regs}"
            ),
        )
        .to_compile_error()
        .into();
    }

    let fn_name = format_ident!("microkernel_{}x{}_avx512_gen", mr, nr);

    // Generate accumulator identifiers: c{row}_{half}
    let mut acc_idents = Vec::new();
    for row in 0..mr {
        for h in 0..b_regs {
            acc_idents.push(format_ident!("c{}_{}", row, h));
        }
    }

    // Generate C load statements
    let c_loads = generate_c_loads(mr, b_regs, &acc_idents);

    // Generate the inner K-loop body
    let k_body = generate_k_body(mr, nr, b_regs, &acc_idents);

    // Generate C store statements
    let c_stores = generate_c_stores(mr, b_regs, &acc_idents);

    let doc = format!(
        "Generated {mr}x{nr} AVX-512 microkernel ({acc_count} zmm accumulators, \
         {b_regs} B loads, {} FMAs/K-step). Contract: cgp-gemm-codegen-v1.yaml.",
        mr * b_regs
    );

    let output = quote! {
        #[doc = #doc]
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx512f", enable = "fma")]
        pub unsafe fn #fn_name(
            k: usize,
            a: *const f32,
            b: *const f32,
            c: *mut f32,
            ldc: usize,
        ) {
            use std::arch::x86_64::*;

            // Load C accumulators
            #(#c_loads)*

            // Main K loop
            for p in 0..k {
                #(#k_body)*
            }

            // Store C accumulators
            #(#c_stores)*
        }
    };

    output.into()
}

/// Generate C load statements for all accumulators.
fn generate_c_loads(
    mr: usize,
    b_regs: usize,
    acc_idents: &[proc_macro2::Ident],
) -> Vec<TokenStream2> {
    let mut loads = Vec::new();
    for row in 0..mr {
        for h in 0..b_regs {
            let ident = &acc_idents[row * b_regs + h];
            let offset = if row == 0 && h == 0 {
                quote! { c }
            } else if h == 0 {
                let row_val = row;
                quote! { c.add(#row_val * ldc) }
            } else {
                let row_val = row;
                let col_offset = h * 16;
                quote! { c.add(#row_val * ldc + #col_offset) }
            };
            loads.push(quote! {
                let mut #ident = _mm512_loadu_ps(#offset);
            });
        }
    }
    loads
}

/// Generate the inner K-loop body.
fn generate_k_body(
    mr: usize,
    nr: usize,
    b_regs: usize,
    acc_idents: &[proc_macro2::Ident],
) -> Vec<TokenStream2> {
    let mut body = Vec::new();

    // Load B registers
    let nr_val = nr;
    for h in 0..b_regs {
        let b_ident = format_ident!("b{}", h);
        let offset = h * 16;
        body.push(quote! {
            let #b_ident = _mm512_loadu_ps(b.add(p * #nr_val + #offset));
        });
    }

    // Broadcast A and FMA for each row
    let mr_val = mr;
    for row in 0..mr {
        let a_ident = format_ident!("a{}", row);
        body.push(quote! {
            let #a_ident = _mm512_set1_ps(*a.add(p * #mr_val + #row));
        });
        for h in 0..b_regs {
            let c_ident = &acc_idents[row * b_regs + h];
            let b_ident = format_ident!("b{}", h);
            body.push(quote! {
                #c_ident = _mm512_fmadd_ps(#a_ident, #b_ident, #c_ident);
            });
        }
    }

    body
}

/// Generate an AVX-512 broadcast-B microkernel (faer-style).
///
/// Layout: A is MR×K packed column-major (MR contiguous per K step),
/// B is K×NR packed row-major (NR contiguous per K step).
/// C is MR×NR with stride `ldc`, stored in MR/16 zmm chunks per column.
///
/// Strategy (broadcast-B):
/// - Each K step: load MR/16 zmm from A, broadcast NR B scalars
/// - Each accumulator holds 16 elements of one column of C
/// - Total accumulators = (MR/16) × NR
/// - Per K step: MR/16 A loads + NR B broadcasts + (MR/16)*NR FMAs
///
/// Advantage over broadcast-A: NR can be small (6), keeping B panel tiny,
/// allowing KC to stay large (256+). This matches faer's nano-gemm approach.
#[proc_macro]
pub fn avx512_microkernel_broadcast_b(input: TokenStream) -> TokenStream {
    let params = syn::parse_macro_input!(input as MicrokernelParams);
    let mr = params.mr;
    let nr = params.nr;

    if mr % 16 != 0 {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            format!("broadcast-B requires MR divisible by 16, got MR={mr}"),
        )
        .to_compile_error()
        .into();
    }

    let a_regs = mr / 16; // zmm registers for A loads
    let acc_count = a_regs * nr; // Total accumulator registers
    let total_regs = acc_count + a_regs + 4; // +4 headroom for B broadcasts

    if total_regs > 32 {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            format!(
                "broadcast-B {mr}x{nr} needs {total_regs} zmm registers (max 32). \
                 Reduce MR or NR. accumulators={acc_count}, A_loads={a_regs}"
            ),
        )
        .to_compile_error()
        .into();
    }

    let fn_name = format_ident!("microkernel_{}x{}_avx512_bcast_b", mr, nr);

    // Accumulator identifiers: c{a_chunk}_{col}
    let mut acc_idents = Vec::new();
    for col in 0..nr {
        for chunk in 0..a_regs {
            acc_idents.push(format_ident!("c{}_{}", chunk, col));
        }
    }

    // C load: load MR/16 zmm per column, NR columns
    let c_loads = generate_bcast_b_c_loads(mr, nr, a_regs, &acc_idents);

    // K-loop body
    let k_body = generate_bcast_b_k_body(mr, nr, a_regs, &acc_idents);

    // C store
    let c_stores = generate_bcast_b_c_stores(mr, nr, a_regs, &acc_idents);

    let doc = format!(
        "Generated {mr}x{nr} AVX-512 broadcast-B microkernel ({acc_count} zmm accumulators, \
         {a_regs} A loads, {nr} B broadcasts, {} FMAs/K-step). \
         Faer-style: small NR keeps B panel tiny, large KC.",
        a_regs * nr
    );

    let output = quote! {
        #[doc = #doc]
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx512f", enable = "fma")]
        pub unsafe fn #fn_name(
            k: usize,
            a: *const f32,
            b: *const f32,
            c: *mut f32,
            ldc: usize,
        ) {
            use std::arch::x86_64::*;

            // Load C accumulators (column-major within tile)
            #(#c_loads)*

            // Main K loop
            for p in 0..k {
                #(#k_body)*
            }

            // Store C accumulators
            #(#c_stores)*
        }
    };

    output.into()
}

/// Generate C loads for broadcast-B: each column j has MR/16 zmm registers.
/// C layout: row-major with stride ldc. C[i][j] = c[i*ldc + j].
/// For column j, chunk c: load C[c*16..c*16+15][j] — but C is row-major,
/// so we need 16 individual loads and a gather. Instead, store the tile in
/// a transposed layout: accumulate as column-major, then scatter-store at end.
///
/// Actually, for simplicity and performance, we'll load/store C row-major:
/// acc[chunk][col] holds C[chunk*16 + i][col] for i=0..15 — but that requires
/// gather/scatter since C rows are ldc apart.
///
/// Better approach: just use scalar loads into accumulators, accumulate, scalar store.
/// No — the whole point is SIMD accumulation.
///
/// faer approach: C tile is stored in a local buffer (column-major), then written
/// back to C row-major at the end. This avoids gather/scatter in the hot loop.
///
/// We'll follow faer: accumulate in registers (column-major tile), then write
/// back to row-major C via scalar scatter at the end (NR is small, so scatter cost
/// is amortized over many K iterations).
fn generate_bcast_b_c_loads(
    _mr: usize,
    nr: usize,
    a_regs: usize,
    acc_idents: &[proc_macro2::Ident],
) -> Vec<TokenStream2> {
    let mut loads = Vec::new();
    // Initialize accumulators to zero (we'll add existing C values at the end)
    // This is simpler and avoids gather in the hot setup path.
    for col in 0..nr {
        for chunk in 0..a_regs {
            let ident = &acc_idents[col * a_regs + chunk];
            loads.push(quote! {
                let mut #ident = _mm512_setzero_ps();
            });
        }
    }
    loads
}

/// Generate K-loop body for broadcast-B.
fn generate_bcast_b_k_body(
    mr: usize,
    nr: usize,
    a_regs: usize,
    acc_idents: &[proc_macro2::Ident],
) -> Vec<TokenStream2> {
    let mut body = Vec::new();
    let mr_val = mr;
    let nr_val = nr;

    // Load A: MR/16 zmm registers
    for chunk in 0..a_regs {
        let a_ident = format_ident!("a{}", chunk);
        let offset = chunk * 16;
        body.push(quote! {
            let #a_ident = _mm512_loadu_ps(a.add(p * #mr_val + #offset));
        });
    }

    // For each B column: broadcast B[k][j], FMA with all A chunks
    for col in 0..nr {
        let b_ident = format_ident!("b{}", col);
        body.push(quote! {
            let #b_ident = _mm512_set1_ps(*b.add(p * #nr_val + #col));
        });
        for chunk in 0..a_regs {
            let c_ident = &acc_idents[col * a_regs + chunk];
            let a_ident = format_ident!("a{}", chunk);
            body.push(quote! {
                #c_ident = _mm512_fmadd_ps(#a_ident, #b_ident, #c_ident);
            });
        }
    }

    body
}

/// Generate C store statements for broadcast-B.
/// Accumulators are column-major: acc[col][chunk] holds 16 contiguous rows.
/// C is row-major: C[i][j] = c[i*ldc + j].
/// We must scatter: for each row i in chunk, store acc element to C[row][col].
/// Since NR is small (6), we extract each f32 and store individually.
/// This is the scatter cost — amortized over K iterations (typically 128-256).
fn generate_bcast_b_c_stores(
    mr: usize,
    nr: usize,
    a_regs: usize,
    acc_idents: &[proc_macro2::Ident],
) -> Vec<TokenStream2> {
    let mut stores = Vec::new();

    // For each accumulator, extract 16 f32 values and add to C row-major.
    // Use _mm512_storeu_ps to a temp buffer, then scatter to C.
    for col in 0..nr {
        for chunk in 0..a_regs {
            let ident = &acc_idents[col * a_regs + chunk];
            let base_row = chunk * 16;

            // Build scatter indices for this chunk
            let mut scatter_stmts = Vec::new();
            for i in 0..16 {
                let row = base_row + i;
                if row < mr {
                    scatter_stmts.push(quote! {
                        *c.add(#row * ldc + #col) += tmp[#i];
                    });
                }
            }

            stores.push(quote! {
                {
                    let mut tmp = [0.0f32; 16];
                    _mm512_storeu_ps(tmp.as_mut_ptr(), #ident);
                    #(#scatter_stmts)*
                }
            });
        }
    }

    stores
}

/// Generate C store statements.
fn generate_c_stores(
    mr: usize,
    b_regs: usize,
    acc_idents: &[proc_macro2::Ident],
) -> Vec<TokenStream2> {
    let mut stores = Vec::new();
    for row in 0..mr {
        for h in 0..b_regs {
            let ident = &acc_idents[row * b_regs + h];
            let offset = if row == 0 && h == 0 {
                quote! { c }
            } else if h == 0 {
                let row_val = row;
                quote! { c.add(#row_val * ldc) }
            } else {
                let row_val = row;
                let col_offset = h * 16;
                quote! { c.add(#row_val * ldc + #col_offset) }
            };
            stores.push(quote! {
                _mm512_storeu_ps(#offset, #ident);
            });
        }
    }
    stores
}
