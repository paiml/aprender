//! Phase 5 falsification tests for optimizer, sequence, and classical ML kernels.
//!
//! Covers `AdamW`, Conv1d, SSM, `KMeans`, `PageRank`, LBFGS, CMA-ES, and Gated Delta Net
//! with 47 tests total. Each test targets a specific mathematical invariant that
//! would break if the implementation contains a common bug class.

mod common;

use proptest::prelude::*;
use provable_contracts::kernels::adamw::*;
use provable_contracts::kernels::cma_es::*;
use provable_contracts::kernels::conv1d::*;
use provable_contracts::kernels::gated_delta_net::*;
use provable_contracts::kernels::kmeans::*;
use provable_contracts::kernels::lbfgs::*;
use provable_contracts::kernels::pagerank::*;
use provable_contracts::kernels::ssm::*;

// Part 1: AdamW + Conv1d (12 tests)
include!("includes/falsify_optseq_adamw_conv.rs");

// Part 2: SSM + KMeans (12 tests)
include!("includes/falsify_optseq_ssm_kmeans.rs");

// Part 3: PageRank + LBFGS (12 tests)
include!("includes/falsify_optseq_pagerank_lbfgs.rs");

// Part 4: CMA-ES + GDN (11 tests)
include!("includes/falsify_optseq_cma_gdn.rs");
