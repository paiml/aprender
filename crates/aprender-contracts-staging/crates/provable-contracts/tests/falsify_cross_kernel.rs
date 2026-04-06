//! Cross-kernel falsification tests: isolation + mutation detection.
//!
//! 42 tests total:
//! - 21 isolation tests: prove kernels compute distinct functions by feeding one kernel's
//!   input through a different kernel and verifying invariant violation.
//! - 21 mutation detection tests: implement broken (mutated) kernel variants inline and
//!   verify the correct kernel produces different output, proving the test suite would
//!   catch implementation bugs.

mod common;

use provable_contracts::kernels::activation::{gelu_scalar, relu_scalar, silu_scalar};
use provable_contracts::kernels::adamw::adamw_step_scalar;
use provable_contracts::kernels::attention::attention_scalar;
use provable_contracts::kernels::batchnorm::batchnorm_scalar;
use provable_contracts::kernels::cma_es::cma_sample_scalar;
use provable_contracts::kernels::conv1d::conv1d_scalar;
use provable_contracts::kernels::cross_entropy::{cross_entropy_scalar, log_softmax_scalar};
use provable_contracts::kernels::flash_attention::flash_attention_scalar;
use provable_contracts::kernels::gated_delta_net::gdn_recurrence_scalar;
use provable_contracts::kernels::gqa::gqa_scalar;
use provable_contracts::kernels::kmeans::kmeans_assign_scalar;
use provable_contracts::kernels::layernorm::layernorm_scalar;
use provable_contracts::kernels::lbfgs::lbfgs_direction_scalar;
use provable_contracts::kernels::matmul::matmul_scalar;
use provable_contracts::kernels::pagerank::pagerank_iterate_scalar;
use provable_contracts::kernels::rmsnorm::rmsnorm_scalar;
use provable_contracts::kernels::rope::rope_scalar;
use provable_contracts::kernels::silu_standalone::silu_standalone_scalar;
use provable_contracts::kernels::softmax::softmax_scalar;
use provable_contracts::kernels::ssm::ssm_scan_scalar;
use provable_contracts::kernels::swiglu::swiglu_scalar;

// Part 1: Isolation Tests (21 tests)
include!("includes/falsify_cross_isolation.rs");

// Part 2a: Mutation Detection Tests (11 tests)
include!("includes/falsify_cross_mutation1.rs");

// Part 2b: Mutation Detection Tests (10 tests)
include!("includes/falsify_cross_mutation2.rs");
