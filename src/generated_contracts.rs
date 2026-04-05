// Auto-generated contract assertions from YAML — DO NOT EDIT.
// Zero cost in release builds (debug_assert!).
// Regenerate: pv codegen contracts/ -o src/generated_contracts.rs
// Include:   #[macro_use] #[allow(unused_macros)] mod generated_contracts;

// Auto-generated from contracts/absolute-position-v1.yaml — DO NOT EDIT
// Contract: absolute-position-v1

/// Preconditions for equation `absolute_position_add`.
/// Domain-specific. Call: `contract_pre_absolute_position_add!(slice_expr)`
macro_rules! contract_pre_absolute_position_add {
    () => {{}};
    ($input:expr) => {{
        let _pv_indices = &$input;
        debug_assert!(_pv_indices.len() > 0,
            "Contract absolute_position_add: precondition violated — indices.len() > 0");
    }};
}

// Auto-generated from contracts/activation-kernel-v1.yaml — DO NOT EDIT
// Contract: activation-kernel-v1

/// Preconditions for equation `gelu`.
/// Domain-specific. Call: `contract_pre_gelu!(slice_expr)`
macro_rules! contract_pre_gelu {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
        debug_assert!(_pv_x.iter().all(|v| v.is_finite()),
            "Contract gelu: precondition violated — x.iter().all(|v| v.is_finite())");
        debug_assert!(_pv_x.len() > 0,
            "Contract gelu: precondition violated — x.len() > 0");
    }};
}

/// Preconditions for equation `relu`.
/// Domain-specific. Call: `contract_pre_relu!(slice_expr)`
macro_rules! contract_pre_relu {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
        debug_assert!(_pv_x.iter().all(|v| v.is_finite()),
            "Contract relu: precondition violated — x.iter().all(|v| v.is_finite())");
        debug_assert!(_pv_x.len() > 0,
            "Contract relu: precondition violated — x.len() > 0");
    }};
}

/// Preconditions for equation `silu`.
/// Domain-specific. Call: `contract_pre_silu!(slice_expr)`
macro_rules! contract_pre_silu {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
        debug_assert!(_pv_x.iter().all(|v| v.is_finite()),
            "Contract silu: precondition violated — x.iter().all(|v| v.is_finite())");
        debug_assert!(_pv_x.len() > 0,
            "Contract silu: precondition violated — x.len() > 0");
    }};
}

// Auto-generated from contracts/active-learning-v1.yaml — DO NOT EDIT
// Contract: active-learning-v1

/// Preconditions for equation `entropy_score`.
/// Domain-specific. Call: `contract_pre_entropy_score!(slice_expr)`
macro_rules! contract_pre_entropy_score {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract entropy_score: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract entropy_score: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `margin_score`.
/// Domain-specific. Call: `contract_pre_margin_score!(slice_expr)`
macro_rules! contract_pre_margin_score {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract margin_score: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract margin_score: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `qbc_score`.
/// Domain-specific. Call: `contract_pre_qbc_score!(slice_expr)`
macro_rules! contract_pre_qbc_score {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract qbc_score: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract qbc_score: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `uncertainty_score`.
/// Domain-specific. Call: `contract_pre_uncertainty_score!(slice_expr)`
macro_rules! contract_pre_uncertainty_score {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract uncertainty_score: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract uncertainty_score: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

// Auto-generated from contracts/adamw-kernel-v1.yaml — DO NOT EDIT
// Contract: adamw-kernel-v1

/// Preconditions for equation `adam_moments`.
/// Domain-specific. Call: `contract_pre_adam_moments!(slice_expr)`
macro_rules! contract_pre_adam_moments {
    () => {{}};
    ($input:expr) => {{
        let _pv_params = &$input;
        debug_assert!(_pv_params.len() > 0,
            "Contract adam_moments: precondition violated — params.len() > 0");
    }};
}

/// Preconditions for equation `adam_variance`.
/// Domain-specific. Call: `contract_pre_adam_variance!(slice_expr)`
macro_rules! contract_pre_adam_variance {
    () => {{}};
    ($input:expr) => {{
        let _pv_params = &$input;
        debug_assert!(_pv_params.len() > 0,
            "Contract adam_variance: precondition violated — params.len() > 0");
    }};
}

/// Preconditions for equation `bias_correction`.
/// Domain-specific. Call: `contract_pre_bias_correction!(slice_expr)`
macro_rules! contract_pre_bias_correction {
    () => {{}};
    ($input:expr) => {{
        let _pv_params = &$input;
        debug_assert!(_pv_params.len() > 0,
            "Contract bias_correction: precondition violated — params.len() > 0");
    }};
}

/// Preconditions for equation `weight_update`.
/// Domain-specific. Call: `contract_pre_weight_update!(slice_expr)`
macro_rules! contract_pre_weight_update {
    () => {{}};
    ($input:expr) => {{
        let _pv_params = &$input;
        debug_assert!(_pv_params.len() > 0,
            "Contract weight_update: precondition violated — params.len() > 0");
    }};
}

// Auto-generated from contracts/agent-loop-v1.yaml — DO NOT EDIT
// Contract: agent-loop-v1

/// Preconditions for equation `context_compaction`.
/// Call at function entry: `contract_pre_context_compaction!(input_expr)`
macro_rules! contract_pre_context_compaction {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `context_compaction`.
/// Call before return: `contract_post_context_compaction!(result_expr)`
macro_rules! contract_post_context_compaction {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `context_compaction`.
macro_rules! contract_context_compaction {
    ($input:expr, $body:expr) => {{
        contract_pre_context_compaction!($input);
        let _contract_result = $body;
        contract_post_context_compaction!(_contract_result);
        _contract_result
    }};
}

/// Postconditions for equation `hook_ordering`.
/// Call before return: `contract_post_hook_ordering!(result_expr)`
macro_rules! contract_post_hook_ordering {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Preconditions for equation `loop_termination`.
/// Domain-specific. Call: `contract_pre_loop_termination!(slice_expr)`
macro_rules! contract_pre_loop_termination {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
    }};
}

/// Postconditions for equation `loop_termination`.
/// Call before return: `contract_post_loop_termination!(result_expr)`
macro_rules! contract_post_loop_termination {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `loop_termination`.
macro_rules! contract_loop_termination {
    ($input:expr, $body:expr) => {{
        contract_pre_loop_termination!($input);
        let _contract_result = $body;
        contract_post_loop_termination!(_contract_result);
        _contract_result
    }};
}

/// Postconditions for equation `parallel_tool_safety`.
/// Call before return: `contract_post_parallel_tool_safety!(result_expr)`
macro_rules! contract_post_parallel_tool_safety {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Preconditions for equation `sandbox_enforcement`.
/// Domain-specific. Call: `contract_pre_sandbox_enforcement!(slice_expr)`
macro_rules! contract_pre_sandbox_enforcement {
    () => {{}};
    ($input:expr) => {{
        let _pv_manifest = &$input;
    }};
}

/// Postconditions for equation `sandbox_enforcement`.
/// Call before return: `contract_post_sandbox_enforcement!(result_expr)`
macro_rules! contract_post_sandbox_enforcement {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `sandbox_enforcement`.
macro_rules! contract_sandbox_enforcement {
    ($input:expr, $body:expr) => {{
        contract_pre_sandbox_enforcement!($input);
        let _contract_result = $body;
        contract_post_sandbox_enforcement!(_contract_result);
        _contract_result
    }};
}

/// Postconditions for equation `session_crash_recovery`.
/// Call before return: `contract_post_session_crash_recovery!(result_expr)`
macro_rules! contract_post_session_crash_recovery {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Postconditions for equation `state_machine`.
/// Call before return: `contract_post_state_machine!(result_expr)`
macro_rules! contract_post_state_machine {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

// Auto-generated from contracts/agent-orchestration-v1.yaml — DO NOT EDIT
// Contract: agent-orchestration-v1

/// Preconditions for equation `daemon_lifecycle`.
/// Domain-specific. Call: `contract_pre_daemon_lifecycle!(slice_expr)`
macro_rules! contract_pre_daemon_lifecycle {
    () => {{}};
    ($input:expr) => {{
        let _pv_config = &$input;
    }};
}

/// Preconditions for equation `error_classification`.
/// Call at function entry: `contract_pre_error_classification!(input_expr)`
macro_rules! contract_pre_error_classification {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `manager_registration`.
/// Call at function entry: `contract_pre_manager_registration!(input_expr)`
macro_rules! contract_pre_manager_registration {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `red_metrics`.
/// Call at function entry: `contract_pre_red_metrics!(input_expr)`
macro_rules! contract_pre_red_metrics {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `restart_policy`.
/// Domain-specific. Call: `contract_pre_restart_policy!(slice_expr)`
macro_rules! contract_pre_restart_policy {
    () => {{}};
    ($input:expr) => {{
        let _pv_BackoffConfig = &$input;
    }};
}

/// Preconditions for equation `signal_handling`.
/// Call at function entry: `contract_pre_signal_handling!(input_expr)`
macro_rules! contract_pre_signal_handling {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

// Auto-generated from contracts/agent-ux-v1.yaml — DO NOT EDIT
// Contract: agent-ux-v1

/// Preconditions for equation `brick_verification`.
/// Domain-specific. Call: `contract_pre_brick_verification!(slice_expr)`
macro_rules! contract_pre_brick_verification {
    () => {{}};
    ($input:expr) => {{
        let _pv_brick = &$input;
    }};
}

/// Postconditions for equation `brick_verification`.
/// Call before return: `contract_post_brick_verification!(result_expr)`
macro_rules! contract_post_brick_verification {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `brick_verification`.
macro_rules! contract_brick_verification {
    ($input:expr, $body:expr) => {{
        contract_pre_brick_verification!($input);
        let _contract_result = $body;
        contract_post_brick_verification!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `contrast_accessibility`.
/// Call at function entry: `contract_pre_contrast_accessibility!(input_expr)`
macro_rules! contract_pre_contrast_accessibility {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `contrast_accessibility`.
/// Call before return: `contract_post_contrast_accessibility!(result_expr)`
macro_rules! contract_post_contrast_accessibility {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `contrast_accessibility`.
macro_rules! contract_contrast_accessibility {
    ($input:expr, $body:expr) => {{
        contract_pre_contrast_accessibility!($input);
        let _contract_result = $body;
        contract_post_contrast_accessibility!(_contract_result);
        _contract_result
    }};
}

/// Postconditions for equation `cost_display_accuracy`.
/// Call before return: `contract_post_cost_display_accuracy!(result_expr)`
macro_rules! contract_post_cost_display_accuracy {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Preconditions for equation `frame_budget`.
/// Domain-specific. Call: `contract_pre_frame_budget!(slice_expr)`
macro_rules! contract_pre_frame_budget {
    () => {{}};
    ($input:expr) => {{
        let _pv_panels = &$input;
        debug_assert!(_pv_panels.len() > 0,
            "Contract frame_budget: precondition violated — panels.len() > 0");
    }};
}

/// Postconditions for equation `frame_budget`.
/// Call before return: `contract_post_frame_budget!(result_expr)`
macro_rules! contract_post_frame_budget {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `frame_budget`.
macro_rules! contract_frame_budget {
    ($input:expr, $body:expr) => {{
        contract_pre_frame_budget!($input);
        let _contract_result = $body;
        contract_post_frame_budget!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `layout_correctness`.
/// Call at function entry: `contract_pre_layout_correctness!(input_expr)`
macro_rules! contract_pre_layout_correctness {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `pixel_coverage`.
/// Call at function entry: `contract_pre_pixel_coverage!(input_expr)`
macro_rules! contract_pre_pixel_coverage {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `pixel_coverage`.
/// Call before return: `contract_post_pixel_coverage!(result_expr)`
macro_rules! contract_post_pixel_coverage {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `pixel_coverage`.
macro_rules! contract_pixel_coverage {
    ($input:expr, $body:expr) => {{
        contract_pre_pixel_coverage!($input);
        let _contract_result = $body;
        contract_post_pixel_coverage!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `streaming_responsiveness`.
/// Domain-specific. Call: `contract_pre_streaming_responsiveness!(slice_expr)`
macro_rules! contract_pre_streaming_responsiveness {
    () => {{}};
    ($input:expr) => {{
        let _pv_provider = &$input;
    }};
}

/// Postconditions for equation `streaming_responsiveness`.
/// Call before return: `contract_post_streaming_responsiveness!(result_expr)`
macro_rules! contract_post_streaming_responsiveness {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `streaming_responsiveness`.
macro_rules! contract_streaming_responsiveness {
    ($input:expr, $body:expr) => {{
        contract_pre_streaming_responsiveness!($input);
        let _contract_result = $body;
        contract_post_streaming_responsiveness!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/alibi-kernel-v1.yaml — DO NOT EDIT
// Contract: alibi-kernel-v1

/// Preconditions for equation `alibi_bias`.
/// Domain-specific. Call: `contract_pre_alibi_bias!(slice_expr)`
macro_rules! contract_pre_alibi_bias {
    () => {{}};
    ($input:expr) => {{
        let _pv_indices = &$input;
        debug_assert!(_pv_indices.len() > 0,
            "Contract alibi_bias: precondition violated — indices.len() > 0");
    }};
}

/// Preconditions for equation `alibi_slopes`.
/// Domain-specific. Call: `contract_pre_alibi_slopes!(slice_expr)`
macro_rules! contract_pre_alibi_slopes {
    () => {{}};
    ($input:expr) => {{
        let _pv_indices = &$input;
        debug_assert!(_pv_indices.len() > 0,
            "Contract alibi_slopes: precondition violated — indices.len() > 0");
    }};
}

// Auto-generated from contracts/apr-architecture-schema-v1.yaml — DO NOT EDIT
// Contract: apr-architecture-schema-v1

/// Preconditions for equation `architecture_config_invariants`.
/// Call at function entry: `contract_pre_architecture_config_invariants!(input_expr)`
macro_rules! contract_pre_architecture_config_invariants {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `architecture_config_invariants`.
/// Call before return: `contract_post_architecture_config_invariants!(result_expr)`
macro_rules! contract_post_architecture_config_invariants {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `architecture_config_invariants`.
macro_rules! contract_architecture_config_invariants {
    ($input:expr, $body:expr) => {{
        contract_pre_architecture_config_invariants!($input);
        let _contract_result = $body;
        contract_post_architecture_config_invariants!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `attention_tensor_shapes`.
/// Domain-specific. Call: `contract_pre_attention_tensor_shapes!(slice_expr)`
macro_rules! contract_pre_attention_tensor_shapes {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
    }};
}

/// Postconditions for equation `attention_tensor_shapes`.
/// Call before return: `contract_post_attention_tensor_shapes!(result_expr)`
macro_rules! contract_post_attention_tensor_shapes {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `attention_tensor_shapes`.
macro_rules! contract_attention_tensor_shapes {
    ($input:expr, $body:expr) => {{
        contract_pre_attention_tensor_shapes!($input);
        let _contract_result = $body;
        contract_post_attention_tensor_shapes!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `embedding_tensor_shapes`.
/// Call at function entry: `contract_pre_embedding_tensor_shapes!(input_expr)`
macro_rules! contract_pre_embedding_tensor_shapes {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `embedding_tensor_shapes`.
/// Call before return: `contract_post_embedding_tensor_shapes!(result_expr)`
macro_rules! contract_post_embedding_tensor_shapes {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `embedding_tensor_shapes`.
macro_rules! contract_embedding_tensor_shapes {
    ($input:expr, $body:expr) => {{
        contract_pre_embedding_tensor_shapes!($input);
        let _contract_result = $body;
        contract_post_embedding_tensor_shapes!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `ffn_tensor_shapes`.
/// Domain-specific. Call: `contract_pre_ffn_tensor_shapes!(slice_expr)`
macro_rules! contract_pre_ffn_tensor_shapes {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
    }};
}

/// Postconditions for equation `ffn_tensor_shapes`.
/// Call before return: `contract_post_ffn_tensor_shapes!(result_expr)`
macro_rules! contract_post_ffn_tensor_shapes {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `ffn_tensor_shapes`.
macro_rules! contract_ffn_tensor_shapes {
    ($input:expr, $body:expr) => {{
        contract_pre_ffn_tensor_shapes!($input);
        let _contract_result = $body;
        contract_post_ffn_tensor_shapes!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `normalization_tensor_shapes`.
/// Domain-specific. Call: `contract_pre_normalization_tensor_shapes!(slice_expr)`
macro_rules! contract_pre_normalization_tensor_shapes {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
    }};
}

/// Postconditions for equation `normalization_tensor_shapes`.
/// Call before return: `contract_post_normalization_tensor_shapes!(result_expr)`
macro_rules! contract_post_normalization_tensor_shapes {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `normalization_tensor_shapes`.
macro_rules! contract_normalization_tensor_shapes {
    ($input:expr, $body:expr) => {{
        contract_pre_normalization_tensor_shapes!($input);
        let _contract_result = $body;
        contract_post_normalization_tensor_shapes!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `rope_position_encoding`.
/// Domain-specific. Call: `contract_pre_rope_position_encoding!(slice_expr)`
macro_rules! contract_pre_rope_position_encoding {
    () => {{}};
    ($input:expr) => {{
        let _pv_config = &$input;
    }};
}

/// Postconditions for equation `rope_position_encoding`.
/// Call before return: `contract_post_rope_position_encoding!(result_expr)`
macro_rules! contract_post_rope_position_encoding {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `rope_position_encoding`.
macro_rules! contract_rope_position_encoding {
    ($input:expr, $body:expr) => {{
        contract_pre_rope_position_encoding!($input);
        let _contract_result = $body;
        contract_post_rope_position_encoding!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `total_tensor_count`.
/// Call at function entry: `contract_pre_total_tensor_count!(input_expr)`
macro_rules! contract_pre_total_tensor_count {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `total_tensor_count`.
/// Call before return: `contract_post_total_tensor_count!(result_expr)`
macro_rules! contract_post_total_tensor_count {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `total_tensor_count`.
macro_rules! contract_total_tensor_count {
    ($input:expr, $body:expr) => {{
        contract_pre_total_tensor_count!($input);
        let _contract_result = $body;
        contract_post_total_tensor_count!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/apr-chat-session-v1.yaml — DO NOT EDIT
// Contract: apr-chat-session-v1

/// Preconditions for equation `chat_template_application`.
/// Call at function entry: `contract_pre_chat_template_application!(input_expr)`
macro_rules! contract_pre_chat_template_application {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `chat_template_application`.
/// Call before return: `contract_post_chat_template_application!(result_expr)`
macro_rules! contract_post_chat_template_application {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `chat_template_application`.
macro_rules! contract_chat_template_application {
    ($input:expr, $body:expr) => {{
        contract_pre_chat_template_application!($input);
        let _contract_result = $body;
        contract_post_chat_template_application!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `kv_cache_management`.
/// Domain-specific. Call: `contract_pre_kv_cache_management!(slice_expr)`
macro_rules! contract_pre_kv_cache_management {
    () => {{}};
    ($input:expr) => {{
        let _pv_new_tokens = &$input;
        debug_assert!(_pv_new_tokens.len() > 0,
            "Contract kv_cache_management: precondition violated — new_tokens.len() > 0");
    }};
}

/// Postconditions for equation `kv_cache_management`.
/// Call before return: `contract_post_kv_cache_management!(result_expr)`
macro_rules! contract_post_kv_cache_management {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `kv_cache_management`.
macro_rules! contract_kv_cache_management {
    ($input:expr, $body:expr) => {{
        contract_pre_kv_cache_management!($input);
        let _contract_result = $body;
        contract_post_kv_cache_management!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `session_persistence`.
/// Call at function entry: `contract_pre_session_persistence!(input_expr)`
macro_rules! contract_pre_session_persistence {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `session_persistence`.
/// Call before return: `contract_post_session_persistence!(result_expr)`
macro_rules! contract_post_session_persistence {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `session_persistence`.
macro_rules! contract_session_persistence {
    ($input:expr, $body:expr) => {{
        contract_pre_session_persistence!($input);
        let _contract_result = $body;
        contract_post_session_persistence!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `session_state_machine`.
/// Call at function entry: `contract_pre_session_state_machine!(input_expr)`
macro_rules! contract_pre_session_state_machine {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `session_state_machine`.
/// Call before return: `contract_post_session_state_machine!(result_expr)`
macro_rules! contract_post_session_state_machine {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `session_state_machine`.
macro_rules! contract_session_state_machine {
    ($input:expr, $body:expr) => {{
        contract_pre_session_state_machine!($input);
        let _contract_result = $body;
        contract_post_session_state_machine!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/apr-checkpoint-v1.yaml — DO NOT EDIT
// Contract: apr-checkpoint-v1

/// Preconditions for equation `load_checkpoint`.
/// Domain-specific. Call: `contract_pre_load_checkpoint!(slice_expr)`
macro_rules! contract_pre_load_checkpoint {
    () => {{}};
    ($input:expr) => {{
        let _pv_path = &$input;
    }};
}

/// Postconditions for equation `load_checkpoint`.
/// Call before return: `contract_post_load_checkpoint!(result_expr)`
macro_rules! contract_post_load_checkpoint {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `load_checkpoint`.
macro_rules! contract_load_checkpoint {
    ($input:expr, $body:expr) => {{
        contract_pre_load_checkpoint!($input);
        let _contract_result = $body;
        contract_post_load_checkpoint!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `save_checkpoint`.
/// Domain-specific. Call: `contract_pre_save_checkpoint!(slice_expr)`
macro_rules! contract_pre_save_checkpoint {
    () => {{}};
    ($input:expr) => {{
        let _pv_data = &$input;
        debug_assert!(!_pv_data.is_empty(),
            "Contract save_checkpoint: precondition violated — !data.is_empty()");
    }};
}

/// Postconditions for equation `save_checkpoint`.
/// Call before return: `contract_post_save_checkpoint!(result_expr)`
macro_rules! contract_post_save_checkpoint {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `save_checkpoint`.
macro_rules! contract_save_checkpoint {
    ($input:expr, $body:expr) => {{
        contract_pre_save_checkpoint!($input);
        let _contract_result = $body;
        contract_post_save_checkpoint!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/apr-cli-operations-v1.yaml — DO NOT EDIT
// Contract: apr-cli-operations-v1

/// Preconditions for equation `concurrent_model_access`.
/// Domain-specific. Call: `contract_pre_concurrent_model_access!(slice_expr)`
macro_rules! contract_pre_concurrent_model_access {
    () => {{}};
    ($input:expr) => {{
        let _pv_requests = &$input;
        debug_assert!(_pv_requests.len() > 0,
            "Contract concurrent_model_access: precondition violated — requests.len() > 0");
    }};
}

/// Postconditions for equation `concurrent_model_access`.
/// Call before return: `contract_post_concurrent_model_access!(result_expr)`
macro_rules! contract_post_concurrent_model_access {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `concurrent_model_access`.
macro_rules! contract_concurrent_model_access {
    ($input:expr, $body:expr) => {{
        contract_pre_concurrent_model_access!($input);
        let _contract_result = $body;
        contract_post_concurrent_model_access!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `inference_determinism`.
/// Call at function entry: `contract_pre_inference_determinism!(input_expr)`
macro_rules! contract_pre_inference_determinism {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `inference_determinism`.
/// Call before return: `contract_post_inference_determinism!(result_expr)`
macro_rules! contract_post_inference_determinism {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `inference_determinism`.
macro_rules! contract_inference_determinism {
    ($input:expr, $body:expr) => {{
        contract_pre_inference_determinism!($input);
        let _contract_result = $body;
        contract_post_inference_determinism!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `progress_reporting`.
/// Call at function entry: `contract_pre_progress_reporting!(input_expr)`
macro_rules! contract_pre_progress_reporting {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `progress_reporting`.
/// Call before return: `contract_post_progress_reporting!(result_expr)`
macro_rules! contract_post_progress_reporting {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `progress_reporting`.
macro_rules! contract_progress_reporting {
    ($input:expr, $body:expr) => {{
        contract_pre_progress_reporting!($input);
        let _contract_result = $body;
        contract_post_progress_reporting!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `resource_cleanup`.
/// Call at function entry: `contract_pre_resource_cleanup!(input_expr)`
macro_rules! contract_pre_resource_cleanup {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `resource_cleanup`.
/// Call before return: `contract_post_resource_cleanup!(result_expr)`
macro_rules! contract_post_resource_cleanup {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `resource_cleanup`.
macro_rules! contract_resource_cleanup {
    ($input:expr, $body:expr) => {{
        contract_pre_resource_cleanup!($input);
        let _contract_result = $body;
        contract_post_resource_cleanup!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `side_effect_classification`.
/// Call at function entry: `contract_pre_side_effect_classification!(input_expr)`
macro_rules! contract_pre_side_effect_classification {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `side_effect_classification`.
/// Call before return: `contract_post_side_effect_classification!(result_expr)`
macro_rules! contract_post_side_effect_classification {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `side_effect_classification`.
macro_rules! contract_side_effect_classification {
    ($input:expr, $body:expr) => {{
        contract_pre_side_effect_classification!($input);
        let _contract_result = $body;
        contract_post_side_effect_classification!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `tokenizer_consistency`.
/// Call at function entry: `contract_pre_tokenizer_consistency!(input_expr)`
macro_rules! contract_pre_tokenizer_consistency {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `tokenizer_consistency`.
/// Call before return: `contract_post_tokenizer_consistency!(result_expr)`
macro_rules! contract_post_tokenizer_consistency {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `tokenizer_consistency`.
macro_rules! contract_tokenizer_consistency {
    ($input:expr, $body:expr) => {{
        contract_pre_tokenizer_consistency!($input);
        let _contract_result = $body;
        contract_post_tokenizer_consistency!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/apr-cli-v1.yaml — DO NOT EDIT
// Contract: apr-cli-v1

/// Preconditions for equation `command_parse_determinism`.
/// Call at function entry: `contract_pre_command_parse_determinism!(input_expr)`
macro_rules! contract_pre_command_parse_determinism {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `command_parse_determinism`.
/// Call before return: `contract_post_command_parse_determinism!(result_expr)`
macro_rules! contract_post_command_parse_determinism {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `command_parse_determinism`.
macro_rules! contract_command_parse_determinism {
    ($input:expr, $body:expr) => {{
        contract_pre_command_parse_determinism!($input);
        let _contract_result = $body;
        contract_post_command_parse_determinism!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `contract_gate_enforcement`.
/// Call at function entry: `contract_pre_contract_gate_enforcement!(input_expr)`
macro_rules! contract_pre_contract_gate_enforcement {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `contract_gate_enforcement`.
/// Call before return: `contract_post_contract_gate_enforcement!(result_expr)`
macro_rules! contract_post_contract_gate_enforcement {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `contract_gate_enforcement`.
macro_rules! contract_contract_gate_enforcement {
    ($input:expr, $body:expr) => {{
        contract_pre_contract_gate_enforcement!($input);
        let _contract_result = $body;
        contract_post_contract_gate_enforcement!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `model_path_resolution`.
/// Call at function entry: `contract_pre_model_path_resolution!(input_expr)`
macro_rules! contract_pre_model_path_resolution {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `model_path_resolution`.
/// Call before return: `contract_post_model_path_resolution!(result_expr)`
macro_rules! contract_post_model_path_resolution {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `model_path_resolution`.
macro_rules! contract_model_path_resolution {
    ($input:expr, $body:expr) => {{
        contract_pre_model_path_resolution!($input);
        let _contract_result = $body;
        contract_post_model_path_resolution!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `pipe_stdin_support`.
/// Call at function entry: `contract_pre_pipe_stdin_support!(input_expr)`
macro_rules! contract_pre_pipe_stdin_support {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `pipe_stdin_support`.
/// Call before return: `contract_post_pipe_stdin_support!(result_expr)`
macro_rules! contract_post_pipe_stdin_support {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `pipe_stdin_support`.
macro_rules! contract_pipe_stdin_support {
    ($input:expr, $body:expr) => {{
        contract_pre_pipe_stdin_support!($input);
        let _contract_result = $body;
        contract_post_pipe_stdin_support!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `tokenizer_training_correctness`.
/// Call at function entry: `contract_pre_tokenizer_training_correctness!(input_expr)`
macro_rules! contract_pre_tokenizer_training_correctness {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `tokenizer_training_correctness`.
/// Call before return: `contract_post_tokenizer_training_correctness!(result_expr)`
macro_rules! contract_post_tokenizer_training_correctness {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `tokenizer_training_correctness`.
macro_rules! contract_tokenizer_training_correctness {
    ($input:expr, $body:expr) => {{
        contract_pre_tokenizer_training_correctness!($input);
        let _contract_result = $body;
        contract_post_tokenizer_training_correctness!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `training_plan_apply_semantics`.
/// Domain-specific. Call: `contract_pre_training_plan_apply_semantics!(slice_expr)`
macro_rules! contract_pre_training_plan_apply_semantics {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
    }};
}

/// Postconditions for equation `training_plan_apply_semantics`.
/// Call before return: `contract_post_training_plan_apply_semantics!(result_expr)`
macro_rules! contract_post_training_plan_apply_semantics {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `training_plan_apply_semantics`.
macro_rules! contract_training_plan_apply_semantics {
    ($input:expr, $body:expr) => {{
        contract_pre_training_plan_apply_semantics!($input);
        let _contract_result = $body;
        contract_post_training_plan_apply_semantics!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/apr-code-v1.yaml — DO NOT EDIT
// Contract: apr-code-v1

/// Preconditions for equation `apr_md_compliance`.
/// Call at function entry: `contract_pre_apr_md_compliance!(input_expr)`
macro_rules! contract_pre_apr_md_compliance {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `apr_md_compliance`.
/// Call before return: `contract_post_apr_md_compliance!(result_expr)`
macro_rules! contract_post_apr_md_compliance {
    ($result:expr) => {{
        let _contract_result = &$result;
        debug_assert!(violated_instructions.len() == 0, "Contract apr_md_compliance: postcondition violated — violated_instructions.len() == 0");
    }};
}

/// Combined pre+post contract for equation `apr_md_compliance`.
macro_rules! contract_apr_md_compliance {
    ($input:expr, $body:expr) => {{
        contract_pre_apr_md_compliance!($input);
        let _contract_result = $body;
        contract_post_apr_md_compliance!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `apr_model_validity`.
/// Domain-specific. Call: `contract_pre_apr_model_validity!(slice_expr)`
macro_rules! contract_pre_apr_model_validity {
    () => {{}};
    ($input:expr) => {{
        let _pv_path = &$input;
    }};
}

/// Postconditions for equation `apr_model_validity`.
/// Call before return: `contract_post_apr_model_validity!(result_expr)`
macro_rules! contract_post_apr_model_validity {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `apr_model_validity`.
macro_rules! contract_apr_model_validity {
    ($input:expr, $body:expr) => {{
        contract_pre_apr_model_validity!($input);
        let _contract_result = $body;
        contract_post_apr_model_validity!(_contract_result);
        _contract_result
    }};
}

/// Postconditions for equation `no_model_error`.
/// Call before return: `contract_post_no_model_error!(result_expr)`
macro_rules! contract_post_no_model_error {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Preconditions for equation `session_integrity`.
/// Domain-specific. Call: `contract_pre_session_integrity!(slice_expr)`
macro_rules! contract_pre_session_integrity {
    () => {{}};
    ($input:expr) => {{
        let _pv_session = &$input;
    }};
}

/// Postconditions for equation `session_integrity`.
/// Call before return: `contract_post_session_integrity!(result_expr)`
macro_rules! contract_post_session_integrity {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `session_integrity`.
macro_rules! contract_session_integrity {
    ($input:expr, $body:expr) => {{
        contract_pre_session_integrity!($input);
        let _contract_result = $body;
        contract_post_session_integrity!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `sovereignty_guarantee`.
/// Domain-specific. Call: `contract_pre_sovereignty_guarantee!(slice_expr)`
macro_rules! contract_pre_sovereignty_guarantee {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
    }};
}

/// Postconditions for equation `sovereignty_guarantee`.
/// Call before return: `contract_post_sovereignty_guarantee!(result_expr)`
macro_rules! contract_post_sovereignty_guarantee {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `sovereignty_guarantee`.
macro_rules! contract_sovereignty_guarantee {
    ($input:expr, $body:expr) => {{
        contract_pre_sovereignty_guarantee!($input);
        let _contract_result = $body;
        contract_post_sovereignty_guarantee!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `startup_latency`.
/// Domain-specific. Call: `contract_pre_startup_latency!(slice_expr)`
macro_rules! contract_pre_startup_latency {
    () => {{}};
    ($input:expr) => {{
        let _pv_project = &$input;
    }};
}

/// Postconditions for equation `startup_latency`.
/// Call before return: `contract_post_startup_latency!(result_expr)`
macro_rules! contract_post_startup_latency {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `startup_latency`.
macro_rules! contract_startup_latency {
    ($input:expr, $body:expr) => {{
        contract_pre_startup_latency!($input);
        let _contract_result = $body;
        contract_post_startup_latency!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `tool_safety`.
/// Domain-specific. Call: `contract_pre_tool_safety!(slice_expr)`
macro_rules! contract_pre_tool_safety {
    () => {{}};
    ($input:expr) => {{
        let _pv_session = &$input;
    }};
}

/// Postconditions for equation `tool_safety`.
/// Call before return: `contract_post_tool_safety!(result_expr)`
macro_rules! contract_post_tool_safety {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `tool_safety`.
macro_rules! contract_tool_safety {
    ($input:expr, $body:expr) => {{
        contract_pre_tool_safety!($input);
        let _contract_result = $body;
        contract_post_tool_safety!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/apr-data-pipeline-v1.yaml — DO NOT EDIT
// Contract: apr-data-pipeline-v1

/// Preconditions for equation `data_split_determinism`.
/// Domain-specific. Call: `contract_pre_data_split_determinism!(slice_expr)`
macro_rules! contract_pre_data_split_determinism {
    () => {{}};
    ($input:expr) => {{
        let _pv_ratios = &$input;
        debug_assert!(_pv_ratios.sum() == 1.0,
            "Contract data_split_determinism: precondition violated — ratios.sum() == 1.0");
    }};
}

/// Postconditions for equation `data_split_determinism`.
/// Call before return: `contract_post_data_split_determinism!(result_expr)`
macro_rules! contract_post_data_split_determinism {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `data_split_determinism`.
macro_rules! contract_data_split_determinism {
    ($input:expr, $body:expr) => {{
        contract_pre_data_split_determinism!($input);
        let _contract_result = $body;
        contract_post_data_split_determinism!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `data_validation`.
/// Domain-specific. Call: `contract_pre_data_validation!(slice_expr)`
macro_rules! contract_pre_data_validation {
    () => {{}};
    ($input:expr) => {{
        let _pv_path = &$input;
    }};
}

/// Postconditions for equation `data_validation`.
/// Call before return: `contract_post_data_validation!(result_expr)`
macro_rules! contract_post_data_validation {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `data_validation`.
macro_rules! contract_data_validation {
    ($input:expr, $body:expr) => {{
        contract_pre_data_validation!($input);
        let _contract_result = $body;
        contract_post_data_validation!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `preprocessing_idempotency`.
/// Call at function entry: `contract_pre_preprocessing_idempotency!(input_expr)`
macro_rules! contract_pre_preprocessing_idempotency {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `preprocessing_idempotency`.
/// Call before return: `contract_post_preprocessing_idempotency!(result_expr)`
macro_rules! contract_post_preprocessing_idempotency {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `preprocessing_idempotency`.
macro_rules! contract_preprocessing_idempotency {
    ($input:expr, $body:expr) => {{
        contract_pre_preprocessing_idempotency!($input);
        let _contract_result = $body;
        contract_post_preprocessing_idempotency!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `streaming_data_loader`.
/// Domain-specific. Call: `contract_pre_streaming_data_loader!(slice_expr)`
macro_rules! contract_pre_streaming_data_loader {
    () => {{}};
    ($input:expr) => {{
        let _pv_dataset = &$input;
        debug_assert!(_pv_dataset.len() > 0,
            "Contract streaming_data_loader: precondition violated — dataset.len() > 0");
    }};
}

/// Postconditions for equation `streaming_data_loader`.
/// Call before return: `contract_post_streaming_data_loader!(result_expr)`
macro_rules! contract_post_streaming_data_loader {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `streaming_data_loader`.
macro_rules! contract_streaming_data_loader {
    ($input:expr, $body:expr) => {{
        contract_pre_streaming_data_loader!($input);
        let _contract_result = $body;
        contract_post_streaming_data_loader!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/apr-finetune-v1.yaml — DO NOT EDIT
// Contract: apr-finetune-v1

/// Preconditions for equation `alpha_rank_ratio`.
/// Call at function entry: `contract_pre_alpha_rank_ratio!(input_expr)`
macro_rules! contract_pre_alpha_rank_ratio {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `alpha_rank_ratio`.
/// Call before return: `contract_post_alpha_rank_ratio!(result_expr)`
macro_rules! contract_post_alpha_rank_ratio {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `alpha_rank_ratio`.
macro_rules! contract_alpha_rank_ratio {
    ($input:expr, $body:expr) => {{
        contract_pre_alpha_rank_ratio!($input);
        let _contract_result = $body;
        contract_post_alpha_rank_ratio!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `checkpoint_metadata_roundtrip`.
/// Call at function entry: `contract_pre_checkpoint_metadata_roundtrip!(input_expr)`
macro_rules! contract_pre_checkpoint_metadata_roundtrip {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `checkpoint_metadata_roundtrip`.
/// Call before return: `contract_post_checkpoint_metadata_roundtrip!(result_expr)`
macro_rules! contract_post_checkpoint_metadata_roundtrip {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `checkpoint_metadata_roundtrip`.
macro_rules! contract_checkpoint_metadata_roundtrip {
    ($input:expr, $body:expr) => {{
        contract_pre_checkpoint_metadata_roundtrip!($input);
        let _contract_result = $body;
        contract_post_checkpoint_metadata_roundtrip!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `merge_tensor_shape`.
/// Call at function entry: `contract_pre_merge_tensor_shape!(input_expr)`
macro_rules! contract_pre_merge_tensor_shape {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `merge_tensor_shape`.
/// Call before return: `contract_post_merge_tensor_shape!(result_expr)`
macro_rules! contract_post_merge_tensor_shape {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `merge_tensor_shape`.
macro_rules! contract_merge_tensor_shape {
    ($input:expr, $body:expr) => {{
        contract_pre_merge_tensor_shape!($input);
        let _contract_result = $body;
        contract_post_merge_tensor_shape!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `rank_bounds_safety`.
/// Call at function entry: `contract_pre_rank_bounds_safety!(input_expr)`
macro_rules! contract_pre_rank_bounds_safety {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `rank_bounds_safety`.
/// Call before return: `contract_post_rank_bounds_safety!(result_expr)`
macro_rules! contract_post_rank_bounds_safety {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `rank_bounds_safety`.
macro_rules! contract_rank_bounds_safety {
    ($input:expr, $body:expr) => {{
        contract_pre_rank_bounds_safety!($input);
        let _contract_result = $body;
        contract_post_rank_bounds_safety!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `vram_feasibility`.
/// Call at function entry: `contract_pre_vram_feasibility!(input_expr)`
macro_rules! contract_pre_vram_feasibility {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `vram_feasibility`.
/// Call before return: `contract_post_vram_feasibility!(result_expr)`
macro_rules! contract_post_vram_feasibility {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `vram_feasibility`.
macro_rules! contract_vram_feasibility {
    ($input:expr, $body:expr) => {{
        contract_pre_vram_feasibility!($input);
        let _contract_result = $body;
        contract_post_vram_feasibility!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/apr-format-invariants-v1.yaml — DO NOT EDIT
// Contract: apr-format-invariants-v1

/// Preconditions for equation `detect_regression`.
/// Domain-specific. Call: `contract_pre_detect_regression!(slice_expr)`
macro_rules! contract_pre_detect_regression {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract detect_regression: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `format_report`.
/// Domain-specific. Call: `contract_pre_format_report!(slice_expr)`
macro_rules! contract_pre_format_report {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract format_report: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `parse_playbook`.
/// Domain-specific. Call: `contract_pre_parse_playbook!(slice_expr)`
macro_rules! contract_pre_parse_playbook {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract parse_playbook: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `serialize_roundtrip`.
/// Domain-specific. Call: `contract_pre_serialize_roundtrip!(slice_expr)`
macro_rules! contract_pre_serialize_roundtrip {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract serialize_roundtrip: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `validate_schema`.
/// Domain-specific. Call: `contract_pre_validate_schema!(slice_expr)`
macro_rules! contract_pre_validate_schema {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract validate_schema: precondition violated — input.len() > 0");
    }};
}

// Auto-generated from contracts/apr-format-safety-v1.yaml — DO NOT EDIT
// Contract: apr-format-safety-v1

/// Preconditions for equation `dtype_coercion_safety`.
/// Call at function entry: `contract_pre_dtype_coercion_safety!(input_expr)`
macro_rules! contract_pre_dtype_coercion_safety {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `dtype_coercion_safety`.
/// Call before return: `contract_post_dtype_coercion_safety!(result_expr)`
macro_rules! contract_post_dtype_coercion_safety {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `dtype_coercion_safety`.
macro_rules! contract_dtype_coercion_safety {
    ($input:expr, $body:expr) => {{
        contract_pre_dtype_coercion_safety!($input);
        let _contract_result = $body;
        contract_post_dtype_coercion_safety!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `header_integrity`.
/// Call at function entry: `contract_pre_header_integrity!(input_expr)`
macro_rules! contract_pre_header_integrity {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `header_integrity`.
/// Call before return: `contract_post_header_integrity!(result_expr)`
macro_rules! contract_post_header_integrity {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `header_integrity`.
macro_rules! contract_header_integrity {
    ($input:expr, $body:expr) => {{
        contract_pre_header_integrity!($input);
        let _contract_result = $body;
        contract_post_header_integrity!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `magic_byte_validation`.
/// Call at function entry: `contract_pre_magic_byte_validation!(input_expr)`
macro_rules! contract_pre_magic_byte_validation {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `magic_byte_validation`.
/// Call before return: `contract_post_magic_byte_validation!(result_expr)`
macro_rules! contract_post_magic_byte_validation {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `magic_byte_validation`.
macro_rules! contract_magic_byte_validation {
    ($input:expr, $body:expr) => {{
        contract_pre_magic_byte_validation!($input);
        let _contract_result = $body;
        contract_post_magic_byte_validation!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `provenance_enforcement`.
/// Call at function entry: `contract_pre_provenance_enforcement!(input_expr)`
macro_rules! contract_pre_provenance_enforcement {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `provenance_enforcement`.
/// Call before return: `contract_post_provenance_enforcement!(result_expr)`
macro_rules! contract_post_provenance_enforcement {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `provenance_enforcement`.
macro_rules! contract_provenance_enforcement {
    ($input:expr, $body:expr) => {{
        contract_pre_provenance_enforcement!($input);
        let _contract_result = $body;
        contract_post_provenance_enforcement!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `strict_import_validation`.
/// Call at function entry: `contract_pre_strict_import_validation!(input_expr)`
macro_rules! contract_pre_strict_import_validation {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `strict_import_validation`.
/// Call before return: `contract_post_strict_import_validation!(result_expr)`
macro_rules! contract_post_strict_import_validation {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `strict_import_validation`.
macro_rules! contract_strict_import_validation {
    ($input:expr, $body:expr) => {{
        contract_pre_strict_import_validation!($input);
        let _contract_result = $body;
        contract_post_strict_import_validation!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `truncation_detection`.
/// Call at function entry: `contract_pre_truncation_detection!(input_expr)`
macro_rules! contract_pre_truncation_detection {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `truncation_detection`.
/// Call before return: `contract_post_truncation_detection!(result_expr)`
macro_rules! contract_post_truncation_detection {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `truncation_detection`.
macro_rules! contract_truncation_detection {
    ($input:expr, $body:expr) => {{
        contract_pre_truncation_detection!($input);
        let _contract_result = $body;
        contract_post_truncation_detection!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/apr-model-discovery-v1.yaml — DO NOT EDIT
// Contract: apr-model-discovery-v1

/// Preconditions for equation `architecture_extraction`.
/// Call at function entry: `contract_pre_architecture_extraction!(input_expr)`
macro_rules! contract_pre_architecture_extraction {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `architecture_extraction`.
/// Call before return: `contract_post_architecture_extraction!(result_expr)`
macro_rules! contract_post_architecture_extraction {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `architecture_extraction`.
macro_rules! contract_architecture_extraction {
    ($input:expr, $body:expr) => {{
        contract_pre_architecture_extraction!($input);
        let _contract_result = $body;
        contract_post_architecture_extraction!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `jidoka_validation`.
/// Call at function entry: `contract_pre_jidoka_validation!(input_expr)`
macro_rules! contract_pre_jidoka_validation {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `jidoka_validation`.
/// Call before return: `contract_post_jidoka_validation!(result_expr)`
macro_rules! contract_post_jidoka_validation {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `jidoka_validation`.
macro_rules! contract_jidoka_validation {
    ($input:expr, $body:expr) => {{
        contract_pre_jidoka_validation!($input);
        let _contract_result = $body;
        contract_post_jidoka_validation!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `no_model_ux`.
/// Call at function entry: `contract_pre_no_model_ux!(input_expr)`
macro_rules! contract_pre_no_model_ux {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `no_model_ux`.
/// Call before return: `contract_post_no_model_ux!(result_expr)`
macro_rules! contract_post_no_model_ux {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `no_model_ux`.
macro_rules! contract_no_model_ux {
    ($input:expr, $body:expr) => {{
        contract_pre_no_model_ux!($input);
        let _contract_result = $body;
        contract_post_no_model_ux!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `search_order`.
/// Call at function entry: `contract_pre_search_order!(input_expr)`
macro_rules! contract_pre_search_order {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `search_order`.
/// Call before return: `contract_post_search_order!(result_expr)`
macro_rules! contract_post_search_order {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `search_order`.
macro_rules! contract_search_order {
    ($input:expr, $body:expr) => {{
        contract_pre_search_order!($input);
        let _contract_result = $body;
        contract_post_search_order!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `sort_priority`.
/// Domain-specific. Call: `contract_pre_sort_priority!(slice_expr)`
macro_rules! contract_pre_sort_priority {
    () => {{}};
    ($input:expr) => {{
        let _pv_candidates = &$input;
        debug_assert!(_pv_candidates.len() > 0,
            "Contract sort_priority: precondition violated — candidates.len() > 0");
    }};
}

/// Postconditions for equation `sort_priority`.
/// Call before return: `contract_post_sort_priority!(result_expr)`
macro_rules! contract_post_sort_priority {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `sort_priority`.
macro_rules! contract_sort_priority {
    ($input:expr, $body:expr) => {{
        contract_pre_sort_priority!($input);
        let _contract_result = $body;
        contract_post_sort_priority!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/apr-model-lifecycle-v1.yaml — DO NOT EDIT
// Contract: apr-model-lifecycle-v1

/// Preconditions for equation `export_roundtrip`.
/// Domain-specific. Call: `contract_pre_export_roundtrip!(slice_expr)`
macro_rules! contract_pre_export_roundtrip {
    () => {{}};
    ($input:expr) => {{
        let _pv_model = &$input;
    }};
}

/// Postconditions for equation `export_roundtrip`.
/// Call before return: `contract_post_export_roundtrip!(result_expr)`
macro_rules! contract_post_export_roundtrip {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `export_roundtrip`.
macro_rules! contract_export_roundtrip {
    ($input:expr, $body:expr) => {{
        contract_pre_export_roundtrip!($input);
        let _contract_result = $body;
        contract_post_export_roundtrip!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `import_format_detection`.
/// Domain-specific. Call: `contract_pre_import_format_detection!(slice_expr)`
macro_rules! contract_pre_import_format_detection {
    () => {{}};
    ($input:expr) => {{
        let _pv_path = &$input;
    }};
}

/// Postconditions for equation `import_format_detection`.
/// Call before return: `contract_post_import_format_detection!(result_expr)`
macro_rules! contract_post_import_format_detection {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `import_format_detection`.
macro_rules! contract_import_format_detection {
    ($input:expr, $body:expr) => {{
        contract_pre_import_format_detection!($input);
        let _contract_result = $body;
        contract_post_import_format_detection!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `merge_weight_conservation`.
/// Domain-specific. Call: `contract_pre_merge_weight_conservation!(slice_expr)`
macro_rules! contract_pre_merge_weight_conservation {
    () => {{}};
    ($input:expr) => {{
        let _pv_models = &$input;
        debug_assert!(_pv_models.len() >= 2,
            "Contract merge_weight_conservation: precondition violated — models.len() >= 2");
    }};
}

/// Postconditions for equation `merge_weight_conservation`.
/// Call before return: `contract_post_merge_weight_conservation!(result_expr)`
macro_rules! contract_post_merge_weight_conservation {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `merge_weight_conservation`.
macro_rules! contract_merge_weight_conservation {
    ($input:expr, $body:expr) => {{
        contract_pre_merge_weight_conservation!($input);
        let _contract_result = $body;
        contract_post_merge_weight_conservation!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `pull_cache_integrity`.
/// Call at function entry: `contract_pre_pull_cache_integrity!(input_expr)`
macro_rules! contract_pre_pull_cache_integrity {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `pull_cache_integrity`.
/// Call before return: `contract_post_pull_cache_integrity!(result_expr)`
macro_rules! contract_post_pull_cache_integrity {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `pull_cache_integrity`.
macro_rules! contract_pull_cache_integrity {
    ($input:expr, $body:expr) => {{
        contract_pre_pull_cache_integrity!($input);
        let _contract_result = $body;
        contract_post_pull_cache_integrity!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `quantize_precision_bound`.
/// Domain-specific. Call: `contract_pre_quantize_precision_bound!(slice_expr)`
macro_rules! contract_pre_quantize_precision_bound {
    () => {{}};
    ($input:expr) => {{
        let _pv_model = &$input;
    }};
}

/// Postconditions for equation `quantize_precision_bound`.
/// Call before return: `contract_post_quantize_precision_bound!(result_expr)`
macro_rules! contract_post_quantize_precision_bound {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `quantize_precision_bound`.
macro_rules! contract_quantize_precision_bound {
    ($input:expr, $body:expr) => {{
        contract_pre_quantize_precision_bound!($input);
        let _contract_result = $body;
        contract_post_quantize_precision_bound!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/apr-model-qa-v1.yaml — DO NOT EDIT
// Contract: apr-model-qa-v1

/// Preconditions for equation `canary_regression_detection`.
/// Domain-specific. Call: `contract_pre_canary_regression_detection!(slice_expr)`
macro_rules! contract_pre_canary_regression_detection {
    () => {{}};
    ($input:expr) => {{
        let _pv_baseline = &$input;
    }};
}

/// Postconditions for equation `canary_regression_detection`.
/// Call before return: `contract_post_canary_regression_detection!(result_expr)`
macro_rules! contract_post_canary_regression_detection {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `canary_regression_detection`.
macro_rules! contract_canary_regression_detection {
    ($input:expr, $body:expr) => {{
        contract_pre_canary_regression_detection!($input);
        let _contract_result = $body;
        contract_post_canary_regression_detection!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `lint_model_conventions`.
/// Domain-specific. Call: `contract_pre_lint_model_conventions!(slice_expr)`
macro_rules! contract_pre_lint_model_conventions {
    () => {{}};
    ($input:expr) => {{
        let _pv_path = &$input;
    }};
}

/// Postconditions for equation `lint_model_conventions`.
/// Call before return: `contract_post_lint_model_conventions!(result_expr)`
macro_rules! contract_post_lint_model_conventions {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `lint_model_conventions`.
macro_rules! contract_lint_model_conventions {
    ($input:expr, $body:expr) => {{
        contract_pre_lint_model_conventions!($input);
        let _contract_result = $body;
        contract_post_lint_model_conventions!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `model_integrity_check`.
/// Domain-specific. Call: `contract_pre_model_integrity_check!(slice_expr)`
macro_rules! contract_pre_model_integrity_check {
    () => {{}};
    ($input:expr) => {{
        let _pv_path = &$input;
    }};
}

/// Postconditions for equation `model_integrity_check`.
/// Call before return: `contract_post_model_integrity_check!(result_expr)`
macro_rules! contract_post_model_integrity_check {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `model_integrity_check`.
macro_rules! contract_model_integrity_check {
    ($input:expr, $body:expr) => {{
        contract_pre_model_integrity_check!($input);
        let _contract_result = $body;
        contract_post_model_integrity_check!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `probar_property_tests`.
/// Domain-specific. Call: `contract_pre_probar_property_tests!(slice_expr)`
macro_rules! contract_pre_probar_property_tests {
    () => {{}};
    ($input:expr) => {{
        let _pv_properties = &$input;
        debug_assert!(_pv_properties.len() > 0,
            "Contract probar_property_tests: precondition violated — properties.len() > 0");
    }};
}

/// Postconditions for equation `probar_property_tests`.
/// Call before return: `contract_post_probar_property_tests!(result_expr)`
macro_rules! contract_post_probar_property_tests {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `probar_property_tests`.
macro_rules! contract_probar_property_tests {
    ($input:expr, $body:expr) => {{
        contract_pre_probar_property_tests!($input);
        let _contract_result = $body;
        contract_post_probar_property_tests!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `qa_gate_composition`.
/// Call at function entry: `contract_pre_qa_gate_composition!(input_expr)`
macro_rules! contract_pre_qa_gate_composition {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `qa_gate_composition`.
/// Call before return: `contract_post_qa_gate_composition!(result_expr)`
macro_rules! contract_post_qa_gate_composition {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `qa_gate_composition`.
macro_rules! contract_qa_gate_composition {
    ($input:expr, $body:expr) => {{
        contract_pre_qa_gate_composition!($input);
        let _contract_result = $body;
        contract_post_qa_gate_composition!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/apr-serve-v1.yaml — DO NOT EDIT
// Contract: apr-serve-v1

/// Preconditions for equation `chat_template_dispatch`.
/// Domain-specific. Call: `contract_pre_chat_template_dispatch!(slice_expr)`
macro_rules! contract_pre_chat_template_dispatch {
    () => {{}};
    ($input:expr) => {{
        let _pv_state = &$input;
    }};
}

/// Postconditions for equation `chat_template_dispatch`.
/// Call before return: `contract_post_chat_template_dispatch!(result_expr)`
macro_rules! contract_post_chat_template_dispatch {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `chat_template_dispatch`.
macro_rules! contract_chat_template_dispatch {
    ($input:expr, $body:expr) => {{
        contract_pre_chat_template_dispatch!($input);
        let _contract_result = $body;
        contract_post_chat_template_dispatch!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `concurrent_inference_isolation`.
/// Call at function entry: `contract_pre_concurrent_inference_isolation!(input_expr)`
macro_rules! contract_pre_concurrent_inference_isolation {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `concurrent_inference_isolation`.
/// Call before return: `contract_post_concurrent_inference_isolation!(result_expr)`
macro_rules! contract_post_concurrent_inference_isolation {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `concurrent_inference_isolation`.
macro_rules! contract_concurrent_inference_isolation {
    ($input:expr, $body:expr) => {{
        contract_pre_concurrent_inference_isolation!($input);
        let _contract_result = $body;
        contract_post_concurrent_inference_isolation!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `format_detection`.
/// Call at function entry: `contract_pre_format_detection!(input_expr)`
macro_rules! contract_pre_format_detection {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `format_detection`.
/// Call before return: `contract_post_format_detection!(result_expr)`
macro_rules! contract_post_format_detection {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `format_detection`.
macro_rules! contract_format_detection {
    ($input:expr, $body:expr) => {{
        contract_pre_format_detection!($input);
        let _contract_result = $body;
        contract_post_format_detection!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `graceful_shutdown`.
/// Call at function entry: `contract_pre_graceful_shutdown!(input_expr)`
macro_rules! contract_pre_graceful_shutdown {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `graceful_shutdown`.
/// Call before return: `contract_post_graceful_shutdown!(result_expr)`
macro_rules! contract_post_graceful_shutdown {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `graceful_shutdown`.
macro_rules! contract_graceful_shutdown {
    ($input:expr, $body:expr) => {{
        contract_pre_graceful_shutdown!($input);
        let _contract_result = $body;
        contract_post_graceful_shutdown!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `request_routing`.
/// Call at function entry: `contract_pre_request_routing!(input_expr)`
macro_rules! contract_pre_request_routing {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `request_routing`.
/// Call before return: `contract_post_request_routing!(result_expr)`
macro_rules! contract_post_request_routing {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `request_routing`.
macro_rules! contract_request_routing {
    ($input:expr, $body:expr) => {{
        contract_pre_request_routing!($input);
        let _contract_result = $body;
        contract_post_request_routing!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `server_lifecycle`.
/// Domain-specific. Call: `contract_pre_server_lifecycle!(slice_expr)`
macro_rules! contract_pre_server_lifecycle {
    () => {{}};
    ($input:expr) => {{
        let _pv_config = &$input;
    }};
}

/// Postconditions for equation `server_lifecycle`.
/// Call before return: `contract_post_server_lifecycle!(result_expr)`
macro_rules! contract_post_server_lifecycle {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `server_lifecycle`.
macro_rules! contract_server_lifecycle {
    ($input:expr, $body:expr) => {{
        contract_pre_server_lifecycle!($input);
        let _contract_result = $body;
        contract_post_server_lifecycle!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/apr-training-parity-v1.yaml — DO NOT EDIT
// Contract: apr-training-parity-v1

/// Preconditions for equation `gpu_utilization_gate`.
/// Call at function entry: `contract_pre_gpu_utilization_gate!(input_expr)`
macro_rules! contract_pre_gpu_utilization_gate {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `parity_ratio`.
/// Call at function entry: `contract_pre_parity_ratio!(input_expr)`
macro_rules! contract_pre_parity_ratio {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

// Auto-generated from contracts/arch-constraints-v1.yaml — DO NOT EDIT
// Contract: arch-constraints-v1

/// Preconditions for equation `arch_constraint_lookup`.
/// Domain-specific. Call: `contract_pre_arch_constraint_lookup!(slice_expr)`
macro_rules! contract_pre_arch_constraint_lookup {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract arch_constraint_lookup: precondition violated — input.len() > 0");
    }};
}

// Auto-generated from contracts/architecture-requirements-v1.yaml — DO NOT EDIT
// Contract: architecture-requirements-v1

/// Preconditions for equation `constraint_matrix_exhaustiveness`.
/// Domain-specific. Call: `contract_pre_constraint_matrix_exhaustiveness!(slice_expr)`
macro_rules! contract_pre_constraint_matrix_exhaustiveness {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract constraint_matrix_exhaustiveness: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `role_mapping`.
/// Domain-specific. Call: `contract_pre_role_mapping!(slice_expr)`
macro_rules! contract_pre_role_mapping {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract role_mapping: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `weight_completeness`.
/// Domain-specific. Call: `contract_pre_weight_completeness!(slice_expr)`
macro_rules! contract_pre_weight_completeness {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract weight_completeness: precondition violated — input.len() > 0");
    }};
}

// Auto-generated from contracts/arima-v1.yaml — DO NOT EDIT
// Contract: arima-v1

/// Preconditions for equation `ar_forecast`.
/// Domain-specific. Call: `contract_pre_ar_forecast!(slice_expr)`
macro_rules! contract_pre_ar_forecast {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract ar_forecast: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract ar_forecast: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `differencing`.
/// Domain-specific. Call: `contract_pre_differencing!(slice_expr)`
macro_rules! contract_pre_differencing {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract differencing: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract differencing: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `forecast_finite`.
/// Domain-specific. Call: `contract_pre_forecast_finite!(slice_expr)`
macro_rules! contract_pre_forecast_finite {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract forecast_finite: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract forecast_finite: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `ma_filter`.
/// Domain-specific. Call: `contract_pre_ma_filter!(slice_expr)`
macro_rules! contract_pre_ma_filter {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract ma_filter: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract ma_filter: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

// Auto-generated from contracts/async-safety-v1.yaml — DO NOT EDIT
// Contract: async-safety-v1

/// Preconditions for equation `cancellation_safe`.
/// Call at function entry: `contract_pre_cancellation_safe!(input_expr)`
macro_rules! contract_pre_cancellation_safe {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `cancellation_safe`.
/// Call before return: `contract_post_cancellation_safe!(result_expr)`
macro_rules! contract_post_cancellation_safe {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `cancellation_safe`.
macro_rules! contract_cancellation_safe {
    ($input:expr, $body:expr) => {{
        contract_pre_cancellation_safe!($input);
        let _contract_result = $body;
        contract_post_cancellation_safe!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `channel_lossless`.
/// Call at function entry: `contract_pre_channel_lossless!(input_expr)`
macro_rules! contract_pre_channel_lossless {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `channel_lossless`.
/// Call before return: `contract_post_channel_lossless!(result_expr)`
macro_rules! contract_post_channel_lossless {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `channel_lossless`.
macro_rules! contract_channel_lossless {
    ($input:expr, $body:expr) => {{
        contract_pre_channel_lossless!($input);
        let _contract_result = $body;
        contract_post_channel_lossless!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `structured_spawn`.
/// Call at function entry: `contract_pre_structured_spawn!(input_expr)`
macro_rules! contract_pre_structured_spawn {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `structured_spawn`.
/// Call before return: `contract_post_structured_spawn!(result_expr)`
macro_rules! contract_post_structured_spawn {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `structured_spawn`.
macro_rules! contract_structured_spawn {
    ($input:expr, $body:expr) => {{
        contract_pre_structured_spawn!($input);
        let _contract_result = $body;
        contract_post_structured_spawn!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/attention-head-extraction-v1.yaml — DO NOT EDIT
// Contract: attention-head-extraction-v1

/// Preconditions for equation `extract_heads`.
/// Domain-specific. Call: `contract_pre_extract_heads!(slice_expr)`
macro_rules! contract_pre_extract_heads {
    () => {{}};
    ($input:expr) => {{
        let _pv_q = &$input;
        debug_assert!(_pv_q.len() > 0,
            "Contract extract_heads: precondition violated — q.len() > 0");
    }};
}

/// Postconditions for equation `extract_heads`.
/// Call before return: `contract_post_extract_heads!(result_expr)`
macro_rules! contract_post_extract_heads {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `extract_heads`.
macro_rules! contract_extract_heads {
    ($input:expr, $body:expr) => {{
        contract_pre_extract_heads!($input);
        let _contract_result = $body;
        contract_post_extract_heads!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/attention-kernel-v1.yaml — DO NOT EDIT
// Contract: attention-kernel-v1

/// Preconditions for equation `attention`.
/// Domain-specific. Call: `contract_pre_attention!(slice_expr)`
macro_rules! contract_pre_attention {
    () => {{}};
    ($input:expr) => {{
        let _pv_query = &$input;
        debug_assert!(_pv_query.len() > 0,
            "Contract attention: precondition violated — query.len() > 0");
    }};
}

/// Postconditions for equation `attention`.
/// Call before return: `contract_post_attention!(result_expr)`
macro_rules! contract_post_attention {
    ($result:expr) => {{
        let _contract_result = &$result;
        debug_assert!(_contract_result.iter().all(|v| v.is_finite()), "Contract attention: postcondition violated — result.iter().all(|v| v.is_finite())");
    }};
}

/// Combined pre+post contract for equation `attention`.
macro_rules! contract_attention {
    ($input:expr, $body:expr) => {{
        contract_pre_attention!($input);
        let _contract_result = $body;
        contract_post_attention!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/attention-kernel-v1.yaml — DO NOT EDIT
// Contract: attention-kernel-v1

/// Preconditions for equation `rmsnorm`.
/// Domain-specific. Call: `contract_pre_rmsnorm!(slice_expr)`
macro_rules! contract_pre_rmsnorm {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
    }};
}

/// Preconditions for equation `rope_rotation`.
/// Domain-specific. Call: `contract_pre_rope_rotation!(slice_expr)`
macro_rules! contract_pre_rope_rotation {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
        debug_assert!(_pv_x.len() % 2 == 0,
            "Contract rope_rotation: precondition violated — x.len() % 2 == 0");
    }};
}

/// Preconditions for equation `scaled_dot_product`.
/// Domain-specific. Call: `contract_pre_scaled_dot_product!(slice_expr)`
macro_rules! contract_pre_scaled_dot_product {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
    }};
}

// Auto-generated from contracts/attention-scaling-v1.yaml — DO NOT EDIT
// Contract: attention-scaling-v1

/// Preconditions for equation `attention_entropy`.
/// Domain-specific. Call: `contract_pre_attention_entropy!(slice_expr)`
macro_rules! contract_pre_attention_entropy {
    () => {{}};
    ($input:expr) => {{
        let _pv_q = &$input;
        debug_assert!(_pv_q.len() > 0,
            "Contract attention_entropy: precondition violated — q.len() > 0");
    }};
}

/// Preconditions for equation `numerical_stability`.
/// Domain-specific. Call: `contract_pre_numerical_stability!(slice_expr)`
macro_rules! contract_pre_numerical_stability {
    () => {{}};
    ($input:expr) => {{
        let _pv_q = &$input;
        debug_assert!(_pv_q.len() > 0,
            "Contract numerical_stability: precondition violated — q.len() > 0");
    }};
}

/// Preconditions for equation `scaled_dot_product`.
/// Domain-specific. Call: `contract_pre_scaled_dot_product!(slice_expr)`
macro_rules! contract_pre_scaled_dot_product {
    () => {{}};
    ($input:expr) => {{
        let _pv_a = &$input;
        debug_assert!(_pv_a.len() > 0,
            "Contract scaled_dot_product: precondition violated — a.len() > 0");
    }};
}

/// Preconditions for equation `score_bound_with_qknorm`.
/// Domain-specific. Call: `contract_pre_score_bound_with_qknorm!(slice_expr)`
macro_rules! contract_pre_score_bound_with_qknorm {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract score_bound_with_qknorm: precondition violated — input.iter().all(|v| v.is_finite())");
        debug_assert!(_pv_input.len() > 0,
            "Contract score_bound_with_qknorm: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `softmax_saturation`.
/// Domain-specific. Call: `contract_pre_softmax_saturation!(slice_expr)`
macro_rules! contract_pre_softmax_saturation {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
        debug_assert!(_pv_x.iter().all(|v| v.is_finite()),
            "Contract softmax_saturation: precondition violated — x.iter().all(|v| v.is_finite())");
        debug_assert!(_pv_x.len() > 0,
            "Contract softmax_saturation: precondition violated — x.len() > 0");
    }};
}

/// Preconditions for equation `variance_preservation`.
/// Domain-specific. Call: `contract_pre_variance_preservation!(slice_expr)`
macro_rules! contract_pre_variance_preservation {
    () => {{}};
    ($input:expr) => {{
        let _pv_q = &$input;
        debug_assert!(_pv_q.len() > 0,
            "Contract variance_preservation: precondition violated — q.len() > 0");
    }};
}

// Auto-generated from contracts/avx2-fma-dot-v1.yaml — DO NOT EDIT
// Contract: avx2-fma-dot-v1

/// Preconditions for equation `dot_product`.
/// Domain-specific. Call: `contract_pre_dot_product!(slice_expr)`
macro_rules! contract_pre_dot_product {
    () => {{}};
    ($input:expr) => {{
        let _pv_a = &$input;
        debug_assert!(_pv_a.len() > 0,
            "Contract dot_product: precondition violated — a.len() > 0");
    }};
}

/// Preconditions for equation `fma_accumulation`.
/// Domain-specific. Call: `contract_pre_fma_accumulation!(slice_expr)`
macro_rules! contract_pre_fma_accumulation {
    () => {{}};
    ($input:expr) => {{
        let _pv_a = &$input;
        debug_assert!(_pv_a.len() > 0,
            "Contract fma_accumulation: precondition violated — a.len() > 0");
    }};
}

// Auto-generated from contracts/backend-dispatch-v1.yaml — DO NOT EDIT
// Contract: backend-dispatch-v1

/// Preconditions for equation `garbage_oracle`.
/// Domain-specific. Call: `contract_pre_garbage_oracle!(slice_expr)`
macro_rules! contract_pre_garbage_oracle {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract garbage_oracle: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract garbage_oracle: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `gpu_threshold`.
/// Domain-specific. Call: `contract_pre_gpu_threshold!(slice_expr)`
macro_rules! contract_pre_gpu_threshold {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract gpu_threshold: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract gpu_threshold: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `qk_norm_score_bound`.
/// Domain-specific. Call: `contract_pre_qk_norm_score_bound!(slice_expr)`
macro_rules! contract_pre_qk_norm_score_bound {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract qk_norm_score_bound: precondition violated — input.iter().all(|v| v.is_finite())");
        debug_assert!(_pv_input.len() > 0,
            "Contract qk_norm_score_bound: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `simd_only_threshold`.
/// Domain-specific. Call: `contract_pre_simd_only_threshold!(slice_expr)`
macro_rules! contract_pre_simd_only_threshold {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract simd_only_threshold: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract simd_only_threshold: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

// Auto-generated from contracts/batch-training-v1.yaml — DO NOT EDIT
// Contract: batch-training-v1

/// Preconditions for equation `batch_loss`.
/// Domain-specific. Call: `contract_pre_batch_loss!(slice_expr)`
macro_rules! contract_pre_batch_loss {
    () => {{}};
    ($input:expr) => {{
        let _pv_predicted = &$input;
        debug_assert!(_pv_predicted.len() > 0,
            "Contract batch_loss: precondition violated — predicted.len() > 0");
    }};
}

/// Preconditions for equation `gradient_accumulation`.
/// Domain-specific. Call: `contract_pre_gradient_accumulation!(slice_expr)`
macro_rules! contract_pre_gradient_accumulation {
    () => {{}};
    ($input:expr) => {{
        let _pv_params = &$input;
        debug_assert!(_pv_params.len() > 0,
            "Contract gradient_accumulation: precondition violated — params.len() > 0");
    }};
}

/// Preconditions for equation `gradient_clipping`.
/// Domain-specific. Call: `contract_pre_gradient_clipping!(slice_expr)`
macro_rules! contract_pre_gradient_clipping {
    () => {{}};
    ($input:expr) => {{
        let _pv_params = &$input;
        debug_assert!(_pv_params.len() > 0,
            "Contract gradient_clipping: precondition violated — params.len() > 0");
    }};
}

// Auto-generated from contracts/batched-beam-search-v1.yaml — DO NOT EDIT
// Contract: batched-beam-search-v1

/// Preconditions for equation `batched_beam_projection`.
/// Domain-specific. Call: `contract_pre_batched_beam_projection!(slice_expr)`
macro_rules! contract_pre_batched_beam_projection {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract batched_beam_projection: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `beam_selection`.
/// Domain-specific. Call: `contract_pre_beam_selection!(slice_expr)`
macro_rules! contract_pre_beam_selection {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract beam_selection: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `sequential_beam_projection`.
/// Domain-specific. Call: `contract_pre_sequential_beam_projection!(slice_expr)`
macro_rules! contract_pre_sequential_beam_projection {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract sequential_beam_projection: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `termination`.
/// Domain-specific. Call: `contract_pre_termination!(slice_expr)`
macro_rules! contract_pre_termination {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract termination: precondition violated — input.len() > 0");
    }};
}

// Auto-generated from contracts/batchnorm-kernel-v1.yaml — DO NOT EDIT
// Contract: batchnorm-kernel-v1

/// Preconditions for equation `batchnorm_eval`.
/// Domain-specific. Call: `contract_pre_batchnorm_eval!(slice_expr)`
macro_rules! contract_pre_batchnorm_eval {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract batchnorm_eval: precondition violated — input.iter().all(|v| v.is_finite())");
        debug_assert!(_pv_input.len() > 0,
            "Contract batchnorm_eval: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `batchnorm_train`.
/// Domain-specific. Call: `contract_pre_batchnorm_train!(slice_expr)`
macro_rules! contract_pre_batchnorm_train {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract batchnorm_train: precondition violated — input.iter().all(|v| v.is_finite())");
        debug_assert!(_pv_input.len() > 0,
            "Contract batchnorm_train: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `running_stats`.
/// Domain-specific. Call: `contract_pre_running_stats!(slice_expr)`
macro_rules! contract_pre_running_stats {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract running_stats: precondition violated — input.iter().all(|v| v.is_finite())");
        debug_assert!(_pv_input.len() > 0,
            "Contract running_stats: precondition violated — input.len() > 0");
    }};
}

// Auto-generated from contracts/bayesian-v1.yaml — DO NOT EDIT
// Contract: bayesian-v1

/// Preconditions for equation `blr_predict`.
/// Domain-specific. Call: `contract_pre_blr_predict!(slice_expr)`
macro_rules! contract_pre_blr_predict {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract blr_predict: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract blr_predict: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `conjugate_update`.
/// Domain-specific. Call: `contract_pre_conjugate_update!(slice_expr)`
macro_rules! contract_pre_conjugate_update {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract conjugate_update: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract conjugate_update: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `posterior_predictive`.
/// Domain-specific. Call: `contract_pre_posterior_predictive!(slice_expr)`
macro_rules! contract_pre_posterior_predictive {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract posterior_predictive: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract posterior_predictive: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `posterior_valid`.
/// Domain-specific. Call: `contract_pre_posterior_valid!(slice_expr)`
macro_rules! contract_pre_posterior_valid {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract posterior_valid: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract posterior_valid: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

// Auto-generated from contracts/beacon-dispatch-v1.yaml — DO NOT EDIT
// Contract: beacon-dispatch-v1

/// Preconditions for equation `bm25_ranking`.
/// Domain-specific. Call: `contract_pre_bm25_ranking!(slice_expr)`
macro_rules! contract_pre_bm25_ranking {
    () => {{}};
    ($input:expr) => {{
        let _pv_0 = &$input;
    }};
}

/// Preconditions for equation `index_insert_retrieve`.
/// Domain-specific. Call: `contract_pre_index_insert_retrieve!(slice_expr)`
macro_rules! contract_pre_index_insert_retrieve {
    () => {{}};
    ($input:expr) => {{
        let _pv_doc = &$input;
    }};
}

/// Preconditions for equation `robots_compliance`.
/// Domain-specific. Call: `contract_pre_robots_compliance!(slice_expr)`
macro_rules! contract_pre_robots_compliance {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
    }};
}

/// Preconditions for equation `tokenize_normalization`.
/// Call at function entry: `contract_pre_tokenize_normalization!(input_expr)`
macro_rules! contract_pre_tokenize_normalization {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `tokenize_normalization`.
/// Call before return: `contract_post_tokenize_normalization!(result_expr)`
macro_rules! contract_post_tokenize_normalization {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `tokenize_normalization`.
macro_rules! contract_tokenize_normalization {
    ($input:expr, $body:expr) => {{
        contract_pre_tokenize_normalization!($input);
        let _contract_result = $body;
        contract_post_tokenize_normalization!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/bias-add-v1.yaml — DO NOT EDIT
// Contract: bias-add-v1

/// Preconditions for equation `bias_add`.
/// Domain-specific. Call: `contract_pre_bias_add!(slice_expr)`
macro_rules! contract_pre_bias_add {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract bias_add: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract bias_add: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

// Auto-generated from contracts/bidirectional-attention-v1.yaml — DO NOT EDIT
// Contract: bidirectional-attention-v1

/// Preconditions for equation `bidirectional_attention`.
/// Domain-specific. Call: `contract_pre_bidirectional_attention!(slice_expr)`
macro_rules! contract_pre_bidirectional_attention {
    () => {{}};
    ($input:expr) => {{
        let _pv_q = &$input;
        debug_assert!(_pv_q.len() > 0,
            "Contract bidirectional_attention: precondition violated — q.len() > 0");
    }};
}

// Auto-generated from contracts/blake3-state-v1.yaml — DO NOT EDIT
// Contract: blake3-state-v1

/// Preconditions for equation `composite_hash`.
/// Domain-specific. Call: `contract_pre_composite_hash!(slice_expr)`
macro_rules! contract_pre_composite_hash {
    () => {{}};
    ($input:expr) => {{
        let _pv_parts = &$input;
        debug_assert!(_pv_parts.len() > 0,
            "Contract composite_hash: precondition violated — parts.len() > 0");
    }};
}

/// Preconditions for equation `hash_file`.
/// Call at function entry: `contract_pre_hash_file!(input_expr)`
macro_rules! contract_pre_hash_file {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `hash_string`.
/// Domain-specific. Call: `contract_pre_hash_string!(slice_expr)`
macro_rules! contract_pre_hash_string {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(!_pv_input.is_empty(),
            "Contract hash_string: precondition violated — !input.is_empty()");
        debug_assert!(_pv_input.len() <= 1_073_741_824,
            "Contract hash_string: precondition violated — input.len() <= 1_073_741_824");
    }};
}

// Auto-generated from contracts/bpe-tokenization-v1.yaml — DO NOT EDIT
// Contract: bpe-tokenization-v1

/// Preconditions for equation `decode`.
/// Domain-specific. Call: `contract_pre_decode!(slice_expr)`
macro_rules! contract_pre_decode {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract decode: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `encode`.
/// Domain-specific. Call: `contract_pre_encode!(slice_expr)`
macro_rules! contract_pre_encode {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract encode: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `merge_rule`.
/// Domain-specific. Call: `contract_pre_merge_rule!(slice_expr)`
macro_rules! contract_pre_merge_rule {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract merge_rule: precondition violated — input.len() > 0");
    }};
}

// Auto-generated from contracts/builder-pattern-v1.yaml — DO NOT EDIT
// Contract: builder-pattern-v1

/// Preconditions for equation `builder_pattern`.
/// Call at function entry: `contract_pre_builder_pattern!(input_expr)`
macro_rules! contract_pre_builder_pattern {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `builder_pattern`.
/// Call before return: `contract_post_builder_pattern!(result_expr)`
macro_rules! contract_post_builder_pattern {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `builder_pattern`.
macro_rules! contract_builder_pattern {
    ($input:expr, $body:expr) => {{
        contract_pre_builder_pattern!($input);
        let _contract_result = $body;
        contract_post_builder_pattern!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/builder-pattern-v1.yaml — DO NOT EDIT
// Contract: builder-pattern-v1

/// Preconditions for equation `build`.
/// Domain-specific. Call: `contract_pre_build!(slice_expr)`
macro_rules! contract_pre_build {
    () => {{}};
    ($input:expr) => {{
        let _pv_builder = &$input;
    }};
}

// Auto-generated from contracts/calibration-v1.yaml — DO NOT EDIT
// Contract: calibration-v1

/// Preconditions for equation `expected_calibration_error`.
/// Domain-specific. Call: `contract_pre_expected_calibration_error!(slice_expr)`
macro_rules! contract_pre_expected_calibration_error {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract expected_calibration_error: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract expected_calibration_error: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `isotonic_regression`.
/// Domain-specific. Call: `contract_pre_isotonic_regression!(slice_expr)`
macro_rules! contract_pre_isotonic_regression {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract isotonic_regression: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract isotonic_regression: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `maximum_calibration_error`.
/// Domain-specific. Call: `contract_pre_maximum_calibration_error!(slice_expr)`
macro_rules! contract_pre_maximum_calibration_error {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract maximum_calibration_error: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract maximum_calibration_error: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `platt_scaling`.
/// Domain-specific. Call: `contract_pre_platt_scaling!(slice_expr)`
macro_rules! contract_pre_platt_scaling {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract platt_scaling: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract platt_scaling: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `reliability_diagram`.
/// Domain-specific. Call: `contract_pre_reliability_diagram!(slice_expr)`
macro_rules! contract_pre_reliability_diagram {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract reliability_diagram: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract reliability_diagram: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

// Auto-generated from contracts/canary-metrics-schema-v1.yaml — DO NOT EDIT
// Contract: canary-metrics-schema-v1

/// Preconditions for equation `domain_loss`.
/// Call at function entry: `contract_pre_domain_loss!(input_expr)`
macro_rules! contract_pre_domain_loss {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `domain_throughput`.
/// Call at function entry: `contract_pre_domain_throughput!(input_expr)`
macro_rules! contract_pre_domain_throughput {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `schema_completeness`.
/// Domain-specific. Call: `contract_pre_schema_completeness!(slice_expr)`
macro_rules! contract_pre_schema_completeness {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
    }};
}

// Auto-generated from contracts/canary-score-gate-v1.yaml — DO NOT EDIT
// Contract: canary-score-gate-v1

/// Preconditions for equation `parity_gate`.
/// Call at function entry: `contract_pre_parity_gate!(input_expr)`
macro_rules! contract_pre_parity_gate {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `throughput_gate`.
/// Call at function entry: `contract_pre_throughput_gate!(input_expr)`
macro_rules! contract_pre_throughput_gate {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `vram_gate`.
/// Call at function entry: `contract_pre_vram_gate!(input_expr)`
macro_rules! contract_pre_vram_gate {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

// Auto-generated from contracts/chat-template-v1.yaml — DO NOT EDIT
// Contract: chat-template-v1

/// Preconditions for equation `appstate_architecture_cache`.
/// Domain-specific. Call: `contract_pre_appstate_architecture_cache!(slice_expr)`
macro_rules! contract_pre_appstate_architecture_cache {
    () => {{}};
    ($input:expr) => {{
        let _pv_quantized_model = &$input;
    }};
}

/// Postconditions for equation `appstate_architecture_cache`.
/// Call before return: `contract_post_appstate_architecture_cache!(result_expr)`
macro_rules! contract_post_appstate_architecture_cache {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `appstate_architecture_cache`.
macro_rules! contract_appstate_architecture_cache {
    ($input:expr, $body:expr) => {{
        contract_pre_appstate_architecture_cache!($input);
        let _contract_result = $body;
        contract_post_appstate_architecture_cache!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `architecture_aware_selection`.
/// Call at function entry: `contract_pre_architecture_aware_selection!(input_expr)`
macro_rules! contract_pre_architecture_aware_selection {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `architecture_aware_selection`.
/// Call before return: `contract_post_architecture_aware_selection!(result_expr)`
macro_rules! contract_post_architecture_aware_selection {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `architecture_aware_selection`.
macro_rules! contract_architecture_aware_selection {
    ($input:expr, $body:expr) => {{
        contract_pre_architecture_aware_selection!($input);
        let _contract_result = $body;
        contract_post_architecture_aware_selection!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `format_conversation_determinism`.
/// Domain-specific. Call: `contract_pre_format_conversation_determinism!(slice_expr)`
macro_rules! contract_pre_format_conversation_determinism {
    () => {{}};
    ($input:expr) => {{
        let _pv_messages = &$input;
        debug_assert!(_pv_messages.len() >= 0,
            "Contract format_conversation_determinism: precondition violated — messages.len() >= 0");
    }};
}

/// Postconditions for equation `format_conversation_determinism`.
/// Call before return: `contract_post_format_conversation_determinism!(result_expr)`
macro_rules! contract_post_format_conversation_determinism {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `format_conversation_determinism`.
macro_rules! contract_format_conversation_determinism {
    ($input:expr, $body:expr) => {{
        contract_pre_format_conversation_determinism!($input);
        let _contract_result = $body;
        contract_post_format_conversation_determinism!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `thinking_block_suppression`.
/// Call at function entry: `contract_pre_thinking_block_suppression!(input_expr)`
macro_rules! contract_pre_thinking_block_suppression {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `thinking_block_suppression`.
/// Call before return: `contract_post_thinking_block_suppression!(result_expr)`
macro_rules! contract_post_thinking_block_suppression {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `thinking_block_suppression`.
macro_rules! contract_thinking_block_suppression {
    ($input:expr, $body:expr) => {{
        contract_pre_thinking_block_suppression!($input);
        let _contract_result = $body;
        contract_post_thinking_block_suppression!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `trait_completeness`.
/// Call at function entry: `contract_pre_trait_completeness!(input_expr)`
macro_rules! contract_pre_trait_completeness {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `trait_completeness`.
/// Call before return: `contract_post_trait_completeness!(result_expr)`
macro_rules! contract_post_trait_completeness {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `trait_completeness`.
macro_rules! contract_trait_completeness {
    ($input:expr, $body:expr) => {{
        contract_pre_trait_completeness!($input);
        let _contract_result = $body;
        contract_post_trait_completeness!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/classification-finetune-v1.yaml — DO NOT EDIT
// Contract: classification-finetune-v1

/// Preconditions for equation `classifier_weight_shape`.
/// Domain-specific. Call: `contract_pre_classifier_weight_shape!(slice_expr)`
macro_rules! contract_pre_classifier_weight_shape {
    () => {{}};
    ($input:expr) => {{
        let _pv_a = &$input;
        debug_assert!(_pv_a.len() > 0,
            "Contract classifier_weight_shape: precondition violated — a.len() > 0");
    }};
}

/// Preconditions for equation `label_bounds`.
/// Domain-specific. Call: `contract_pre_label_bounds!(slice_expr)`
macro_rules! contract_pre_label_bounds {
    () => {{}};
    ($input:expr) => {{
        let _pv_a = &$input;
        debug_assert!(_pv_a.len() > 0,
            "Contract label_bounds: precondition violated — a.len() > 0");
    }};
}

/// Preconditions for equation `logit_shape`.
/// Domain-specific. Call: `contract_pre_logit_shape!(slice_expr)`
macro_rules! contract_pre_logit_shape {
    () => {{}};
    ($input:expr) => {{
        let _pv_a = &$input;
        debug_assert!(_pv_a.len() > 0,
            "Contract logit_shape: precondition violated — a.len() > 0");
    }};
}

/// Preconditions for equation `softmax_sum`.
/// Domain-specific. Call: `contract_pre_softmax_sum!(slice_expr)`
macro_rules! contract_pre_softmax_sum {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
        debug_assert!(_pv_x.iter().all(|v| v.is_finite()),
            "Contract softmax_sum: precondition violated — x.iter().all(|v| v.is_finite())");
        debug_assert!(_pv_x.len() > 0,
            "Contract softmax_sum: precondition violated — x.len() > 0");
    }};
}

// Auto-generated from contracts/classifier-pipeline-v1.yaml — DO NOT EDIT
// Contract: classifier-pipeline-v1

/// Preconditions for equation `embedding_extraction`.
/// Domain-specific. Call: `contract_pre_embedding_extraction!(slice_expr)`
macro_rules! contract_pre_embedding_extraction {
    () => {{}};
    ($input:expr) => {{
        let _pv_indices = &$input;
        debug_assert!(_pv_indices.len() > 0,
            "Contract embedding_extraction: precondition violated — indices.len() > 0");
    }};
}

/// Preconditions for equation `evaluation`.
/// Call at function entry: `contract_pre_evaluation!(input_expr)`
macro_rules! contract_pre_evaluation {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `linear_probe`.
/// Call at function entry: `contract_pre_linear_probe!(input_expr)`
macro_rules! contract_pre_linear_probe {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

// Auto-generated from contracts/cleanup-safety-v1.yaml — DO NOT EDIT
// Contract: cleanup-safety-v1

/// Preconditions for equation `duplicate_detection`.
/// Call at function entry: `contract_pre_duplicate_detection!(input_expr)`
macro_rules! contract_pre_duplicate_detection {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `outlier_detection`.
/// Call at function entry: `contract_pre_outlier_detection!(input_expr)`
macro_rules! contract_pre_outlier_detection {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `scan_completeness`.
/// Call at function entry: `contract_pre_scan_completeness!(input_expr)`
macro_rules! contract_pre_scan_completeness {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

// Auto-generated from contracts/cli-dispatch-v1.yaml — DO NOT EDIT
// Contract: cli-dispatch-v1

/// Preconditions for equation `dispatch_completeness`.
/// Domain-specific. Call: `contract_pre_dispatch_completeness!(slice_expr)`
macro_rules! contract_pre_dispatch_completeness {
    () => {{}};
    ($input:expr) => {{
        let _pv_args = &$input;
    }};
}

/// Preconditions for equation `exit_code_semantics`.
/// Call at function entry: `contract_pre_exit_code_semantics!(input_expr)`
macro_rules! contract_pre_exit_code_semantics {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `feature_gated_dispatch`.
/// Call at function entry: `contract_pre_feature_gated_dispatch!(input_expr)`
macro_rules! contract_pre_feature_gated_dispatch {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `feature_gated_dispatch`.
/// Call before return: `contract_post_feature_gated_dispatch!(result_expr)`
macro_rules! contract_post_feature_gated_dispatch {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `feature_gated_dispatch`.
macro_rules! contract_feature_gated_dispatch {
    ($input:expr, $body:expr) => {{
        contract_pre_feature_gated_dispatch!($input);
        let _contract_result = $body;
        contract_post_feature_gated_dispatch!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `idempotent_inspection`.
/// Call at function entry: `contract_pre_idempotent_inspection!(input_expr)`
macro_rules! contract_pre_idempotent_inspection {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `output_format_fidelity`.
/// Call at function entry: `contract_pre_output_format_fidelity!(input_expr)`
macro_rules! contract_pre_output_format_fidelity {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

// Auto-generated from contracts/cli-interface-v1.yaml — DO NOT EDIT
// Contract: cli-interface-v1

/// Preconditions for equation `exit_code_semantics`.
/// Call at function entry: `contract_pre_exit_code_semantics!(input_expr)`
macro_rules! contract_pre_exit_code_semantics {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `exit_code_semantics`.
/// Call before return: `contract_post_exit_code_semantics!(result_expr)`
macro_rules! contract_post_exit_code_semantics {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `exit_code_semantics`.
macro_rules! contract_exit_code_semantics {
    ($input:expr, $body:expr) => {{
        contract_pre_exit_code_semantics!($input);
        let _contract_result = $body;
        contract_post_exit_code_semantics!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `output_format_fidelity`.
/// Call at function entry: `contract_pre_output_format_fidelity!(input_expr)`
macro_rules! contract_pre_output_format_fidelity {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `output_format_fidelity`.
/// Call before return: `contract_post_output_format_fidelity!(result_expr)`
macro_rules! contract_post_output_format_fidelity {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `output_format_fidelity`.
macro_rules! contract_output_format_fidelity {
    ($input:expr, $body:expr) => {{
        contract_pre_output_format_fidelity!($input);
        let _contract_result = $body;
        contract_post_output_format_fidelity!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `result_cardinality`.
/// Call at function entry: `contract_pre_result_cardinality!(input_expr)`
macro_rules! contract_pre_result_cardinality {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `result_cardinality`.
/// Call before return: `contract_post_result_cardinality!(result_expr)`
macro_rules! contract_post_result_cardinality {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `result_cardinality`.
macro_rules! contract_result_cardinality {
    ($input:expr, $body:expr) => {{
        contract_pre_result_cardinality!($input);
        let _contract_result = $body;
        contract_post_result_cardinality!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `timeout_honoring`.
/// Call at function entry: `contract_pre_timeout_honoring!(input_expr)`
macro_rules! contract_pre_timeout_honoring {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `timeout_honoring`.
/// Call before return: `contract_post_timeout_honoring!(result_expr)`
macro_rules! contract_post_timeout_honoring {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `timeout_honoring`.
macro_rules! contract_timeout_honoring {
    ($input:expr, $body:expr) => {{
        contract_pre_timeout_honoring!($input);
        let _contract_result = $body;
        contract_post_timeout_honoring!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/cli-lint-v1.yaml — DO NOT EDIT
// Contract: cli-lint-v1

/// Preconditions for equation `exit_code_dispatch`.
/// Domain-specific. Call: `contract_pre_exit_code_dispatch!(slice_expr)`
macro_rules! contract_pre_exit_code_dispatch {
    () => {{}};
    ($input:expr) => {{
        let _pv_args = &$input;
        debug_assert!(_pv_args.len() >= 2,
            "Contract exit_code_dispatch: precondition violated — args.len() >= 2");
        debug_assert!(_pv_args[0] == "lint",
            "Contract exit_code_dispatch: precondition violated — args[0] == \"lint\"");
    }};
}

/// Preconditions for equation `finding_determinism`.
/// Call at function entry: `contract_pre_finding_determinism!(input_expr)`
macro_rules! contract_pre_finding_determinism {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `output_format_validity`.
/// Call at function entry: `contract_pre_output_format_validity!(input_expr)`
macro_rules! contract_pre_output_format_validity {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `severity_ordering`.
/// Domain-specific. Call: `contract_pre_severity_ordering!(slice_expr)`
macro_rules! contract_pre_severity_ordering {
    () => {{}};
    ($input:expr) => {{
        let _pv_diagnostics = &$input;
        debug_assert!(_pv_diagnostics.len() >= 0,
            "Contract severity_ordering: precondition violated — diagnostics.len() >= 0");
    }};
}

// Auto-generated from contracts/cli-oracle-v1.yaml — DO NOT EDIT
// Contract: cli-oracle-v1

/// Preconditions for equation `dispatch_correctness`.
/// Call at function entry: `contract_pre_dispatch_correctness!(input_expr)`
macro_rules! contract_pre_dispatch_correctness {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `index_freshness`.
/// Call at function entry: `contract_pre_index_freshness!(input_expr)`
macro_rules! contract_pre_index_freshness {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `rag_query_correctness`.
/// Domain-specific. Call: `contract_pre_rag_query_correctness!(slice_expr)`
macro_rules! contract_pre_rag_query_correctness {
    () => {{}};
    ($input:expr) => {{
        let _pv_query = &$input;
        debug_assert!(_pv_query.len() > 0,
            "Contract rag_query_correctness: precondition violated — query.len() > 0");
    }};
}

// Auto-generated from contracts/cli-transpile-v1.yaml — DO NOT EDIT
// Contract: cli-transpile-v1

/// Preconditions for equation `exit_code_dispatch`.
/// Domain-specific. Call: `contract_pre_exit_code_dispatch!(slice_expr)`
macro_rules! contract_pre_exit_code_dispatch {
    () => {{}};
    ($input:expr) => {{
        let _pv_args = &$input;
        debug_assert!(_pv_args.len() >= 2,
            "Contract exit_code_dispatch: precondition violated — args.len() >= 2");
        debug_assert!(_pv_args[0] == "transpile",
            "Contract exit_code_dispatch: precondition violated — args[0] == \"transpile\"");
    }};
}

/// Preconditions for equation `input_validation`.
/// Call at function entry: `contract_pre_input_validation!(input_expr)`
macro_rules! contract_pre_input_validation {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `output_validity`.
/// Domain-specific. Call: `contract_pre_output_validity!(slice_expr)`
macro_rules! contract_pre_output_validity {
    () => {{}};
    ($input:expr) => {{
        let _pv_rust_source = &$input;
        debug_assert!(!_pv_rust_source.is_empty(),
            "Contract output_validity: precondition violated — !rust_source.is_empty()");
        debug_assert!(_pv_rust_source.len() <= 10_000_000,
            "Contract output_validity: precondition violated — rust_source.len() <= 10_000_000");
    }};
}

/// Preconditions for equation `transpilation_determinism`.
/// Call at function entry: `contract_pre_transpilation_determinism!(input_expr)`
macro_rules! contract_pre_transpilation_determinism {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

// Auto-generated from contracts/cma-es-kernel-v1.yaml — DO NOT EDIT
// Contract: cma-es-kernel-v1

/// Preconditions for equation `covariance_update`.
/// Domain-specific. Call: `contract_pre_covariance_update!(slice_expr)`
macro_rules! contract_pre_covariance_update {
    () => {{}};
    ($input:expr) => {{
        let _pv_params = &$input;
        debug_assert!(_pv_params.len() > 0,
            "Contract covariance_update: precondition violated — params.len() > 0");
    }};
}

/// Preconditions for equation `mean_update`.
/// Domain-specific. Call: `contract_pre_mean_update!(slice_expr)`
macro_rules! contract_pre_mean_update {
    () => {{}};
    ($input:expr) => {{
        let _pv_params = &$input;
        debug_assert!(_pv_params.len() > 0,
            "Contract mean_update: precondition violated — params.len() > 0");
    }};
}

/// Preconditions for equation `sample`.
/// Domain-specific. Call: `contract_pre_sample!(slice_expr)`
macro_rules! contract_pre_sample {
    () => {{}};
    ($input:expr) => {{
        let _pv_params = &$input;
        debug_assert!(_pv_params.len() > 0,
            "Contract sample: precondition violated — params.len() > 0");
    }};
}

// Auto-generated from contracts/codebert-tokenizer-validation-v1.yaml — DO NOT EDIT
// Contract: codebert-tokenizer-validation-v1

/// Preconditions for equation `tokenizer_adequacy`.
/// Domain-specific. Call: `contract_pre_tokenizer_adequacy!(slice_expr)`
macro_rules! contract_pre_tokenizer_adequacy {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract tokenizer_adequacy: precondition violated — input.len() > 0");
    }};
}

// Auto-generated from contracts/codegen-dispatch-v1.yaml — DO NOT EDIT
// Contract: codegen-dispatch-v1

/// Preconditions for equation `apply_script`.
/// Call at function entry: `contract_pre_apply_script!(input_expr)`
macro_rules! contract_pre_apply_script {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `check_script`.
/// Call at function entry: `contract_pre_check_script!(input_expr)`
macro_rules! contract_pre_check_script {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `state_query_script`.
/// Call at function entry: `contract_pre_state_query_script!(input_expr)`
macro_rules! contract_pre_state_query_script {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

// Auto-generated from contracts/columnar-storage-v1.yaml — DO NOT EDIT
// Contract: columnar-storage-v1

/// Preconditions for equation `insert_get_consistency`.
/// Call at function entry: `contract_pre_insert_get_consistency!(input_expr)`
macro_rules! contract_pre_insert_get_consistency {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `query_correctness`.
/// Call at function entry: `contract_pre_query_correctness!(input_expr)`
macro_rules! contract_pre_query_correctness {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `wasm_parity`.
/// Call at function entry: `contract_pre_wasm_parity!(input_expr)`
macro_rules! contract_pre_wasm_parity {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

// Auto-generated from contracts/comply-check-v1.yaml — DO NOT EDIT
// Contract: comply-check-v1

/// Preconditions for equation `aggregate_score`.
/// Domain-specific. Call: `contract_pre_aggregate_score!(slice_expr)`
macro_rules! contract_pre_aggregate_score {
    () => {{}};
    ($input:expr) => {{
        let _pv_checks = &$input;
        debug_assert!(_pv_checks.len() > 0,
            "Contract aggregate_score: precondition violated — checks.len() > 0");
    }};
}

/// Postconditions for equation `aggregate_score`.
/// Call before return: `contract_post_aggregate_score!(result_expr)`
macro_rules! contract_post_aggregate_score {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `aggregate_score`.
macro_rules! contract_aggregate_score {
    ($input:expr, $body:expr) => {{
        contract_pre_aggregate_score!($input);
        let _contract_result = $body;
        contract_post_aggregate_score!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `run_checks`.
/// Call at function entry: `contract_pre_run_checks!(input_expr)`
macro_rules! contract_pre_run_checks {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `run_checks`.
/// Call before return: `contract_post_run_checks!(result_expr)`
macro_rules! contract_post_run_checks {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `run_checks`.
macro_rules! contract_run_checks {
    ($input:expr, $body:expr) => {{
        contract_pre_run_checks!($input);
        let _contract_result = $body;
        contract_post_run_checks!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/compression-codec-v1.yaml — DO NOT EDIT
// Contract: compression-codec-v1

/// Preconditions for equation `batch_correctness`.
/// Domain-specific. Call: `contract_pre_batch_correctness!(slice_expr)`
macro_rules! contract_pre_batch_correctness {
    () => {{}};
    ($input:expr) => {{
        let _pv_B = &$input;
        debug_assert!(_pv_B.len() > 0,
            "Contract batch_correctness: precondition violated — B.len() > 0");
    }};
}

/// Preconditions for equation `roundtrip_identity`.
/// Domain-specific. Call: `contract_pre_roundtrip_identity!(slice_expr)`
macro_rules! contract_pre_roundtrip_identity {
    () => {{}};
    ($input:expr) => {{
        let _pv_data = &$input;
        debug_assert!(!_pv_data.is_empty(),
            "Contract roundtrip_identity: precondition violated — !data.is_empty()");
    }};
}

/// Preconditions for equation `simd_scalar_parity`.
/// Domain-specific. Call: `contract_pre_simd_scalar_parity!(slice_expr)`
macro_rules! contract_pre_simd_scalar_parity {
    () => {{}};
    ($input:expr) => {{
        let _pv_data = &$input;
        debug_assert!(_pv_data.len() > 0,
            "Contract simd_scalar_parity: precondition violated — data.len() > 0");
    }};
}

// Auto-generated from contracts/compression-roundtrip-v1.yaml — DO NOT EDIT
// Contract: compression-roundtrip-v1

/// Preconditions for equation `compression_ratio`.
/// Domain-specific. Call: `contract_pre_compression_ratio!(slice_expr)`
macro_rules! contract_pre_compression_ratio {
    () => {{}};
    ($input:expr) => {{
        let _pv_data = &$input;
        debug_assert!(_pv_data.len() > 0,
            "Contract compression_ratio: precondition violated — data.len() > 0");
    }};
}

/// Preconditions for equation `page_state`.
/// Call at function entry: `contract_pre_page_state!(input_expr)`
macro_rules! contract_pre_page_state {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `roundtrip_identity`.
/// Domain-specific. Call: `contract_pre_roundtrip_identity!(slice_expr)`
macro_rules! contract_pre_roundtrip_identity {
    () => {{}};
    ($input:expr) => {{
        let _pv_data = &$input;
        debug_assert!(_pv_data.len() > 0,
            "Contract roundtrip_identity: precondition violated — data.len() > 0");
    }};
}

// Auto-generated from contracts/compression-roundtrip-v1.yaml — DO NOT EDIT
// Contract: compression-roundtrip-v1

/// Preconditions for equation `lz4_roundtrip`.
/// Call at function entry: `contract_pre_lz4_roundtrip!(input_expr)`
macro_rules! contract_pre_lz4_roundtrip {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `lz4_roundtrip`.
/// Call before return: `contract_post_lz4_roundtrip!(result_expr)`
macro_rules! contract_post_lz4_roundtrip {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `lz4_roundtrip`.
macro_rules! contract_lz4_roundtrip {
    ($input:expr, $body:expr) => {{
        contract_pre_lz4_roundtrip!($input);
        let _contract_result = $body;
        contract_post_lz4_roundtrip!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `sqlite_migration`.
/// Call at function entry: `contract_pre_sqlite_migration!(input_expr)`
macro_rules! contract_pre_sqlite_migration {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `sqlite_migration`.
/// Call before return: `contract_post_sqlite_migration!(result_expr)`
macro_rules! contract_post_sqlite_migration {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `sqlite_migration`.
macro_rules! contract_sqlite_migration {
    ($input:expr, $body:expr) => {{
        contract_pre_sqlite_migration!($input);
        let _contract_result = $body;
        contract_post_sqlite_migration!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/compute-parity-v1.yaml — DO NOT EDIT
// Contract: compute-parity-v1

/// Preconditions for equation `backend_dispatch_complete`.
/// Call at function entry: `contract_pre_backend_dispatch_complete!(input_expr)`
macro_rules! contract_pre_backend_dispatch_complete {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `backend_dispatch_complete`.
/// Call before return: `contract_post_backend_dispatch_complete!(result_expr)`
macro_rules! contract_post_backend_dispatch_complete {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `backend_dispatch_complete`.
macro_rules! contract_backend_dispatch_complete {
    ($input:expr, $body:expr) => {{
        contract_pre_backend_dispatch_complete!($input);
        let _contract_result = $body;
        contract_post_backend_dispatch_complete!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `gpu_cpu_parity`.
/// Call at function entry: `contract_pre_gpu_cpu_parity!(input_expr)`
macro_rules! contract_pre_gpu_cpu_parity {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `gpu_cpu_parity`.
/// Call before return: `contract_post_gpu_cpu_parity!(result_expr)`
macro_rules! contract_post_gpu_cpu_parity {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `gpu_cpu_parity`.
macro_rules! contract_gpu_cpu_parity {
    ($input:expr, $body:expr) => {{
        contract_pre_gpu_cpu_parity!($input);
        let _contract_result = $body;
        contract_post_gpu_cpu_parity!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `simd_scalar_parity`.
/// Call at function entry: `contract_pre_simd_scalar_parity!(input_expr)`
macro_rules! contract_pre_simd_scalar_parity {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `simd_scalar_parity`.
/// Call before return: `contract_post_simd_scalar_parity!(result_expr)`
macro_rules! contract_post_simd_scalar_parity {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `simd_scalar_parity`.
macro_rules! contract_simd_scalar_parity {
    ($input:expr, $body:expr) => {{
        contract_pre_simd_scalar_parity!($input);
        let _contract_result = $body;
        contract_post_simd_scalar_parity!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/concurrency-safety-v1.yaml — DO NOT EDIT
// Contract: concurrency-safety-v1

/// Preconditions for equation `channel_lossless`.
/// Call at function entry: `contract_pre_channel_lossless!(input_expr)`
macro_rules! contract_pre_channel_lossless {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `channel_lossless`.
/// Call before return: `contract_post_channel_lossless!(result_expr)`
macro_rules! contract_post_channel_lossless {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `channel_lossless`.
macro_rules! contract_channel_lossless {
    ($input:expr, $body:expr) => {{
        contract_pre_channel_lossless!($input);
        let _contract_result = $body;
        contract_post_channel_lossless!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `parallel_determinism`.
/// Call at function entry: `contract_pre_parallel_determinism!(input_expr)`
macro_rules! contract_pre_parallel_determinism {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `parallel_determinism`.
/// Call before return: `contract_post_parallel_determinism!(result_expr)`
macro_rules! contract_post_parallel_determinism {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `parallel_determinism`.
macro_rules! contract_parallel_determinism {
    ($input:expr, $body:expr) => {{
        contract_pre_parallel_determinism!($input);
        let _contract_result = $body;
        contract_post_parallel_determinism!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `task_cancellation_cleanup`.
/// Call at function entry: `contract_pre_task_cancellation_cleanup!(input_expr)`
macro_rules! contract_pre_task_cancellation_cleanup {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `task_cancellation_cleanup`.
/// Call before return: `contract_post_task_cancellation_cleanup!(result_expr)`
macro_rules! contract_post_task_cancellation_cleanup {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `task_cancellation_cleanup`.
macro_rules! contract_task_cancellation_cleanup {
    ($input:expr, $body:expr) => {{
        contract_pre_task_cancellation_cleanup!($input);
        let _contract_result = $body;
        contract_post_task_cancellation_cleanup!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/configuration-schema-v1.yaml — DO NOT EDIT
// Contract: configuration-schema-v1

/// Preconditions for equation `threshold_invariants`.
/// Call at function entry: `contract_pre_threshold_invariants!(input_expr)`
macro_rules! contract_pre_threshold_invariants {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `threshold_invariants`.
/// Call before return: `contract_post_threshold_invariants!(result_expr)`
macro_rules! contract_post_threshold_invariants {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `threshold_invariants`.
macro_rules! contract_threshold_invariants {
    ($input:expr, $body:expr) => {{
        contract_pre_threshold_invariants!($input);
        let _contract_result = $body;
        contract_post_threshold_invariants!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `unknown_key_rejection`.
/// Call at function entry: `contract_pre_unknown_key_rejection!(input_expr)`
macro_rules! contract_pre_unknown_key_rejection {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `unknown_key_rejection`.
/// Call before return: `contract_post_unknown_key_rejection!(result_expr)`
macro_rules! contract_post_unknown_key_rejection {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `unknown_key_rejection`.
macro_rules! contract_unknown_key_rejection {
    ($input:expr, $body:expr) => {{
        contract_pre_unknown_key_rejection!($input);
        let _contract_result = $body;
        contract_post_unknown_key_rejection!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/configuration-v1.yaml — DO NOT EDIT
// Contract: configuration-v1

/// Preconditions for equation `configuration`.
/// Domain-specific. Call: `contract_pre_configuration!(slice_expr)`
macro_rules! contract_pre_configuration {
    () => {{}};
    ($input:expr) => {{
        let _pv_path = &$input;
    }};
}

/// Postconditions for equation `configuration`.
/// Call before return: `contract_post_configuration!(result_expr)`
macro_rules! contract_post_configuration {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `configuration`.
macro_rules! contract_configuration {
    ($input:expr, $body:expr) => {{
        contract_pre_configuration!($input);
        let _contract_result = $body;
        contract_post_configuration!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/configuration-v1.yaml — DO NOT EDIT
// Contract: configuration-v1

/// Preconditions for equation `bfs`.
/// Call at function entry: `contract_pre_bfs!(input_expr)`
macro_rules! contract_pre_bfs {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

// Auto-generated from contracts/configuration-v1.yaml — DO NOT EDIT
// Contract: configuration-v1

/// Preconditions for equation `connect`.
/// Call at function entry: `contract_pre_connect!(input_expr)`
macro_rules! contract_pre_connect {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

// Auto-generated from contracts/configuration-v1.yaml — DO NOT EDIT
// Contract: configuration-v1

/// Preconditions for equation `validate_index`.
/// Call at function entry: `contract_pre_validate_index!(input_expr)`
macro_rules! contract_pre_validate_index {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `validate_size`.
/// Call at function entry: `contract_pre_validate_size!(input_expr)`
macro_rules! contract_pre_validate_size {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

// Auto-generated from contracts/configuration-v1.yaml — DO NOT EDIT
// Contract: configuration-v1

// Auto-generated from contracts/configuration-v1.yaml — DO NOT EDIT
// Contract: configuration-v1

/// Preconditions for equation `insert`.
/// Call at function entry: `contract_pre_insert!(input_expr)`
macro_rules! contract_pre_insert {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `query`.
/// Call at function entry: `contract_pre_query!(input_expr)`
macro_rules! contract_pre_query {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

// Auto-generated from contracts/configuration-v1.yaml — DO NOT EDIT
// Contract: configuration-v1

/// Preconditions for equation `config`.
/// Call at function entry: `contract_pre_config!(input_expr)`
macro_rules! contract_pre_config {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

// Auto-generated from contracts/context-generation-v1.yaml — DO NOT EDIT
// Contract: context-generation-v1

/// Preconditions for equation `generate_context`.
/// Call at function entry: `contract_pre_generate_context!(input_expr)`
macro_rules! contract_pre_generate_context {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `generate_context`.
/// Call before return: `contract_post_generate_context!(result_expr)`
macro_rules! contract_post_generate_context {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `generate_context`.
macro_rules! contract_generate_context {
    ($input:expr, $body:expr) => {{
        contract_pre_generate_context!($input);
        let _contract_result = $body;
        contract_post_generate_context!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `index_persistence`.
/// Call at function entry: `contract_pre_index_persistence!(input_expr)`
macro_rules! contract_pre_index_persistence {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `index_persistence`.
/// Call before return: `contract_post_index_persistence!(result_expr)`
macro_rules! contract_post_index_persistence {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `index_persistence`.
macro_rules! contract_index_persistence {
    ($input:expr, $body:expr) => {{
        contract_pre_index_persistence!($input);
        let _contract_result = $body;
        contract_post_index_persistence!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/continuous-batching-v1.yaml — DO NOT EDIT
// Contract: continuous-batching-v1

/// Preconditions for equation `chunked_prefill`.
/// Call at function entry: `contract_pre_chunked_prefill!(input_expr)`
macro_rules! contract_pre_chunked_prefill {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `correctness_under_batching`.
/// Call at function entry: `contract_pre_correctness_under_batching!(input_expr)`
macro_rules! contract_pre_correctness_under_batching {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `decode_degradation`.
/// Domain-specific. Call: `contract_pre_decode_degradation!(slice_expr)`
macro_rules! contract_pre_decode_degradation {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract decode_degradation: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `request_state`.
/// Call at function entry: `contract_pre_request_state!(input_expr)`
macro_rules! contract_pre_request_state {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `scheduling_fairness`.
/// Call at function entry: `contract_pre_scheduling_fairness!(input_expr)`
macro_rules! contract_pre_scheduling_fairness {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `throughput_scaling`.
/// Call at function entry: `contract_pre_throughput_scaling!(input_expr)`
macro_rules! contract_pre_throughput_scaling {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `token_budget`.
/// Domain-specific. Call: `contract_pre_token_budget!(slice_expr)`
macro_rules! contract_pre_token_budget {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract token_budget: precondition violated — input.len() > 0");
    }};
}

// Auto-generated from contracts/conv1d-kernel-v1.yaml — DO NOT EDIT
// Contract: conv1d-kernel-v1

/// Preconditions for equation `conv1d`.
/// Domain-specific. Call: `contract_pre_conv1d!(slice_expr)`
macro_rules! contract_pre_conv1d {
    () => {{}};
    ($input:expr) => {{
        let _pv_a = &$input;
        debug_assert!(_pv_a.len() > 0,
            "Contract conv1d: precondition violated — a.len() > 0");
    }};
}

// Auto-generated from contracts/conversation-generation-v1.yaml — DO NOT EDIT
// Contract: conversation-generation-v1

/// Preconditions for equation `chatml_format`.
/// Domain-specific. Call: `contract_pre_chatml_format!(slice_expr)`
macro_rules! contract_pre_chatml_format {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract chatml_format: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `conversation_types`.
/// Domain-specific. Call: `contract_pre_conversation_types!(slice_expr)`
macro_rules! contract_pre_conversation_types {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract conversation_types: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `quality_gate`.
/// Domain-specific. Call: `contract_pre_quality_gate!(slice_expr)`
macro_rules! contract_pre_quality_gate {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract quality_gate: precondition violated — input.len() > 0");
    }};
}

// Auto-generated from contracts/cooperative-matrix-gemm-v1.yaml — DO NOT EDIT
// Contract: cooperative-matrix-gemm-v1

// Auto-generated from contracts/copia-delta-v1.yaml — DO NOT EDIT
// Contract: copia-delta-v1

/// Preconditions for equation `block_reuse`.
/// Domain-specific. Call: `contract_pre_block_reuse!(slice_expr)`
macro_rules! contract_pre_block_reuse {
    () => {{}};
    ($input:expr) => {{
        let _pv_old_idx = &$input;
    }};
}

/// Preconditions for equation `delta_correctness`.
/// Call at function entry: `contract_pre_delta_correctness!(input_expr)`
macro_rules! contract_pre_delta_correctness {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `identity_sync`.
/// Call at function entry: `contract_pre_identity_sync!(input_expr)`
macro_rules! contract_pre_identity_sync {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `transfer_minimality`.
/// Domain-specific. Call: `contract_pre_transfer_minimality!(slice_expr)`
macro_rules! contract_pre_transfer_minimality {
    () => {{}};
    ($input:expr) => {{
        let _pv_delta = &$input;
    }};
}

// Auto-generated from contracts/cpp-type-preservation-v1.yaml — DO NOT EDIT
// Contract: cpp-type-preservation-v1

/// Preconditions for equation `class_to_struct`.
/// Call at function entry: `contract_pre_class_to_struct!(input_expr)`
macro_rules! contract_pre_class_to_struct {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `class_to_struct`.
/// Call before return: `contract_post_class_to_struct!(result_expr)`
macro_rules! contract_post_class_to_struct {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `class_to_struct`.
macro_rules! contract_class_to_struct {
    ($input:expr, $body:expr) => {{
        contract_pre_class_to_struct!($input);
        let _contract_result = $body;
        contract_post_class_to_struct!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `inheritance_to_composition`.
/// Call at function entry: `contract_pre_inheritance_to_composition!(input_expr)`
macro_rules! contract_pre_inheritance_to_composition {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `inheritance_to_composition`.
/// Call before return: `contract_post_inheritance_to_composition!(result_expr)`
macro_rules! contract_post_inheritance_to_composition {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `inheritance_to_composition`.
macro_rules! contract_inheritance_to_composition {
    ($input:expr, $body:expr) => {{
        contract_pre_inheritance_to_composition!($input);
        let _contract_result = $body;
        contract_post_inheritance_to_composition!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `namespace_to_mod`.
/// Call at function entry: `contract_pre_namespace_to_mod!(input_expr)`
macro_rules! contract_pre_namespace_to_mod {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `namespace_to_mod`.
/// Call before return: `contract_post_namespace_to_mod!(result_expr)`
macro_rules! contract_post_namespace_to_mod {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `namespace_to_mod`.
macro_rules! contract_namespace_to_mod {
    ($input:expr, $body:expr) => {{
        contract_pre_namespace_to_mod!($input);
        let _contract_result = $body;
        contract_post_namespace_to_mod!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `operator_to_trait`.
/// Domain-specific. Call: `contract_pre_operator_to_trait!(slice_expr)`
macro_rules! contract_pre_operator_to_trait {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
    }};
}

/// Postconditions for equation `operator_to_trait`.
/// Call before return: `contract_post_operator_to_trait!(result_expr)`
macro_rules! contract_post_operator_to_trait {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `operator_to_trait`.
macro_rules! contract_operator_to_trait {
    ($input:expr, $body:expr) => {{
        contract_pre_operator_to_trait!($input);
        let _contract_result = $body;
        contract_post_operator_to_trait!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/cpu-q4k-activation-quant-v1.yaml — DO NOT EDIT
// Contract: cpu-q4k-activation-quant-v1

/// Preconditions for equation `current_path`.
/// Domain-specific. Call: `contract_pre_current_path!(slice_expr)`
macro_rules! contract_pre_current_path {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
        debug_assert!(_pv_x.iter().all(|v| v.is_finite()),
            "Contract current_path: precondition violated — x.iter().all(|v| v.is_finite())");
        debug_assert!(_pv_x.len() > 0,
            "Contract current_path: precondition violated — x.len() > 0");
    }};
}

/// Preconditions for equation `speedup_bound`.
/// Domain-specific. Call: `contract_pre_speedup_bound!(slice_expr)`
macro_rules! contract_pre_speedup_bound {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
        debug_assert!(_pv_x.iter().all(|v| v.is_finite()),
            "Contract speedup_bound: precondition violated — x.iter().all(|v| v.is_finite())");
        debug_assert!(_pv_x.len() > 0,
            "Contract speedup_bound: precondition violated — x.len() > 0");
    }};
}

/// Preconditions for equation `target_path`.
/// Domain-specific. Call: `contract_pre_target_path!(slice_expr)`
macro_rules! contract_pre_target_path {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
        debug_assert!(_pv_x.iter().all(|v| v.is_finite()),
            "Contract target_path: precondition violated — x.iter().all(|v| v.is_finite())");
        debug_assert!(_pv_x.len() > 0,
            "Contract target_path: precondition violated — x.len() > 0");
    }};
}

// Auto-generated from contracts/cpu-work-stealing-v1.yaml — DO NOT EDIT
// Contract: cpu-work-stealing-v1

/// Preconditions for equation `l1_tiling`.
/// Domain-specific. Call: `contract_pre_l1_tiling!(slice_expr)`
macro_rules! contract_pre_l1_tiling {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract l1_tiling: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract l1_tiling: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `rayon_overhead`.
/// Domain-specific. Call: `contract_pre_rayon_overhead!(slice_expr)`
macro_rules! contract_pre_rayon_overhead {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract rayon_overhead: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract rayon_overhead: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

// Auto-generated from contracts/cross-entropy-kernel-v1.yaml — DO NOT EDIT
// Contract: cross-entropy-kernel-v1

/// Preconditions for equation `cross_entropy`.
/// Domain-specific. Call: `contract_pre_cross_entropy!(slice_expr)`
macro_rules! contract_pre_cross_entropy {
    () => {{}};
    ($input:expr) => {{
        let _pv_logits = &$input;
        debug_assert!(_pv_logits.len() > 0,
            "Contract cross_entropy: precondition violated — logits.len() > 0");
        debug_assert!(_pv_logits.iter().all(|v| v.is_finite()),
            "Contract cross_entropy: precondition violated — logits.iter().all(|v| v.is_finite())");
    }};
}

/// Postconditions for equation `cross_entropy`.
/// Call before return: `contract_post_cross_entropy!(result_expr)`
macro_rules! contract_post_cross_entropy {
    ($result:expr) => {{
        let _contract_result = &$result;
        debug_assert!(_contract_result.is_finite(), "Contract cross_entropy: postcondition violated — result.is_finite()");
        debug_assert!(*_contract_result >= 0.0, "Contract cross_entropy: postcondition violated — result >= 0.0");
    }};
}

/// Combined pre+post contract for equation `cross_entropy`.
macro_rules! contract_cross_entropy {
    ($input:expr, $body:expr) => {{
        contract_pre_cross_entropy!($input);
        let _contract_result = $body;
        contract_post_cross_entropy!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `log_softmax`.
/// Domain-specific. Call: `contract_pre_log_softmax!(slice_expr)`
macro_rules! contract_pre_log_softmax {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
        debug_assert!(_pv_x.iter().all(|v| v.is_finite()),
            "Contract log_softmax: precondition violated — x.iter().all(|v| v.is_finite())");
        debug_assert!(_pv_x.len() > 0,
            "Contract log_softmax: precondition violated — x.len() > 0");
    }};
}

// Auto-generated from contracts/cuda-classify-training-v1.yaml — DO NOT EDIT
// Contract: cuda-classify-training-v1

/// Preconditions for equation `device_dispatch`.
/// Call at function entry: `contract_pre_device_dispatch!(input_expr)`
macro_rules! contract_pre_device_dispatch {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `gpu_forward`.
/// Call at function entry: `contract_pre_gpu_forward!(input_expr)`
macro_rules! contract_pre_gpu_forward {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `weight_roundtrip`.
/// Domain-specific. Call: `contract_pre_weight_roundtrip!(slice_expr)`
macro_rules! contract_pre_weight_roundtrip {
    () => {{}};
    ($input:expr) => {{
        let _pv_weights = &$input;
        debug_assert!(_pv_weights.len() > 0,
            "Contract weight_roundtrip: precondition violated — weights.len() > 0");
    }};
}

// Auto-generated from contracts/cuda-kernel-safety-v1.yaml — DO NOT EDIT
// Contract: cuda-kernel-safety-v1

/// Preconditions for equation `host_transpilation`.
/// Call at function entry: `contract_pre_host_transpilation!(input_expr)`
macro_rules! contract_pre_host_transpilation {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `host_transpilation`.
/// Call before return: `contract_post_host_transpilation!(result_expr)`
macro_rules! contract_post_host_transpilation {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `host_transpilation`.
macro_rules! contract_host_transpilation {
    ($input:expr, $body:expr) => {{
        contract_pre_host_transpilation!($input);
        let _contract_result = $body;
        contract_post_host_transpilation!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `kernel_ffi`.
/// Call at function entry: `contract_pre_kernel_ffi!(input_expr)`
macro_rules! contract_pre_kernel_ffi {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `kernel_ffi`.
/// Call before return: `contract_post_kernel_ffi!(result_expr)`
macro_rules! contract_post_kernel_ffi {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `kernel_ffi`.
macro_rules! contract_kernel_ffi {
    ($input:expr, $body:expr) => {{
        contract_pre_kernel_ffi!($input);
        let _contract_result = $body;
        contract_post_kernel_ffi!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `qualifier_preservation`.
/// Call at function entry: `contract_pre_qualifier_preservation!(input_expr)`
macro_rules! contract_pre_qualifier_preservation {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `qualifier_preservation`.
/// Call before return: `contract_post_qualifier_preservation!(result_expr)`
macro_rules! contract_post_qualifier_preservation {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `qualifier_preservation`.
macro_rules! contract_qualifier_preservation {
    ($input:expr, $body:expr) => {{
        contract_pre_qualifier_preservation!($input);
        let _contract_result = $body;
        contract_post_qualifier_preservation!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/dag-ordering-v1.yaml — DO NOT EDIT
// Contract: dag-ordering-v1

/// Preconditions for equation `kahn_sort`.
/// Call at function entry: `contract_pre_kahn_sort!(input_expr)`
macro_rules! contract_pre_kahn_sort {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `topological_sort`.
/// Call at function entry: `contract_pre_topological_sort!(input_expr)`
macro_rules! contract_pre_topological_sort {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

// Auto-generated from contracts/data-feed-v1.yaml — DO NOT EDIT
// Contract: data-feed-v1

/// Preconditions for equation `config_validity`.
/// Call at function entry: `contract_pre_config_validity!(input_expr)`
macro_rules! contract_pre_config_validity {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `serialize_roundtrip`.
/// Call at function entry: `contract_pre_serialize_roundtrip!(input_expr)`
macro_rules! contract_pre_serialize_roundtrip {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

// Auto-generated from contracts/decision-engine-v1.yaml — DO NOT EDIT
// Contract: decision-engine-v1

/// Preconditions for equation `include_resolution`.
/// Call at function entry: `contract_pre_include_resolution!(input_expr)`
macro_rules! contract_pre_include_resolution {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `transpile_dispatch`.
/// Call at function entry: `contract_pre_transpile_dispatch!(input_expr)`
macro_rules! contract_pre_transpile_dispatch {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `type_preservation`.
/// Call at function entry: `contract_pre_type_preservation!(input_expr)`
macro_rules! contract_pre_type_preservation {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

// Auto-generated from contracts/decision-tree-v1.yaml — DO NOT EDIT
// Contract: decision-tree-v1

/// Preconditions for equation `gini_impurity`.
/// Domain-specific. Call: `contract_pre_gini_impurity!(slice_expr)`
macro_rules! contract_pre_gini_impurity {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract gini_impurity: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract gini_impurity: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `gini_split`.
/// Domain-specific. Call: `contract_pre_gini_split!(slice_expr)`
macro_rules! contract_pre_gini_split {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract gini_split: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract gini_split: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `mse_split`.
/// Domain-specific. Call: `contract_pre_mse_split!(slice_expr)`
macro_rules! contract_pre_mse_split {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract mse_split: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract mse_split: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `prediction`.
/// Domain-specific. Call: `contract_pre_prediction!(slice_expr)`
macro_rules! contract_pre_prediction {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract prediction: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract prediction: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

// Auto-generated from contracts/delta-sync-v1.yaml — DO NOT EDIT
// Contract: delta-sync-v1

/// Preconditions for equation `delta_computation`.
/// Domain-specific. Call: `contract_pre_delta_computation!(slice_expr)`
macro_rules! contract_pre_delta_computation {
    () => {{}};
    ($input:expr) => {{
        let _pv_signature = &$input;
    }};
}

/// Preconditions for equation `patch_apply`.
/// Domain-specific. Call: `contract_pre_patch_apply!(slice_expr)`
macro_rules! contract_pre_patch_apply {
    () => {{}};
    ($input:expr) => {{
        let _pv_delta = &$input;
    }};
}

/// Preconditions for equation `rolling_checksum`.
/// Domain-specific. Call: `contract_pre_rolling_checksum!(slice_expr)`
macro_rules! contract_pre_rolling_checksum {
    () => {{}};
    ($input:expr) => {{
        let _pv_window = &$input;
    }};
}

// Auto-generated from contracts/display-format-v1.yaml — DO NOT EDIT
// Contract: display-format-v1

/// Preconditions for equation `display_format`.
/// Call at function entry: `contract_pre_display_format!(input_expr)`
macro_rules! contract_pre_display_format {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `display_format`.
/// Call before return: `contract_post_display_format!(result_expr)`
macro_rules! contract_post_display_format {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `display_format`.
macro_rules! contract_display_format {
    ($input:expr, $body:expr) => {{
        contract_pre_display_format!($input);
        let _contract_result = $body;
        contract_post_display_format!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `render`.
/// Call at function entry: `contract_pre_render!(input_expr)`
macro_rules! contract_pre_render {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `render`.
/// Call before return: `contract_post_render!(result_expr)`
macro_rules! contract_post_render {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `render`.
macro_rules! contract_render {
    ($input:expr, $body:expr) => {{
        contract_pre_render!($input);
        let _contract_result = $body;
        contract_post_render!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/distributed-training-v1.yaml — DO NOT EDIT
// Contract: distributed-training-v1

/// Preconditions for equation `gradient_allreduce`.
/// Domain-specific. Call: `contract_pre_gradient_allreduce!(slice_expr)`
macro_rules! contract_pre_gradient_allreduce {
    () => {{}};
    ($input:expr) => {{
        let _pv_params = &$input;
        debug_assert!(_pv_params.len() > 0,
            "Contract gradient_allreduce: precondition violated — params.len() > 0");
    }};
}

/// Preconditions for equation `lora_gradient_size`.
/// Domain-specific. Call: `contract_pre_lora_gradient_size!(slice_expr)`
macro_rules! contract_pre_lora_gradient_size {
    () => {{}};
    ($input:expr) => {{
        let _pv_grad_output = &$input;
        debug_assert!(_pv_grad_output.len() > 0,
            "Contract lora_gradient_size: precondition violated — grad_output.len() > 0");
        debug_assert!(_pv_grad_output.iter().all(|v| v.is_finite()),
            "Contract lora_gradient_size: precondition violated — grad_output.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `sharding`.
/// Domain-specific. Call: `contract_pre_sharding!(slice_expr)`
macro_rules! contract_pre_sharding {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract sharding: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract sharding: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `swiglu_ffn`.
/// Domain-specific. Call: `contract_pre_swiglu_ffn!(slice_expr)`
macro_rules! contract_pre_swiglu_ffn {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract swiglu_ffn: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract swiglu_ffn: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `weighted_loss`.
/// Domain-specific. Call: `contract_pre_weighted_loss!(slice_expr)`
macro_rules! contract_pre_weighted_loss {
    () => {{}};
    ($input:expr) => {{
        let _pv_predicted = &$input;
        debug_assert!(_pv_predicted.len() > 0,
            "Contract weighted_loss: precondition violated — predicted.len() > 0");
    }};
}

// Auto-generated from contracts/distribution-v1.yaml — DO NOT EDIT
// Contract: distribution-v1

/// Preconditions for equation `build_integrity`.
/// Call at function entry: `contract_pre_build_integrity!(input_expr)`
macro_rules! contract_pre_build_integrity {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `distribution_delivery`.
/// Call at function entry: `contract_pre_distribution_delivery!(input_expr)`
macro_rules! contract_pre_distribution_delivery {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

// Auto-generated from contracts/dpo-loss-v1.yaml — DO NOT EDIT
// Contract: dpo-loss-v1

/// Preconditions for equation `dpo_loss`.
/// Domain-specific. Call: `contract_pre_dpo_loss!(slice_expr)`
macro_rules! contract_pre_dpo_loss {
    () => {{}};
    ($input:expr) => {{
        let _pv_predicted = &$input;
        debug_assert!(_pv_predicted.len() > 0,
            "Contract dpo_loss: precondition violated — predicted.len() > 0");
    }};
}

/// Preconditions for equation `implicit_reward`.
/// Domain-specific. Call: `contract_pre_implicit_reward!(slice_expr)`
macro_rules! contract_pre_implicit_reward {
    () => {{}};
    ($input:expr) => {{
        let _pv_predicted = &$input;
        debug_assert!(_pv_predicted.len() > 0,
            "Contract implicit_reward: precondition violated — predicted.len() > 0");
    }};
}

/// Preconditions for equation `log_ratio`.
/// Domain-specific. Call: `contract_pre_log_ratio!(slice_expr)`
macro_rules! contract_pre_log_ratio {
    () => {{}};
    ($input:expr) => {{
        let _pv_predicted = &$input;
        debug_assert!(_pv_predicted.len() > 0,
            "Contract log_ratio: precondition violated — predicted.len() > 0");
    }};
}

// Auto-generated from contracts/drift-detection-v1.yaml — DO NOT EDIT
// Contract: drift-detection-v1

/// Preconditions for equation `classify_drift`.
/// Domain-specific. Call: `contract_pre_classify_drift!(slice_expr)`
macro_rules! contract_pre_classify_drift {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
    }};
}

/// Preconditions for equation `min_samples_guard`.
/// Domain-specific. Call: `contract_pre_min_samples_guard!(slice_expr)`
macro_rules! contract_pre_min_samples_guard {
    () => {{}};
    ($input:expr) => {{
        let _pv_params = &$input;
        debug_assert!(_pv_params.len() > 0,
            "Contract min_samples_guard: precondition violated — params.len() > 0");
    }};
}

/// Preconditions for equation `performance_drift`.
/// Domain-specific. Call: `contract_pre_performance_drift!(slice_expr)`
macro_rules! contract_pre_performance_drift {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract performance_drift: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `univariate_drift`.
/// Domain-specific. Call: `contract_pre_univariate_drift!(slice_expr)`
macro_rules! contract_pre_univariate_drift {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract univariate_drift: precondition violated — input.len() > 0");
    }};
}

// Auto-generated from contracts/dropout-v1.yaml — DO NOT EDIT
// Contract: dropout-v1

/// Preconditions for equation `dropout_eval`.
/// Domain-specific. Call: `contract_pre_dropout_eval!(slice_expr)`
macro_rules! contract_pre_dropout_eval {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
        debug_assert!(_pv_x.iter().all(|v| v.is_finite()),
            "Contract dropout_eval: precondition violated — x.iter().all(|v| v.is_finite())");
        debug_assert!(_pv_x.len() > 0,
            "Contract dropout_eval: precondition violated — x.len() > 0");
    }};
}

/// Preconditions for equation `dropout_train`.
/// Domain-specific. Call: `contract_pre_dropout_train!(slice_expr)`
macro_rules! contract_pre_dropout_train {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
        debug_assert!(_pv_x.iter().all(|v| v.is_finite()),
            "Contract dropout_train: precondition violated — x.iter().all(|v| v.is_finite())");
        debug_assert!(_pv_x.len() > 0,
            "Contract dropout_train: precondition violated — x.len() > 0");
    }};
}

// Auto-generated from contracts/embedding-algebra-v1.yaml — DO NOT EDIT
// Contract: embedding-algebra-v1

/// Preconditions for equation `embedding_lookup`.
/// Domain-specific. Call: `contract_pre_embedding_lookup!(slice_expr)`
macro_rules! contract_pre_embedding_lookup {
    () => {{}};
    ($input:expr) => {{
        let _pv_indices = &$input;
        debug_assert!(_pv_indices.len() > 0,
            "Contract embedding_lookup: precondition violated — indices.len() > 0");
    }};
}

/// Preconditions for equation `embedding_norm`.
/// Domain-specific. Call: `contract_pre_embedding_norm!(slice_expr)`
macro_rules! contract_pre_embedding_norm {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract embedding_norm: precondition violated — input.iter().all(|v| v.is_finite())");
        debug_assert!(_pv_input.len() > 0,
            "Contract embedding_norm: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `logit_temperature`.
/// Domain-specific. Call: `contract_pre_logit_temperature!(slice_expr)`
macro_rules! contract_pre_logit_temperature {
    () => {{}};
    ($input:expr) => {{
        let _pv_indices = &$input;
        debug_assert!(_pv_indices.len() > 0,
            "Contract logit_temperature: precondition violated — indices.len() > 0");
    }};
}

/// Preconditions for equation `tied_weights`.
/// Domain-specific. Call: `contract_pre_tied_weights!(slice_expr)`
macro_rules! contract_pre_tied_weights {
    () => {{}};
    ($input:expr) => {{
        let _pv_indices = &$input;
        debug_assert!(_pv_indices.len() > 0,
            "Contract tied_weights: precondition violated — indices.len() > 0");
    }};
}

/// Preconditions for equation `unembedding_projection`.
/// Domain-specific. Call: `contract_pre_unembedding_projection!(slice_expr)`
macro_rules! contract_pre_unembedding_projection {
    () => {{}};
    ($input:expr) => {{
        let _pv_indices = &$input;
        debug_assert!(_pv_indices.len() > 0,
            "Contract unembedding_projection: precondition violated — indices.len() > 0");
    }};
}

/// Preconditions for equation `vocabulary_bounds`.
/// Domain-specific. Call: `contract_pre_vocabulary_bounds!(slice_expr)`
macro_rules! contract_pre_vocabulary_bounds {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract vocabulary_bounds: precondition violated — input.len() > 0");
    }};
}

// Auto-generated from contracts/embedding-lookup-v1.yaml — DO NOT EDIT
// Contract: embedding-lookup-v1

/// Preconditions for equation `embedding_lookup`.
/// Domain-specific. Call: `contract_pre_embedding_lookup!(slice_expr)`
macro_rules! contract_pre_embedding_lookup {
    () => {{}};
    ($input:expr) => {{
        let _pv_token_ids = &$input;
    }};
}

/// Postconditions for equation `embedding_lookup`.
/// Call before return: `contract_post_embedding_lookup!(result_expr)`
macro_rules! contract_post_embedding_lookup {
    ($result:expr) => {{
        let _contract_result = &$result;
        debug_assert!(_contract_result.iter().all(|v| v.is_finite()), "Contract embedding_lookup: postcondition violated — result.iter().all(|v| v.is_finite())");
    }};
}

/// Combined pre+post contract for equation `embedding_lookup`.
macro_rules! contract_embedding_lookup {
    ($input:expr, $body:expr) => {{
        contract_pre_embedding_lookup!($input);
        let _contract_result = $body;
        contract_post_embedding_lookup!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/encoder-forward-v1.yaml — DO NOT EDIT
// Contract: encoder-forward-v1

/// Preconditions for equation `cls_pooling`.
/// Domain-specific. Call: `contract_pre_cls_pooling!(slice_expr)`
macro_rules! contract_pre_cls_pooling {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract cls_pooling: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `encoder_layer`.
/// Domain-specific. Call: `contract_pre_encoder_layer!(slice_expr)`
macro_rules! contract_pre_encoder_layer {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract encoder_layer: precondition violated — input.len() > 0");
    }};
}

// Auto-generated from contracts/encoder-roundtrip-v1.yaml — DO NOT EDIT
// Contract: encoder-roundtrip-v1

/// Preconditions for equation `emit_posix`.
/// Call at function entry: `contract_pre_emit_posix!(input_expr)`
macro_rules! contract_pre_emit_posix {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
        debug_assert!(!_contract_input.is_empty(),
            "Contract emit_posix: precondition violated — !input.is_empty()");
    }};
}

/// Preconditions for equation `emit_purified`.
/// Call at function entry: `contract_pre_emit_purified!(input_expr)`
macro_rules! contract_pre_emit_purified {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
        debug_assert!(!_contract_input.is_empty(),
            "Contract emit_purified: precondition violated — !input.is_empty()");
    }};
}

/// Preconditions for equation `roundtrip`.
/// Call at function entry: `contract_pre_roundtrip!(input_expr)`
macro_rules! contract_pre_roundtrip {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
        debug_assert!(!_contract_input.is_empty(),
            "Contract roundtrip: precondition violated — !input.is_empty()");
    }};
}

// Auto-generated from contracts/encoder-roundtrip-v1.yaml — DO NOT EDIT
// Contract: encoder-roundtrip-v1

/// Preconditions for equation `decode`.
/// Domain-specific. Call: `contract_pre_decode!(slice_expr)`
macro_rules! contract_pre_decode {
    () => {{}};
    ($input:expr) => {{
        let _pv_bitstream = &$input;
        debug_assert!(_pv_bitstream.len() > 0,
            "Contract decode: precondition violated — bitstream.len() > 0");
    }};
}

/// Preconditions for equation `encode`.
/// Domain-specific. Call: `contract_pre_encode!(slice_expr)`
macro_rules! contract_pre_encode {
    () => {{}};
    ($input:expr) => {{
        let _pv_frame = &$input;
    }};
}

/// Preconditions for equation `encoder_resolution`.
/// Call at function entry: `contract_pre_encoder_resolution!(input_expr)`
macro_rules! contract_pre_encoder_resolution {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

// Auto-generated from contracts/error-handling-v1.yaml — DO NOT EDIT
// Contract: error-handling-v1

/// Preconditions for equation `send`.
/// Domain-specific. Call: `contract_pre_send!(slice_expr)`
macro_rules! contract_pre_send {
    () => {{}};
    ($input:expr) => {{
        let _pv_conn = &$input;
        debug_assert!(_pv_conn.is_active(),
            "Contract send: precondition violated — conn.is_active()");
    }};
}

/// Preconditions for equation `send_error_propagation`.
/// Domain-specific. Call: `contract_pre_send_error_propagation!(slice_expr)`
macro_rules! contract_pre_send_error_propagation {
    () => {{}};
    ($input:expr) => {{
        let _pv_send_result = &$input;
        debug_assert!(_pv_send_result.is_err(),
            "Contract send_error_propagation: precondition violated — send_result.is_err()");
    }};
}

// Auto-generated from contracts/error-handling-v1.yaml — DO NOT EDIT
// Contract: error-handling-v1

/// Preconditions for equation `error_handling`.
/// Call at function entry: `contract_pre_error_handling!(input_expr)`
macro_rules! contract_pre_error_handling {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `error_handling`.
/// Call before return: `contract_post_error_handling!(result_expr)`
macro_rules! contract_post_error_handling {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `error_handling`.
macro_rules! contract_error_handling {
    ($input:expr, $body:expr) => {{
        contract_pre_error_handling!($input);
        let _contract_result = $body;
        contract_post_error_handling!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/event-rulebook-v1.yaml — DO NOT EDIT
// Contract: event-rulebook-v1

/// Preconditions for equation `action_ordering`.
/// Domain-specific. Call: `contract_pre_action_ordering!(slice_expr)`
macro_rules! contract_pre_action_ordering {
    () => {{}};
    ($input:expr) => {{
        let _pv_rule = &$input;
    }};
}

/// Preconditions for equation `cooldown_deduplication`.
/// Domain-specific. Call: `contract_pre_cooldown_deduplication!(slice_expr)`
macro_rules! contract_pre_cooldown_deduplication {
    () => {{}};
    ($input:expr) => {{
        let _pv_rule = &$input;
    }};
}

/// Preconditions for equation `trigger_dispatch_completeness`.
/// Call at function entry: `contract_pre_trigger_dispatch_completeness!(input_expr)`
macro_rules! contract_pre_trigger_dispatch_completeness {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

// Auto-generated from contracts/execution-safety-v1.yaml — DO NOT EDIT
// Contract: execution-safety-v1

/// Preconditions for equation `atomic_write`.
/// Call at function entry: `contract_pre_atomic_write!(input_expr)`
macro_rules! contract_pre_atomic_write {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `jidoka_stop`.
/// Call at function entry: `contract_pre_jidoka_stop!(input_expr)`
macro_rules! contract_pre_jidoka_stop {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

// Auto-generated from contracts/f16-conversion-v1.yaml — DO NOT EDIT
// Contract: f16-conversion-v1

/// Preconditions for equation `f16_to_f32_bias`.
/// Domain-specific. Call: `contract_pre_f16_to_f32_bias!(slice_expr)`
macro_rules! contract_pre_f16_to_f32_bias {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract f16_to_f32_bias: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `roundtrip`.
/// Domain-specific. Call: `contract_pre_roundtrip!(slice_expr)`
macro_rules! contract_pre_roundtrip {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract roundtrip: precondition violated — input.len() > 0");
    }};
}

// Auto-generated from contracts/flash-attention-v1.yaml — DO NOT EDIT
// Contract: flash-attention-v1

/// Preconditions for equation `flash_attention`.
/// Domain-specific. Call: `contract_pre_flash_attention!(slice_expr)`
macro_rules! contract_pre_flash_attention {
    () => {{}};
    ($input:expr) => {{
        let _pv_q = &$input;
        debug_assert!(_pv_q.len() > 0,
            "Contract flash_attention: precondition violated — q.len() > 0");
    }};
}

// Auto-generated from contracts/format-parity-v1.yaml — DO NOT EDIT
// Contract: format-parity-v1

/// Preconditions for equation `element_count`.
/// Domain-specific. Call: `contract_pre_element_count!(slice_expr)`
macro_rules! contract_pre_element_count {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract element_count: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `identity_1d`.
/// Domain-specific. Call: `contract_pre_identity_1d!(slice_expr)`
macro_rules! contract_pre_identity_1d {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract identity_1d: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract identity_1d: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `name_bijection`.
/// Domain-specific. Call: `contract_pre_name_bijection!(slice_expr)`
macro_rules! contract_pre_name_bijection {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract name_bijection: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `transpose_involution`.
/// Domain-specific. Call: `contract_pre_transpose_involution!(slice_expr)`
macro_rules! contract_pre_transpose_involution {
    () => {{}};
    ($input:expr) => {{
        let _pv_a = &$input;
        debug_assert!(_pv_a.len() > 0,
            "Contract transpose_involution: precondition violated — a.len() > 0");
    }};
}

// Auto-generated from contracts/fp8-interchange-v1.yaml — DO NOT EDIT
// Contract: fp8-interchange-v1

/// Preconditions for equation `e4m3_encode`.
/// Domain-specific. Call: `contract_pre_e4m3_encode!(slice_expr)`
macro_rules! contract_pre_e4m3_encode {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract e4m3_encode: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `e5m2_encode`.
/// Domain-specific. Call: `contract_pre_e5m2_encode!(slice_expr)`
macro_rules! contract_pre_e5m2_encode {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract e5m2_encode: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `roundtrip`.
/// Domain-specific. Call: `contract_pre_roundtrip!(slice_expr)`
macro_rules! contract_pre_roundtrip {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract roundtrip: precondition violated — input.len() > 0");
    }};
}

// Auto-generated from contracts/fused-qkv-projection-v1.yaml — DO NOT EDIT
// Contract: fused-qkv-projection-v1

/// Preconditions for equation `fused_qkv`.
/// Domain-specific. Call: `contract_pre_fused_qkv!(slice_expr)`
macro_rules! contract_pre_fused_qkv {
    () => {{}};
    ($input:expr) => {{
        let _pv_a = &$input;
        debug_assert!(_pv_a.len() > 0,
            "Contract fused_qkv: precondition violated — a.len() > 0");
    }};
}

/// Preconditions for equation `separate_qkv`.
/// Domain-specific. Call: `contract_pre_separate_qkv!(slice_expr)`
macro_rules! contract_pre_separate_qkv {
    () => {{}};
    ($input:expr) => {{
        let _pv_a = &$input;
        debug_assert!(_pv_a.len() > 0,
            "Contract separate_qkv: precondition violated — a.len() > 0");
    }};
}

/// Preconditions for equation `shared_q8_qkv`.
/// Domain-specific. Call: `contract_pre_shared_q8_qkv!(slice_expr)`
macro_rules! contract_pre_shared_q8_qkv {
    () => {{}};
    ($input:expr) => {{
        let _pv_a = &$input;
        debug_assert!(_pv_a.len() > 0,
            "Contract shared_q8_qkv: precondition violated — a.len() > 0");
    }};
}

// Auto-generated from contracts/gated-delta-net-v1.yaml — DO NOT EDIT
// Contract: gated-delta-net-v1

/// Preconditions for equation `decay`.
/// Domain-specific. Call: `contract_pre_decay!(slice_expr)`
macro_rules! contract_pre_decay {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
        debug_assert!(_pv_x.iter().all(|v| v.is_finite()),
            "Contract decay: precondition violated — x.iter().all(|v| v.is_finite())");
        debug_assert!(_pv_x.len() > 0,
            "Contract decay: precondition violated — x.len() > 0");
    }};
}

/// Preconditions for equation `delta`.
/// Domain-specific. Call: `contract_pre_delta!(slice_expr)`
macro_rules! contract_pre_delta {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract delta: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract delta: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `output`.
/// Domain-specific. Call: `contract_pre_output!(slice_expr)`
macro_rules! contract_pre_output {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract output: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract output: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `read`.
/// Domain-specific. Call: `contract_pre_read!(slice_expr)`
macro_rules! contract_pre_read {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract read: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract read: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `write`.
/// Domain-specific. Call: `contract_pre_write!(slice_expr)`
macro_rules! contract_pre_write {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract write: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract write: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

// Auto-generated from contracts/gbm-v1.yaml — DO NOT EDIT
// Contract: gbm-v1

/// Preconditions for equation `gradient_boost`.
/// Domain-specific. Call: `contract_pre_gradient_boost!(slice_expr)`
macro_rules! contract_pre_gradient_boost {
    () => {{}};
    ($input:expr) => {{
        let _pv_grad_output = &$input;
        debug_assert!(_pv_grad_output.len() > 0,
            "Contract gradient_boost: precondition violated — grad_output.len() > 0");
        debug_assert!(_pv_grad_output.iter().all(|v| v.is_finite()),
            "Contract gradient_boost: precondition violated — grad_output.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `negative_gradient`.
/// Domain-specific. Call: `contract_pre_negative_gradient!(slice_expr)`
macro_rules! contract_pre_negative_gradient {
    () => {{}};
    ($input:expr) => {{
        let _pv_grad_output = &$input;
        debug_assert!(_pv_grad_output.len() > 0,
            "Contract negative_gradient: precondition violated — grad_output.len() > 0");
        debug_assert!(_pv_grad_output.iter().all(|v| v.is_finite()),
            "Contract negative_gradient: precondition violated — grad_output.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `predict`.
/// Domain-specific. Call: `contract_pre_predict!(slice_expr)`
macro_rules! contract_pre_predict {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract predict: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract predict: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `training_loss`.
/// Domain-specific. Call: `contract_pre_training_loss!(slice_expr)`
macro_rules! contract_pre_training_loss {
    () => {{}};
    ($input:expr) => {{
        let _pv_predicted = &$input;
        debug_assert!(_pv_predicted.len() > 0,
            "Contract training_loss: precondition violated — predicted.len() > 0");
    }};
}

// Auto-generated from contracts/gelu-kernel-v1.yaml — DO NOT EDIT
// Contract: gelu-kernel-v1

/// Preconditions for equation `gelu`.
/// Domain-specific. Call: `contract_pre_gelu!(slice_expr)`
macro_rules! contract_pre_gelu {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
        debug_assert!(_pv_x.iter().all(|v| v.is_finite()),
            "Contract gelu: precondition violated — x.iter().all(|v| v.is_finite())");
        debug_assert!(_pv_x.len() > 0,
            "Contract gelu: precondition violated — x.len() > 0");
    }};
}

/// Preconditions for equation `gelu_tanh_approx`.
/// Domain-specific. Call: `contract_pre_gelu_tanh_approx!(slice_expr)`
macro_rules! contract_pre_gelu_tanh_approx {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
        debug_assert!(_pv_x.iter().all(|v| v.is_finite()),
            "Contract gelu_tanh_approx: precondition violated — x.iter().all(|v| v.is_finite())");
        debug_assert!(_pv_x.len() > 0,
            "Contract gelu_tanh_approx: precondition violated — x.len() > 0");
    }};
}

// Auto-generated from contracts/gemm-backward-tiled-v1.yaml — DO NOT EDIT
// Contract: gemm-backward-tiled-v1

/// Preconditions for equation `backward_a_gemm`.
/// Domain-specific. Call: `contract_pre_backward_a_gemm!(slice_expr)`
macro_rules! contract_pre_backward_a_gemm {
    () => {{}};
    ($input:expr) => {{
        let _pv_a = &$input;
        debug_assert!(_pv_a.len() > 0,
            "Contract backward_a_gemm: precondition violated — a.len() > 0");
    }};
}

/// Preconditions for equation `backward_b_gemm`.
/// Domain-specific. Call: `contract_pre_backward_b_gemm!(slice_expr)`
macro_rules! contract_pre_backward_b_gemm {
    () => {{}};
    ($input:expr) => {{
        let _pv_a = &$input;
        debug_assert!(_pv_a.len() > 0,
            "Contract backward_b_gemm: precondition violated — a.len() > 0");
    }};
}

/// Preconditions for equation `shared_memory_per_tile`.
/// Domain-specific. Call: `contract_pre_shared_memory_per_tile!(slice_expr)`
macro_rules! contract_pre_shared_memory_per_tile {
    () => {{}};
    ($input:expr) => {{
        let _pv_a = &$input;
        debug_assert!(_pv_a.len() > 0,
            "Contract shared_memory_per_tile: precondition violated — a.len() > 0");
    }};
}

/// Preconditions for equation `tiled_gemm_arithmetic_intensity`.
/// Domain-specific. Call: `contract_pre_tiled_gemm_arithmetic_intensity!(slice_expr)`
macro_rules! contract_pre_tiled_gemm_arithmetic_intensity {
    () => {{}};
    ($input:expr) => {{
        let _pv_a = &$input;
        debug_assert!(_pv_a.len() > 0,
            "Contract tiled_gemm_arithmetic_intensity: precondition violated — a.len() > 0");
    }};
}

/// Preconditions for equation `unrolled_instruction_ratio`.
/// Domain-specific. Call: `contract_pre_unrolled_instruction_ratio!(slice_expr)`
macro_rules! contract_pre_unrolled_instruction_ratio {
    () => {{}};
    ($input:expr) => {{
        let _pv_a = &$input;
        debug_assert!(_pv_a.len() > 0,
            "Contract unrolled_instruction_ratio: precondition violated — a.len() > 0");
    }};
}

// Auto-generated from contracts/gguf-cpu-cache-v1.yaml — DO NOT EDIT
// Contract: gguf-cpu-cache-v1

/// Preconditions for equation `autoregressive_generation`.
/// Call at function entry: `contract_pre_autoregressive_generation!(input_expr)`
macro_rules! contract_pre_autoregressive_generation {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

// Auto-generated from contracts/gguf-format-safety-v1.yaml — DO NOT EDIT
// Contract: gguf-format-safety-v1

/// Preconditions for equation `alignment_enforcement`.
/// Call at function entry: `contract_pre_alignment_enforcement!(input_expr)`
macro_rules! contract_pre_alignment_enforcement {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `alignment_enforcement`.
/// Call before return: `contract_post_alignment_enforcement!(result_expr)`
macro_rules! contract_post_alignment_enforcement {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `alignment_enforcement`.
macro_rules! contract_alignment_enforcement {
    ($input:expr, $body:expr) => {{
        contract_pre_alignment_enforcement!($input);
        let _contract_result = $body;
        contract_post_alignment_enforcement!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `magic_validation`.
/// Call at function entry: `contract_pre_magic_validation!(input_expr)`
macro_rules! contract_pre_magic_validation {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `magic_validation`.
/// Call before return: `contract_post_magic_validation!(result_expr)`
macro_rules! contract_post_magic_validation {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `magic_validation`.
macro_rules! contract_magic_validation {
    ($input:expr, $body:expr) => {{
        contract_pre_magic_validation!($input);
        let _contract_result = $body;
        contract_post_magic_validation!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `metadata_kv_safety`.
/// Call at function entry: `contract_pre_metadata_kv_safety!(input_expr)`
macro_rules! contract_pre_metadata_kv_safety {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `metadata_kv_safety`.
/// Call before return: `contract_post_metadata_kv_safety!(result_expr)`
macro_rules! contract_post_metadata_kv_safety {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `metadata_kv_safety`.
macro_rules! contract_metadata_kv_safety {
    ($input:expr, $body:expr) => {{
        contract_pre_metadata_kv_safety!($input);
        let _contract_result = $body;
        contract_post_metadata_kv_safety!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `tensor_metadata_integrity`.
/// Domain-specific. Call: `contract_pre_tensor_metadata_integrity!(slice_expr)`
macro_rules! contract_pre_tensor_metadata_integrity {
    () => {{}};
    ($input:expr) => {{
        let _pv_header = &$input;
    }};
}

/// Postconditions for equation `tensor_metadata_integrity`.
/// Call before return: `contract_post_tensor_metadata_integrity!(result_expr)`
macro_rules! contract_post_tensor_metadata_integrity {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `tensor_metadata_integrity`.
macro_rules! contract_tensor_metadata_integrity {
    ($input:expr, $body:expr) => {{
        contract_pre_tensor_metadata_integrity!($input);
        let _contract_result = $body;
        contract_post_tensor_metadata_integrity!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `version_compatibility`.
/// Call at function entry: `contract_pre_version_compatibility!(input_expr)`
macro_rules! contract_pre_version_compatibility {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `version_compatibility`.
/// Call before return: `contract_post_version_compatibility!(result_expr)`
macro_rules! contract_post_version_compatibility {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `version_compatibility`.
macro_rules! contract_version_compatibility {
    ($input:expr, $body:expr) => {{
        contract_pre_version_compatibility!($input);
        let _contract_result = $body;
        contract_post_version_compatibility!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/glm-v1.yaml — DO NOT EDIT
// Contract: glm-v1

/// Preconditions for equation `binomial_link`.
/// Domain-specific. Call: `contract_pre_binomial_link!(slice_expr)`
macro_rules! contract_pre_binomial_link {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract binomial_link: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract binomial_link: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `gamma_link`.
/// Domain-specific. Call: `contract_pre_gamma_link!(slice_expr)`
macro_rules! contract_pre_gamma_link {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract gamma_link: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract gamma_link: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `irls_fit`.
/// Domain-specific. Call: `contract_pre_irls_fit!(slice_expr)`
macro_rules! contract_pre_irls_fit {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract irls_fit: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract irls_fit: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `poisson_link`.
/// Domain-specific. Call: `contract_pre_poisson_link!(slice_expr)`
macro_rules! contract_pre_poisson_link {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract poisson_link: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract poisson_link: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

// Auto-generated from contracts/gnn-v1.yaml — DO NOT EDIT
// Contract: gnn-v1

/// Preconditions for equation `gcn_aggregate`.
/// Call at function entry: `contract_pre_gcn_aggregate!(input_expr)`
macro_rules! contract_pre_gcn_aggregate {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `global_max_pool`.
/// Call at function entry: `contract_pre_global_max_pool!(input_expr)`
macro_rules! contract_pre_global_max_pool {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `global_mean_pool`.
/// Call at function entry: `contract_pre_global_mean_pool!(input_expr)`
macro_rules! contract_pre_global_mean_pool {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `message_passing`.
/// Call at function entry: `contract_pre_message_passing!(input_expr)`
macro_rules! contract_pre_message_passing {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

// Auto-generated from contracts/golden-trace-v1.yaml — DO NOT EDIT
// Contract: golden-trace-v1

/// Preconditions for equation `adaptive_sampling`.
/// Domain-specific. Call: `contract_pre_adaptive_sampling!(slice_expr)`
macro_rules! contract_pre_adaptive_sampling {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
    }};
}

/// Preconditions for equation `trace_capture`.
/// Call at function entry: `contract_pre_trace_capture!(input_expr)`
macro_rules! contract_pre_trace_capture {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `trace_validate`.
/// Call at function entry: `contract_pre_trace_validate!(input_expr)`
macro_rules! contract_pre_trace_validate {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

// Auto-generated from contracts/gpu-context-health-v1.yaml — DO NOT EDIT
// Contract: gpu-context-health-v1

/// Preconditions for equation `context_health`.
/// Domain-specific. Call: `contract_pre_context_health!(slice_expr)`
macro_rules! contract_pre_context_health {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract context_health: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `cuda_graph_guard`.
/// Call at function entry: `contract_pre_cuda_graph_guard!(input_expr)`
macro_rules! contract_pre_cuda_graph_guard {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `culink_skip`.
/// Domain-specific. Call: `contract_pre_culink_skip!(slice_expr)`
macro_rules! contract_pre_culink_skip {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract culink_skip: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `fp8_architecture_guard`.
/// Domain-specific. Call: `contract_pre_fp8_architecture_guard!(slice_expr)`
macro_rules! contract_pre_fp8_architecture_guard {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract fp8_architecture_guard: precondition violated — input.len() > 0");
    }};
}

// Auto-generated from contracts/gpu-decode-profiling-v1.yaml — DO NOT EDIT
// Contract: gpu-decode-profiling-v1

/// Preconditions for equation `decode_audio`.
/// Domain-specific. Call: `contract_pre_decode_audio!(slice_expr)`
macro_rules! contract_pre_decode_audio {
    () => {{}};
    ($input:expr) => {{
        let _pv_packet = &$input;
    }};
}

/// Preconditions for equation `decode_video`.
/// Domain-specific. Call: `contract_pre_decode_video!(slice_expr)`
macro_rules! contract_pre_decode_video {
    () => {{}};
    ($input:expr) => {{
        let _pv_packet = &$input;
    }};
}

// Auto-generated from contracts/gpu-decode-profiling-v1.yaml — DO NOT EDIT
// Contract: gpu-decode-profiling-v1

/// Preconditions for equation `brick_ordering`.
/// Domain-specific. Call: `contract_pre_brick_ordering!(slice_expr)`
macro_rules! contract_pre_brick_ordering {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract brick_ordering: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `graph_disable`.
/// Domain-specific. Call: `contract_pre_graph_disable!(slice_expr)`
macro_rules! contract_pre_graph_disable {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract graph_disable: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `report_completeness`.
/// Domain-specific. Call: `contract_pre_report_completeness!(slice_expr)`
macro_rules! contract_pre_report_completeness {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract report_completeness: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `report_denominator`.
/// Domain-specific. Call: `contract_pre_report_denominator!(slice_expr)`
macro_rules! contract_pre_report_denominator {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract report_denominator: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `report_fidelity`.
/// Domain-specific. Call: `contract_pre_report_fidelity!(slice_expr)`
macro_rules! contract_pre_report_fidelity {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract report_fidelity: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `report_metadata`.
/// Domain-specific. Call: `contract_pre_report_metadata!(slice_expr)`
macro_rules! contract_pre_report_metadata {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract report_metadata: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `sync_verification`.
/// Domain-specific. Call: `contract_pre_sync_verification!(slice_expr)`
macro_rules! contract_pre_sync_verification {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract sync_verification: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `token_accounting`.
/// Domain-specific. Call: `contract_pre_token_accounting!(slice_expr)`
macro_rules! contract_pre_token_accounting {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract token_accounting: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `wall_coverage`.
/// Domain-specific. Call: `contract_pre_wall_coverage!(slice_expr)`
macro_rules! contract_pre_wall_coverage {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract wall_coverage: precondition violated — input.len() > 0");
    }};
}

// Auto-generated from contracts/gpu-multi-backend-parity-v1.yaml — DO NOT EDIT
// Contract: gpu-multi-backend-parity-v1

/// Preconditions for equation `backend_priority`.
/// Domain-specific. Call: `contract_pre_backend_priority!(slice_expr)`
macro_rules! contract_pre_backend_priority {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract backend_priority: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `bandwidth_bound_theorem`.
/// Domain-specific. Call: `contract_pre_bandwidth_bound_theorem!(slice_expr)`
macro_rules! contract_pre_bandwidth_bound_theorem {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract bandwidth_bound_theorem: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `jit_compilation_correctness`.
/// Domain-specific. Call: `contract_pre_jit_compilation_correctness!(slice_expr)`
macro_rules! contract_pre_jit_compilation_correctness {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract jit_compilation_correctness: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `multi_backend_parity`.
/// Domain-specific. Call: `contract_pre_multi_backend_parity!(slice_expr)`
macro_rules! contract_pre_multi_backend_parity {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract multi_backend_parity: precondition violated — input.len() > 0");
    }};
}

// Auto-generated from contracts/gpu-weight-residency-v1.yaml — DO NOT EDIT
// Contract: gpu-weight-residency-v1

/// Preconditions for equation `pcie_overhead`.
/// Domain-specific. Call: `contract_pre_pcie_overhead!(slice_expr)`
macro_rules! contract_pre_pcie_overhead {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract pcie_overhead: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `throughput_target`.
/// Domain-specific. Call: `contract_pre_throughput_target!(slice_expr)`
macro_rules! contract_pre_throughput_target {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract throughput_target: precondition violated — input.len() > 0");
    }};
}

// Auto-generated from contracts/gqa-kernel-v1.yaml — DO NOT EDIT
// Contract: gqa-kernel-v1

/// Preconditions for equation `gqa`.
/// Domain-specific. Call: `contract_pre_gqa!(slice_expr)`
macro_rules! contract_pre_gqa {
    () => {{}};
    ($input:expr) => {{
        let _pv_q = &$input;
        debug_assert!(_pv_q.len() > 0,
            "Contract gqa: precondition violated — q.len() > 0");
    }};
}

// Auto-generated from contracts/graph-centrality-v1.yaml — DO NOT EDIT
// Contract: graph-centrality-v1

/// Preconditions for equation `betweenness`.
/// Call at function entry: `contract_pre_betweenness!(input_expr)`
macro_rules! contract_pre_betweenness {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `closeness`.
/// Call at function entry: `contract_pre_closeness!(input_expr)`
macro_rules! contract_pre_closeness {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `degree`.
/// Call at function entry: `contract_pre_degree!(input_expr)`
macro_rules! contract_pre_degree {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `eigenvector`.
/// Call at function entry: `contract_pre_eigenvector!(input_expr)`
macro_rules! contract_pre_eigenvector {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `harmonic`.
/// Call at function entry: `contract_pre_harmonic!(input_expr)`
macro_rules! contract_pre_harmonic {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `katz`.
/// Call at function entry: `contract_pre_katz!(input_expr)`
macro_rules! contract_pre_katz {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

// Auto-generated from contracts/graph-index-v1.yaml — DO NOT EDIT
// Contract: graph-index-v1

/// Preconditions for equation `bm25_scoring`.
/// Call at function entry: `contract_pre_bm25_scoring!(input_expr)`
macro_rules! contract_pre_bm25_scoring {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `bm25_scoring`.
/// Call before return: `contract_post_bm25_scoring!(result_expr)`
macro_rules! contract_post_bm25_scoring {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `bm25_scoring`.
macro_rules! contract_bm25_scoring {
    ($input:expr, $body:expr) => {{
        contract_pre_bm25_scoring!($input);
        let _contract_result = $body;
        contract_post_bm25_scoring!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `csr_construction`.
/// Call at function entry: `contract_pre_csr_construction!(input_expr)`
macro_rules! contract_pre_csr_construction {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `csr_construction`.
/// Call before return: `contract_post_csr_construction!(result_expr)`
macro_rules! contract_post_csr_construction {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `csr_construction`.
macro_rules! contract_csr_construction {
    ($input:expr, $body:expr) => {{
        contract_pre_csr_construction!($input);
        let _contract_result = $body;
        contract_post_csr_construction!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `fts5_consistency`.
/// Domain-specific. Call: `contract_pre_fts5_consistency!(slice_expr)`
macro_rules! contract_pre_fts5_consistency {
    () => {{}};
    ($input:expr) => {{
        let _pv_doc = &$input;
    }};
}

/// Postconditions for equation `fts5_consistency`.
/// Call before return: `contract_post_fts5_consistency!(result_expr)`
macro_rules! contract_post_fts5_consistency {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `fts5_consistency`.
macro_rules! contract_fts5_consistency {
    ($input:expr, $body:expr) => {{
        contract_pre_fts5_consistency!($input);
        let _contract_result = $body;
        contract_post_fts5_consistency!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `pagerank_convergence`.
/// Domain-specific. Call: `contract_pre_pagerank_convergence!(slice_expr)`
macro_rules! contract_pre_pagerank_convergence {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
    }};
}

/// Postconditions for equation `pagerank_convergence`.
/// Call before return: `contract_post_pagerank_convergence!(result_expr)`
macro_rules! contract_post_pagerank_convergence {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `pagerank_convergence`.
macro_rules! contract_pagerank_convergence {
    ($input:expr, $body:expr) => {{
        contract_pre_pagerank_convergence!($input);
        let _contract_result = $body;
        contract_post_pagerank_convergence!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `sqlite_roundtrip`.
/// Call at function entry: `contract_pre_sqlite_roundtrip!(input_expr)`
macro_rules! contract_pre_sqlite_roundtrip {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `sqlite_roundtrip`.
/// Call before return: `contract_post_sqlite_roundtrip!(result_expr)`
macro_rules! contract_post_sqlite_roundtrip {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `sqlite_roundtrip`.
macro_rules! contract_sqlite_roundtrip {
    ($input:expr, $body:expr) => {{
        contract_pre_sqlite_roundtrip!($input);
        let _contract_result = $body;
        contract_post_sqlite_roundtrip!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/graph-query-v1.yaml — DO NOT EDIT
// Contract: graph-query-v1

/// Preconditions for equation `bfs_correctness`.
/// Call at function entry: `contract_pre_bfs_correctness!(input_expr)`
macro_rules! contract_pre_bfs_correctness {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `pagerank_convergence`.
/// Domain-specific. Call: `contract_pre_pagerank_convergence!(slice_expr)`
macro_rules! contract_pre_pagerank_convergence {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
    }};
}

// Auto-generated from contracts/http-api-v1.yaml — DO NOT EDIT
// Contract: http-api-v1

/// Preconditions for equation `cors_negotiation`.
/// Call at function entry: `contract_pre_cors_negotiation!(input_expr)`
macro_rules! contract_pre_cors_negotiation {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `cors_negotiation`.
/// Call before return: `contract_post_cors_negotiation!(result_expr)`
macro_rules! contract_post_cors_negotiation {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `cors_negotiation`.
macro_rules! contract_cors_negotiation {
    ($input:expr, $body:expr) => {{
        contract_pre_cors_negotiation!($input);
        let _contract_result = $body;
        contract_post_cors_negotiation!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `error_envelope_preservation`.
/// Call at function entry: `contract_pre_error_envelope_preservation!(input_expr)`
macro_rules! contract_pre_error_envelope_preservation {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `error_envelope_preservation`.
/// Call before return: `contract_post_error_envelope_preservation!(result_expr)`
macro_rules! contract_post_error_envelope_preservation {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `error_envelope_preservation`.
macro_rules! contract_error_envelope_preservation {
    ($input:expr, $body:expr) => {{
        contract_pre_error_envelope_preservation!($input);
        let _contract_result = $body;
        contract_post_error_envelope_preservation!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `request_response_schema`.
/// Call at function entry: `contract_pre_request_response_schema!(input_expr)`
macro_rules! contract_pre_request_response_schema {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `request_response_schema`.
/// Call before return: `contract_post_request_response_schema!(result_expr)`
macro_rules! contract_post_request_response_schema {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `request_response_schema`.
macro_rules! contract_request_response_schema {
    ($input:expr, $body:expr) => {{
        contract_pre_request_response_schema!($input);
        let _contract_result = $body;
        contract_post_request_response_schema!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `timeout_honoring`.
/// Call at function entry: `contract_pre_timeout_honoring!(input_expr)`
macro_rules! contract_pre_timeout_honoring {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `timeout_honoring`.
/// Call before return: `contract_post_timeout_honoring!(result_expr)`
macro_rules! contract_post_timeout_honoring {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `timeout_honoring`.
macro_rules! contract_timeout_honoring {
    ($input:expr, $body:expr) => {{
        contract_pre_timeout_honoring!($input);
        let _contract_result = $body;
        contract_post_timeout_honoring!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/http-api-v1.yaml — DO NOT EDIT
// Contract: http-api-v1

/// Preconditions for equation `body_schema_compliance`.
/// Call at function entry: `contract_pre_body_schema_compliance!(input_expr)`
macro_rules! contract_pre_body_schema_compliance {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `body_schema_compliance`.
/// Call before return: `contract_post_body_schema_compliance!(result_expr)`
macro_rules! contract_post_body_schema_compliance {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `body_schema_compliance`.
macro_rules! contract_body_schema_compliance {
    ($input:expr, $body:expr) => {{
        contract_pre_body_schema_compliance!($input);
        let _contract_result = $body;
        contract_post_body_schema_compliance!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `max_tokens_cap`.
/// Call at function entry: `contract_pre_max_tokens_cap!(input_expr)`
macro_rules! contract_pre_max_tokens_cap {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `max_tokens_cap`.
/// Call before return: `contract_post_max_tokens_cap!(result_expr)`
macro_rules! contract_post_max_tokens_cap {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `max_tokens_cap`.
macro_rules! contract_max_tokens_cap {
    ($input:expr, $body:expr) => {{
        contract_pre_max_tokens_cap!($input);
        let _contract_result = $body;
        contract_post_max_tokens_cap!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `response_schema`.
/// Call at function entry: `contract_pre_response_schema!(input_expr)`
macro_rules! contract_pre_response_schema {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `response_schema`.
/// Call before return: `contract_post_response_schema!(result_expr)`
macro_rules! contract_post_response_schema {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `response_schema`.
macro_rules! contract_response_schema {
    ($input:expr, $body:expr) => {{
        contract_pre_response_schema!($input);
        let _contract_result = $body;
        contract_post_response_schema!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `thinking_block_strip`.
/// Call at function entry: `contract_pre_thinking_block_strip!(input_expr)`
macro_rules! contract_pre_thinking_block_strip {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `thinking_block_strip`.
/// Call before return: `contract_post_thinking_block_strip!(result_expr)`
macro_rules! contract_post_thinking_block_strip {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `thinking_block_strip`.
macro_rules! contract_thinking_block_strip {
    ($input:expr, $body:expr) => {{
        contract_pre_thinking_block_strip!($input);
        let _contract_result = $body;
        contract_post_thinking_block_strip!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `tool_format_fidelity`.
/// Domain-specific. Call: `contract_pre_tool_format_fidelity!(slice_expr)`
macro_rules! contract_pre_tool_format_fidelity {
    () => {{}};
    ($input:expr) => {{
        let _pv_tools = &$input;
        debug_assert!(_pv_tools.len() > 0,
            "Contract tool_format_fidelity: precondition violated — tools.len() > 0");
    }};
}

/// Postconditions for equation `tool_format_fidelity`.
/// Call before return: `contract_post_tool_format_fidelity!(result_expr)`
macro_rules! contract_post_tool_format_fidelity {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `tool_format_fidelity`.
macro_rules! contract_tool_format_fidelity {
    ($input:expr, $body:expr) => {{
        contract_pre_tool_format_fidelity!($input);
        let _contract_result = $body;
        contract_post_tool_format_fidelity!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/http-client-v1.yaml — DO NOT EDIT
// Contract: http-client-v1

/// Preconditions for equation `error_propagation`.
/// Call at function entry: `contract_pre_error_propagation!(input_expr)`
macro_rules! contract_pre_error_propagation {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `error_propagation`.
/// Call before return: `contract_post_error_propagation!(result_expr)`
macro_rules! contract_post_error_propagation {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `error_propagation`.
macro_rules! contract_error_propagation {
    ($input:expr, $body:expr) => {{
        contract_pre_error_propagation!($input);
        let _contract_result = $body;
        contract_post_error_propagation!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `lru_cache_eviction`.
/// Call at function entry: `contract_pre_lru_cache_eviction!(input_expr)`
macro_rules! contract_pre_lru_cache_eviction {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `lru_cache_eviction`.
/// Call before return: `contract_post_lru_cache_eviction!(result_expr)`
macro_rules! contract_post_lru_cache_eviction {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `lru_cache_eviction`.
macro_rules! contract_lru_cache_eviction {
    ($input:expr, $body:expr) => {{
        contract_pre_lru_cache_eviction!($input);
        let _contract_result = $body;
        contract_post_lru_cache_eviction!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `multi_tier_routing`.
/// Call at function entry: `contract_pre_multi_tier_routing!(input_expr)`
macro_rules! contract_pre_multi_tier_routing {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `multi_tier_routing`.
/// Call before return: `contract_post_multi_tier_routing!(result_expr)`
macro_rules! contract_post_multi_tier_routing {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `multi_tier_routing`.
macro_rules! contract_multi_tier_routing {
    ($input:expr, $body:expr) => {{
        contract_pre_multi_tier_routing!($input);
        let _contract_result = $body;
        contract_post_multi_tier_routing!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `request_construction`.
/// Call at function entry: `contract_pre_request_construction!(input_expr)`
macro_rules! contract_pre_request_construction {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `request_construction`.
/// Call before return: `contract_post_request_construction!(result_expr)`
macro_rules! contract_post_request_construction {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `request_construction`.
macro_rules! contract_request_construction {
    ($input:expr, $body:expr) => {{
        contract_pre_request_construction!($input);
        let _contract_result = $body;
        contract_post_request_construction!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `response_parsing`.
/// Call at function entry: `contract_pre_response_parsing!(input_expr)`
macro_rules! contract_pre_response_parsing {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `response_parsing`.
/// Call before return: `contract_post_response_parsing!(result_expr)`
macro_rules! contract_post_response_parsing {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `response_parsing`.
macro_rules! contract_response_parsing {
    ($input:expr, $body:expr) => {{
        contract_pre_response_parsing!($input);
        let _contract_result = $body;
        contract_post_response_parsing!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `ssrf_prevention`.
/// Call at function entry: `contract_pre_ssrf_prevention!(input_expr)`
macro_rules! contract_pre_ssrf_prevention {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `ssrf_prevention`.
/// Call before return: `contract_post_ssrf_prevention!(result_expr)`
macro_rules! contract_post_ssrf_prevention {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `ssrf_prevention`.
macro_rules! contract_ssrf_prevention {
    ($input:expr, $body:expr) => {{
        contract_pre_ssrf_prevention!($input);
        let _contract_result = $body;
        contract_post_ssrf_prevention!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `url_validation`.
/// Call at function entry: `contract_pre_url_validation!(input_expr)`
macro_rules! contract_pre_url_validation {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `url_validation`.
/// Call before return: `contract_post_url_validation!(result_expr)`
macro_rules! contract_post_url_validation {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `url_validation`.
macro_rules! contract_url_validation {
    ($input:expr, $body:expr) => {{
        contract_pre_url_validation!($input);
        let _contract_result = $body;
        contract_post_url_validation!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/hybrid-layer-dispatch-v1.yaml — DO NOT EDIT
// Contract: hybrid-layer-dispatch-v1

/// Preconditions for equation `conv1d_causal`.
/// Domain-specific. Call: `contract_pre_conv1d_causal!(slice_expr)`
macro_rules! contract_pre_conv1d_causal {
    () => {{}};
    ($input:expr) => {{
        let _pv_a = &$input;
        debug_assert!(_pv_a.len() > 0,
            "Contract conv1d_causal: precondition violated — a.len() > 0");
    }};
}

/// Preconditions for equation `head_grouping`.
/// Domain-specific. Call: `contract_pre_head_grouping!(slice_expr)`
macro_rules! contract_pre_head_grouping {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract head_grouping: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract head_grouping: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `hybrid_dispatch`.
/// Domain-specific. Call: `contract_pre_hybrid_dispatch!(slice_expr)`
macro_rules! contract_pre_hybrid_dispatch {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract hybrid_dispatch: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract hybrid_dispatch: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `linear_associativity`.
/// Domain-specific. Call: `contract_pre_linear_associativity!(slice_expr)`
macro_rules! contract_pre_linear_associativity {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract linear_associativity: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract linear_associativity: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `linear_no_softmax`.
/// Domain-specific. Call: `contract_pre_linear_no_softmax!(slice_expr)`
macro_rules! contract_pre_linear_no_softmax {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
        debug_assert!(_pv_x.iter().all(|v| v.is_finite()),
            "Contract linear_no_softmax: precondition violated — x.iter().all(|v| v.is_finite())");
        debug_assert!(_pv_x.len() > 0,
            "Contract linear_no_softmax: precondition violated — x.len() > 0");
    }};
}

/// Preconditions for equation `linear_shapes`.
/// Domain-specific. Call: `contract_pre_linear_shapes!(slice_expr)`
macro_rules! contract_pre_linear_shapes {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract linear_shapes: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract linear_shapes: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

// Auto-generated from contracts/ica-v1.yaml — DO NOT EDIT
// Contract: ica-v1

/// Preconditions for equation `fastica`.
/// Domain-specific. Call: `contract_pre_fastica!(slice_expr)`
macro_rules! contract_pre_fastica {
    () => {{}};
    ($input:expr) => {{
        let _pv_a = &$input;
        debug_assert!(_pv_a.len() > 0,
            "Contract fastica: precondition violated — a.len() > 0");
    }};
}

/// Preconditions for equation `mixing`.
/// Domain-specific. Call: `contract_pre_mixing!(slice_expr)`
macro_rules! contract_pre_mixing {
    () => {{}};
    ($input:expr) => {{
        let _pv_a = &$input;
        debug_assert!(_pv_a.len() > 0,
            "Contract mixing: precondition violated — a.len() > 0");
    }};
}

/// Preconditions for equation `unmixing`.
/// Domain-specific. Call: `contract_pre_unmixing!(slice_expr)`
macro_rules! contract_pre_unmixing {
    () => {{}};
    ($input:expr) => {{
        let _pv_a = &$input;
        debug_assert!(_pv_a.len() > 0,
            "Contract unmixing: precondition violated — a.len() > 0");
    }};
}

// Auto-generated from contracts/inference-pipeline-v1.yaml — DO NOT EDIT
// Contract: inference-pipeline-v1

/// Preconditions for equation `decode_step`.
/// Domain-specific. Call: `contract_pre_decode_step!(slice_expr)`
macro_rules! contract_pre_decode_step {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract decode_step: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `hybrid_layer_schedule`.
/// Call at function entry: `contract_pre_hybrid_layer_schedule!(input_expr)`
macro_rules! contract_pre_hybrid_layer_schedule {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `kv_cache_growth`.
/// Domain-specific. Call: `contract_pre_kv_cache_growth!(slice_expr)`
macro_rules! contract_pre_kv_cache_growth {
    () => {{}};
    ($input:expr) => {{
        let _pv_q = &$input;
        debug_assert!(_pv_q.len() > 0,
            "Contract kv_cache_growth: precondition violated — q.len() > 0");
    }};
}

/// Preconditions for equation `layer_composition`.
/// Domain-specific. Call: `contract_pre_layer_composition!(slice_expr)`
macro_rules! contract_pre_layer_composition {
    () => {{}};
    ($input:expr) => {{
        let _pv_indices = &$input;
        debug_assert!(_pv_indices.len() > 0,
            "Contract layer_composition: precondition violated — indices.len() > 0");
    }};
}

/// Preconditions for equation `prefill_phase`.
/// Call at function entry: `contract_pre_prefill_phase!(input_expr)`
macro_rules! contract_pre_prefill_phase {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `residual_stream`.
/// Call at function entry: `contract_pre_residual_stream!(input_expr)`
macro_rules! contract_pre_residual_stream {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

// Auto-generated from contracts/inference-pipeline-v1.yaml — DO NOT EDIT
// Contract: inference-pipeline-v1

/// Preconditions for equation `decode_step`.
/// Call at function entry: `contract_pre_decode_step!(input_expr)`
macro_rules! contract_pre_decode_step {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `prefill_phase`.
/// Domain-specific. Call: `contract_pre_prefill_phase!(slice_expr)`
macro_rules! contract_pre_prefill_phase {
    () => {{}};
    ($input:expr) => {{
        let _pv_tokens = &$input;
        debug_assert!(_pv_tokens.len() > 0,
            "Contract prefill_phase: precondition violated — tokens.len() > 0");
    }};
}

/// Preconditions for equation `sampling_temperature`.
/// Domain-specific. Call: `contract_pre_sampling_temperature!(slice_expr)`
macro_rules! contract_pre_sampling_temperature {
    () => {{}};
    ($input:expr) => {{
        let _pv_logits = &$input;
    }};
}

// Auto-generated from contracts/int8-symmetric-quant-v1.yaml — DO NOT EDIT
// Contract: int8-symmetric-quant-v1

/// Preconditions for equation `dequant_dot`.
/// Domain-specific. Call: `contract_pre_dequant_dot!(slice_expr)`
macro_rules! contract_pre_dequant_dot {
    () => {{}};
    ($input:expr) => {{
        let _pv_a = &$input;
        debug_assert!(_pv_a.len() > 0,
            "Contract dequant_dot: precondition violated — a.len() > 0");
    }};
}

/// Preconditions for equation `per_row_scale`.
/// Domain-specific. Call: `contract_pre_per_row_scale!(slice_expr)`
macro_rules! contract_pre_per_row_scale {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract per_row_scale: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `quantize`.
/// Domain-specific. Call: `contract_pre_quantize!(slice_expr)`
macro_rules! contract_pre_quantize {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract quantize: precondition violated — input.len() > 0");
    }};
}

// Auto-generated from contracts/iterator-v1.yaml — DO NOT EDIT
// Contract: iterator-v1

/// Preconditions for equation `iterator`.
/// Domain-specific. Call: `contract_pre_iterator!(slice_expr)`
macro_rules! contract_pre_iterator {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract iterator: precondition violated — input.len() > 0");
    }};
}

// Auto-generated from contracts/kernel-fusion-v1.yaml — DO NOT EDIT
// Contract: kernel-fusion-v1

/// Preconditions for equation `fusion_decision_registry`.
/// Call at function entry: `contract_pre_fusion_decision_registry!(input_expr)`
macro_rules! contract_pre_fusion_decision_registry {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `fusion_performance`.
/// Domain-specific. Call: `contract_pre_fusion_performance!(slice_expr)`
macro_rules! contract_pre_fusion_performance {
    () => {{}};
    ($input:expr) => {{
        let _pv_benchmark = &$input;
    }};
}

/// Preconditions for equation `identity`.
/// Domain-specific. Call: `contract_pre_identity!(slice_expr)`
macro_rules! contract_pre_identity {
    () => {{}};
    ($input:expr) => {{
        let _pv_q = &$input;
        debug_assert!(_pv_q.len() > 0,
            "Contract identity: precondition violated — q.len() > 0");
    }};
}

// Auto-generated from contracts/kernel-launch-budget-v1.yaml — DO NOT EDIT
// Contract: kernel-launch-budget-v1

/// Preconditions for equation `bsum_budget`.
/// Domain-specific. Call: `contract_pre_bsum_budget!(slice_expr)`
macro_rules! contract_pre_bsum_budget {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract bsum_budget: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `per_layer_decomposition`.
/// Domain-specific. Call: `contract_pre_per_layer_decomposition!(slice_expr)`
macro_rules! contract_pre_per_layer_decomposition {
    () => {{}};
    ($input:expr) => {{
        let _pv_indices = &$input;
        debug_assert!(_pv_indices.len() > 0,
            "Contract per_layer_decomposition: precondition violated — indices.len() > 0");
    }};
}

/// Preconditions for equation `per_token_launches`.
/// Domain-specific. Call: `contract_pre_per_token_launches!(slice_expr)`
macro_rules! contract_pre_per_token_launches {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract per_token_launches: precondition violated — input.len() > 0");
    }};
}

// Auto-generated from contracts/kmeans-kernel-v1.yaml — DO NOT EDIT
// Contract: kmeans-kernel-v1

/// Preconditions for equation `assignment`.
/// Domain-specific. Call: `contract_pre_assignment!(slice_expr)`
macro_rules! contract_pre_assignment {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract assignment: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract assignment: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `objective`.
/// Domain-specific. Call: `contract_pre_objective!(slice_expr)`
macro_rules! contract_pre_objective {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract objective: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract objective: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `update`.
/// Domain-specific. Call: `contract_pre_update!(slice_expr)`
macro_rules! contract_pre_update {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract update: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract update: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

// Auto-generated from contracts/kv-cache-equivalence-v1.yaml — DO NOT EDIT
// Contract: kv-cache-equivalence-v1

/// Preconditions for equation `batched_serial_equivalence`.
/// Call at function entry: `contract_pre_batched_serial_equivalence!(input_expr)`
macro_rules! contract_pre_batched_serial_equivalence {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `fused_kernel`.
/// Call at function entry: `contract_pre_fused_kernel!(input_expr)`
macro_rules! contract_pre_fused_kernel {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `page_shape`.
/// Call at function entry: `contract_pre_page_shape!(input_expr)`
macro_rules! contract_pre_page_shape {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `prefill_incremental`.
/// Call at function entry: `contract_pre_prefill_incremental!(input_expr)`
macro_rules! contract_pre_prefill_incremental {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

// Auto-generated from contracts/kv-cache-sizing-v1.yaml — DO NOT EDIT
// Contract: kv-cache-sizing-v1

/// Preconditions for equation `bias_absence`.
/// Call at function entry: `contract_pre_bias_absence!(input_expr)`
macro_rules! contract_pre_bias_absence {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `hybrid_accounting`.
/// Call at function entry: `contract_pre_hybrid_accounting!(input_expr)`
macro_rules! contract_pre_hybrid_accounting {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `per_token_per_layer`.
/// Domain-specific. Call: `contract_pre_per_token_per_layer!(slice_expr)`
macro_rules! contract_pre_per_token_per_layer {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract per_token_per_layer: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `total_kv_memory`.
/// Call at function entry: `contract_pre_total_kv_memory!(input_expr)`
macro_rules! contract_pre_total_kv_memory {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `zero_input_identity`.
/// Call at function entry: `contract_pre_zero_input_identity!(input_expr)`
macro_rules! contract_pre_zero_input_identity {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

// Auto-generated from contracts/layer-parity-v1.yaml — DO NOT EDIT
// Contract: layer-parity-v1

/// Preconditions for equation `cosine_parity_gate`.
/// Domain-specific. Call: `contract_pre_cosine_parity_gate!(slice_expr)`
macro_rules! contract_pre_cosine_parity_gate {
    () => {{}};
    ($input:expr) => {{
        let _pv_cpu_logits = &$input;
        debug_assert!(_pv_cpu_logits.len() > 0,
            "Contract cosine_parity_gate: precondition violated — cpu_logits.len() > 0");
    }};
}

/// Preconditions for equation `identity`.
/// Domain-specific. Call: `contract_pre_identity!(slice_expr)`
macro_rules! contract_pre_identity {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
    }};
}

/// Preconditions for equation `layer_parity`.
/// Domain-specific. Call: `contract_pre_layer_parity!(slice_expr)`
macro_rules! contract_pre_layer_parity {
    () => {{}};
    ($input:expr) => {{
        let _pv_cpu_output = &$input;
    }};
}

// Auto-generated from contracts/layernorm-kernel-v1.yaml — DO NOT EDIT
// Contract: layernorm-kernel-v1

/// Preconditions for equation `layernorm`.
/// Domain-specific. Call: `contract_pre_layernorm!(slice_expr)`
macro_rules! contract_pre_layernorm {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
        debug_assert!(_pv_x.len() > 0,
            "Contract layernorm: precondition violated — x.len() > 0");
        debug_assert!(_pv_x.iter().all(|v| v.is_finite()),
            "Contract layernorm: precondition violated — x.iter().all(|v| v.is_finite())");
    }};
}

/// Postconditions for equation `layernorm`.
/// Call before return: `contract_post_layernorm!(result_expr)`
macro_rules! contract_post_layernorm {
    ($result:expr) => {{
        let _contract_result = &$result;
        debug_assert!(_contract_result.iter().all(|v| v.is_finite()), "Contract layernorm: postcondition violated — result.iter().all(|v| v.is_finite())");
    }};
}

/// Combined pre+post contract for equation `layernorm`.
macro_rules! contract_layernorm {
    ($input:expr, $body:expr) => {{
        contract_pre_layernorm!($input);
        let _contract_result = $body;
        contract_post_layernorm!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `statistics`.
/// Domain-specific. Call: `contract_pre_statistics!(slice_expr)`
macro_rules! contract_pre_statistics {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract statistics: precondition violated — input.iter().all(|v| v.is_finite())");
        debug_assert!(_pv_input.len() > 0,
            "Contract statistics: precondition violated — input.len() > 0");
    }};
}

// Auto-generated from contracts/lbfgs-kernel-v1.yaml — DO NOT EDIT
// Contract: lbfgs-kernel-v1

/// Preconditions for equation `line_search`.
/// Domain-specific. Call: `contract_pre_line_search!(slice_expr)`
macro_rules! contract_pre_line_search {
    () => {{}};
    ($input:expr) => {{
        let _pv_params = &$input;
        debug_assert!(_pv_params.len() > 0,
            "Contract line_search: precondition violated — params.len() > 0");
    }};
}

/// Preconditions for equation `secant_condition`.
/// Domain-specific. Call: `contract_pre_secant_condition!(slice_expr)`
macro_rules! contract_pre_secant_condition {
    () => {{}};
    ($input:expr) => {{
        let _pv_params = &$input;
        debug_assert!(_pv_params.len() > 0,
            "Contract secant_condition: precondition violated — params.len() > 0");
    }};
}

/// Preconditions for equation `two_loop_recursion`.
/// Domain-specific. Call: `contract_pre_two_loop_recursion!(slice_expr)`
macro_rules! contract_pre_two_loop_recursion {
    () => {{}};
    ($input:expr) => {{
        let _pv_params = &$input;
        debug_assert!(_pv_params.len() > 0,
            "Contract two_loop_recursion: precondition violated — params.len() > 0");
    }};
}

// Auto-generated from contracts/learned-position-embedding-v1.yaml — DO NOT EDIT
// Contract: learned-position-embedding-v1

/// Preconditions for equation `position_embedding`.
/// Domain-specific. Call: `contract_pre_position_embedding!(slice_expr)`
macro_rules! contract_pre_position_embedding {
    () => {{}};
    ($input:expr) => {{
        let _pv_indices = &$input;
        debug_assert!(_pv_indices.len() > 0,
            "Contract position_embedding: precondition violated — indices.len() > 0");
    }};
}

// Auto-generated from contracts/linear-models-v1.yaml — DO NOT EDIT
// Contract: linear-models-v1

/// Preconditions for equation `logistic_predict_proba`.
/// Domain-specific. Call: `contract_pre_logistic_predict_proba!(slice_expr)`
macro_rules! contract_pre_logistic_predict_proba {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract logistic_predict_proba: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract logistic_predict_proba: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `ols_fit`.
/// Domain-specific. Call: `contract_pre_ols_fit!(slice_expr)`
macro_rules! contract_pre_ols_fit {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract ols_fit: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract ols_fit: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `ols_predict`.
/// Domain-specific. Call: `contract_pre_ols_predict!(slice_expr)`
macro_rules! contract_pre_ols_predict {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract ols_predict: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract ols_predict: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `r_squared_training`.
/// Domain-specific. Call: `contract_pre_r_squared_training!(slice_expr)`
macro_rules! contract_pre_r_squared_training {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract r_squared_training: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract r_squared_training: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

// Auto-generated from contracts/linear-probe-classifier-v1.yaml — DO NOT EDIT
// Contract: linear-probe-classifier-v1

/// Preconditions for equation `linear_probe`.
/// Domain-specific. Call: `contract_pre_linear_probe!(slice_expr)`
macro_rules! contract_pre_linear_probe {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract linear_probe: precondition violated — input.len() > 0");
    }};
}

// Auto-generated from contracts/linear-projection-v1.yaml — DO NOT EDIT
// Contract: linear-projection-v1

/// Preconditions for equation `linear_forward`.
/// Domain-specific. Call: `contract_pre_linear_forward!(slice_expr)`
macro_rules! contract_pre_linear_forward {
    () => {{}};
    ($input:expr) => {{
        let _pv_a = &$input;
        debug_assert!(_pv_a.len() > 0,
            "Contract linear_forward: precondition violated — a.len() > 0");
    }};
}

/// Preconditions for equation `linear_no_bias`.
/// Domain-specific. Call: `contract_pre_linear_no_bias!(slice_expr)`
macro_rules! contract_pre_linear_no_bias {
    () => {{}};
    ($input:expr) => {{
        let _pv_a = &$input;
        debug_assert!(_pv_a.len() > 0,
            "Contract linear_no_bias: precondition violated — a.len() > 0");
    }};
}

// Auto-generated from contracts/lora-algebra-v1.yaml — DO NOT EDIT
// Contract: lora-algebra-v1

/// Preconditions for equation `dare_unbiased`.
/// Domain-specific. Call: `contract_pre_dare_unbiased!(slice_expr)`
macro_rules! contract_pre_dare_unbiased {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract dare_unbiased: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract dare_unbiased: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `eckart_young`.
/// Domain-specific. Call: `contract_pre_eckart_young!(slice_expr)`
macro_rules! contract_pre_eckart_young {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract eckart_young: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract eckart_young: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `lora_shape`.
/// Domain-specific. Call: `contract_pre_lora_shape!(slice_expr)`
macro_rules! contract_pre_lora_shape {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract lora_shape: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract lora_shape: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `shape_preservation`.
/// Domain-specific. Call: `contract_pre_shape_preservation!(slice_expr)`
macro_rules! contract_pre_shape_preservation {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract shape_preservation: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract shape_preservation: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `task_vector`.
/// Domain-specific. Call: `contract_pre_task_vector!(slice_expr)`
macro_rules! contract_pre_task_vector {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract task_vector: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract task_vector: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

// Auto-generated from contracts/lora-gradient-flow-v1.yaml — DO NOT EDIT
// Contract: lora-gradient-flow-v1

/// Preconditions for equation `lora_forward`.
/// Domain-specific. Call: `contract_pre_lora_forward!(slice_expr)`
macro_rules! contract_pre_lora_forward {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
    }};
}

/// Postconditions for equation `lora_forward`.
/// Call before return: `contract_post_lora_forward!(result_expr)`
macro_rules! contract_post_lora_forward {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `lora_forward`.
macro_rules! contract_lora_forward {
    ($input:expr, $body:expr) => {{
        contract_pre_lora_forward!($input);
        let _contract_result = $body;
        contract_post_lora_forward!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/lora-target-selection-v1.yaml — DO NOT EDIT
// Contract: lora-target-selection-v1

// Auto-generated from contracts/loss-functions-v1.yaml — DO NOT EDIT
// Contract: loss-functions-v1

/// Preconditions for equation `bce`.
/// Domain-specific. Call: `contract_pre_bce!(slice_expr)`
macro_rules! contract_pre_bce {
    () => {{}};
    ($input:expr) => {{
        let _pv_predicted = &$input;
        debug_assert!(_pv_predicted.len() > 0,
            "Contract bce: precondition violated — predicted.len() > 0");
    }};
}

/// Preconditions for equation `huber`.
/// Domain-specific. Call: `contract_pre_huber!(slice_expr)`
macro_rules! contract_pre_huber {
    () => {{}};
    ($input:expr) => {{
        let _pv_predicted = &$input;
        debug_assert!(_pv_predicted.len() > 0,
            "Contract huber: precondition violated — predicted.len() > 0");
    }};
}

/// Preconditions for equation `l1_loss`.
/// Domain-specific. Call: `contract_pre_l1_loss!(slice_expr)`
macro_rules! contract_pre_l1_loss {
    () => {{}};
    ($input:expr) => {{
        let _pv_predicted = &$input;
        debug_assert!(_pv_predicted.len() > 0,
            "Contract l1_loss: precondition violated — predicted.len() > 0");
    }};
}

/// Preconditions for equation `mse_loss`.
/// Domain-specific. Call: `contract_pre_mse_loss!(slice_expr)`
macro_rules! contract_pre_mse_loss {
    () => {{}};
    ($input:expr) => {{
        let _pv_predicted = &$input;
        debug_assert!(_pv_predicted.len() > 0,
            "Contract mse_loss: precondition violated — predicted.len() > 0");
    }};
}

/// Preconditions for equation `nll`.
/// Domain-specific. Call: `contract_pre_nll!(slice_expr)`
macro_rules! contract_pre_nll {
    () => {{}};
    ($input:expr) => {{
        let _pv_predicted = &$input;
        debug_assert!(_pv_predicted.len() > 0,
            "Contract nll: precondition violated — predicted.len() > 0");
    }};
}

/// Preconditions for equation `smooth_l1`.
/// Domain-specific. Call: `contract_pre_smooth_l1!(slice_expr)`
macro_rules! contract_pre_smooth_l1 {
    () => {{}};
    ($input:expr) => {{
        let _pv_predicted = &$input;
        debug_assert!(_pv_predicted.len() > 0,
            "Contract smooth_l1: precondition violated — predicted.len() > 0");
    }};
}

// Auto-generated from contracts/matmul-kernel-v1.yaml — DO NOT EDIT
// Contract: matmul-kernel-v1

/// Preconditions for equation `matmul`.
/// Domain-specific. Call: `contract_pre_matmul!(slice_expr)`
macro_rules! contract_pre_matmul {
    () => {{}};
    ($input:expr) => {{
        let _pv_a = &$input;
    }};
}

/// Postconditions for equation `matmul`.
/// Call before return: `contract_post_matmul!(result_expr)`
macro_rules! contract_post_matmul {
    ($result:expr) => {{
        let _contract_result = &$result;
        debug_assert!(_contract_result.iter().all(|v| v.is_finite()), "Contract matmul: postcondition violated — result.iter().all(|v| v.is_finite())");
    }};
}

/// Combined pre+post contract for equation `matmul`.
macro_rules! contract_matmul {
    ($input:expr, $body:expr) => {{
        contract_pre_matmul!($input);
        let _contract_result = $body;
        contract_post_matmul!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `quantized_dot`.
/// Domain-specific. Call: `contract_pre_quantized_dot!(slice_expr)`
macro_rules! contract_pre_quantized_dot {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract quantized_dot: precondition violated — input.len() > 0");
    }};
}

// Auto-generated from contracts/mcp-protocol-sdk-v1.yaml — DO NOT EDIT
// Contract: mcp-protocol-sdk-v1

/// Preconditions for equation `batch_request_ordering`.
/// Call at function entry: `contract_pre_batch_request_ordering!(input_expr)`
macro_rules! contract_pre_batch_request_ordering {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `batch_request_ordering`.
/// Call before return: `contract_post_batch_request_ordering!(result_expr)`
macro_rules! contract_post_batch_request_ordering {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `batch_request_ordering`.
macro_rules! contract_batch_request_ordering {
    ($input:expr, $body:expr) => {{
        contract_pre_batch_request_ordering!($input);
        let _contract_result = $body;
        contract_post_batch_request_ordering!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `cancellation_safety`.
/// Call at function entry: `contract_pre_cancellation_safety!(input_expr)`
macro_rules! contract_pre_cancellation_safety {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `cancellation_safety`.
/// Call before return: `contract_post_cancellation_safety!(result_expr)`
macro_rules! contract_post_cancellation_safety {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `cancellation_safety`.
macro_rules! contract_cancellation_safety {
    ($input:expr, $body:expr) => {{
        contract_pre_cancellation_safety!($input);
        let _contract_result = $body;
        contract_post_cancellation_safety!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `error_code_mapping`.
/// Call at function entry: `contract_pre_error_code_mapping!(input_expr)`
macro_rules! contract_pre_error_code_mapping {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `error_code_mapping`.
/// Call before return: `contract_post_error_code_mapping!(result_expr)`
macro_rules! contract_post_error_code_mapping {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `error_code_mapping`.
macro_rules! contract_error_code_mapping {
    ($input:expr, $body:expr) => {{
        contract_pre_error_code_mapping!($input);
        let _contract_result = $body;
        contract_post_error_code_mapping!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `jsonrpc_framing`.
/// Call at function entry: `contract_pre_jsonrpc_framing!(input_expr)`
macro_rules! contract_pre_jsonrpc_framing {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `jsonrpc_framing`.
/// Call before return: `contract_post_jsonrpc_framing!(result_expr)`
macro_rules! contract_post_jsonrpc_framing {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `jsonrpc_framing`.
macro_rules! contract_jsonrpc_framing {
    ($input:expr, $body:expr) => {{
        contract_pre_jsonrpc_framing!($input);
        let _contract_result = $body;
        contract_post_jsonrpc_framing!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `payload_limits`.
/// Call at function entry: `contract_pre_payload_limits!(input_expr)`
macro_rules! contract_pre_payload_limits {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `payload_limits`.
/// Call before return: `contract_post_payload_limits!(result_expr)`
macro_rules! contract_post_payload_limits {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `payload_limits`.
macro_rules! contract_payload_limits {
    ($input:expr, $body:expr) => {{
        contract_pre_payload_limits!($input);
        let _contract_result = $body;
        contract_post_payload_limits!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `protocol_version_negotiation`.
/// Call at function entry: `contract_pre_protocol_version_negotiation!(input_expr)`
macro_rules! contract_pre_protocol_version_negotiation {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `protocol_version_negotiation`.
/// Call before return: `contract_post_protocol_version_negotiation!(result_expr)`
macro_rules! contract_post_protocol_version_negotiation {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `protocol_version_negotiation`.
macro_rules! contract_protocol_version_negotiation {
    ($input:expr, $body:expr) => {{
        contract_pre_protocol_version_negotiation!($input);
        let _contract_result = $body;
        contract_post_protocol_version_negotiation!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `session_lifecycle`.
/// Call at function entry: `contract_pre_session_lifecycle!(input_expr)`
macro_rules! contract_pre_session_lifecycle {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `session_lifecycle`.
/// Call before return: `contract_post_session_lifecycle!(result_expr)`
macro_rules! contract_post_session_lifecycle {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `session_lifecycle`.
macro_rules! contract_session_lifecycle {
    ($input:expr, $body:expr) => {{
        contract_pre_session_lifecycle!($input);
        let _contract_result = $body;
        contract_post_session_lifecycle!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `tool_dispatch_integrity`.
/// Call at function entry: `contract_pre_tool_dispatch_integrity!(input_expr)`
macro_rules! contract_pre_tool_dispatch_integrity {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `tool_dispatch_integrity`.
/// Call before return: `contract_post_tool_dispatch_integrity!(result_expr)`
macro_rules! contract_post_tool_dispatch_integrity {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `tool_dispatch_integrity`.
macro_rules! contract_tool_dispatch_integrity {
    ($input:expr, $body:expr) => {{
        contract_pre_tool_dispatch_integrity!($input);
        let _contract_result = $body;
        contract_post_tool_dispatch_integrity!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `transport_abstraction`.
/// Call at function entry: `contract_pre_transport_abstraction!(input_expr)`
macro_rules! contract_pre_transport_abstraction {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `transport_abstraction`.
/// Call before return: `contract_post_transport_abstraction!(result_expr)`
macro_rules! contract_post_transport_abstraction {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `transport_abstraction`.
macro_rules! contract_transport_abstraction {
    ($input:expr, $body:expr) => {{
        contract_pre_transport_abstraction!($input);
        let _contract_result = $body;
        contract_post_transport_abstraction!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/mcp-protocol-v1.yaml — DO NOT EDIT
// Contract: mcp-protocol-v1

/// Preconditions for equation `error_mapping_lossless`.
/// Call at function entry: `contract_pre_error_mapping_lossless!(input_expr)`
macro_rules! contract_pre_error_mapping_lossless {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `error_mapping_lossless`.
/// Call before return: `contract_post_error_mapping_lossless!(result_expr)`
macro_rules! contract_post_error_mapping_lossless {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `error_mapping_lossless`.
macro_rules! contract_error_mapping_lossless {
    ($input:expr, $body:expr) => {{
        contract_pre_error_mapping_lossless!($input);
        let _contract_result = $body;
        contract_post_error_mapping_lossless!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `idempotency`.
/// Call at function entry: `contract_pre_idempotency!(input_expr)`
macro_rules! contract_pre_idempotency {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `idempotency`.
/// Call before return: `contract_post_idempotency!(result_expr)`
macro_rules! contract_post_idempotency {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `idempotency`.
macro_rules! contract_idempotency {
    ($input:expr, $body:expr) => {{
        contract_pre_idempotency!($input);
        let _contract_result = $body;
        contract_post_idempotency!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `session_lifecycle`.
/// Call at function entry: `contract_pre_session_lifecycle!(input_expr)`
macro_rules! contract_pre_session_lifecycle {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `session_lifecycle`.
/// Call before return: `contract_post_session_lifecycle!(result_expr)`
macro_rules! contract_post_session_lifecycle {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `session_lifecycle`.
macro_rules! contract_session_lifecycle {
    ($input:expr, $body:expr) => {{
        contract_pre_session_lifecycle!($input);
        let _contract_result = $body;
        contract_post_session_lifecycle!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `tool_schema_fidelity`.
/// Call at function entry: `contract_pre_tool_schema_fidelity!(input_expr)`
macro_rules! contract_pre_tool_schema_fidelity {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `tool_schema_fidelity`.
/// Call before return: `contract_post_tool_schema_fidelity!(result_expr)`
macro_rules! contract_post_tool_schema_fidelity {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `tool_schema_fidelity`.
macro_rules! contract_tool_schema_fidelity {
    ($input:expr, $body:expr) => {{
        contract_pre_tool_schema_fidelity!($input);
        let _contract_result = $body;
        contract_post_tool_schema_fidelity!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/mcp-tool-schema-v1.yaml — DO NOT EDIT
// Contract: mcp-tool-schema-v1

/// Preconditions for equation `error_mapping`.
/// Call at function entry: `contract_pre_error_mapping!(input_expr)`
macro_rules! contract_pre_error_mapping {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `idempotency_classification`.
/// Call at function entry: `contract_pre_idempotency_classification!(input_expr)`
macro_rules! contract_pre_idempotency_classification {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `session_state_machine`.
/// Call at function entry: `contract_pre_session_state_machine!(input_expr)`
macro_rules! contract_pre_session_state_machine {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `tool_schema_fidelity`.
/// Domain-specific. Call: `contract_pre_tool_schema_fidelity!(slice_expr)`
macro_rules! contract_pre_tool_schema_fidelity {
    () => {{}};
    ($input:expr) => {{
        let _pv_tool = &$input;
    }};
}

// Auto-generated from contracts/media-pipeline-v1.yaml — DO NOT EDIT
// Contract: media-pipeline-v1

/// Preconditions for equation `codec_dispatch`.
/// Call at function entry: `contract_pre_codec_dispatch!(input_expr)`
macro_rules! contract_pre_codec_dispatch {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `encode_decode_roundtrip`.
/// Call at function entry: `contract_pre_encode_decode_roundtrip!(input_expr)`
macro_rules! contract_pre_encode_decode_roundtrip {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `frame_integrity`.
/// Call at function entry: `contract_pre_frame_integrity!(input_expr)`
macro_rules! contract_pre_frame_integrity {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

// Auto-generated from contracts/memory-safety-v1.yaml — DO NOT EDIT
// Contract: memory-safety-v1

/// Preconditions for equation `bounds_safety`.
/// Domain-specific. Call: `contract_pre_bounds_safety!(slice_expr)`
macro_rules! contract_pre_bounds_safety {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract bounds_safety: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `drop_safety`.
/// Domain-specific. Call: `contract_pre_drop_safety!(slice_expr)`
macro_rules! contract_pre_drop_safety {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract drop_safety: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `escape_analysis`.
/// Domain-specific. Call: `contract_pre_escape_analysis!(slice_expr)`
macro_rules! contract_pre_escape_analysis {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract escape_analysis: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `lifetime_safety`.
/// Domain-specific. Call: `contract_pre_lifetime_safety!(slice_expr)`
macro_rules! contract_pre_lifetime_safety {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract lifetime_safety: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `ownership_invariant`.
/// Domain-specific. Call: `contract_pre_ownership_invariant!(slice_expr)`
macro_rules! contract_pre_ownership_invariant {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract ownership_invariant: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `use_after_move`.
/// Domain-specific. Call: `contract_pre_use_after_move!(slice_expr)`
macro_rules! contract_pre_use_after_move {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract use_after_move: precondition violated — input.len() > 0");
    }};
}

// Auto-generated from contracts/memory-safety-v1.yaml — DO NOT EDIT
// Contract: memory-safety-v1

/// Preconditions for equation `arena_lifecycle`.
/// Call at function entry: `contract_pre_arena_lifecycle!(input_expr)`
macro_rules! contract_pre_arena_lifecycle {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `arena_lifecycle`.
/// Call before return: `contract_post_arena_lifecycle!(result_expr)`
macro_rules! contract_post_arena_lifecycle {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `arena_lifecycle`.
macro_rules! contract_arena_lifecycle {
    ($input:expr, $body:expr) => {{
        contract_pre_arena_lifecycle!($input);
        let _contract_result = $body;
        contract_post_arena_lifecycle!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `index_memory_budget`.
/// Call at function entry: `contract_pre_index_memory_budget!(input_expr)`
macro_rules! contract_pre_index_memory_budget {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `index_memory_budget`.
/// Call before return: `contract_post_index_memory_budget!(result_expr)`
macro_rules! contract_post_index_memory_budget {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `index_memory_budget`.
macro_rules! contract_index_memory_budget {
    ($input:expr, $body:expr) => {{
        contract_pre_index_memory_budget!($input);
        let _contract_result = $body;
        contract_post_index_memory_budget!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `lru_eviction_correctness`.
/// Call at function entry: `contract_pre_lru_eviction_correctness!(input_expr)`
macro_rules! contract_pre_lru_eviction_correctness {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `lru_eviction_correctness`.
/// Call before return: `contract_post_lru_eviction_correctness!(result_expr)`
macro_rules! contract_post_lru_eviction_correctness {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `lru_eviction_correctness`.
macro_rules! contract_lru_eviction_correctness {
    ($input:expr, $body:expr) => {{
        contract_pre_lru_eviction_correctness!($input);
        let _contract_result = $body;
        contract_post_lru_eviction_correctness!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/metaheuristics-v1.yaml — DO NOT EDIT
// Contract: metaheuristics-v1

/// Preconditions for equation `best_monotone`.
/// Domain-specific. Call: `contract_pre_best_monotone!(slice_expr)`
macro_rules! contract_pre_best_monotone {
    () => {{}};
    ($input:expr) => {{
        let _pv_params = &$input;
        debug_assert!(_pv_params.len() > 0,
            "Contract best_monotone: precondition violated — params.len() > 0");
    }};
}

/// Preconditions for equation `ga_crossover`.
/// Domain-specific. Call: `contract_pre_ga_crossover!(slice_expr)`
macro_rules! contract_pre_ga_crossover {
    () => {{}};
    ($input:expr) => {{
        let _pv_params = &$input;
        debug_assert!(_pv_params.len() > 0,
            "Contract ga_crossover: precondition violated — params.len() > 0");
    }};
}

/// Preconditions for equation `pso_velocity`.
/// Domain-specific. Call: `contract_pre_pso_velocity!(slice_expr)`
macro_rules! contract_pre_pso_velocity {
    () => {{}};
    ($input:expr) => {{
        let _pv_params = &$input;
        debug_assert!(_pv_params.len() > 0,
            "Contract pso_velocity: precondition violated — params.len() > 0");
    }};
}

/// Preconditions for equation `sa_acceptance`.
/// Domain-specific. Call: `contract_pre_sa_acceptance!(slice_expr)`
macro_rules! contract_pre_sa_acceptance {
    () => {{}};
    ($input:expr) => {{
        let _pv_params = &$input;
        debug_assert!(_pv_params.len() > 0,
            "Contract sa_acceptance: precondition violated — params.len() > 0");
    }};
}

// Auto-generated from contracts/metrics-classification-v1.yaml — DO NOT EDIT
// Contract: metrics-classification-v1

/// Preconditions for equation `accuracy`.
/// Domain-specific. Call: `contract_pre_accuracy!(slice_expr)`
macro_rules! contract_pre_accuracy {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract accuracy: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract accuracy: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `confusion_matrix`.
/// Domain-specific. Call: `contract_pre_confusion_matrix!(slice_expr)`
macro_rules! contract_pre_confusion_matrix {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract confusion_matrix: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract confusion_matrix: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `f1_score`.
/// Domain-specific. Call: `contract_pre_f1_score!(slice_expr)`
macro_rules! contract_pre_f1_score {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract f1_score: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract f1_score: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `precision`.
/// Domain-specific. Call: `contract_pre_precision!(slice_expr)`
macro_rules! contract_pre_precision {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract precision: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract precision: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `recall`.
/// Domain-specific. Call: `contract_pre_recall!(slice_expr)`
macro_rules! contract_pre_recall {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract recall: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract recall: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

// Auto-generated from contracts/metrics-clustering-v1.yaml — DO NOT EDIT
// Contract: metrics-clustering-v1

/// Preconditions for equation `inertia`.
/// Call at function entry: `contract_pre_inertia!(input_expr)`
macro_rules! contract_pre_inertia {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `silhouette_coefficient`.
/// Call at function entry: `contract_pre_silhouette_coefficient!(input_expr)`
macro_rules! contract_pre_silhouette_coefficient {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `silhouette_score`.
/// Call at function entry: `contract_pre_silhouette_score!(input_expr)`
macro_rules! contract_pre_silhouette_score {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

// Auto-generated from contracts/metrics-ranking-v1.yaml — DO NOT EDIT
// Contract: metrics-ranking-v1

/// Preconditions for equation `hit_at_k`.
/// Domain-specific. Call: `contract_pre_hit_at_k!(slice_expr)`
macro_rules! contract_pre_hit_at_k {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract hit_at_k: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `mrr`.
/// Domain-specific. Call: `contract_pre_mrr!(slice_expr)`
macro_rules! contract_pre_mrr {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract mrr: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `ndcg_at_k`.
/// Domain-specific. Call: `contract_pre_ndcg_at_k!(slice_expr)`
macro_rules! contract_pre_ndcg_at_k {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract ndcg_at_k: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `reciprocal_rank`.
/// Domain-specific. Call: `contract_pre_reciprocal_rank!(slice_expr)`
macro_rules! contract_pre_reciprocal_rank {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract reciprocal_rank: precondition violated — input.len() > 0");
    }};
}

// Auto-generated from contracts/metrics-regression-v1.yaml — DO NOT EDIT
// Contract: metrics-regression-v1

/// Preconditions for equation `mae`.
/// Domain-specific. Call: `contract_pre_mae!(slice_expr)`
macro_rules! contract_pre_mae {
    () => {{}};
    ($input:expr) => {{
        let _pv_predicted = &$input;
        debug_assert!(_pv_predicted.len() > 0,
            "Contract mae: precondition violated — predicted.len() > 0");
    }};
}

/// Preconditions for equation `mse`.
/// Domain-specific. Call: `contract_pre_mse!(slice_expr)`
macro_rules! contract_pre_mse {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract mse: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract mse: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `r_squared`.
/// Domain-specific. Call: `contract_pre_r_squared!(slice_expr)`
macro_rules! contract_pre_r_squared {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract r_squared: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract r_squared: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `rmse`.
/// Domain-specific. Call: `contract_pre_rmse!(slice_expr)`
macro_rules! contract_pre_rmse {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract rmse: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract rmse: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

// Auto-generated from contracts/model-config-algebra-v1.yaml — DO NOT EDIT
// Contract: model-config-algebra-v1

/// Preconditions for equation `bounds`.
/// Domain-specific. Call: `contract_pre_bounds!(slice_expr)`
macro_rules! contract_pre_bounds {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract bounds: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `cross_constraint`.
/// Domain-specific. Call: `contract_pre_cross_constraint!(slice_expr)`
macro_rules! contract_pre_cross_constraint {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract cross_constraint: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `divisibility`.
/// Domain-specific. Call: `contract_pre_divisibility!(slice_expr)`
macro_rules! contract_pre_divisibility {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract divisibility: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `non_degeneracy`.
/// Domain-specific. Call: `contract_pre_non_degeneracy!(slice_expr)`
macro_rules! contract_pre_non_degeneracy {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract non_degeneracy: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `ordering`.
/// Domain-specific. Call: `contract_pre_ordering!(slice_expr)`
macro_rules! contract_pre_ordering {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract ordering: precondition violated — input.len() > 0");
    }};
}

// Auto-generated from contracts/model-format-conversion-v1.yaml — DO NOT EDIT
// Contract: model-format-conversion-v1

/// Preconditions for equation `apr_tokenizer_embedding`.
/// Domain-specific. Call: `contract_pre_apr_tokenizer_embedding!(slice_expr)`
macro_rules! contract_pre_apr_tokenizer_embedding {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
    }};
}

/// Postconditions for equation `apr_tokenizer_embedding`.
/// Call before return: `contract_post_apr_tokenizer_embedding!(result_expr)`
macro_rules! contract_post_apr_tokenizer_embedding {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `apr_tokenizer_embedding`.
macro_rules! contract_apr_tokenizer_embedding {
    ($input:expr, $body:expr) => {{
        contract_pre_apr_tokenizer_embedding!($input);
        let _contract_result = $body;
        contract_post_apr_tokenizer_embedding!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `export_fidelity`.
/// Call at function entry: `contract_pre_export_fidelity!(input_expr)`
macro_rules! contract_pre_export_fidelity {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `export_fidelity`.
/// Call before return: `contract_post_export_fidelity!(result_expr)`
macro_rules! contract_post_export_fidelity {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `export_fidelity`.
macro_rules! contract_export_fidelity {
    ($input:expr, $body:expr) => {{
        contract_pre_export_fidelity!($input);
        let _contract_result = $body;
        contract_post_export_fidelity!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `format_conversion_roundtrip`.
/// Call at function entry: `contract_pre_format_conversion_roundtrip!(input_expr)`
macro_rules! contract_pre_format_conversion_roundtrip {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `format_conversion_roundtrip`.
/// Call before return: `contract_post_format_conversion_roundtrip!(result_expr)`
macro_rules! contract_post_format_conversion_roundtrip {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `format_conversion_roundtrip`.
macro_rules! contract_format_conversion_roundtrip {
    ($input:expr, $body:expr) => {{
        contract_pre_format_conversion_roundtrip!($input);
        let _contract_result = $body;
        contract_post_format_conversion_roundtrip!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `import_integrity`.
/// Call at function entry: `contract_pre_import_integrity!(input_expr)`
macro_rules! contract_pre_import_integrity {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `import_integrity`.
/// Call before return: `contract_post_import_integrity!(result_expr)`
macro_rules! contract_post_import_integrity {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `import_integrity`.
macro_rules! contract_import_integrity {
    ($input:expr, $body:expr) => {{
        contract_pre_import_integrity!($input);
        let _contract_result = $body;
        contract_post_import_integrity!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `merge_weight_algebra`.
/// Domain-specific. Call: `contract_pre_merge_weight_algebra!(slice_expr)`
macro_rules! contract_pre_merge_weight_algebra {
    () => {{}};
    ($input:expr) => {{
        let _pv_models = &$input;
        debug_assert!(_pv_models.len() >= 2,
            "Contract merge_weight_algebra: precondition violated — models.len() >= 2");
    }};
}

/// Postconditions for equation `merge_weight_algebra`.
/// Call before return: `contract_post_merge_weight_algebra!(result_expr)`
macro_rules! contract_post_merge_weight_algebra {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `merge_weight_algebra`.
macro_rules! contract_merge_weight_algebra {
    ($input:expr, $body:expr) => {{
        contract_pre_merge_weight_algebra!($input);
        let _contract_result = $body;
        contract_post_merge_weight_algebra!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `quantization_bounds`.
/// Call at function entry: `contract_pre_quantization_bounds!(input_expr)`
macro_rules! contract_pre_quantization_bounds {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `quantization_bounds`.
/// Call before return: `contract_post_quantization_bounds!(result_expr)`
macro_rules! contract_post_quantization_bounds {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `quantization_bounds`.
macro_rules! contract_quantization_bounds {
    ($input:expr, $body:expr) => {{
        contract_pre_quantization_bounds!($input);
        let _contract_result = $body;
        contract_post_quantization_bounds!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/model-metadata-bounds-v1.yaml — DO NOT EDIT
// Contract: model-metadata-bounds-v1

/// Preconditions for equation `config_bounds_check`.
/// Domain-specific. Call: `contract_pre_config_bounds_check!(slice_expr)`
macro_rules! contract_pre_config_bounds_check {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract config_bounds_check: precondition violated — input.len() > 0");
    }};
}

// Auto-generated from contracts/model-qa-v1.yaml — DO NOT EDIT
// Contract: model-qa-v1

/// Preconditions for equation `grade_assignment`.
/// Domain-specific. Call: `contract_pre_grade_assignment!(slice_expr)`
macro_rules! contract_pre_grade_assignment {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
    }};
}

/// Preconditions for equation `mqs_scoring`.
/// Call at function entry: `contract_pre_mqs_scoring!(input_expr)`
macro_rules! contract_pre_mqs_scoring {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `regression_detection`.
/// Call at function entry: `contract_pre_regression_detection!(input_expr)`
macro_rules! contract_pre_regression_detection {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

// Auto-generated from contracts/monitor-metrics-v1.yaml — DO NOT EDIT
// Contract: monitor-metrics-v1

/// Preconditions for equation `cpu_utilization`.
/// Call at function entry: `contract_pre_cpu_utilization!(input_expr)`
macro_rules! contract_pre_cpu_utilization {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `history_persistence`.
/// Call at function entry: `contract_pre_history_persistence!(input_expr)`
macro_rules! contract_pre_history_persistence {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `memory_usage`.
/// Call at function entry: `contract_pre_memory_usage!(input_expr)`
macro_rules! contract_pre_memory_usage {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

// Auto-generated from contracts/mqs-scoring-v1.yaml — DO NOT EDIT
// Contract: mqs-scoring-v1

/// Preconditions for equation `mqs_composite`.
/// Domain-specific. Call: `contract_pre_mqs_composite!(slice_expr)`
macro_rules! contract_pre_mqs_composite {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract mqs_composite: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract mqs_composite: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `mqs_deterministic`.
/// Domain-specific. Call: `contract_pre_mqs_deterministic!(slice_expr)`
macro_rules! contract_pre_mqs_deterministic {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract mqs_deterministic: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract mqs_deterministic: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `mqs_grade`.
/// Domain-specific. Call: `contract_pre_mqs_grade!(slice_expr)`
macro_rules! contract_pre_mqs_grade {
    () => {{}};
    ($input:expr) => {{
        let _pv_grad_output = &$input;
        debug_assert!(_pv_grad_output.len() > 0,
            "Contract mqs_grade: precondition violated — grad_output.len() > 0");
        debug_assert!(_pv_grad_output.iter().all(|v| v.is_finite()),
            "Contract mqs_grade: precondition violated — grad_output.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `mqs_pass_rate`.
/// Domain-specific. Call: `contract_pre_mqs_pass_rate!(slice_expr)`
macro_rules! contract_pre_mqs_pass_rate {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract mqs_pass_rate: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract mqs_pass_rate: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

// Auto-generated from contracts/mqs-scoring-v1.yaml — DO NOT EDIT
// Contract: mqs-scoring-v1

/// Preconditions for equation `mqs_composite`.
/// Domain-specific. Call: `contract_pre_mqs_composite!(slice_expr)`
macro_rules! contract_pre_mqs_composite {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
    }};
}

/// Postconditions for equation `mqs_composite`.
/// Call before return: `contract_post_mqs_composite!(result_expr)`
macro_rules! contract_post_mqs_composite {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `mqs_composite`.
macro_rules! contract_mqs_composite {
    ($input:expr, $body:expr) => {{
        contract_pre_mqs_composite!($input);
        let _contract_result = $body;
        contract_post_mqs_composite!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `mqs_deterministic`.
/// Call at function entry: `contract_pre_mqs_deterministic!(input_expr)`
macro_rules! contract_pre_mqs_deterministic {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `mqs_deterministic`.
/// Call before return: `contract_post_mqs_deterministic!(result_expr)`
macro_rules! contract_post_mqs_deterministic {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `mqs_deterministic`.
macro_rules! contract_mqs_deterministic {
    ($input:expr, $body:expr) => {{
        contract_pre_mqs_deterministic!($input);
        let _contract_result = $body;
        contract_post_mqs_deterministic!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `mqs_grade`.
/// Domain-specific. Call: `contract_pre_mqs_grade!(slice_expr)`
macro_rules! contract_pre_mqs_grade {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
    }};
}

/// Postconditions for equation `mqs_grade`.
/// Call before return: `contract_post_mqs_grade!(result_expr)`
macro_rules! contract_post_mqs_grade {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `mqs_grade`.
macro_rules! contract_mqs_grade {
    ($input:expr, $body:expr) => {{
        contract_pre_mqs_grade!($input);
        let _contract_result = $body;
        contract_post_mqs_grade!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/naive-bayes-v1.yaml — DO NOT EDIT
// Contract: naive-bayes-v1

/// Preconditions for equation `class_prior`.
/// Domain-specific. Call: `contract_pre_class_prior!(slice_expr)`
macro_rules! contract_pre_class_prior {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract class_prior: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract class_prior: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `gaussian_likelihood`.
/// Domain-specific. Call: `contract_pre_gaussian_likelihood!(slice_expr)`
macro_rules! contract_pre_gaussian_likelihood {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract gaussian_likelihood: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract gaussian_likelihood: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `log_posterior`.
/// Domain-specific. Call: `contract_pre_log_posterior!(slice_expr)`
macro_rules! contract_pre_log_posterior {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract log_posterior: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract log_posterior: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

// Auto-generated from contracts/namespace-isolation-v1.yaml — DO NOT EDIT
// Contract: namespace-isolation-v1

/// Preconditions for equation `connect_lifecycle`.
/// Call at function entry: `contract_pre_connect_lifecycle!(input_expr)`
macro_rules! contract_pre_connect_lifecycle {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `send_isolation`.
/// Domain-specific. Call: `contract_pre_send_isolation!(slice_expr)`
macro_rules! contract_pre_send_isolation {
    () => {{}};
    ($input:expr) => {{
        let _pv_data = &$input;
        debug_assert!(_pv_data.len() > 0,
            "Contract send_isolation: precondition violated — data.len() > 0");
    }};
}

// Auto-generated from contracts/oci-manifest-v1.yaml — DO NOT EDIT
// Contract: oci-manifest-v1

/// Preconditions for equation `layer_cache_hit`.
/// Call at function entry: `contract_pre_layer_cache_hit!(input_expr)`
macro_rules! contract_pre_layer_cache_hit {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `layer_ordering`.
/// Domain-specific. Call: `contract_pre_layer_ordering!(slice_expr)`
macro_rules! contract_pre_layer_ordering {
    () => {{}};
    ($input:expr) => {{
        let _pv_layers = &$input;
        debug_assert!(_pv_layers.len() > 0,
            "Contract layer_ordering: precondition violated — layers.len() > 0");
    }};
}

/// Preconditions for equation `manifest_digest_consistency`.
/// Domain-specific. Call: `contract_pre_manifest_digest_consistency!(slice_expr)`
macro_rules! contract_pre_manifest_digest_consistency {
    () => {{}};
    ($input:expr) => {{
        let _pv_manifest = &$input;
    }};
}

/// Preconditions for equation `reproducible_build`.
/// Call at function entry: `contract_pre_reproducible_build!(input_expr)`
macro_rules! contract_pre_reproducible_build {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

// Auto-generated from contracts/online-softmax-v1.yaml — DO NOT EDIT
// Contract: online-softmax-v1

/// Preconditions for equation `online_normalizer`.
/// Domain-specific. Call: `contract_pre_online_normalizer!(slice_expr)`
macro_rules! contract_pre_online_normalizer {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
        debug_assert!(_pv_x.iter().all(|v| v.is_finite()),
            "Contract online_normalizer: precondition violated — x.iter().all(|v| v.is_finite())");
        debug_assert!(_pv_x.len() > 0,
            "Contract online_normalizer: precondition violated — x.len() > 0");
    }};
}

/// Preconditions for equation `standard_softmax`.
/// Domain-specific. Call: `contract_pre_standard_softmax!(slice_expr)`
macro_rules! contract_pre_standard_softmax {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
        debug_assert!(_pv_x.iter().all(|v| v.is_finite()),
            "Contract standard_softmax: precondition violated — x.iter().all(|v| v.is_finite())");
        debug_assert!(_pv_x.len() > 0,
            "Contract standard_softmax: precondition violated — x.len() > 0");
    }};
}

// Auto-generated from contracts/optimization-v1.yaml — DO NOT EDIT
// Contract: optimization-v1

/// Preconditions for equation `cg_minimize`.
/// Domain-specific. Call: `contract_pre_cg_minimize!(slice_expr)`
macro_rules! contract_pre_cg_minimize {
    () => {{}};
    ($input:expr) => {{
        let _pv_params = &$input;
        debug_assert!(_pv_params.len() > 0,
            "Contract cg_minimize: precondition violated — params.len() > 0");
    }};
}

/// Preconditions for equation `convergence`.
/// Domain-specific. Call: `contract_pre_convergence!(slice_expr)`
macro_rules! contract_pre_convergence {
    () => {{}};
    ($input:expr) => {{
        let _pv_params = &$input;
        debug_assert!(_pv_params.len() > 0,
            "Contract convergence: precondition violated — params.len() > 0");
    }};
}

/// Preconditions for equation `line_search`.
/// Domain-specific. Call: `contract_pre_line_search!(slice_expr)`
macro_rules! contract_pre_line_search {
    () => {{}};
    ($input:expr) => {{
        let _pv_params = &$input;
        debug_assert!(_pv_params.len() > 0,
            "Contract line_search: precondition violated — params.len() > 0");
    }};
}

// Auto-generated from contracts/package-resolve-v1.yaml — DO NOT EDIT
// Contract: package-resolve-v1

/// Preconditions for equation `pull_resolve`.
/// Call at function entry: `contract_pre_pull_resolve!(input_expr)`
macro_rules! contract_pre_pull_resolve {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `registry_list`.
/// Call at function entry: `contract_pre_registry_list!(input_expr)`
macro_rules! contract_pre_registry_list {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `run_tracking`.
/// Call at function entry: `contract_pre_run_tracking!(input_expr)`
macro_rules! contract_pre_run_tracking {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

// Auto-generated from contracts/paged-attention-v1.yaml — DO NOT EDIT
// Contract: paged-attention-v1

/// Preconditions for equation `block_allocation`.
/// Domain-specific. Call: `contract_pre_block_allocation!(slice_expr)`
macro_rules! contract_pre_block_allocation {
    () => {{}};
    ($input:expr) => {{
        let _pv_q = &$input;
        debug_assert!(_pv_q.len() > 0,
            "Contract block_allocation: precondition violated — q.len() > 0");
    }};
}

/// Preconditions for equation `block_table_lookup`.
/// Domain-specific. Call: `contract_pre_block_table_lookup!(slice_expr)`
macro_rules! contract_pre_block_table_lookup {
    () => {{}};
    ($input:expr) => {{
        let _pv_q = &$input;
        debug_assert!(_pv_q.len() > 0,
            "Contract block_table_lookup: precondition violated — q.len() > 0");
    }};
}

/// Preconditions for equation `copy_on_write`.
/// Domain-specific. Call: `contract_pre_copy_on_write!(slice_expr)`
macro_rules! contract_pre_copy_on_write {
    () => {{}};
    ($input:expr) => {{
        let _pv_q = &$input;
        debug_assert!(_pv_q.len() > 0,
            "Contract copy_on_write: precondition violated — q.len() > 0");
    }};
}

// Auto-generated from contracts/paged-kv-cache-v1.yaml — DO NOT EDIT
// Contract: paged-kv-cache-v1

/// Preconditions for equation `block_allocation`.
/// Domain-specific. Call: `contract_pre_block_allocation!(slice_expr)`
macro_rules! contract_pre_block_allocation {
    () => {{}};
    ($input:expr) => {{
        let _pv_q = &$input;
        debug_assert!(_pv_q.len() > 0,
            "Contract block_allocation: precondition violated — q.len() > 0");
    }};
}

/// Preconditions for equation `block_table_invariant`.
/// Domain-specific. Call: `contract_pre_block_table_invariant!(slice_expr)`
macro_rules! contract_pre_block_table_invariant {
    () => {{}};
    ($input:expr) => {{
        let _pv_q = &$input;
        debug_assert!(_pv_q.len() > 0,
            "Contract block_table_invariant: precondition violated — q.len() > 0");
    }};
}

/// Preconditions for equation `fragmentation_free`.
/// Domain-specific. Call: `contract_pre_fragmentation_free!(slice_expr)`
macro_rules! contract_pre_fragmentation_free {
    () => {{}};
    ($input:expr) => {{
        let _pv_q = &$input;
        debug_assert!(_pv_q.len() > 0,
            "Contract fragmentation_free: precondition violated — q.len() > 0");
    }};
}

/// Preconditions for equation `graph_compatibility`.
/// Call at function entry: `contract_pre_graph_compatibility!(input_expr)`
macro_rules! contract_pre_graph_compatibility {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `paged_contiguous_equivalence`.
/// Call at function entry: `contract_pre_paged_contiguous_equivalence!(input_expr)`
macro_rules! contract_pre_paged_contiguous_equivalence {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `slot_mapping`.
/// Domain-specific. Call: `contract_pre_slot_mapping!(slice_expr)`
macro_rules! contract_pre_slot_mapping {
    () => {{}};
    ($input:expr) => {{
        let _pv_q = &$input;
        debug_assert!(_pv_q.len() > 0,
            "Contract slot_mapping: precondition violated — q.len() > 0");
    }};
}

// Auto-generated from contracts/pagerank-kernel-v1.yaml — DO NOT EDIT
// Contract: pagerank-kernel-v1

/// Preconditions for equation `bfs`.
/// Call at function entry: `contract_pre_bfs!(input_expr)`
macro_rules! contract_pre_bfs {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `pagerank`.
/// Domain-specific. Call: `contract_pre_pagerank!(slice_expr)`
macro_rules! contract_pre_pagerank {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
    }};
}

// Auto-generated from contracts/pagerank-kernel-v1.yaml — DO NOT EDIT
// Contract: pagerank-kernel-v1

/// Preconditions for equation `pagerank`.
/// Call at function entry: `contract_pre_pagerank!(input_expr)`
macro_rules! contract_pre_pagerank {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `power_iteration`.
/// Call at function entry: `contract_pre_power_iteration!(input_expr)`
macro_rules! contract_pre_power_iteration {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

// Auto-generated from contracts/parser-soundness-v1.yaml — DO NOT EDIT
// Contract: parser-soundness-v1

/// Preconditions for equation `lex`.
/// Domain-specific. Call: `contract_pre_lex!(slice_expr)`
macro_rules! contract_pre_lex {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract lex: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `parse`.
/// Domain-specific. Call: `contract_pre_parse!(slice_expr)`
macro_rules! contract_pre_parse {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract parse: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `semantic_analyze`.
/// Domain-specific. Call: `contract_pre_semantic_analyze!(slice_expr)`
macro_rules! contract_pre_semantic_analyze {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract semantic_analyze: precondition violated — input.len() > 0");
    }};
}

// Auto-generated from contracts/parser-soundness-v1.yaml — DO NOT EDIT
// Contract: parser-soundness-v1

/// Preconditions for equation `block_scoping`.
/// Call at function entry: `contract_pre_block_scoping!(input_expr)`
macro_rules! contract_pre_block_scoping {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `parse_correctness`.
/// Call at function entry: `contract_pre_parse_correctness!(input_expr)`
macro_rules! contract_pre_parse_correctness {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `transpile_roundtrip`.
/// Call at function entry: `contract_pre_transpile_roundtrip!(input_expr)`
macro_rules! contract_pre_transpile_roundtrip {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

// Auto-generated from contracts/pca-v1.yaml — DO NOT EDIT
// Contract: pca-v1

/// Preconditions for equation `explained_variance`.
/// Domain-specific. Call: `contract_pre_explained_variance!(slice_expr)`
macro_rules! contract_pre_explained_variance {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract explained_variance: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract explained_variance: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `pca_transform`.
/// Domain-specific. Call: `contract_pre_pca_transform!(slice_expr)`
macro_rules! contract_pre_pca_transform {
    () => {{}};
    ($input:expr) => {{
        let _pv_a = &$input;
        debug_assert!(_pv_a.len() > 0,
            "Contract pca_transform: precondition violated — a.len() > 0");
    }};
}

/// Preconditions for equation `reconstruction`.
/// Domain-specific. Call: `contract_pre_reconstruction!(slice_expr)`
macro_rules! contract_pre_reconstruction {
    () => {{}};
    ($input:expr) => {{
        let _pv_a = &$input;
        debug_assert!(_pv_a.len() > 0,
            "Contract reconstruction: precondition violated — a.len() > 0");
    }};
}

// Auto-generated from contracts/performance-grading-v1.yaml — DO NOT EDIT
// Contract: performance-grading-v1

/// Preconditions for equation `concrete_instance`.
/// Domain-specific. Call: `contract_pre_concrete_instance!(slice_expr)`
macro_rules! contract_pre_concrete_instance {
    () => {{}};
    ($input:expr) => {{
        let _pv_grad_output = &$input;
        debug_assert!(_pv_grad_output.len() > 0,
            "Contract concrete_instance: precondition violated — grad_output.len() > 0");
        debug_assert!(_pv_grad_output.iter().all(|v| v.is_finite()),
            "Contract concrete_instance: precondition violated — grad_output.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `efficiency_grade`.
/// Domain-specific. Call: `contract_pre_efficiency_grade!(slice_expr)`
macro_rules! contract_pre_efficiency_grade {
    () => {{}};
    ($input:expr) => {{
        let _pv_grad_output = &$input;
        debug_assert!(_pv_grad_output.len() > 0,
            "Contract efficiency_grade: precondition violated — grad_output.len() > 0");
        debug_assert!(_pv_grad_output.iter().all(|v| v.is_finite()),
            "Contract efficiency_grade: precondition violated — grad_output.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `llamacpp_parity`.
/// Domain-specific. Call: `contract_pre_llamacpp_parity!(slice_expr)`
macro_rules! contract_pre_llamacpp_parity {
    () => {{}};
    ($input:expr) => {{
        let _pv_grad_output = &$input;
        debug_assert!(_pv_grad_output.len() > 0,
            "Contract llamacpp_parity: precondition violated — grad_output.len() > 0");
        debug_assert!(_pv_grad_output.iter().all(|v| v.is_finite()),
            "Contract llamacpp_parity: precondition violated — grad_output.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `ollama_parity`.
/// Domain-specific. Call: `contract_pre_ollama_parity!(slice_expr)`
macro_rules! contract_pre_ollama_parity {
    () => {{}};
    ($input:expr) => {{
        let _pv_grad_output = &$input;
        debug_assert!(_pv_grad_output.len() > 0,
            "Contract ollama_parity: precondition violated — grad_output.len() > 0");
        debug_assert!(_pv_grad_output.iter().all(|v| v.is_finite()),
            "Contract ollama_parity: precondition violated — grad_output.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `vllm_parity`.
/// Domain-specific. Call: `contract_pre_vllm_parity!(slice_expr)`
macro_rules! contract_pre_vllm_parity {
    () => {{}};
    ($input:expr) => {{
        let _pv_grad_output = &$input;
        debug_assert!(_pv_grad_output.len() > 0,
            "Contract vllm_parity: precondition violated — grad_output.len() > 0");
        debug_assert!(_pv_grad_output.iter().all(|v| v.is_finite()),
            "Contract vllm_parity: precondition violated — grad_output.iter().all(|v| v.is_finite())");
    }};
}

// Auto-generated from contracts/pipeline-cache-v1.yaml — DO NOT EDIT
// Contract: pipeline-cache-v1

/// Preconditions for equation `identity`.
/// Call at function entry: `contract_pre_identity!(input_expr)`
macro_rules! contract_pre_identity {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

// Auto-generated from contracts/plugin-lifecycle-v1.yaml — DO NOT EDIT
// Contract: plugin-lifecycle-v1

/// Preconditions for equation `lifecycle_state_machine`.
/// Call at function entry: `contract_pre_lifecycle_state_machine!(input_expr)`
macro_rules! contract_pre_lifecycle_state_machine {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `permission_scoping`.
/// Domain-specific. Call: `contract_pre_permission_scoping!(slice_expr)`
macro_rules! contract_pre_permission_scoping {
    () => {{}};
    ($input:expr) => {{
        let _pv_plugin = &$input;
    }};
}

/// Preconditions for equation `schema_validation`.
/// Domain-specific. Call: `contract_pre_schema_validation!(slice_expr)`
macro_rules! contract_pre_schema_validation {
    () => {{}};
    ($input:expr) => {{
        let _pv_schema = &$input;
    }};
}

// Auto-generated from contracts/pmat-work-lifecycle-v1.yaml — DO NOT EDIT
// Contract: pmat-work-lifecycle-v1

/// Preconditions for equation `baseline_integrity`.
/// Call at function entry: `contract_pre_baseline_integrity!(input_expr)`
macro_rules! contract_pre_baseline_integrity {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `baseline_integrity`.
/// Call before return: `contract_post_baseline_integrity!(result_expr)`
macro_rules! contract_post_baseline_integrity {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `baseline_integrity`.
macro_rules! contract_baseline_integrity {
    ($input:expr, $body:expr) => {{
        contract_pre_baseline_integrity!($input);
        let _contract_result = $body;
        contract_post_baseline_integrity!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `contract_immutability`.
/// Domain-specific. Call: `contract_pre_contract_immutability!(slice_expr)`
macro_rules! contract_pre_contract_immutability {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
    }};
}

/// Postconditions for equation `contract_immutability`.
/// Call before return: `contract_post_contract_immutability!(result_expr)`
macro_rules! contract_post_contract_immutability {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `contract_immutability`.
macro_rules! contract_contract_immutability {
    ($input:expr, $body:expr) => {{
        contract_pre_contract_immutability!($input);
        let _contract_result = $body;
        contract_post_contract_immutability!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `falsification_completeness`.
/// Call at function entry: `contract_pre_falsification_completeness!(input_expr)`
macro_rules! contract_pre_falsification_completeness {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `falsification_completeness`.
/// Call before return: `contract_post_falsification_completeness!(result_expr)`
macro_rules! contract_post_falsification_completeness {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `falsification_completeness`.
macro_rules! contract_falsification_completeness {
    ($input:expr, $body:expr) => {{
        contract_pre_falsification_completeness!($input);
        let _contract_result = $body;
        contract_post_falsification_completeness!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `monotonic_ledger`.
/// Call at function entry: `contract_pre_monotonic_ledger!(input_expr)`
macro_rules! contract_pre_monotonic_ledger {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `monotonic_ledger`.
/// Call before return: `contract_post_monotonic_ledger!(result_expr)`
macro_rules! contract_post_monotonic_ledger {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `monotonic_ledger`.
macro_rules! contract_monotonic_ledger {
    ($input:expr, $body:expr) => {{
        contract_pre_monotonic_ledger!($input);
        let _contract_result = $body;
        contract_post_monotonic_ledger!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `profile_determinism`.
/// Call at function entry: `contract_pre_profile_determinism!(input_expr)`
macro_rules! contract_pre_profile_determinism {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `profile_determinism`.
/// Call before return: `contract_post_profile_determinism!(result_expr)`
macro_rules! contract_post_profile_determinism {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `profile_determinism`.
macro_rules! contract_profile_determinism {
    ($input:expr, $body:expr) => {{
        contract_pre_profile_determinism!($input);
        let _contract_result = $body;
        contract_post_profile_determinism!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `rescue_bound`.
/// Call at function entry: `contract_pre_rescue_bound!(input_expr)`
macro_rules! contract_pre_rescue_bound {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `rescue_bound`.
/// Call before return: `contract_post_rescue_bound!(result_expr)`
macro_rules! contract_post_rescue_bound {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `rescue_bound`.
macro_rules! contract_rescue_bound {
    ($input:expr, $body:expr) => {{
        contract_pre_rescue_bound!($input);
        let _contract_result = $body;
        contract_post_rescue_bound!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `subcontracting_soundness`.
/// Call at function entry: `contract_pre_subcontracting_soundness!(input_expr)`
macro_rules! contract_pre_subcontracting_soundness {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `subcontracting_soundness`.
/// Call before return: `contract_post_subcontracting_soundness!(result_expr)`
macro_rules! contract_post_subcontracting_soundness {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `subcontracting_soundness`.
macro_rules! contract_subcontracting_soundness {
    ($input:expr, $body:expr) => {{
        contract_pre_subcontracting_soundness!($input);
        let _contract_result = $body;
        contract_post_subcontracting_soundness!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/preprocessing-normalization-v1.yaml — DO NOT EDIT
// Contract: preprocessing-normalization-v1

/// Preconditions for equation `minmax_scaler`.
/// Domain-specific. Call: `contract_pre_minmax_scaler!(slice_expr)`
macro_rules! contract_pre_minmax_scaler {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract minmax_scaler: precondition violated — input.iter().all(|v| v.is_finite())");
        debug_assert!(_pv_input.len() > 0,
            "Contract minmax_scaler: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `robust_scaler`.
/// Domain-specific. Call: `contract_pre_robust_scaler!(slice_expr)`
macro_rules! contract_pre_robust_scaler {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract robust_scaler: precondition violated — input.iter().all(|v| v.is_finite())");
        debug_assert!(_pv_input.len() > 0,
            "Contract robust_scaler: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `standard_scaler`.
/// Domain-specific. Call: `contract_pre_standard_scaler!(slice_expr)`
macro_rules! contract_pre_standard_scaler {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract standard_scaler: precondition violated — input.iter().all(|v| v.is_finite())");
        debug_assert!(_pv_input.len() > 0,
            "Contract standard_scaler: precondition violated — input.len() > 0");
    }};
}

// Auto-generated from contracts/property-testing-v1.yaml — DO NOT EDIT
// Contract: property-testing-v1

/// Preconditions for equation `assertion_evaluation`.
/// Domain-specific. Call: `contract_pre_assertion_evaluation!(slice_expr)`
macro_rules! contract_pre_assertion_evaluation {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
    }};
}

/// Preconditions for equation `coverage_collection`.
/// Call at function entry: `contract_pre_coverage_collection!(input_expr)`
macro_rules! contract_pre_coverage_collection {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `playbook_state_machine`.
/// Call at function entry: `contract_pre_playbook_state_machine!(input_expr)`
macro_rules! contract_pre_playbook_state_machine {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `retry_assertion`.
/// Call at function entry: `contract_pre_retry_assertion!(input_expr)`
macro_rules! contract_pre_retry_assertion {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `soft_assertion_collection`.
/// Call at function entry: `contract_pre_soft_assertion_collection!(input_expr)`
macro_rules! contract_pre_soft_assertion_collection {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `test_result_reporting`.
/// Call at function entry: `contract_pre_test_result_reporting!(input_expr)`
macro_rules! contract_pre_test_result_reporting {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

// Auto-generated from contracts/provider-routing-v1.yaml — DO NOT EDIT
// Contract: provider-routing-v1

/// Preconditions for equation `backoff_jitter`.
/// Call at function entry: `contract_pre_backoff_jitter!(input_expr)`
macro_rules! contract_pre_backoff_jitter {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `cost_budget`.
/// Call before return: `contract_post_cost_budget!(result_expr)`
macro_rules! contract_post_cost_budget {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Preconditions for equation `failover_cascade`.
/// Domain-specific. Call: `contract_pre_failover_cascade!(slice_expr)`
macro_rules! contract_pre_failover_cascade {
    () => {{}};
    ($input:expr) => {{
        let _pv_providers = &$input;
        debug_assert!(_pv_providers.len() > 0,
            "Contract failover_cascade: precondition violated — providers.len() > 0");
    }};
}

/// Preconditions for equation `privacy_enforcement`.
/// Domain-specific. Call: `contract_pre_privacy_enforcement!(slice_expr)`
macro_rules! contract_pre_privacy_enforcement {
    () => {{}};
    ($input:expr) => {{
        let _pv_request = &$input;
    }};
}

/// Postconditions for equation `privacy_enforcement`.
/// Call before return: `contract_post_privacy_enforcement!(result_expr)`
macro_rules! contract_post_privacy_enforcement {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `privacy_enforcement`.
macro_rules! contract_privacy_enforcement {
    ($input:expr, $body:expr) => {{
        contract_pre_privacy_enforcement!($input);
        let _contract_result = $body;
        contract_post_privacy_enforcement!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/ptx-target-parity-v1.yaml — DO NOT EDIT
// Contract: ptx-target-parity-v1

/// Preconditions for equation `jit_compilation_success`.
/// Domain-specific. Call: `contract_pre_jit_compilation_success!(slice_expr)`
macro_rules! contract_pre_jit_compilation_success {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract jit_compilation_success: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract jit_compilation_success: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `no_hardcoded_targets`.
/// Domain-specific. Call: `contract_pre_no_hardcoded_targets!(slice_expr)`
macro_rules! contract_pre_no_hardcoded_targets {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract no_hardcoded_targets: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract no_hardcoded_targets: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `target_parity`.
/// Domain-specific. Call: `contract_pre_target_parity!(slice_expr)`
macro_rules! contract_pre_target_parity {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract target_parity: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract target_parity: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

// Auto-generated from contracts/q4k-q6k-superblock-v1.yaml — DO NOT EDIT
// Contract: q4k-q6k-superblock-v1

/// Preconditions for equation `bsum`.
/// Domain-specific. Call: `contract_pre_bsum!(slice_expr)`
macro_rules! contract_pre_bsum {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract bsum: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `dequantization`.
/// Domain-specific. Call: `contract_pre_dequantization!(slice_expr)`
macro_rules! contract_pre_dequantization {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract dequantization: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `q4k_superblock`.
/// Domain-specific. Call: `contract_pre_q4k_superblock!(slice_expr)`
macro_rules! contract_pre_q4k_superblock {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract q4k_superblock: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `q6k_superblock`.
/// Domain-specific. Call: `contract_pre_q6k_superblock!(slice_expr)`
macro_rules! contract_pre_q6k_superblock {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract q6k_superblock: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `total_bytes`.
/// Domain-specific. Call: `contract_pre_total_bytes!(slice_expr)`
macro_rules! contract_pre_total_bytes {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract total_bytes: precondition violated — input.len() > 0");
    }};
}

// Auto-generated from contracts/qk-norm-apr-loader-v1.yaml — DO NOT EDIT
// Contract: qk-norm-apr-loader-v1

/// Preconditions for equation `qk_norm_load`.
/// Domain-specific. Call: `contract_pre_qk_norm_load!(slice_expr)`
macro_rules! contract_pre_qk_norm_load {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract qk_norm_load: precondition violated — input.iter().all(|v| v.is_finite())");
        debug_assert!(_pv_input.len() > 0,
            "Contract qk_norm_load: precondition violated — input.len() > 0");
    }};
}

// Auto-generated from contracts/qk-norm-v1.yaml — DO NOT EDIT
// Contract: qk-norm-v1

/// Preconditions for equation `qk_rmsnorm`.
/// Domain-specific. Call: `contract_pre_qk_rmsnorm!(slice_expr)`
macro_rules! contract_pre_qk_rmsnorm {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract qk_rmsnorm: precondition violated — input.iter().all(|v| v.is_finite())");
        debug_assert!(_pv_input.len() > 0,
            "Contract qk_rmsnorm: precondition violated — input.len() > 0");
    }};
}

// Auto-generated from contracts/qlora-hyperparameters-v1.yaml — DO NOT EDIT
// Contract: qlora-hyperparameters-v1

/// Preconditions for equation `effective_batch_size`.
/// Domain-specific. Call: `contract_pre_effective_batch_size!(slice_expr)`
macro_rules! contract_pre_effective_batch_size {
    () => {{}};
    ($input:expr) => {{
        let _pv_params = &$input;
        debug_assert!(_pv_params.len() > 0,
            "Contract effective_batch_size: precondition violated — params.len() > 0");
    }};
}

/// Preconditions for equation `epoch_count_imbalanced`.
/// Domain-specific. Call: `contract_pre_epoch_count_imbalanced!(slice_expr)`
macro_rules! contract_pre_epoch_count_imbalanced {
    () => {{}};
    ($input:expr) => {{
        let _pv_params = &$input;
        debug_assert!(_pv_params.len() > 0,
            "Contract epoch_count_imbalanced: precondition violated — params.len() > 0");
    }};
}

/// Preconditions for equation `gradient_clip_bound`.
/// Domain-specific. Call: `contract_pre_gradient_clip_bound!(slice_expr)`
macro_rules! contract_pre_gradient_clip_bound {
    () => {{}};
    ($input:expr) => {{
        let _pv_params = &$input;
        debug_assert!(_pv_params.len() > 0,
            "Contract gradient_clip_bound: precondition violated — params.len() > 0");
    }};
}

/// Preconditions for equation `learning_rate_scaling`.
/// Domain-specific. Call: `contract_pre_learning_rate_scaling!(slice_expr)`
macro_rules! contract_pre_learning_rate_scaling {
    () => {{}};
    ($input:expr) => {{
        let _pv_params = &$input;
        debug_assert!(_pv_params.len() > 0,
            "Contract learning_rate_scaling: precondition violated — params.len() > 0");
    }};
}

/// Preconditions for equation `lora_alpha_ratio`.
/// Domain-specific. Call: `contract_pre_lora_alpha_ratio!(slice_expr)`
macro_rules! contract_pre_lora_alpha_ratio {
    () => {{}};
    ($input:expr) => {{
        let _pv_params = &$input;
        debug_assert!(_pv_params.len() > 0,
            "Contract lora_alpha_ratio: precondition violated — params.len() > 0");
    }};
}

/// Preconditions for equation `seq_len_from_data`.
/// Domain-specific. Call: `contract_pre_seq_len_from_data!(slice_expr)`
macro_rules! contract_pre_seq_len_from_data {
    () => {{}};
    ($input:expr) => {{
        let _pv_params = &$input;
        debug_assert!(_pv_params.len() > 0,
            "Contract seq_len_from_data: precondition violated — params.len() > 0");
    }};
}

/// Preconditions for equation `warmup_fraction`.
/// Domain-specific. Call: `contract_pre_warmup_fraction!(slice_expr)`
macro_rules! contract_pre_warmup_fraction {
    () => {{}};
    ($input:expr) => {{
        let _pv_params = &$input;
        debug_assert!(_pv_params.len() > 0,
            "Contract warmup_fraction: precondition violated — params.len() > 0");
    }};
}

// Auto-generated from contracts/quality-validation-v1.yaml — DO NOT EDIT
// Contract: quality-validation-v1

/// Preconditions for equation `gate_composition`.
/// Call at function entry: `contract_pre_gate_composition!(input_expr)`
macro_rules! contract_pre_gate_composition {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `validate_index`.
/// Call at function entry: `contract_pre_validate_index!(input_expr)`
macro_rules! contract_pre_validate_index {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `validate_size`.
/// Call at function entry: `contract_pre_validate_size!(input_expr)`
macro_rules! contract_pre_validate_size {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

// Auto-generated from contracts/quantization-ordering-v1.yaml — DO NOT EDIT
// Contract: quantization-ordering-v1

/// Preconditions for equation `alpha_scaling`.
/// Domain-specific. Call: `contract_pre_alpha_scaling!(slice_expr)`
macro_rules! contract_pre_alpha_scaling {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract alpha_scaling: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `bytes_per_param`.
/// Domain-specific. Call: `contract_pre_bytes_per_param!(slice_expr)`
macro_rules! contract_pre_bytes_per_param {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract bytes_per_param: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `dropout_expectation`.
/// Domain-specific. Call: `contract_pre_dropout_expectation!(slice_expr)`
macro_rules! contract_pre_dropout_expectation {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
        debug_assert!(_pv_x.iter().all(|v| v.is_finite()),
            "Contract dropout_expectation: precondition violated — x.iter().all(|v| v.is_finite())");
        debug_assert!(_pv_x.len() > 0,
            "Contract dropout_expectation: precondition violated — x.len() > 0");
    }};
}

/// Preconditions for equation `size_ordering`.
/// Domain-specific. Call: `contract_pre_size_ordering!(slice_expr)`
macro_rules! contract_pre_size_ordering {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract size_ordering: precondition violated — input.len() > 0");
    }};
}

// Auto-generated from contracts/quantized-dot-product-v1.yaml — DO NOT EDIT
// Contract: quantized-dot-product-v1

/// Preconditions for equation `bsum_decomposition`.
/// Domain-specific. Call: `contract_pre_bsum_decomposition!(slice_expr)`
macro_rules! contract_pre_bsum_decomposition {
    () => {{}};
    ($input:expr) => {{
        let _pv_activations = &$input;
    }};
}

/// Preconditions for equation `format_isolation`.
/// Call at function entry: `contract_pre_format_isolation!(input_expr)`
macro_rules! contract_pre_format_isolation {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `identity`.
/// Domain-specific. Call: `contract_pre_identity!(slice_expr)`
macro_rules! contract_pre_identity {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract identity: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `simd_scalar_equivalence`.
/// Domain-specific. Call: `contract_pre_simd_scalar_equivalence!(slice_expr)`
macro_rules! contract_pre_simd_scalar_equivalence {
    () => {{}};
    ($input:expr) => {{
        let _pv_data = &$input;
    }};
}

// Auto-generated from contracts/qwen2-e2e-verification-v1.yaml — DO NOT EDIT
// Contract: qwen2-e2e-verification-v1

/// Preconditions for equation `contract_composition`.
/// Domain-specific. Call: `contract_pre_contract_composition!(slice_expr)`
macro_rules! contract_pre_contract_composition {
    () => {{}};
    ($input:expr) => {{
        let _pv_indices = &$input;
        debug_assert!(_pv_indices.len() > 0,
            "Contract contract_composition: precondition violated — indices.len() > 0");
    }};
}

/// Preconditions for equation `flops_per_token`.
/// Domain-specific. Call: `contract_pre_flops_per_token!(slice_expr)`
macro_rules! contract_pre_flops_per_token {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract flops_per_token: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `memory_breakdown`.
/// Domain-specific. Call: `contract_pre_memory_breakdown!(slice_expr)`
macro_rules! contract_pre_memory_breakdown {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract memory_breakdown: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `model_parameter_count`.
/// Domain-specific. Call: `contract_pre_model_parameter_count!(slice_expr)`
macro_rules! contract_pre_model_parameter_count {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract model_parameter_count: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `throughput_model`.
/// Domain-specific. Call: `contract_pre_throughput_model!(slice_expr)`
macro_rules! contract_pre_throughput_model {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract throughput_model: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `verification_ladder`.
/// Domain-specific. Call: `contract_pre_verification_ladder!(slice_expr)`
macro_rules! contract_pre_verification_ladder {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract verification_ladder: precondition violated — input.len() > 0");
    }};
}

// Auto-generated from contracts/qwen2-shapes-v1.yaml — DO NOT EDIT
// Contract: qwen2-shapes-v1

/// Preconditions for equation `head_dim_consistency`.
/// Domain-specific. Call: `contract_pre_head_dim_consistency!(slice_expr)`
macro_rules! contract_pre_head_dim_consistency {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract head_dim_consistency: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `kv_projection_shape`.
/// Domain-specific. Call: `contract_pre_kv_projection_shape!(slice_expr)`
macro_rules! contract_pre_kv_projection_shape {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract kv_projection_shape: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `o_projection_transpose`.
/// Domain-specific. Call: `contract_pre_o_projection_transpose!(slice_expr)`
macro_rules! contract_pre_o_projection_transpose {
    () => {{}};
    ($input:expr) => {{
        let _pv_a = &$input;
        debug_assert!(_pv_a.len() > 0,
            "Contract o_projection_transpose: precondition violated — a.len() > 0");
    }};
}

/// Preconditions for equation `q_projection_shape`.
/// Domain-specific. Call: `contract_pre_q_projection_shape!(slice_expr)`
macro_rules! contract_pre_q_projection_shape {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract q_projection_shape: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `rope_frequency`.
/// Domain-specific. Call: `contract_pre_rope_frequency!(slice_expr)`
macro_rules! contract_pre_rope_frequency {
    () => {{}};
    ($input:expr) => {{
        let _pv_indices = &$input;
        debug_assert!(_pv_indices.len() > 0,
            "Contract rope_frequency: precondition violated — indices.len() > 0");
    }};
}

/// Preconditions for equation `swiglu_ratio`.
/// Domain-specific. Call: `contract_pre_swiglu_ratio!(slice_expr)`
macro_rules! contract_pre_swiglu_ratio {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract swiglu_ratio: precondition violated — input.len() > 0");
    }};
}

// Auto-generated from contracts/qwen2-weight-loading-v1.yaml — DO NOT EDIT
// Contract: qwen2-weight-loading-v1

/// Preconditions for equation `kv_projection`.
/// Call at function entry: `contract_pre_kv_projection!(input_expr)`
macro_rules! contract_pre_kv_projection {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `q_projection`.
/// Call at function entry: `contract_pre_q_projection!(input_expr)`
macro_rules! contract_pre_q_projection {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `swiglu_expansion`.
/// Call at function entry: `contract_pre_swiglu_expansion!(input_expr)`
macro_rules! contract_pre_swiglu_expansion {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `total_parameters`.
/// Call at function entry: `contract_pre_total_parameters!(input_expr)`
macro_rules! contract_pre_total_parameters {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

// Auto-generated from contracts/qwen3-e2e-verification-v1.yaml — DO NOT EDIT
// Contract: qwen3-e2e-verification-v1

/// Preconditions for equation `contract_composition`.
/// Domain-specific. Call: `contract_pre_contract_composition!(slice_expr)`
macro_rules! contract_pre_contract_composition {
    () => {{}};
    ($input:expr) => {{
        let _pv_indices = &$input;
        debug_assert!(_pv_indices.len() > 0,
            "Contract contract_composition: precondition violated — indices.len() > 0");
    }};
}

/// Preconditions for equation `flops_per_token`.
/// Domain-specific. Call: `contract_pre_flops_per_token!(slice_expr)`
macro_rules! contract_pre_flops_per_token {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract flops_per_token: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `memory_breakdown`.
/// Domain-specific. Call: `contract_pre_memory_breakdown!(slice_expr)`
macro_rules! contract_pre_memory_breakdown {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract memory_breakdown: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `model_parameter_count`.
/// Domain-specific. Call: `contract_pre_model_parameter_count!(slice_expr)`
macro_rules! contract_pre_model_parameter_count {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract model_parameter_count: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `throughput_model`.
/// Domain-specific. Call: `contract_pre_throughput_model!(slice_expr)`
macro_rules! contract_pre_throughput_model {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract throughput_model: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `verification_ladder`.
/// Domain-specific. Call: `contract_pre_verification_ladder!(slice_expr)`
macro_rules! contract_pre_verification_ladder {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract verification_ladder: precondition violated — input.len() > 0");
    }};
}

// Auto-generated from contracts/qwen3-shapes-v1.yaml — DO NOT EDIT
// Contract: qwen3-shapes-v1

/// Preconditions for equation `head_dim_consistency`.
/// Domain-specific. Call: `contract_pre_head_dim_consistency!(slice_expr)`
macro_rules! contract_pre_head_dim_consistency {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract head_dim_consistency: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `kv_projection_shape`.
/// Domain-specific. Call: `contract_pre_kv_projection_shape!(slice_expr)`
macro_rules! contract_pre_kv_projection_shape {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract kv_projection_shape: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `o_projection_transpose`.
/// Domain-specific. Call: `contract_pre_o_projection_transpose!(slice_expr)`
macro_rules! contract_pre_o_projection_transpose {
    () => {{}};
    ($input:expr) => {{
        let _pv_a = &$input;
        debug_assert!(_pv_a.len() > 0,
            "Contract o_projection_transpose: precondition violated — a.len() > 0");
    }};
}

/// Preconditions for equation `q_projection_shape`.
/// Domain-specific. Call: `contract_pre_q_projection_shape!(slice_expr)`
macro_rules! contract_pre_q_projection_shape {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract q_projection_shape: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `rope_frequency`.
/// Domain-specific. Call: `contract_pre_rope_frequency!(slice_expr)`
macro_rules! contract_pre_rope_frequency {
    () => {{}};
    ($input:expr) => {{
        let _pv_indices = &$input;
        debug_assert!(_pv_indices.len() > 0,
            "Contract rope_frequency: precondition violated — indices.len() > 0");
    }};
}

/// Preconditions for equation `swiglu_ratio`.
/// Domain-specific. Call: `contract_pre_swiglu_ratio!(slice_expr)`
macro_rules! contract_pre_swiglu_ratio {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract swiglu_ratio: precondition violated — input.len() > 0");
    }};
}

// Auto-generated from contracts/qwen35-e2e-verification-v1.yaml — DO NOT EDIT
// Contract: qwen35-e2e-verification-v1

/// Preconditions for equation `contract_composition`.
/// Domain-specific. Call: `contract_pre_contract_composition!(slice_expr)`
macro_rules! contract_pre_contract_composition {
    () => {{}};
    ($input:expr) => {{
        let _pv_indices = &$input;
        debug_assert!(_pv_indices.len() > 0,
            "Contract contract_composition: precondition violated — indices.len() > 0");
    }};
}

/// Preconditions for equation `flops_per_token`.
/// Domain-specific. Call: `contract_pre_flops_per_token!(slice_expr)`
macro_rules! contract_pre_flops_per_token {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract flops_per_token: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `memory_breakdown`.
/// Domain-specific. Call: `contract_pre_memory_breakdown!(slice_expr)`
macro_rules! contract_pre_memory_breakdown {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract memory_breakdown: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `model_parameter_count`.
/// Domain-specific. Call: `contract_pre_model_parameter_count!(slice_expr)`
macro_rules! contract_pre_model_parameter_count {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract model_parameter_count: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `throughput_model`.
/// Domain-specific. Call: `contract_pre_throughput_model!(slice_expr)`
macro_rules! contract_pre_throughput_model {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract throughput_model: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `verification_ladder`.
/// Domain-specific. Call: `contract_pre_verification_ladder!(slice_expr)`
macro_rules! contract_pre_verification_ladder {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract verification_ladder: precondition violated — input.len() > 0");
    }};
}

// Auto-generated from contracts/qwen35-hybrid-forward-v1.yaml — DO NOT EDIT
// Contract: qwen35-hybrid-forward-v1

/// Preconditions for equation `activation_magnitude`.
/// Domain-specific. Call: `contract_pre_activation_magnitude!(slice_expr)`
macro_rules! contract_pre_activation_magnitude {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
        debug_assert!(_pv_x.iter().all(|v| v.is_finite()),
            "Contract activation_magnitude: precondition violated — x.iter().all(|v| v.is_finite())");
        debug_assert!(_pv_x.len() > 0,
            "Contract activation_magnitude: precondition violated — x.len() > 0");
    }};
}

/// Preconditions for equation `attention_sublayer`.
/// Domain-specific. Call: `contract_pre_attention_sublayer!(slice_expr)`
macro_rules! contract_pre_attention_sublayer {
    () => {{}};
    ($input:expr) => {{
        let _pv_q = &$input;
        debug_assert!(_pv_q.len() > 0,
            "Contract attention_sublayer: precondition violated — q.len() > 0");
    }};
}

/// Preconditions for equation `ffn_sublayer`.
/// Domain-specific. Call: `contract_pre_ffn_sublayer!(slice_expr)`
macro_rules! contract_pre_ffn_sublayer {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract ffn_sublayer: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `gdn_sublayer`.
/// Domain-specific. Call: `contract_pre_gdn_sublayer!(slice_expr)`
macro_rules! contract_pre_gdn_sublayer {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract gdn_sublayer: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `gradient_flow`.
/// Domain-specific. Call: `contract_pre_gradient_flow!(slice_expr)`
macro_rules! contract_pre_gradient_flow {
    () => {{}};
    ($input:expr) => {{
        let _pv_grad_output = &$input;
        debug_assert!(_pv_grad_output.len() > 0,
            "Contract gradient_flow: precondition violated — grad_output.len() > 0");
        debug_assert!(_pv_grad_output.iter().all(|v| v.is_finite()),
            "Contract gradient_flow: precondition violated — grad_output.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `hybrid_block`.
/// Domain-specific. Call: `contract_pre_hybrid_block!(slice_expr)`
macro_rules! contract_pre_hybrid_block {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract hybrid_block: precondition violated — input.len() > 0");
    }};
}

// Auto-generated from contracts/qwen35-shapes-v1.yaml — DO NOT EDIT
// Contract: qwen35-shapes-v1

/// Preconditions for equation `kv_projection_shape`.
/// Domain-specific. Call: `contract_pre_kv_projection_shape!(slice_expr)`
macro_rules! contract_pre_kv_projection_shape {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract kv_projection_shape: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `o_projection_transpose`.
/// Domain-specific. Call: `contract_pre_o_projection_transpose!(slice_expr)`
macro_rules! contract_pre_o_projection_transpose {
    () => {{}};
    ($input:expr) => {{
        let _pv_a = &$input;
        debug_assert!(_pv_a.len() > 0,
            "Contract o_projection_transpose: precondition violated — a.len() > 0");
    }};
}

/// Preconditions for equation `q_projection_shape`.
/// Domain-specific. Call: `contract_pre_q_projection_shape!(slice_expr)`
macro_rules! contract_pre_q_projection_shape {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract q_projection_shape: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `rope_frequency`.
/// Domain-specific. Call: `contract_pre_rope_frequency!(slice_expr)`
macro_rules! contract_pre_rope_frequency {
    () => {{}};
    ($input:expr) => {{
        let _pv_indices = &$input;
        debug_assert!(_pv_indices.len() > 0,
            "Contract rope_frequency: precondition violated — indices.len() > 0");
    }};
}

/// Preconditions for equation `swiglu_ratio`.
/// Domain-specific. Call: `contract_pre_swiglu_ratio!(slice_expr)`
macro_rules! contract_pre_swiglu_ratio {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract swiglu_ratio: precondition violated — input.len() > 0");
    }};
}

// Auto-generated from contracts/qwen3moe-e2e-verification-v1.yaml — DO NOT EDIT
// Contract: qwen3moe-e2e-verification-v1

/// Preconditions for equation `active_parameter_count`.
/// Domain-specific. Call: `contract_pre_active_parameter_count!(slice_expr)`
macro_rules! contract_pre_active_parameter_count {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract active_parameter_count: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `contract_composition`.
/// Domain-specific. Call: `contract_pre_contract_composition!(slice_expr)`
macro_rules! contract_pre_contract_composition {
    () => {{}};
    ($input:expr) => {{
        let _pv_indices = &$input;
        debug_assert!(_pv_indices.len() > 0,
            "Contract contract_composition: precondition violated — indices.len() > 0");
    }};
}

/// Preconditions for equation `flops_per_token`.
/// Domain-specific. Call: `contract_pre_flops_per_token!(slice_expr)`
macro_rules! contract_pre_flops_per_token {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract flops_per_token: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `memory_breakdown`.
/// Domain-specific. Call: `contract_pre_memory_breakdown!(slice_expr)`
macro_rules! contract_pre_memory_breakdown {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract memory_breakdown: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `model_parameter_count`.
/// Domain-specific. Call: `contract_pre_model_parameter_count!(slice_expr)`
macro_rules! contract_pre_model_parameter_count {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract model_parameter_count: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `throughput_model`.
/// Domain-specific. Call: `contract_pre_throughput_model!(slice_expr)`
macro_rules! contract_pre_throughput_model {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract throughput_model: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `verification_ladder`.
/// Domain-specific. Call: `contract_pre_verification_ladder!(slice_expr)`
macro_rules! contract_pre_verification_ladder {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract verification_ladder: precondition violated — input.len() > 0");
    }};
}

// Auto-generated from contracts/qwen3moe-shapes-v1.yaml — DO NOT EDIT
// Contract: qwen3moe-shapes-v1

/// Preconditions for equation `kv_projection_shape`.
/// Domain-specific. Call: `contract_pre_kv_projection_shape!(slice_expr)`
macro_rules! contract_pre_kv_projection_shape {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract kv_projection_shape: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `moe_expert_shape`.
/// Domain-specific. Call: `contract_pre_moe_expert_shape!(slice_expr)`
macro_rules! contract_pre_moe_expert_shape {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract moe_expert_shape: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `moe_router_shape`.
/// Domain-specific. Call: `contract_pre_moe_router_shape!(slice_expr)`
macro_rules! contract_pre_moe_router_shape {
    () => {{}};
    ($input:expr) => {{
        let _pv_a = &$input;
        debug_assert!(_pv_a.len() > 0,
            "Contract moe_router_shape: precondition violated — a.len() > 0");
    }};
}

/// Preconditions for equation `o_projection_transpose`.
/// Domain-specific. Call: `contract_pre_o_projection_transpose!(slice_expr)`
macro_rules! contract_pre_o_projection_transpose {
    () => {{}};
    ($input:expr) => {{
        let _pv_a = &$input;
        debug_assert!(_pv_a.len() > 0,
            "Contract o_projection_transpose: precondition violated — a.len() > 0");
    }};
}

/// Preconditions for equation `q_projection_shape`.
/// Domain-specific. Call: `contract_pre_q_projection_shape!(slice_expr)`
macro_rules! contract_pre_q_projection_shape {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract q_projection_shape: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `rope_frequency`.
/// Domain-specific. Call: `contract_pre_rope_frequency!(slice_expr)`
macro_rules! contract_pre_rope_frequency {
    () => {{}};
    ($input:expr) => {{
        let _pv_indices = &$input;
        debug_assert!(_pv_indices.len() > 0,
            "Contract rope_frequency: precondition violated — indices.len() > 0");
    }};
}

/// Preconditions for equation `swiglu_ratio`.
/// Domain-specific. Call: `contract_pre_swiglu_ratio!(slice_expr)`
macro_rules! contract_pre_swiglu_ratio {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract swiglu_ratio: precondition violated — input.len() > 0");
    }};
}

// Auto-generated from contracts/rag-pipeline-v1.yaml — DO NOT EDIT
// Contract: rag-pipeline-v1

/// Preconditions for equation `embed_insert`.
/// Call at function entry: `contract_pre_embed_insert!(input_expr)`
macro_rules! contract_pre_embed_insert {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `metric_correctness`.
/// Call at function entry: `contract_pre_metric_correctness!(input_expr)`
macro_rules! contract_pre_metric_correctness {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `retrieve_rank`.
/// Call at function entry: `contract_pre_retrieve_rank!(input_expr)`
macro_rules! contract_pre_retrieve_rank {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

// Auto-generated from contracts/random-forest-v1.yaml — DO NOT EDIT
// Contract: random-forest-v1

/// Preconditions for equation `bootstrap_sample`.
/// Domain-specific. Call: `contract_pre_bootstrap_sample!(slice_expr)`
macro_rules! contract_pre_bootstrap_sample {
    () => {{}};
    ($input:expr) => {{
        let _pv_params = &$input;
        debug_assert!(_pv_params.len() > 0,
            "Contract bootstrap_sample: precondition violated — params.len() > 0");
    }};
}

/// Preconditions for equation `ensemble_size`.
/// Domain-specific. Call: `contract_pre_ensemble_size!(slice_expr)`
macro_rules! contract_pre_ensemble_size {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract ensemble_size: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract ensemble_size: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `majority_vote`.
/// Domain-specific. Call: `contract_pre_majority_vote!(slice_expr)`
macro_rules! contract_pre_majority_vote {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract majority_vote: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract majority_vote: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `predict`.
/// Domain-specific. Call: `contract_pre_predict!(slice_expr)`
macro_rules! contract_pre_predict {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract predict: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract predict: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

// Auto-generated from contracts/recipe-determinism-v1.yaml — DO NOT EDIT
// Contract: recipe-determinism-v1

/// Preconditions for equation `expand_recipe`.
/// Call at function entry: `contract_pre_expand_recipe!(input_expr)`
macro_rules! contract_pre_expand_recipe {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `validate_input_type`.
/// Call at function entry: `contract_pre_validate_input_type!(input_expr)`
macro_rules! contract_pre_validate_input_type {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `validate_inputs`.
/// Domain-specific. Call: `contract_pre_validate_inputs!(slice_expr)`
macro_rules! contract_pre_validate_inputs {
    () => {{}};
    ($input:expr) => {{
        let _pv_inputs = &$input;
        debug_assert!(_pv_inputs.len() > 0,
            "Contract validate_inputs: precondition violated — inputs.len() > 0");
    }};
}

// Auto-generated from contracts/regex-contract-example-v1.yaml — DO NOT EDIT
// Contract: regex-contract-example-v1

/// Preconditions for equation `format_iso_timestamp`.
/// Call at function entry: `contract_pre_format_iso_timestamp!(input_expr)`
macro_rules! contract_pre_format_iso_timestamp {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `format_iso_timestamp`.
/// Call before return: `contract_post_format_iso_timestamp!(result_expr)`
macro_rules! contract_post_format_iso_timestamp {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `format_iso_timestamp`.
macro_rules! contract_format_iso_timestamp {
    ($input:expr, $body:expr) => {{
        contract_pre_format_iso_timestamp!($input);
        let _contract_result = $body;
        contract_post_format_iso_timestamp!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `format_semver`.
/// Call at function entry: `contract_pre_format_semver!(input_expr)`
macro_rules! contract_pre_format_semver {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `format_semver`.
/// Call before return: `contract_post_format_semver!(result_expr)`
macro_rules! contract_post_format_semver {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `format_semver`.
macro_rules! contract_format_semver {
    ($input:expr, $body:expr) => {{
        contract_pre_format_semver!($input);
        let _contract_result = $body;
        contract_post_format_semver!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `format_ticket_id`.
/// Call at function entry: `contract_pre_format_ticket_id!(input_expr)`
macro_rules! contract_pre_format_ticket_id {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `format_ticket_id`.
/// Call before return: `contract_post_format_ticket_id!(result_expr)`
macro_rules! contract_post_format_ticket_id {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `format_ticket_id`.
macro_rules! contract_format_ticket_id {
    ($input:expr, $body:expr) => {{
        contract_pre_format_ticket_id!($input);
        let _contract_result = $body;
        contract_post_format_ticket_id!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/registry-integrity-v1.yaml — DO NOT EDIT
// Contract: registry-integrity-v1

/// Preconditions for equation `pull_idempotency`.
/// Call at function entry: `contract_pre_pull_idempotency!(input_expr)`
macro_rules! contract_pre_pull_idempotency {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `run_lifecycle`.
/// Call at function entry: `contract_pre_run_lifecycle!(input_expr)`
macro_rules! contract_pre_run_lifecycle {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

// Auto-generated from contracts/render-primitives-v1.yaml — DO NOT EDIT
// Contract: render-primitives-v1

/// Preconditions for equation `draw_bounds`.
/// Call at function entry: `contract_pre_draw_bounds!(input_expr)`
macro_rules! contract_pre_draw_bounds {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `layout_area_conservation`.
/// Domain-specific. Call: `contract_pre_layout_area_conservation!(slice_expr)`
macro_rules! contract_pre_layout_area_conservation {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
    }};
}

/// Preconditions for equation `line_connectivity`.
/// Call at function entry: `contract_pre_line_connectivity!(input_expr)`
macro_rules! contract_pre_line_connectivity {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

// Auto-generated from contracts/retrieval-quality-v1.yaml — DO NOT EDIT
// Contract: retrieval-quality-v1

/// Preconditions for equation `embedding_insert`.
/// Domain-specific. Call: `contract_pre_embedding_insert!(slice_expr)`
macro_rules! contract_pre_embedding_insert {
    () => {{}};
    ($input:expr) => {{
        let _pv_embedding = &$input;
    }};
}

/// Preconditions for equation `metric_bounds`.
/// Domain-specific. Call: `contract_pre_metric_bounds!(slice_expr)`
macro_rules! contract_pre_metric_bounds {
    () => {{}};
    ($input:expr) => {{
        let _pv_relevant = &$input;
    }};
}

/// Preconditions for equation `retrieval_ranking`.
/// Call at function entry: `contract_pre_retrieval_ranking!(input_expr)`
macro_rules! contract_pre_retrieval_ranking {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

// Auto-generated from contracts/rmsnorm-kernel-v1.yaml — DO NOT EDIT
// Contract: rmsnorm-kernel-v1

/// Preconditions for equation `rmsnorm`.
/// Domain-specific. Call: `contract_pre_rmsnorm!(slice_expr)`
macro_rules! contract_pre_rmsnorm {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
        debug_assert!(_pv_x.len() > 0,
            "Contract rmsnorm: precondition violated — x.len() > 0");
        debug_assert!(_pv_x.iter().all(|v| v.is_finite()),
            "Contract rmsnorm: precondition violated — x.iter().all(|v| v.is_finite())");
    }};
}

/// Postconditions for equation `rmsnorm`.
/// Call before return: `contract_post_rmsnorm!(result_expr)`
macro_rules! contract_post_rmsnorm {
    ($result:expr) => {{
        let _contract_result = &$result;
        debug_assert!(_contract_result.iter().all(|v| v.is_finite()), "Contract rmsnorm: postcondition violated — result.iter().all(|v| v.is_finite())");
    }};
}

/// Combined pre+post contract for equation `rmsnorm`.
macro_rules! contract_rmsnorm {
    ($input:expr, $body:expr) => {{
        contract_pre_rmsnorm!($input);
        let _contract_result = $body;
        contract_post_rmsnorm!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/roofline-model-v1.yaml — DO NOT EDIT
// Contract: roofline-model-v1

/// Preconditions for equation `bandwidth_ceiling`.
/// Domain-specific. Call: `contract_pre_bandwidth_ceiling!(slice_expr)`
macro_rules! contract_pre_bandwidth_ceiling {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract bandwidth_ceiling: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `compute_ceiling`.
/// Domain-specific. Call: `contract_pre_compute_ceiling!(slice_expr)`
macro_rules! contract_pre_compute_ceiling {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract compute_ceiling: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `model_bytes`.
/// Domain-specific. Call: `contract_pre_model_bytes!(slice_expr)`
macro_rules! contract_pre_model_bytes {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract model_bytes: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `throughput_bound`.
/// Domain-specific. Call: `contract_pre_throughput_bound!(slice_expr)`
macro_rules! contract_pre_throughput_bound {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract throughput_bound: precondition violated — input.len() > 0");
    }};
}

// Auto-generated from contracts/rope-extrapolation-v1.yaml — DO NOT EDIT
// Contract: rope-extrapolation-v1

/// Preconditions for equation `base_frequency`.
/// Domain-specific. Call: `contract_pre_base_frequency!(slice_expr)`
macro_rules! contract_pre_base_frequency {
    () => {{}};
    ($input:expr) => {{
        let _pv_indices = &$input;
        debug_assert!(_pv_indices.len() > 0,
            "Contract base_frequency: precondition violated — indices.len() > 0");
    }};
}

/// Preconditions for equation `linear_interpolation`.
/// Domain-specific. Call: `contract_pre_linear_interpolation!(slice_expr)`
macro_rules! contract_pre_linear_interpolation {
    () => {{}};
    ($input:expr) => {{
        let _pv_indices = &$input;
        debug_assert!(_pv_indices.len() > 0,
            "Contract linear_interpolation: precondition violated — indices.len() > 0");
    }};
}

/// Preconditions for equation `ntk_scaled_base`.
/// Domain-specific. Call: `contract_pre_ntk_scaled_base!(slice_expr)`
macro_rules! contract_pre_ntk_scaled_base {
    () => {{}};
    ($input:expr) => {{
        let _pv_indices = &$input;
        debug_assert!(_pv_indices.len() > 0,
            "Contract ntk_scaled_base: precondition violated — indices.len() > 0");
    }};
}

/// Preconditions for equation `rotation_matrix`.
/// Domain-specific. Call: `contract_pre_rotation_matrix!(slice_expr)`
macro_rules! contract_pre_rotation_matrix {
    () => {{}};
    ($input:expr) => {{
        let _pv_indices = &$input;
        debug_assert!(_pv_indices.len() > 0,
            "Contract rotation_matrix: precondition violated — indices.len() > 0");
    }};
}

/// Preconditions for equation `yarn_mixed_frequency`.
/// Domain-specific. Call: `contract_pre_yarn_mixed_frequency!(slice_expr)`
macro_rules! contract_pre_yarn_mixed_frequency {
    () => {{}};
    ($input:expr) => {{
        let _pv_indices = &$input;
        debug_assert!(_pv_indices.len() > 0,
            "Contract yarn_mixed_frequency: precondition violated — indices.len() > 0");
    }};
}

/// Preconditions for equation `yarn_ramp`.
/// Domain-specific. Call: `contract_pre_yarn_ramp!(slice_expr)`
macro_rules! contract_pre_yarn_ramp {
    () => {{}};
    ($input:expr) => {{
        let _pv_indices = &$input;
        debug_assert!(_pv_indices.len() > 0,
            "Contract yarn_ramp: precondition violated — indices.len() > 0");
    }};
}

// Auto-generated from contracts/rope-kernel-v1.yaml — DO NOT EDIT
// Contract: rope-kernel-v1

/// Preconditions for equation `rope`.
/// Domain-specific. Call: `contract_pre_rope!(slice_expr)`
macro_rules! contract_pre_rope {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
        debug_assert!(_pv_x.len() > 0,
            "Contract rope: precondition violated — x.len() > 0");
        debug_assert!(_pv_x.len() % 2 == 0,
            "Contract rope: precondition violated — x.len() % 2 == 0");
    }};
}

/// Postconditions for equation `rope`.
/// Call before return: `contract_post_rope!(result_expr)`
macro_rules! contract_post_rope {
    ($result:expr) => {{
        let _contract_result = &$result;
        debug_assert!(_contract_result.iter().all(|v| v.is_finite()), "Contract rope: postcondition violated — result.iter().all(|v| v.is_finite())");
    }};
}

/// Combined pre+post contract for equation `rope`.
macro_rules! contract_rope {
    ($input:expr, $body:expr) => {{
        contract_pre_rope!($input);
        let _contract_result = $body;
        contract_post_rope!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/safetensors-cpu-dispatch-v1.yaml — DO NOT EDIT
// Contract: safetensors-cpu-dispatch-v1

/// Preconditions for equation `format_parity`.
/// Domain-specific. Call: `contract_pre_format_parity!(slice_expr)`
macro_rules! contract_pre_format_parity {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract format_parity: precondition violated — input.len() > 0");
    }};
}

// Auto-generated from contracts/safetensors-format-safety-v1.yaml — DO NOT EDIT
// Contract: safetensors-format-safety-v1

/// Preconditions for equation `dtype_consistency`.
/// Call at function entry: `contract_pre_dtype_consistency!(input_expr)`
macro_rules! contract_pre_dtype_consistency {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `dtype_consistency`.
/// Call before return: `contract_post_dtype_consistency!(result_expr)`
macro_rules! contract_post_dtype_consistency {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `dtype_consistency`.
macro_rules! contract_dtype_consistency {
    ($input:expr, $body:expr) => {{
        contract_pre_dtype_consistency!($input);
        let _contract_result = $body;
        contract_post_dtype_consistency!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `header_size_validation`.
/// Call at function entry: `contract_pre_header_size_validation!(input_expr)`
macro_rules! contract_pre_header_size_validation {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `header_size_validation`.
/// Call before return: `contract_post_header_size_validation!(result_expr)`
macro_rules! contract_post_header_size_validation {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `header_size_validation`.
macro_rules! contract_header_size_validation {
    ($input:expr, $body:expr) => {{
        contract_pre_header_size_validation!($input);
        let _contract_result = $body;
        contract_post_header_size_validation!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `mmap_zero_copy`.
/// Call at function entry: `contract_pre_mmap_zero_copy!(input_expr)`
macro_rules! contract_pre_mmap_zero_copy {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `mmap_zero_copy`.
/// Call before return: `contract_post_mmap_zero_copy!(result_expr)`
macro_rules! contract_post_mmap_zero_copy {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `mmap_zero_copy`.
macro_rules! contract_mmap_zero_copy {
    ($input:expr, $body:expr) => {{
        contract_pre_mmap_zero_copy!($input);
        let _contract_result = $body;
        contract_post_mmap_zero_copy!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `no_overlap_invariant`.
/// Call at function entry: `contract_pre_no_overlap_invariant!(input_expr)`
macro_rules! contract_pre_no_overlap_invariant {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `no_overlap_invariant`.
/// Call before return: `contract_post_no_overlap_invariant!(result_expr)`
macro_rules! contract_post_no_overlap_invariant {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `no_overlap_invariant`.
macro_rules! contract_no_overlap_invariant {
    ($input:expr, $body:expr) => {{
        contract_pre_no_overlap_invariant!($input);
        let _contract_result = $body;
        contract_post_no_overlap_invariant!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `tensor_offset_bounds`.
/// Call at function entry: `contract_pre_tensor_offset_bounds!(input_expr)`
macro_rules! contract_pre_tensor_offset_bounds {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `tensor_offset_bounds`.
/// Call before return: `contract_post_tensor_offset_bounds!(result_expr)`
macro_rules! contract_post_tensor_offset_bounds {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `tensor_offset_bounds`.
macro_rules! contract_tensor_offset_bounds {
    ($input:expr, $body:expr) => {{
        contract_pre_tensor_offset_bounds!($input);
        let _contract_result = $body;
        contract_post_tensor_offset_bounds!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/safety-classifier-v1.yaml — DO NOT EDIT
// Contract: safety-classifier-v1

/// Preconditions for equation `classify_filesystem`.
/// Domain-specific. Call: `contract_pre_classify_filesystem!(slice_expr)`
macro_rules! contract_pre_classify_filesystem {
    () => {{}};
    ($input:expr) => {{
        let _pv_source = &$input;
        debug_assert!(!_pv_source.is_empty(),
            "Contract classify_filesystem: precondition violated — !source.is_empty()");
        debug_assert!(_pv_source.len() <= 1_000_000,
            "Contract classify_filesystem: precondition violated — source.len() <= 1_000_000");
    }};
}

/// Preconditions for equation `classify_injection`.
/// Domain-specific. Call: `contract_pre_classify_injection!(slice_expr)`
macro_rules! contract_pre_classify_injection {
    () => {{}};
    ($input:expr) => {{
        let _pv_source = &$input;
        debug_assert!(!_pv_source.is_empty(),
            "Contract classify_injection: precondition violated — !source.is_empty()");
        debug_assert!(_pv_source.len() <= 1_000_000,
            "Contract classify_injection: precondition violated — source.len() <= 1_000_000");
    }};
}

/// Preconditions for equation `classify_secrets`.
/// Domain-specific. Call: `contract_pre_classify_secrets!(slice_expr)`
macro_rules! contract_pre_classify_secrets {
    () => {{}};
    ($input:expr) => {{
        let _pv_source = &$input;
        debug_assert!(!_pv_source.is_empty(),
            "Contract classify_secrets: precondition violated — !source.is_empty()");
        debug_assert!(_pv_source.len() <= 1_000_000,
            "Contract classify_secrets: precondition violated — source.len() <= 1_000_000");
    }};
}

/// Preconditions for equation `lint_shell`.
/// Call at function entry: `contract_pre_lint_shell!(input_expr)`
macro_rules! contract_pre_lint_shell {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

// Auto-generated from contracts/sampling-algorithms-v1.yaml — DO NOT EDIT
// Contract: sampling-algorithms-v1

/// Preconditions for equation `greedy`.
/// Domain-specific. Call: `contract_pre_greedy!(slice_expr)`
macro_rules! contract_pre_greedy {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract greedy: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `temperature`.
/// Domain-specific. Call: `contract_pre_temperature!(slice_expr)`
macro_rules! contract_pre_temperature {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract temperature: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `top_k`.
/// Domain-specific. Call: `contract_pre_top_k!(slice_expr)`
macro_rules! contract_pre_top_k {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract top_k: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `top_p`.
/// Domain-specific. Call: `contract_pre_top_p!(slice_expr)`
macro_rules! contract_pre_top_p {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract top_p: precondition violated — input.len() > 0");
    }};
}

// Auto-generated from contracts/sandbox-isolation-v1.yaml — DO NOT EDIT
// Contract: sandbox-isolation-v1

/// Preconditions for equation `filesystem_isolation`.
/// Domain-specific. Call: `contract_pre_filesystem_isolation!(slice_expr)`
macro_rules! contract_pre_filesystem_isolation {
    () => {{}};
    ($input:expr) => {{
        let _pv_config = &$input;
    }};
}

/// Preconditions for equation `network_isolation`.
/// Domain-specific. Call: `contract_pre_network_isolation!(slice_expr)`
macro_rules! contract_pre_network_isolation {
    () => {{}};
    ($input:expr) => {{
        let _pv_config = &$input;
    }};
}

/// Preconditions for equation `overlay_capture`.
/// Domain-specific. Call: `contract_pre_overlay_capture!(slice_expr)`
macro_rules! contract_pre_overlay_capture {
    () => {{}};
    ($input:expr) => {{
        let _pv_overlay = &$input;
    }};
}

// Auto-generated from contracts/score-composite-v1.yaml — DO NOT EDIT
// Contract: score-composite-v1

/// Preconditions for equation `geometric_mean`.
/// Domain-specific. Call: `contract_pre_geometric_mean!(slice_expr)`
macro_rules! contract_pre_geometric_mean {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract geometric_mean: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract geometric_mean: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Postconditions for equation `geometric_mean`.
/// Call before return: `contract_post_geometric_mean!(result_expr)`
macro_rules! contract_post_geometric_mean {
    ($result:expr) => {{
        let _contract_result = &$result;
        debug_assert!(*_contract_result >= 0.0 && *_contract_result <= 100.0, "Contract geometric_mean: postcondition violated — result >= 0.0 && result <= 100.0");
    }};
}

/// Combined pre+post contract for equation `geometric_mean`.
macro_rules! contract_geometric_mean {
    ($input:expr, $body:expr) => {{
        contract_pre_geometric_mean!($input);
        let _contract_result = $body;
        contract_post_geometric_mean!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `grade_from_score`.
/// Domain-specific. Call: `contract_pre_grade_from_score!(slice_expr)`
macro_rules! contract_pre_grade_from_score {
    () => {{}};
    ($input:expr) => {{
        let _pv_grad_output = &$input;
        debug_assert!(_pv_grad_output.len() > 0,
            "Contract grade_from_score: precondition violated — grad_output.len() > 0");
        debug_assert!(_pv_grad_output.iter().all(|v| v.is_finite()),
            "Contract grade_from_score: precondition violated — grad_output.iter().all(|v| v.is_finite())");
    }};
}

/// Postconditions for equation `grade_from_score`.
/// Call before return: `contract_post_grade_from_score!(result_expr)`
macro_rules! contract_post_grade_from_score {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `grade_from_score`.
macro_rules! contract_grade_from_score {
    ($input:expr, $body:expr) => {{
        contract_pre_grade_from_score!($input);
        let _contract_result = $body;
        contract_post_grade_from_score!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/secret-provider-v1.yaml — DO NOT EDIT
// Contract: secret-provider-v1

/// Preconditions for equation `drift_detection`.
/// Call at function entry: `contract_pre_drift_detection!(input_expr)`
macro_rules! contract_pre_drift_detection {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `ephemeral_cleanup`.
/// Domain-specific. Call: `contract_pre_ephemeral_cleanup!(slice_expr)`
macro_rules! contract_pre_ephemeral_cleanup {
    () => {{}};
    ($input:expr) => {{
        let _pv_secret = &$input;
    }};
}

/// Preconditions for equation `provider_dispatch`.
/// Domain-specific. Call: `contract_pre_provider_dispatch!(slice_expr)`
macro_rules! contract_pre_provider_dispatch {
    () => {{}};
    ($input:expr) => {{
        let _pv_ref = &$input;
    }};
}

// Auto-generated from contracts/semantic-equivalence-v1.yaml — DO NOT EDIT
// Contract: semantic-equivalence-v1

/// Preconditions for equation `comprehension_equivalence`.
/// Domain-specific. Call: `contract_pre_comprehension_equivalence!(slice_expr)`
macro_rules! contract_pre_comprehension_equivalence {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract comprehension_equivalence: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `control_flow_equivalence`.
/// Domain-specific. Call: `contract_pre_control_flow_equivalence!(slice_expr)`
macro_rules! contract_pre_control_flow_equivalence {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract control_flow_equivalence: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `expression_equivalence`.
/// Domain-specific. Call: `contract_pre_expression_equivalence!(slice_expr)`
macro_rules! contract_pre_expression_equivalence {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract expression_equivalence: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `observational_equivalence`.
/// Domain-specific. Call: `contract_pre_observational_equivalence!(slice_expr)`
macro_rules! contract_pre_observational_equivalence {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract observational_equivalence: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `statement_equivalence`.
/// Domain-specific. Call: `contract_pre_statement_equivalence!(slice_expr)`
macro_rules! contract_pre_statement_equivalence {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract statement_equivalence: precondition violated — input.len() > 0");
    }};
}

// Auto-generated from contracts/serialization-v1.yaml — DO NOT EDIT
// Contract: serialization-v1

/// Preconditions for equation `deserialize`.
/// Domain-specific. Call: `contract_pre_deserialize!(slice_expr)`
macro_rules! contract_pre_deserialize {
    () => {{}};
    ($input:expr) => {{
        let _pv_bytes = &$input;
        debug_assert!(_pv_bytes.len() > 0,
            "Contract deserialize: precondition violated — bytes.len() > 0");
    }};
}

/// Preconditions for equation `serialize`.
/// Call at function entry: `contract_pre_serialize!(input_expr)`
macro_rules! contract_pre_serialize {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

// Auto-generated from contracts/serialization-v1.yaml — DO NOT EDIT
// Contract: serialization-v1

/// Preconditions for equation `serialization`.
/// Domain-specific. Call: `contract_pre_serialization!(slice_expr)`
macro_rules! contract_pre_serialization {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract serialization: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract serialization: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

// Auto-generated from contracts/session-v1.yaml — DO NOT EDIT
// Contract: session-v1

/// Preconditions for equation `age_filter`.
/// Call at function entry: `contract_pre_age_filter!(input_expr)`
macro_rules! contract_pre_age_filter {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `age_filter`.
/// Call before return: `contract_post_age_filter!(result_expr)`
macro_rules! contract_post_age_filter {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `age_filter`.
macro_rules! contract_age_filter {
    ($input:expr, $body:expr) => {{
        contract_pre_age_filter!($input);
        let _contract_result = $body;
        contract_post_age_filter!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `append_only`.
/// Call at function entry: `contract_pre_append_only!(input_expr)`
macro_rules! contract_pre_append_only {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `append_only`.
/// Call before return: `contract_post_append_only!(result_expr)`
macro_rules! contract_post_append_only {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `append_only`.
macro_rules! contract_append_only {
    ($input:expr, $body:expr) => {{
        contract_pre_append_only!($input);
        let _contract_result = $body;
        contract_post_append_only!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `jsonl_roundtrip`.
/// Domain-specific. Call: `contract_pre_jsonl_roundtrip!(slice_expr)`
macro_rules! contract_pre_jsonl_roundtrip {
    () => {{}};
    ($input:expr) => {{
        let _pv_messages = &$input;
        debug_assert!(_pv_messages.len() > 0,
            "Contract jsonl_roundtrip: precondition violated — messages.len() > 0");
    }};
}

/// Postconditions for equation `jsonl_roundtrip`.
/// Call before return: `contract_post_jsonl_roundtrip!(result_expr)`
macro_rules! contract_post_jsonl_roundtrip {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `jsonl_roundtrip`.
macro_rules! contract_jsonl_roundtrip {
    ($input:expr, $body:expr) => {{
        contract_pre_jsonl_roundtrip!($input);
        let _contract_result = $body;
        contract_post_jsonl_roundtrip!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `manifest_serde`.
/// Domain-specific. Call: `contract_pre_manifest_serde!(slice_expr)`
macro_rules! contract_pre_manifest_serde {
    () => {{}};
    ($input:expr) => {{
        let _pv_manifest = &$input;
    }};
}

/// Postconditions for equation `manifest_serde`.
/// Call before return: `contract_post_manifest_serde!(result_expr)`
macro_rules! contract_post_manifest_serde {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `manifest_serde`.
macro_rules! contract_manifest_serde {
    ($input:expr, $body:expr) => {{
        contract_pre_manifest_serde!($input);
        let _contract_result = $body;
        contract_post_manifest_serde!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/shannon-entropy-v1.yaml — DO NOT EDIT
// Contract: shannon-entropy-v1

/// Preconditions for equation `entropy`.
/// Domain-specific. Call: `contract_pre_entropy!(slice_expr)`
macro_rules! contract_pre_entropy {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract entropy: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract entropy: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `uniform_entropy`.
/// Domain-specific. Call: `contract_pre_uniform_entropy!(slice_expr)`
macro_rules! contract_pre_uniform_entropy {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract uniform_entropy: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract uniform_entropy: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

// Auto-generated from contracts/shell-execution-v1.yaml — DO NOT EDIT
// Contract: shell-execution-v1

/// Preconditions for equation `config_validation`.
/// Call at function entry: `contract_pre_config_validation!(input_expr)`
macro_rules! contract_pre_config_validation {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `parser_correctness`.
/// Domain-specific. Call: `contract_pre_parser_correctness!(slice_expr)`
macro_rules! contract_pre_parser_correctness {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() <= 1_048_576,
            "Contract parser_correctness: precondition violated — input.len() <= 1_048_576");
    }};
}

/// Preconditions for equation `startup_budget`.
/// Call at function entry: `contract_pre_startup_budget!(input_expr)`
macro_rules! contract_pre_startup_budget {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

// Auto-generated from contracts/silu-kernel-v1.yaml — DO NOT EDIT
// Contract: silu-kernel-v1

/// Preconditions for equation `sigmoid`.
/// Domain-specific. Call: `contract_pre_sigmoid!(slice_expr)`
macro_rules! contract_pre_sigmoid {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
        debug_assert!(_pv_x.iter().all(|v| v.is_finite()),
            "Contract sigmoid: precondition violated — x.iter().all(|v| v.is_finite())");
        debug_assert!(_pv_x.len() > 0,
            "Contract sigmoid: precondition violated — x.len() > 0");
    }};
}

/// Preconditions for equation `silu`.
/// Domain-specific. Call: `contract_pre_silu!(slice_expr)`
macro_rules! contract_pre_silu {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
        debug_assert!(_pv_x.iter().all(|v| v.is_finite()),
            "Contract silu: precondition violated — x.iter().all(|v| v.is_finite())");
        debug_assert!(_pv_x.len() > 0,
            "Contract silu: precondition violated — x.len() > 0");
    }};
}

// Auto-generated from contracts/simulation-determinism-v1.yaml — DO NOT EDIT
// Contract: simulation-determinism-v1

/// Preconditions for equation `audit_trail`.
/// Call at function entry: `contract_pre_audit_trail!(input_expr)`
macro_rules! contract_pre_audit_trail {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `step_determinism`.
/// Domain-specific. Call: `contract_pre_step_determinism!(slice_expr)`
macro_rules! contract_pre_step_determinism {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
    }};
}

/// Preconditions for equation `time_advancement`.
/// Domain-specific. Call: `contract_pre_time_advancement!(slice_expr)`
macro_rules! contract_pre_time_advancement {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
    }};
}

// Auto-generated from contracts/simulation-step-v1.yaml — DO NOT EDIT
// Contract: simulation-step-v1

/// Preconditions for equation `audit_completeness`.
/// Call at function entry: `contract_pre_audit_completeness!(input_expr)`
macro_rules! contract_pre_audit_completeness {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `simulate_convergence`.
/// Domain-specific. Call: `contract_pre_simulate_convergence!(slice_expr)`
macro_rules! contract_pre_simulate_convergence {
    () => {{}};
    ($input:expr) => {{
        let _pv_params = &$input;
    }};
}

/// Preconditions for equation `step_monotonicity`.
/// Domain-specific. Call: `contract_pre_step_monotonicity!(slice_expr)`
macro_rules! contract_pre_step_monotonicity {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
    }};
}

// Auto-generated from contracts/sliding-window-attention-v1.yaml — DO NOT EDIT
// Contract: sliding-window-attention-v1

/// Preconditions for equation `attention_sparsity`.
/// Domain-specific. Call: `contract_pre_attention_sparsity!(slice_expr)`
macro_rules! contract_pre_attention_sparsity {
    () => {{}};
    ($input:expr) => {{
        let _pv_q = &$input;
        debug_assert!(_pv_q.len() > 0,
            "Contract attention_sparsity: precondition violated — q.len() > 0");
    }};
}

/// Preconditions for equation `causal_window_mask`.
/// Domain-specific. Call: `contract_pre_causal_window_mask!(slice_expr)`
macro_rules! contract_pre_causal_window_mask {
    () => {{}};
    ($input:expr) => {{
        let _pv_q = &$input;
        debug_assert!(_pv_q.len() > 0,
            "Contract causal_window_mask: precondition violated — q.len() > 0");
    }};
}

/// Preconditions for equation `effective_context`.
/// Domain-specific. Call: `contract_pre_effective_context!(slice_expr)`
macro_rules! contract_pre_effective_context {
    () => {{}};
    ($input:expr) => {{
        let _pv_q = &$input;
        debug_assert!(_pv_q.len() > 0,
            "Contract effective_context: precondition violated — q.len() > 0");
    }};
}

/// Preconditions for equation `multi_layer_receptive_field`.
/// Domain-specific. Call: `contract_pre_multi_layer_receptive_field!(slice_expr)`
macro_rules! contract_pre_multi_layer_receptive_field {
    () => {{}};
    ($input:expr) => {{
        let _pv_q = &$input;
        debug_assert!(_pv_q.len() > 0,
            "Contract multi_layer_receptive_field: precondition violated — q.len() > 0");
    }};
}

/// Preconditions for equation `window_mask`.
/// Domain-specific. Call: `contract_pre_window_mask!(slice_expr)`
macro_rules! contract_pre_window_mask {
    () => {{}};
    ($input:expr) => {{
        let _pv_q = &$input;
        debug_assert!(_pv_q.len() > 0,
            "Contract window_mask: precondition violated — q.len() > 0");
    }};
}

// Auto-generated from contracts/softmax-kernel-v1.yaml — DO NOT EDIT
// Contract: softmax-kernel-v1

/// Preconditions for equation `softmax`.
/// Domain-specific. Call: `contract_pre_softmax!(slice_expr)`
macro_rules! contract_pre_softmax {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
        debug_assert!(_pv_x.len() > 0,
            "Contract softmax: precondition violated — x.len() > 0");
        debug_assert!(_pv_x.iter().all(|v| v.is_finite()),
            "Contract softmax: precondition violated — x.iter().all(|v| v.is_finite())");
    }};
}

/// Postconditions for equation `softmax`.
/// Call before return: `contract_post_softmax!(result_expr)`
macro_rules! contract_post_softmax {
    ($result:expr) => {{
        let _contract_result = &$result;
        debug_assert!(_contract_result.iter().all(|v| *v > 0.0), "Contract softmax: postcondition violated — result.iter().all(|v| *v > 0.0)");
        debug_assert!((_contract_result.iter().sum::<f32>() - 1.0).abs() < 1e-5, "Contract softmax: postcondition violated — (result.iter().sum::<f32>() - 1.0).abs() < 1e-5");
    }};
}

/// Combined pre+post contract for equation `softmax`.
macro_rules! contract_softmax {
    ($input:expr, $body:expr) => {{
        contract_pre_softmax!($input);
        let _contract_result = $body;
        contract_post_softmax!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/sovereign-tensor-v1.yaml — DO NOT EDIT
// Contract: sovereign-tensor-v1

// Auto-generated from contracts/special-tokens-registry-v1.yaml — DO NOT EDIT
// Contract: special-tokens-registry-v1

/// Preconditions for equation `token_bounds`.
/// Domain-specific. Call: `contract_pre_token_bounds!(slice_expr)`
macro_rules! contract_pre_token_bounds {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract token_bounds: precondition violated — input.len() > 0");
    }};
}

// Auto-generated from contracts/speculative-decoding-v1.yaml — DO NOT EDIT
// Contract: speculative-decoding-v1

/// Preconditions for equation `acceptance_probability`.
/// Domain-specific. Call: `contract_pre_acceptance_probability!(slice_expr)`
macro_rules! contract_pre_acceptance_probability {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract acceptance_probability: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `output_equivalence`.
/// Domain-specific. Call: `contract_pre_output_equivalence!(slice_expr)`
macro_rules! contract_pre_output_equivalence {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract output_equivalence: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `token_acceptance`.
/// Domain-specific. Call: `contract_pre_token_acceptance!(slice_expr)`
macro_rules! contract_pre_token_acceptance {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract token_acceptance: precondition violated — input.len() > 0");
    }};
}

// Auto-generated from contracts/ssm-kernel-v1.yaml — DO NOT EDIT
// Contract: ssm-kernel-v1

/// Preconditions for equation `selective_gate`.
/// Domain-specific. Call: `contract_pre_selective_gate!(slice_expr)`
macro_rules! contract_pre_selective_gate {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract selective_gate: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract selective_gate: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `ssm_discretize`.
/// Domain-specific. Call: `contract_pre_ssm_discretize!(slice_expr)`
macro_rules! contract_pre_ssm_discretize {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
        debug_assert!(_pv_x.iter().all(|v| v.is_finite()),
            "Contract ssm_discretize: precondition violated — x.iter().all(|v| v.is_finite())");
        debug_assert!(_pv_x.len() > 0,
            "Contract ssm_discretize: precondition violated — x.len() > 0");
    }};
}

/// Preconditions for equation `ssm_scan`.
/// Domain-specific. Call: `contract_pre_ssm_scan!(slice_expr)`
macro_rules! contract_pre_ssm_scan {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract ssm_scan: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract ssm_scan: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

// Auto-generated from contracts/state-machine-v1.yaml — DO NOT EDIT
// Contract: state-machine-v1

/// Preconditions for equation `event_store_append_only`.
/// Call at function entry: `contract_pre_event_store_append_only!(input_expr)`
macro_rules! contract_pre_event_store_append_only {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `event_store_append_only`.
/// Call before return: `contract_post_event_store_append_only!(result_expr)`
macro_rules! contract_post_event_store_append_only {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `event_store_append_only`.
macro_rules! contract_event_store_append_only {
    ($input:expr, $body:expr) => {{
        contract_pre_event_store_append_only!($input);
        let _contract_result = $body;
        contract_post_event_store_append_only!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `refactor_transitions`.
/// Call at function entry: `contract_pre_refactor_transitions!(input_expr)`
macro_rules! contract_pre_refactor_transitions {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `refactor_transitions`.
/// Call before return: `contract_post_refactor_transitions!(result_expr)`
macro_rules! contract_post_refactor_transitions {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `refactor_transitions`.
macro_rules! contract_refactor_transitions {
    ($input:expr, $body:expr) => {{
        contract_pre_refactor_transitions!($input);
        let _contract_result = $body;
        contract_post_refactor_transitions!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `snapshot_recovery`.
/// Call at function entry: `contract_pre_snapshot_recovery!(input_expr)`
macro_rules! contract_pre_snapshot_recovery {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `snapshot_recovery`.
/// Call before return: `contract_post_snapshot_recovery!(result_expr)`
macro_rules! contract_post_snapshot_recovery {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `snapshot_recovery`.
macro_rules! contract_snapshot_recovery {
    ($input:expr, $body:expr) => {{
        contract_pre_snapshot_recovery!($input);
        let _contract_result = $body;
        contract_post_snapshot_recovery!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/store-cas-v1.yaml — DO NOT EDIT
// Contract: store-cas-v1

/// Preconditions for equation `closure_completeness`.
/// Domain-specific. Call: `contract_pre_closure_completeness!(slice_expr)`
macro_rules! contract_pre_closure_completeness {
    () => {{}};
    ($input:expr) => {{
        let _pv_entry = &$input;
    }};
}

/// Preconditions for equation `derivation_determinism`.
/// Domain-specific. Call: `contract_pre_derivation_determinism!(slice_expr)`
macro_rules! contract_pre_derivation_determinism {
    () => {{}};
    ($input:expr) => {{
        let _pv_d = &$input;
    }};
}

/// Preconditions for equation `far_archive_roundtrip`.
/// Domain-specific. Call: `contract_pre_far_archive_roundtrip!(slice_expr)`
macro_rules! contract_pre_far_archive_roundtrip {
    () => {{}};
    ($input:expr) => {{
        let _pv_dir = &$input;
        debug_assert!(_pv_dir.is__pv_dir(),
            "Contract far_archive_roundtrip: precondition violated — dir.is_dir()");
    }};
}

/// Preconditions for equation `gc_safety`.
/// Domain-specific. Call: `contract_pre_gc_safety!(slice_expr)`
macro_rules! contract_pre_gc_safety {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
    }};
}

/// Preconditions for equation `purity_monotonicity`.
/// Domain-specific. Call: `contract_pre_purity_monotonicity!(slice_expr)`
macro_rules! contract_pre_purity_monotonicity {
    () => {{}};
    ($input:expr) => {{
        let _pv_d = &$input;
    }};
}

// Auto-generated from contracts/streaming-tpot-v1.yaml — DO NOT EDIT
// Contract: streaming-tpot-v1

/// Preconditions for equation `tpot_definition`.
/// Domain-specific. Call: `contract_pre_tpot_definition!(slice_expr)`
macro_rules! contract_pre_tpot_definition {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract tpot_definition: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract tpot_definition: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

// Auto-generated from contracts/svm-v1.yaml — DO NOT EDIT
// Contract: svm-v1

/// Preconditions for equation `decision_function`.
/// Domain-specific. Call: `contract_pre_decision_function!(slice_expr)`
macro_rules! contract_pre_decision_function {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract decision_function: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract decision_function: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `hinge_loss`.
/// Domain-specific. Call: `contract_pre_hinge_loss!(slice_expr)`
macro_rules! contract_pre_hinge_loss {
    () => {{}};
    ($input:expr) => {{
        let _pv_predicted = &$input;
        debug_assert!(_pv_predicted.len() > 0,
            "Contract hinge_loss: precondition violated — predicted.len() > 0");
    }};
}

/// Preconditions for equation `margin`.
/// Domain-specific. Call: `contract_pre_margin!(slice_expr)`
macro_rules! contract_pre_margin {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract margin: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract margin: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `svm_predict`.
/// Domain-specific. Call: `contract_pre_svm_predict!(slice_expr)`
macro_rules! contract_pre_svm_predict {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract svm_predict: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract svm_predict: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

// Auto-generated from contracts/swiglu-kernel-v1.yaml — DO NOT EDIT
// Contract: swiglu-kernel-v1

/// Preconditions for equation `silu`.
/// Domain-specific. Call: `contract_pre_silu!(slice_expr)`
macro_rules! contract_pre_silu {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
        debug_assert!(_pv_x.iter().all(|v| v.is_finite()),
            "Contract silu: precondition violated — x.iter().all(|v| v.is_finite())");
        debug_assert!(_pv_x.len() > 0,
            "Contract silu: precondition violated — x.len() > 0");
    }};
}

/// Preconditions for equation `swiglu`.
/// Domain-specific. Call: `contract_pre_swiglu!(slice_expr)`
macro_rules! contract_pre_swiglu {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
        debug_assert!(_pv_x.len() > 0,
            "Contract swiglu: precondition violated — x.len() > 0");
        debug_assert!(_pv_x.iter().all(|v| v.is_finite()),
            "Contract swiglu: precondition violated — x.iter().all(|v| v.is_finite())");
    }};
}

/// Postconditions for equation `swiglu`.
/// Call before return: `contract_post_swiglu!(result_expr)`
macro_rules! contract_post_swiglu {
    ($result:expr) => {{
        let _contract_result = &$result;
        debug_assert!(_contract_result.iter().all(|v| v.is_finite()), "Contract swiglu: postcondition violated — result.iter().all(|v| v.is_finite())");
    }};
}

/// Combined pre+post contract for equation `swiglu`.
macro_rules! contract_swiglu {
    ($input:expr, $body:expr) => {{
        contract_pre_swiglu!($input);
        let _contract_result = $body;
        contract_post_swiglu!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/task-pipeline-v1.yaml — DO NOT EDIT
// Contract: task-pipeline-v1

/// Preconditions for equation `health_check_retry`.
/// Domain-specific. Call: `contract_pre_health_check_retry!(slice_expr)`
macro_rules! contract_pre_health_check_retry {
    () => {{}};
    ($input:expr) => {{
        let _pv_hc = &$input;
    }};
}

/// Preconditions for equation `pipeline_dag_execution`.
/// Domain-specific. Call: `contract_pre_pipeline_dag_execution!(slice_expr)`
macro_rules! contract_pre_pipeline_dag_execution {
    () => {{}};
    ($input:expr) => {{
        let _pv_stages = &$input;
        debug_assert!(_pv_stages.len() > 0,
            "Contract pipeline_dag_execution: precondition violated — stages.len() > 0");
    }};
}

/// Preconditions for equation `quality_gate_enforcement`.
/// Domain-specific. Call: `contract_pre_quality_gate_enforcement!(slice_expr)`
macro_rules! contract_pre_quality_gate_enforcement {
    () => {{}};
    ($input:expr) => {{
        let _pv_gate = &$input;
    }};
}

/// Preconditions for equation `task_status_terminal`.
/// Call at function entry: `contract_pre_task_status_terminal!(input_expr)`
macro_rules! contract_pre_task_status_terminal {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

// Auto-generated from contracts/tdg-scoring-v1.yaml — DO NOT EDIT
// Contract: tdg-scoring-v1

/// Preconditions for equation `calculate_tdg`.
/// Call at function entry: `contract_pre_calculate_tdg!(input_expr)`
macro_rules! contract_pre_calculate_tdg {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `calculate_tdg`.
/// Call before return: `contract_post_calculate_tdg!(result_expr)`
macro_rules! contract_post_calculate_tdg {
    ($result:expr) => {{
        let _contract_result = &$result;
        debug_assert!(*_contract_result >= 0.0 && *_contract_result <= 100.0, "Contract calculate_tdg: postcondition violated — result >= 0.0 && result <= 100.0");
    }};
}

/// Combined pre+post contract for equation `calculate_tdg`.
macro_rules! contract_calculate_tdg {
    ($input:expr, $body:expr) => {{
        contract_pre_calculate_tdg!($input);
        let _contract_result = $body;
        contract_post_calculate_tdg!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `letter_grade`.
/// Domain-specific. Call: `contract_pre_letter_grade!(slice_expr)`
macro_rules! contract_pre_letter_grade {
    () => {{}};
    ($input:expr) => {{
        let _pv_grad_output = &$input;
        debug_assert!(_pv_grad_output.len() > 0,
            "Contract letter_grade: precondition violated — grad_output.len() > 0");
        debug_assert!(_pv_grad_output.iter().all(|v| v.is_finite()),
            "Contract letter_grade: precondition violated — grad_output.iter().all(|v| v.is_finite())");
    }};
}

/// Postconditions for equation `letter_grade`.
/// Call before return: `contract_post_letter_grade!(result_expr)`
macro_rules! contract_post_letter_grade {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `letter_grade`.
macro_rules! contract_letter_grade {
    ($input:expr, $body:expr) => {{
        contract_pre_letter_grade!($input);
        let _contract_result = $body;
        contract_post_letter_grade!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/tensor-inventory-v1.yaml — DO NOT EDIT
// Contract: tensor-inventory-v1

/// Preconditions for equation `architecture_delta`.
/// Domain-specific. Call: `contract_pre_architecture_delta!(slice_expr)`
macro_rules! contract_pre_architecture_delta {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract architecture_delta: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract architecture_delta: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `parameter_decomposition`.
/// Domain-specific. Call: `contract_pre_parameter_decomposition!(slice_expr)`
macro_rules! contract_pre_parameter_decomposition {
    () => {{}};
    ($input:expr) => {{
        let _pv_indices = &$input;
        debug_assert!(_pv_indices.len() > 0,
            "Contract parameter_decomposition: precondition violated — indices.len() > 0");
    }};
}

/// Preconditions for equation `quantization_bytes`.
/// Domain-specific. Call: `contract_pre_quantization_bytes!(slice_expr)`
macro_rules! contract_pre_quantization_bytes {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract quantization_bytes: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `tensor_count`.
/// Domain-specific. Call: `contract_pre_tensor_count!(slice_expr)`
macro_rules! contract_pre_tensor_count {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract tensor_count: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract tensor_count: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `tied_embeddings`.
/// Domain-specific. Call: `contract_pre_tied_embeddings!(slice_expr)`
macro_rules! contract_pre_tied_embeddings {
    () => {{}};
    ($input:expr) => {{
        let _pv_indices = &$input;
        debug_assert!(_pv_indices.len() > 0,
            "Contract tied_embeddings: precondition violated — indices.len() > 0");
    }};
}

// Auto-generated from contracts/tensor-layout-v1.yaml — DO NOT EDIT
// Contract: tensor-layout-v1

/// Preconditions for equation `identity`.
/// Domain-specific. Call: `contract_pre_identity!(slice_expr)`
macro_rules! contract_pre_identity {
    () => {{}};
    ($input:expr) => {{
        let _pv_a = &$input;
        debug_assert!(_pv_a.len() > 0,
            "Contract identity: precondition violated — a.len() > 0");
    }};
}

/// Preconditions for equation `quant_dispatch_exhaustiveness`.
/// Call at function entry: `contract_pre_quant_dispatch_exhaustiveness!(input_expr)`
macro_rules! contract_pre_quant_dispatch_exhaustiveness {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `transpose_invariant`.
/// Call at function entry: `contract_pre_transpose_invariant!(input_expr)`
macro_rules! contract_pre_transpose_invariant {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `validated_tensor_construction`.
/// Domain-specific. Call: `contract_pre_validated_tensor_construction!(slice_expr)`
macro_rules! contract_pre_validated_tensor_construction {
    () => {{}};
    ($input:expr) => {{
        let _pv_data = &$input;
        debug_assert!(_pv_data.len() > 0,
            "Contract validated_tensor_construction: precondition violated — data.len() > 0");
    }};
}

// Auto-generated from contracts/tensor-names-v1.yaml — DO NOT EDIT
// Contract: tensor-names-v1

/// Preconditions for equation `architecture_normalization`.
/// Domain-specific. Call: `contract_pre_architecture_normalization!(slice_expr)`
macro_rules! contract_pre_architecture_normalization {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract architecture_normalization: precondition violated — input.iter().all(|v| v.is_finite())");
        debug_assert!(_pv_input.len() > 0,
            "Contract architecture_normalization: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `name_resolution`.
/// Domain-specific. Call: `contract_pre_name_resolution!(slice_expr)`
macro_rules! contract_pre_name_resolution {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract name_resolution: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract name_resolution: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

// Auto-generated from contracts/tensor-rc-data-v1.yaml — DO NOT EDIT
// Contract: tensor-rc-data-v1

/// Preconditions for equation `identity`.
/// Domain-specific. Call: `contract_pre_identity!(slice_expr)`
macro_rules! contract_pre_identity {
    () => {{}};
    ($input:expr) => {{
        let _pv_a = &$input;
        debug_assert!(_pv_a.len() > 0,
            "Contract identity: precondition violated — a.len() > 0");
    }};
}

// Auto-generated from contracts/tensor-shape-flow-v1.yaml — DO NOT EDIT
// Contract: tensor-shape-flow-v1

/// Preconditions for equation `gqa_grouping`.
/// Domain-specific. Call: `contract_pre_gqa_grouping!(slice_expr)`
macro_rules! contract_pre_gqa_grouping {
    () => {{}};
    ($input:expr) => {{
        let _pv_q = &$input;
        debug_assert!(_pv_q.len() > 0,
            "Contract gqa_grouping: precondition violated — q.len() > 0");
    }};
}

/// Preconditions for equation `lm_head`.
/// Domain-specific. Call: `contract_pre_lm_head!(slice_expr)`
macro_rules! contract_pre_lm_head {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract lm_head: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract lm_head: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `qkv_projection`.
/// Domain-specific. Call: `contract_pre_qkv_projection!(slice_expr)`
macro_rules! contract_pre_qkv_projection {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract qkv_projection: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract qkv_projection: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `residual`.
/// Domain-specific. Call: `contract_pre_residual!(slice_expr)`
macro_rules! contract_pre_residual {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract residual: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract residual: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `swiglu_shape`.
/// Domain-specific. Call: `contract_pre_swiglu_shape!(slice_expr)`
macro_rules! contract_pre_swiglu_shape {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract swiglu_shape: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract swiglu_shape: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

// Auto-generated from contracts/threading-safety-v1.yaml — DO NOT EDIT
// Contract: threading-safety-v1

/// Preconditions for equation `lock_order_invariant`.
/// Call at function entry: `contract_pre_lock_order_invariant!(input_expr)`
macro_rules! contract_pre_lock_order_invariant {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `lock_order_invariant`.
/// Call before return: `contract_post_lock_order_invariant!(result_expr)`
macro_rules! contract_post_lock_order_invariant {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `lock_order_invariant`.
macro_rules! contract_lock_order_invariant {
    ($input:expr, $body:expr) => {{
        contract_pre_lock_order_invariant!($input);
        let _contract_result = $body;
        contract_post_lock_order_invariant!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `race_freedom`.
/// Call at function entry: `contract_pre_race_freedom!(input_expr)`
macro_rules! contract_pre_race_freedom {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `race_freedom`.
/// Call before return: `contract_post_race_freedom!(result_expr)`
macro_rules! contract_post_race_freedom {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `race_freedom`.
macro_rules! contract_race_freedom {
    ($input:expr, $body:expr) => {{
        contract_pre_race_freedom!($input);
        let _contract_result = $body;
        contract_post_race_freedom!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/tied-embeddings-v1.yaml — DO NOT EDIT
// Contract: tied-embeddings-v1

/// Preconditions for equation `tied_lm_head`.
/// Domain-specific. Call: `contract_pre_tied_lm_head!(slice_expr)`
macro_rules! contract_pre_tied_lm_head {
    () => {{}};
    ($input:expr) => {{
        let _pv_indices = &$input;
        debug_assert!(_pv_indices.len() > 0,
            "Contract tied_lm_head: precondition violated — indices.len() > 0");
    }};
}

// Auto-generated from contracts/tiled-matmul-shader-v1.yaml — DO NOT EDIT
// Contract: tiled-matmul-shader-v1

/// Preconditions for equation `identity`.
/// Domain-specific. Call: `contract_pre_identity!(slice_expr)`
macro_rules! contract_pre_identity {
    () => {{}};
    ($input:expr) => {{
        let _pv_a = &$input;
        debug_assert!(_pv_a.len() > 0,
            "Contract identity: precondition violated — a.len() > 0");
    }};
}

// Auto-generated from contracts/tokenizer-loading-v1.yaml — DO NOT EDIT
// Contract: tokenizer-loading-v1

/// Preconditions for equation `byte_encoder_coverage`.
/// Call at function entry: `contract_pre_byte_encoder_coverage!(input_expr)`
macro_rules! contract_pre_byte_encoder_coverage {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `identity`.
/// Call at function entry: `contract_pre_identity!(input_expr)`
macro_rules! contract_pre_identity {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
        debug_assert!(!_contract_input.is_empty(),
            "Contract identity: precondition violated — !input.is_empty()");
    }};
}

/// Preconditions for equation `roundtrip_encoding`.
/// Call at function entry: `contract_pre_roundtrip_encoding!(input_expr)`
macro_rules! contract_pre_roundtrip_encoding {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

// Auto-generated from contracts/tokenizer-v1.yaml — DO NOT EDIT
// Contract: tokenizer-v1

/// Preconditions for equation `deterministic_encode`.
/// Call at function entry: `contract_pre_deterministic_encode!(input_expr)`
macro_rules! contract_pre_deterministic_encode {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `deterministic_encode`.
/// Call before return: `contract_post_deterministic_encode!(result_expr)`
macro_rules! contract_post_deterministic_encode {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `deterministic_encode`.
macro_rules! contract_deterministic_encode {
    ($input:expr, $body:expr) => {{
        contract_pre_deterministic_encode!($input);
        let _contract_result = $body;
        contract_post_deterministic_encode!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `empty_input`.
/// Call at function entry: `contract_pre_empty_input!(input_expr)`
macro_rules! contract_pre_empty_input {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `empty_input`.
/// Call before return: `contract_post_empty_input!(result_expr)`
macro_rules! contract_post_empty_input {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `empty_input`.
macro_rules! contract_empty_input {
    ($input:expr, $body:expr) => {{
        contract_pre_empty_input!($input);
        let _contract_result = $body;
        contract_post_empty_input!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `roundtrip`.
/// Call at function entry: `contract_pre_roundtrip!(input_expr)`
macro_rules! contract_pre_roundtrip {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `roundtrip`.
/// Call before return: `contract_post_roundtrip!(result_expr)`
macro_rules! contract_post_roundtrip {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `roundtrip`.
macro_rules! contract_roundtrip {
    ($input:expr, $body:expr) => {{
        contract_pre_roundtrip!($input);
        let _contract_result = $body;
        contract_post_roundtrip!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `thread_safety`.
/// Call at function entry: `contract_pre_thread_safety!(input_expr)`
macro_rules! contract_pre_thread_safety {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `thread_safety`.
/// Call before return: `contract_post_thread_safety!(result_expr)`
macro_rules! contract_post_thread_safety {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `thread_safety`.
macro_rules! contract_thread_safety {
    ($input:expr, $body:expr) => {{
        contract_pre_thread_safety!($input);
        let _contract_result = $body;
        contract_post_thread_safety!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `vocab_size_bound`.
/// Call at function entry: `contract_pre_vocab_size_bound!(input_expr)`
macro_rules! contract_pre_vocab_size_bound {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `vocab_size_bound`.
/// Call before return: `contract_post_vocab_size_bound!(result_expr)`
macro_rules! contract_post_vocab_size_bound {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `vocab_size_bound`.
macro_rules! contract_vocab_size_bound {
    ($input:expr, $body:expr) => {{
        contract_pre_vocab_size_bound!($input);
        let _contract_result = $body;
        contract_post_vocab_size_bound!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/trace-integrity-v1.yaml — DO NOT EDIT
// Contract: trace-integrity-v1

/// Preconditions for equation `otel_format`.
/// Call at function entry: `contract_pre_otel_format!(input_expr)`
macro_rules! contract_pre_otel_format {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `trace_capture`.
/// Call at function entry: `contract_pre_trace_capture!(input_expr)`
macro_rules! contract_pre_trace_capture {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `trace_comparison`.
/// Domain-specific. Call: `contract_pre_trace_comparison!(slice_expr)`
macro_rules! contract_pre_trace_comparison {
    () => {{}};
    ($input:expr) => {{
        let _pv_golden = &$input;
        debug_assert!(_pv_golden.len() > 0,
            "Contract trace_comparison: precondition violated — golden.len() > 0");
    }};
}

// Auto-generated from contracts/tracing-observability-v1.yaml — DO NOT EDIT
// Contract: tracing-observability-v1

/// Preconditions for equation `metric_monotonicity`.
/// Call at function entry: `contract_pre_metric_monotonicity!(input_expr)`
macro_rules! contract_pre_metric_monotonicity {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `metric_monotonicity`.
/// Call before return: `contract_post_metric_monotonicity!(result_expr)`
macro_rules! contract_post_metric_monotonicity {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `metric_monotonicity`.
macro_rules! contract_metric_monotonicity {
    ($input:expr, $body:expr) => {{
        contract_pre_metric_monotonicity!($input);
        let _contract_result = $body;
        contract_post_metric_monotonicity!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `renacer_backward_compat`.
/// Call at function entry: `contract_pre_renacer_backward_compat!(input_expr)`
macro_rules! contract_pre_renacer_backward_compat {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `renacer_backward_compat`.
/// Call before return: `contract_post_renacer_backward_compat!(result_expr)`
macro_rules! contract_post_renacer_backward_compat {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `renacer_backward_compat`.
macro_rules! contract_renacer_backward_compat {
    ($input:expr, $body:expr) => {{
        contract_pre_renacer_backward_compat!($input);
        let _contract_result = $body;
        contract_post_renacer_backward_compat!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `span_parentage`.
/// Call at function entry: `contract_pre_span_parentage!(input_expr)`
macro_rules! contract_pre_span_parentage {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `span_parentage`.
/// Call before return: `contract_post_span_parentage!(result_expr)`
macro_rules! contract_post_span_parentage {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `span_parentage`.
macro_rules! contract_span_parentage {
    ($input:expr, $body:expr) => {{
        contract_pre_span_parentage!($input);
        let _contract_result = $body;
        contract_post_span_parentage!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/training-loop-v1.yaml — DO NOT EDIT
// Contract: training-loop-v1

/// Preconditions for equation `ema_loss`.
/// Domain-specific. Call: `contract_pre_ema_loss!(slice_expr)`
macro_rules! contract_pre_ema_loss {
    () => {{}};
    ($input:expr) => {{
        let _pv_predicted = &$input;
        debug_assert!(_pv_predicted.len() > 0,
            "Contract ema_loss: precondition violated — predicted.len() > 0");
    }};
}

/// Preconditions for equation `val_split`.
/// Domain-specific. Call: `contract_pre_val_split!(slice_expr)`
macro_rules! contract_pre_val_split {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract val_split: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract val_split: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `warmup_lr`.
/// Domain-specific. Call: `contract_pre_warmup_lr!(slice_expr)`
macro_rules! contract_pre_warmup_lr {
    () => {{}};
    ($input:expr) => {{
        let _pv_params = &$input;
        debug_assert!(_pv_params.len() > 0,
            "Contract warmup_lr: precondition violated — params.len() > 0");
    }};
}

// Auto-generated from contracts/transpile-pipeline-v1.yaml — DO NOT EDIT
// Contract: transpile-pipeline-v1

/// Preconditions for equation `parse_soundness`.
/// Call at function entry: `contract_pre_parse_soundness!(input_expr)`
macro_rules! contract_pre_parse_soundness {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `transpile_determinism`.
/// Call at function entry: `contract_pre_transpile_determinism!(input_expr)`
macro_rules! contract_pre_transpile_determinism {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `type_preservation`.
/// Call at function entry: `contract_pre_type_preservation!(input_expr)`
macro_rules! contract_pre_type_preservation {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

// Auto-generated from contracts/transpile-soundness-v1.yaml — DO NOT EDIT
// Contract: transpile-soundness-v1

/// Preconditions for equation `ast_to_program`.
/// Call at function entry: `contract_pre_ast_to_program!(input_expr)`
macro_rules! contract_pre_ast_to_program {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `pipeline_composition`.
/// Domain-specific. Call: `contract_pre_pipeline_composition!(slice_expr)`
macro_rules! contract_pre_pipeline_composition {
    () => {{}};
    ($input:expr) => {{
        let _pv_stages = &$input;
        debug_assert!(_pv_stages.len() > 0,
            "Contract pipeline_composition: precondition violated — stages.len() > 0");
    }};
}

/// Preconditions for equation `transpile_determinism`.
/// Call at function entry: `contract_pre_transpile_determinism!(input_expr)`
macro_rules! contract_pre_transpile_determinism {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

// Auto-generated from contracts/transpiler-correctness-v1.yaml — DO NOT EDIT
// Contract: transpiler-correctness-v1

/// Preconditions for equation `semantic_equivalence`.
/// Call at function entry: `contract_pre_semantic_equivalence!(input_expr)`
macro_rules! contract_pre_semantic_equivalence {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `semantic_equivalence`.
/// Call before return: `contract_post_semantic_equivalence!(result_expr)`
macro_rules! contract_post_semantic_equivalence {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `semantic_equivalence`.
macro_rules! contract_semantic_equivalence {
    ($input:expr, $body:expr) => {{
        contract_pre_semantic_equivalence!($input);
        let _contract_result = $body;
        contract_post_semantic_equivalence!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `transpile_determinism`.
/// Call at function entry: `contract_pre_transpile_determinism!(input_expr)`
macro_rules! contract_pre_transpile_determinism {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `transpile_determinism`.
/// Call before return: `contract_post_transpile_determinism!(result_expr)`
macro_rules! contract_post_transpile_determinism {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `transpile_determinism`.
macro_rules! contract_transpile_determinism {
    ($input:expr, $body:expr) => {{
        contract_pre_transpile_determinism!($input);
        let _contract_result = $body;
        contract_post_transpile_determinism!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `type_preservation`.
/// Call at function entry: `contract_pre_type_preservation!(input_expr)`
macro_rules! contract_pre_type_preservation {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `type_preservation`.
/// Call before return: `contract_post_type_preservation!(result_expr)`
macro_rules! contract_post_type_preservation {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `type_preservation`.
macro_rules! contract_type_preservation {
    ($input:expr, $body:expr) => {{
        contract_pre_type_preservation!($input);
        let _contract_result = $body;
        contract_post_type_preservation!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/transpose-kernel-v1.yaml — DO NOT EDIT
// Contract: transpose-kernel-v1

/// Preconditions for equation `transpose`.
/// Domain-specific. Call: `contract_pre_transpose!(slice_expr)`
macro_rules! contract_pre_transpose {
    () => {{}};
    ($input:expr) => {{
        let _pv_a = &$input;
        debug_assert!(_pv_a.len() > 0,
            "Contract transpose: precondition violated — a.len() > 0");
    }};
}

// Auto-generated from contracts/tui-lifecycle-v1.yaml — DO NOT EDIT
// Contract: tui-lifecycle-v1

/// Preconditions for equation `event_dispatch`.
/// Call at function entry: `contract_pre_event_dispatch!(input_expr)`
macro_rules! contract_pre_event_dispatch {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `render_cycle_correctness`.
/// Domain-specific. Call: `contract_pre_render_cycle_correctness!(slice_expr)`
macro_rules! contract_pre_render_cycle_correctness {
    () => {{}};
    ($input:expr) => {{
        let _pv_buffer = &$input;
    }};
}

/// Preconditions for equation `terminal_restore`.
/// Call at function entry: `contract_pre_terminal_restore!(input_expr)`
macro_rules! contract_pre_terminal_restore {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `widget_lifecycle`.
/// Call at function entry: `contract_pre_widget_lifecycle!(input_expr)`
macro_rules! contract_pre_widget_lifecycle {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

// Auto-generated from contracts/tui-panels-v1.yaml — DO NOT EDIT
// Contract: tui-panels-v1

/// Preconditions for equation `adaptive_degradation`.
/// Call at function entry: `contract_pre_adaptive_degradation!(input_expr)`
macro_rules! contract_pre_adaptive_degradation {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `adaptive_degradation`.
/// Call before return: `contract_post_adaptive_degradation!(result_expr)`
macro_rules! contract_post_adaptive_degradation {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `adaptive_degradation`.
macro_rules! contract_adaptive_degradation {
    ($input:expr, $body:expr) => {{
        contract_pre_adaptive_degradation!($input);
        let _contract_result = $body;
        contract_post_adaptive_degradation!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `brick_budget_enforcement`.
/// Domain-specific. Call: `contract_pre_brick_budget_enforcement!(slice_expr)`
macro_rules! contract_pre_brick_budget_enforcement {
    () => {{}};
    ($input:expr) => {{
        let _pv_house = &$input;
    }};
}

/// Postconditions for equation `brick_budget_enforcement`.
/// Call before return: `contract_post_brick_budget_enforcement!(result_expr)`
macro_rules! contract_post_brick_budget_enforcement {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `brick_budget_enforcement`.
macro_rules! contract_brick_budget_enforcement {
    ($input:expr, $body:expr) => {{
        contract_pre_brick_budget_enforcement!($input);
        let _contract_result = $body;
        contract_post_brick_budget_enforcement!(_contract_result);
        _contract_result
    }};
}

/// Postconditions for equation `cost_display_invariants`.
/// Call before return: `contract_post_cost_display_invariants!(result_expr)`
macro_rules! contract_post_cost_display_invariants {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Preconditions for equation `panel_layout_nonoverlap`.
/// Domain-specific. Call: `contract_pre_panel_layout_nonoverlap!(slice_expr)`
macro_rules! contract_pre_panel_layout_nonoverlap {
    () => {{}};
    ($input:expr) => {{
        let _pv_panels = &$input;
        debug_assert!(_pv_panels.len() == 6,
            "Contract panel_layout_nonoverlap: precondition violated — panels.len() == 6");
    }};
}

/// Postconditions for equation `panel_layout_nonoverlap`.
/// Call before return: `contract_post_panel_layout_nonoverlap!(result_expr)`
macro_rules! contract_post_panel_layout_nonoverlap {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `panel_layout_nonoverlap`.
macro_rules! contract_panel_layout_nonoverlap {
    ($input:expr, $body:expr) => {{
        contract_pre_panel_layout_nonoverlap!($input);
        let _contract_result = $body;
        contract_post_panel_layout_nonoverlap!(_contract_result);
        _contract_result
    }};
}

/// Postconditions for equation `sandbox_violation_visibility`.
/// Call before return: `contract_post_sandbox_violation_visibility!(result_expr)`
macro_rules! contract_post_sandbox_violation_visibility {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Postconditions for equation `statusbar_state_display`.
/// Call before return: `contract_post_statusbar_state_display!(result_expr)`
macro_rules! contract_post_statusbar_state_display {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Preconditions for equation `streaming_token_ordering`.
/// Domain-specific. Call: `contract_pre_streaming_token_ordering!(slice_expr)`
macro_rules! contract_pre_streaming_token_ordering {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
    }};
}

/// Postconditions for equation `streaming_token_ordering`.
/// Call before return: `contract_post_streaming_token_ordering!(result_expr)`
macro_rules! contract_post_streaming_token_ordering {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `streaming_token_ordering`.
macro_rules! contract_streaming_token_ordering {
    ($input:expr, $body:expr) => {{
        contract_pre_streaming_token_ordering!($input);
        let _contract_result = $body;
        contract_post_streaming_token_ordering!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `tool_progress_monotonic`.
/// Domain-specific. Call: `contract_pre_tool_progress_monotonic!(slice_expr)`
macro_rules! contract_pre_tool_progress_monotonic {
    () => {{}};
    ($input:expr) => {{
        let _pv_tool_calls = &$input;
        debug_assert!(_pv_tool_calls.len() > 0,
            "Contract tool_progress_monotonic: precondition violated — tool_calls.len() > 0");
    }};
}

/// Postconditions for equation `tool_progress_monotonic`.
/// Call before return: `contract_post_tool_progress_monotonic!(result_expr)`
macro_rules! contract_post_tool_progress_monotonic {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `tool_progress_monotonic`.
macro_rules! contract_tool_progress_monotonic {
    ($input:expr, $body:expr) => {{
        contract_pre_tool_progress_monotonic!($input);
        let _contract_result = $body;
        contract_post_tool_progress_monotonic!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/tui-rendering-v1.yaml — DO NOT EDIT
// Contract: tui-rendering-v1

/// Preconditions for equation `cellbuffer_bounds`.
/// Call at function entry: `contract_pre_cellbuffer_bounds!(input_expr)`
macro_rules! contract_pre_cellbuffer_bounds {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `cellbuffer_bounds`.
/// Call before return: `contract_post_cellbuffer_bounds!(result_expr)`
macro_rules! contract_post_cellbuffer_bounds {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `cellbuffer_bounds`.
macro_rules! contract_cellbuffer_bounds {
    ($input:expr, $body:expr) => {{
        contract_pre_cellbuffer_bounds!($input);
        let _contract_result = $body;
        contract_post_cellbuffer_bounds!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `color_mode_fallback`.
/// Call at function entry: `contract_pre_color_mode_fallback!(input_expr)`
macro_rules! contract_pre_color_mode_fallback {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `color_mode_fallback`.
/// Call before return: `contract_post_color_mode_fallback!(result_expr)`
macro_rules! contract_post_color_mode_fallback {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `color_mode_fallback`.
macro_rules! contract_color_mode_fallback {
    ($input:expr, $body:expr) => {{
        contract_pre_color_mode_fallback!($input);
        let _contract_result = $body;
        contract_post_color_mode_fallback!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `diff_renderer_correctness`.
/// Domain-specific. Call: `contract_pre_diff_renderer_correctness!(slice_expr)`
macro_rules! contract_pre_diff_renderer_correctness {
    () => {{}};
    ($input:expr) => {{
        let _pv_prev = &$input;
    }};
}

/// Postconditions for equation `diff_renderer_correctness`.
/// Call before return: `contract_post_diff_renderer_correctness!(result_expr)`
macro_rules! contract_post_diff_renderer_correctness {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `diff_renderer_correctness`.
macro_rules! contract_diff_renderer_correctness {
    ($input:expr, $body:expr) => {{
        contract_pre_diff_renderer_correctness!($input);
        let _contract_result = $body;
        contract_post_diff_renderer_correctness!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `dirty_tracking`.
/// Domain-specific. Call: `contract_pre_dirty_tracking!(slice_expr)`
macro_rules! contract_pre_dirty_tracking {
    () => {{}};
    ($input:expr) => {{
        let _pv_dirty_mask = &$input;
    }};
}

/// Postconditions for equation `dirty_tracking`.
/// Call before return: `contract_post_dirty_tracking!(result_expr)`
macro_rules! contract_post_dirty_tracking {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `dirty_tracking`.
macro_rules! contract_dirty_tracking {
    ($input:expr, $body:expr) => {{
        contract_pre_dirty_tracking!($input);
        let _contract_result = $body;
        contract_post_dirty_tracking!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `resize_safety`.
/// Call at function entry: `contract_pre_resize_safety!(input_expr)`
macro_rules! contract_pre_resize_safety {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `resize_safety`.
/// Call before return: `contract_post_resize_safety!(result_expr)`
macro_rules! contract_post_resize_safety {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `resize_safety`.
macro_rules! contract_resize_safety {
    ($input:expr, $body:expr) => {{
        contract_pre_resize_safety!($input);
        let _contract_result = $body;
        contract_post_resize_safety!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `unicode_width`.
/// Call at function entry: `contract_pre_unicode_width!(input_expr)`
macro_rules! contract_pre_unicode_width {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `unicode_width`.
/// Call before return: `contract_post_unicode_width!(result_expr)`
macro_rules! contract_post_unicode_width {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `unicode_width`.
macro_rules! contract_unicode_width {
    ($input:expr, $body:expr) => {{
        contract_pre_unicode_width!($input);
        let _contract_result = $body;
        contract_post_unicode_width!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `zero_alloc_render`.
/// Domain-specific. Call: `contract_pre_zero_alloc_render!(slice_expr)`
macro_rules! contract_pre_zero_alloc_render {
    () => {{}};
    ($input:expr) => {{
        let _pv_buffer = &$input;
    }};
}

/// Postconditions for equation `zero_alloc_render`.
/// Call before return: `contract_post_zero_alloc_render!(result_expr)`
macro_rules! contract_post_zero_alloc_render {
    ($result:expr) => {{
        let _contract_result = &$result;
        debug_assert!(allocator_count_during_render == 0, "Contract zero_alloc_render: postcondition violated — allocator_count_during_render == 0");
    }};
}

/// Combined pre+post contract for equation `zero_alloc_render`.
macro_rules! contract_zero_alloc_render {
    ($input:expr, $body:expr) => {{
        contract_pre_zero_alloc_render!($input);
        let _contract_result = $body;
        contract_post_zero_alloc_render!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/type-preservation-v1.yaml — DO NOT EDIT
// Contract: type-preservation-v1

/// Preconditions for equation `container_preservation`.
/// Domain-specific. Call: `contract_pre_container_preservation!(slice_expr)`
macro_rules! contract_pre_container_preservation {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract container_preservation: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `copy_semantics`.
/// Domain-specific. Call: `contract_pre_copy_semantics!(slice_expr)`
macro_rules! contract_pre_copy_semantics {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract copy_semantics: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `numeric_semantics`.
/// Domain-specific. Call: `contract_pre_numeric_semantics!(slice_expr)`
macro_rules! contract_pre_numeric_semantics {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract numeric_semantics: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `type_inference`.
/// Domain-specific. Call: `contract_pre_type_inference!(slice_expr)`
macro_rules! contract_pre_type_inference {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract type_inference: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `type_map`.
/// Domain-specific. Call: `contract_pre_type_map!(slice_expr)`
macro_rules! contract_pre_type_map {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract type_map: precondition violated — input.len() > 0");
    }};
}

// Auto-generated from contracts/validated-tensor-v1.yaml — DO NOT EDIT
// Contract: validated-tensor-v1

/// Preconditions for equation `density_gate`.
/// Domain-specific. Call: `contract_pre_density_gate!(slice_expr)`
macro_rules! contract_pre_density_gate {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract density_gate: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract density_gate: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

/// Preconditions for equation `l2_norm_nondegeneracy`.
/// Domain-specific. Call: `contract_pre_l2_norm_nondegeneracy!(slice_expr)`
macro_rules! contract_pre_l2_norm_nondegeneracy {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract l2_norm_nondegeneracy: precondition violated — input.iter().all(|v| v.is_finite())");
        debug_assert!(_pv_input.len() > 0,
            "Contract l2_norm_nondegeneracy: precondition violated — input.len() > 0");
    }};
}

/// Preconditions for equation `nan_inf_rejection`.
/// Domain-specific. Call: `contract_pre_nan_inf_rejection!(slice_expr)`
macro_rules! contract_pre_nan_inf_rejection {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(_pv_input.len() > 0,
            "Contract nan_inf_rejection: precondition violated — input.len() > 0");
        debug_assert!(_pv_input.iter().all(|v| v.is_finite()),
            "Contract nan_inf_rejection: precondition violated — input.iter().all(|v| v.is_finite())");
    }};
}

// Auto-generated from contracts/verification-engine-v1.yaml — DO NOT EDIT
// Contract: verification-engine-v1

/// Preconditions for equation `generator_coverage`.
/// Domain-specific. Call: `contract_pre_generator_coverage!(slice_expr)`
macro_rules! contract_pre_generator_coverage {
    () => {{}};
    ($input:expr) => {{
        let _pv_strategy = &$input;
    }};
}

/// Preconditions for equation `mutation_soundness`.
/// Call at function entry: `contract_pre_mutation_soundness!(input_expr)`
macro_rules! contract_pre_mutation_soundness {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `oracle_verdict`.
/// Call at function entry: `contract_pre_oracle_verdict!(input_expr)`
macro_rules! contract_pre_oracle_verdict {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

// Auto-generated from contracts/visualization-render-v1.yaml — DO NOT EDIT
// Contract: visualization-render-v1

/// Preconditions for equation `layout_treemap`.
/// Domain-specific. Call: `contract_pre_layout_treemap!(slice_expr)`
macro_rules! contract_pre_layout_treemap {
    () => {{}};
    ($input:expr) => {{
        let _pv_nodes = &$input;
        debug_assert!(_pv_nodes.len() > 0,
            "Contract layout_treemap: precondition violated — nodes.len() > 0");
    }};
}

/// Preconditions for equation `primitive_bounds`.
/// Call at function entry: `contract_pre_primitive_bounds!(input_expr)`
macro_rules! contract_pre_primitive_bounds {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Preconditions for equation `render_output`.
/// Call at function entry: `contract_pre_render_output!(input_expr)`
macro_rules! contract_pre_render_output {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

// Auto-generated from contracts/wgpu-production-training-v1.yaml — DO NOT EDIT
// Contract: wgpu-production-training-v1

// Auto-generated from contracts/wgpu-resident-weights-v1.yaml — DO NOT EDIT
// Contract: wgpu-resident-weights-v1

/// Preconditions for equation `identity`.
/// Domain-specific. Call: `contract_pre_identity!(slice_expr)`
macro_rules! contract_pre_identity {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
    }};
}

// Auto-generated from contracts/work-dbc-v1.yaml — DO NOT EDIT
// Contract: work-dbc-v1

/// Preconditions for equation `checkpoint_verification`.
/// Domain-specific. Call: `contract_pre_checkpoint_verification!(slice_expr)`
macro_rules! contract_pre_checkpoint_verification {
    () => {{}};
    ($input:expr) => {{
        let _pv_contract = &$input;
    }};
}

/// Postconditions for equation `checkpoint_verification`.
/// Call before return: `contract_post_checkpoint_verification!(result_expr)`
macro_rules! contract_post_checkpoint_verification {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `checkpoint_verification`.
macro_rules! contract_checkpoint_verification {
    ($input:expr, $body:expr) => {{
        contract_pre_checkpoint_verification!($input);
        let _contract_result = $body;
        contract_post_checkpoint_verification!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `contract_profile`.
/// Domain-specific. Call: `contract_pre_contract_profile!(slice_expr)`
macro_rules! contract_pre_contract_profile {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
    }};
}

/// Postconditions for equation `contract_profile`.
/// Call before return: `contract_post_contract_profile!(result_expr)`
macro_rules! contract_post_contract_profile {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `contract_profile`.
macro_rules! contract_contract_profile {
    ($input:expr, $body:expr) => {{
        contract_pre_contract_profile!($input);
        let _contract_result = $body;
        contract_post_contract_profile!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `falsifiable_claim`.
/// Domain-specific. Call: `contract_pre_falsifiable_claim!(slice_expr)`
macro_rules! contract_pre_falsifiable_claim {
    () => {{}};
    ($input:expr) => {{
        let _pv_claim = &$input;
    }};
}

/// Postconditions for equation `falsifiable_claim`.
/// Call before return: `contract_post_falsifiable_claim!(result_expr)`
macro_rules! contract_post_falsifiable_claim {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `falsifiable_claim`.
macro_rules! contract_falsifiable_claim {
    ($input:expr, $body:expr) => {{
        contract_pre_falsifiable_claim!($input);
        let _contract_result = $body;
        contract_post_falsifiable_claim!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `meyer_triad`.
/// Domain-specific. Call: `contract_pre_meyer_triad!(slice_expr)`
macro_rules! contract_pre_meyer_triad {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
    }};
}

/// Postconditions for equation `meyer_triad`.
/// Call before return: `contract_post_meyer_triad!(result_expr)`
macro_rules! contract_post_meyer_triad {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `meyer_triad`.
macro_rules! contract_meyer_triad {
    ($input:expr, $body:expr) => {{
        contract_pre_meyer_triad!($input);
        let _contract_result = $body;
        contract_post_meyer_triad!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `override_accountability`.
/// Call at function entry: `contract_pre_override_accountability!(input_expr)`
macro_rules! contract_pre_override_accountability {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `override_accountability`.
/// Call before return: `contract_post_override_accountability!(result_expr)`
macro_rules! contract_post_override_accountability {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `override_accountability`.
macro_rules! contract_override_accountability {
    ($input:expr, $body:expr) => {{
        contract_pre_override_accountability!($input);
        let _contract_result = $body;
        contract_post_override_accountability!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `rescue_protocol`.
/// Call at function entry: `contract_pre_rescue_protocol!(input_expr)`
macro_rules! contract_pre_rescue_protocol {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `rescue_protocol`.
/// Call before return: `contract_post_rescue_protocol!(result_expr)`
macro_rules! contract_post_rescue_protocol {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `rescue_protocol`.
macro_rules! contract_rescue_protocol {
    ($input:expr, $body:expr) => {{
        contract_pre_rescue_protocol!($input);
        let _contract_result = $body;
        contract_post_rescue_protocol!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `work_lifecycle`.
/// Domain-specific. Call: `contract_pre_work_lifecycle!(slice_expr)`
macro_rules! contract_pre_work_lifecycle {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
    }};
}

/// Postconditions for equation `work_lifecycle`.
/// Call before return: `contract_post_work_lifecycle!(result_expr)`
macro_rules! contract_post_work_lifecycle {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `work_lifecycle`.
macro_rules! contract_work_lifecycle {
    ($input:expr, $body:expr) => {{
        contract_pre_work_lifecycle!($input);
        let _contract_result = $body;
        contract_post_work_lifecycle!(_contract_result);
        _contract_result
    }};
}

// Total: 646 preconditions, 15 postconditions from 278 contracts
