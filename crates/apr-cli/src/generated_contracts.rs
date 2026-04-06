// Auto-generated contract assertions from YAML — DO NOT EDIT.
// Zero cost in release builds (debug_assert!).
// Regenerate: pv codegen contracts/ -o src/generated_contracts.rs
// Include:   #[macro_use] #[allow(unused_macros)] mod generated_contracts;

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

/// Invariants for equation `architecture_config_invariants`.
/// Check after computation: `contract_inv_architecture_config_invariants!(result_expr)`
macro_rules! contract_inv_architecture_config_invariants {
    () => {{}};
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

/// Invariants for equation `attention_tensor_shapes`.
/// Check after computation: `contract_inv_attention_tensor_shapes!(result_expr)`
macro_rules! contract_inv_attention_tensor_shapes {
    () => {{}};
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

/// Invariants for equation `embedding_tensor_shapes`.
/// Check after computation: `contract_inv_embedding_tensor_shapes!(result_expr)`
macro_rules! contract_inv_embedding_tensor_shapes {
    () => {{}};
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

/// Invariants for equation `ffn_tensor_shapes`.
/// Check after computation: `contract_inv_ffn_tensor_shapes!(result_expr)`
macro_rules! contract_inv_ffn_tensor_shapes {
    () => {{}};
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

/// Invariants for equation `normalization_tensor_shapes`.
/// Check after computation: `contract_inv_normalization_tensor_shapes!(result_expr)`
macro_rules! contract_inv_normalization_tensor_shapes {
    () => {{}};
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

/// Invariants for equation `rope_position_encoding`.
/// Check after computation: `contract_inv_rope_position_encoding!(result_expr)`
macro_rules! contract_inv_rope_position_encoding {
    () => {{}};
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

/// Invariants for equation `total_tensor_count`.
/// Check after computation: `contract_inv_total_tensor_count!(result_expr)`
macro_rules! contract_inv_total_tensor_count {
    () => {{}};
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

/// Invariants for equation `chat_template_application`.
/// Check after computation: `contract_inv_chat_template_application!(result_expr)`
macro_rules! contract_inv_chat_template_application {
    () => {{}};
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
        debug_assert!(
            _pv_new_tokens.len() > 0,
            "Contract kv_cache_management: precondition violated — new_tokens.len() > 0"
        );
    }};
}

/// Postconditions for equation `kv_cache_management`.
/// Call before return: `contract_post_kv_cache_management!(result_expr)`
macro_rules! contract_post_kv_cache_management {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Invariants for equation `kv_cache_management`.
/// Check after computation: `contract_inv_kv_cache_management!(result_expr)`
macro_rules! contract_inv_kv_cache_management {
    () => {{}};
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

/// Invariants for equation `session_persistence`.
/// Check after computation: `contract_inv_session_persistence!(result_expr)`
macro_rules! contract_inv_session_persistence {
    () => {{}};
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

/// Invariants for equation `session_state_machine`.
/// Check after computation: `contract_inv_session_state_machine!(result_expr)`
macro_rules! contract_inv_session_state_machine {
    () => {{}};
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

// Auto-generated from contracts/apr-cli-operations-v1.yaml — DO NOT EDIT
// Contract: apr-cli-operations-v1

/// Preconditions for equation `concurrent_model_access`.
/// Domain-specific. Call: `contract_pre_concurrent_model_access!(slice_expr)`
macro_rules! contract_pre_concurrent_model_access {
    () => {{}};
    ($input:expr) => {{
        let _pv_requests = &$input;
        debug_assert!(
            _pv_requests.len() > 0,
            "Contract concurrent_model_access: precondition violated — requests.len() > 0"
        );
    }};
}

/// Postconditions for equation `concurrent_model_access`.
/// Call before return: `contract_post_concurrent_model_access!(result_expr)`
macro_rules! contract_post_concurrent_model_access {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Invariants for equation `concurrent_model_access`.
/// Check after computation: `contract_inv_concurrent_model_access!(result_expr)`
macro_rules! contract_inv_concurrent_model_access {
    () => {{}};
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

/// Invariants for equation `inference_determinism`.
/// Check after computation: `contract_inv_inference_determinism!(result_expr)`
macro_rules! contract_inv_inference_determinism {
    () => {{}};
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

/// Invariants for equation `progress_reporting`.
/// Check after computation: `contract_inv_progress_reporting!(result_expr)`
macro_rules! contract_inv_progress_reporting {
    () => {{}};
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

/// Invariants for equation `resource_cleanup`.
/// Check after computation: `contract_inv_resource_cleanup!(result_expr)`
macro_rules! contract_inv_resource_cleanup {
    () => {{}};
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

/// Invariants for equation `side_effect_classification`.
/// Check after computation: `contract_inv_side_effect_classification!(result_expr)`
macro_rules! contract_inv_side_effect_classification {
    () => {{}};
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

/// Invariants for equation `tokenizer_consistency`.
/// Check after computation: `contract_inv_tokenizer_consistency!(result_expr)`
macro_rules! contract_inv_tokenizer_consistency {
    () => {{}};
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

// Auto-generated from contracts/apr-cli-safety-v1.yaml — DO NOT EDIT
// Contract: apr-cli-safety-v1

/// Preconditions for equation `encrypt_guard`.
/// Domain-specific. Call: `contract_pre_encrypt_guard!(slice_expr)`
macro_rules! contract_pre_encrypt_guard {
    () => {{}};
    ($input:expr) => {{
        let _pv_input_path = &$input;
    }};
}

/// Invariants for equation `encrypt_guard`.
/// Check after computation: `contract_inv_encrypt_guard!(result_expr)`
macro_rules! contract_inv_encrypt_guard {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Preconditions for equation `offline_guard`.
/// Domain-specific. Call: `contract_pre_offline_guard!(slice_expr)`
macro_rules! contract_pre_offline_guard {
    () => {{}};
    ($input:expr) => {{
        let _pv_source = &$input;
        debug_assert!(
            _pv_source.len() > 0,
            "Contract offline_guard: precondition violated — source.len() > 0"
        );
    }};
}

/// Invariants for equation `offline_guard`.
/// Check after computation: `contract_inv_offline_guard!(result_expr)`
macro_rules! contract_inv_offline_guard {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Preconditions for equation `validate_exit_code`.
/// Domain-specific. Call: `contract_pre_validate_exit_code!(slice_expr)`
macro_rules! contract_pre_validate_exit_code {
    () => {{}};
    ($input:expr) => {{
        let _pv_path = &$input;
    }};
}

/// Invariants for equation `validate_exit_code`.
/// Check after computation: `contract_inv_validate_exit_code!(result_expr)`
macro_rules! contract_inv_validate_exit_code {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
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

/// Invariants for equation `command_parse_determinism`.
/// Check after computation: `contract_inv_command_parse_determinism!(result_expr)`
macro_rules! contract_inv_command_parse_determinism {
    () => {{}};
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

/// Invariants for equation `contract_gate_enforcement`.
/// Check after computation: `contract_inv_contract_gate_enforcement!(result_expr)`
macro_rules! contract_inv_contract_gate_enforcement {
    () => {{}};
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

/// Invariants for equation `model_path_resolution`.
/// Check after computation: `contract_inv_model_path_resolution!(result_expr)`
macro_rules! contract_inv_model_path_resolution {
    () => {{}};
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

/// Invariants for equation `pipe_stdin_support`.
/// Check after computation: `contract_inv_pipe_stdin_support!(result_expr)`
macro_rules! contract_inv_pipe_stdin_support {
    () => {{}};
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

/// Invariants for equation `tokenizer_training_correctness`.
/// Check after computation: `contract_inv_tokenizer_training_correctness!(result_expr)`
macro_rules! contract_inv_tokenizer_training_correctness {
    () => {{}};
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

/// Invariants for equation `training_plan_apply_semantics`.
/// Check after computation: `contract_inv_training_plan_apply_semantics!(result_expr)`
macro_rules! contract_inv_training_plan_apply_semantics {
    () => {{}};
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

// Auto-generated from contracts/apr-data-pipeline-v1.yaml — DO NOT EDIT
// Contract: apr-data-pipeline-v1

/// Preconditions for equation `data_split_determinism`.
/// Domain-specific. Call: `contract_pre_data_split_determinism!(slice_expr)`
macro_rules! contract_pre_data_split_determinism {
    () => {{}};
    ($input:expr) => {{
        let _pv_ratios = &$input;
        debug_assert!(
            _pv_ratios.sum() == 1.0,
            "Contract data_split_determinism: precondition violated — ratios.sum() == 1.0"
        );
    }};
}

/// Postconditions for equation `data_split_determinism`.
/// Call before return: `contract_post_data_split_determinism!(result_expr)`
macro_rules! contract_post_data_split_determinism {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Invariants for equation `data_split_determinism`.
/// Check after computation: `contract_inv_data_split_determinism!(result_expr)`
macro_rules! contract_inv_data_split_determinism {
    () => {{}};
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

/// Invariants for equation `data_validation`.
/// Check after computation: `contract_inv_data_validation!(result_expr)`
macro_rules! contract_inv_data_validation {
    () => {{}};
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

/// Invariants for equation `preprocessing_idempotency`.
/// Check after computation: `contract_inv_preprocessing_idempotency!(result_expr)`
macro_rules! contract_inv_preprocessing_idempotency {
    () => {{}};
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
        debug_assert!(
            _pv_dataset.len() > 0,
            "Contract streaming_data_loader: precondition violated — dataset.len() > 0"
        );
    }};
}

/// Postconditions for equation `streaming_data_loader`.
/// Call before return: `contract_post_streaming_data_loader!(result_expr)`
macro_rules! contract_post_streaming_data_loader {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Invariants for equation `streaming_data_loader`.
/// Check after computation: `contract_inv_streaming_data_loader!(result_expr)`
macro_rules! contract_inv_streaming_data_loader {
    () => {{}};
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

/// Invariants for equation `dtype_coercion_safety`.
/// Check after computation: `contract_inv_dtype_coercion_safety!(result_expr)`
macro_rules! contract_inv_dtype_coercion_safety {
    () => {{}};
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

/// Invariants for equation `header_integrity`.
/// Check after computation: `contract_inv_header_integrity!(result_expr)`
macro_rules! contract_inv_header_integrity {
    () => {{}};
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

/// Invariants for equation `magic_byte_validation`.
/// Check after computation: `contract_inv_magic_byte_validation!(result_expr)`
macro_rules! contract_inv_magic_byte_validation {
    () => {{}};
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

/// Invariants for equation `provenance_enforcement`.
/// Check after computation: `contract_inv_provenance_enforcement!(result_expr)`
macro_rules! contract_inv_provenance_enforcement {
    () => {{}};
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

/// Invariants for equation `strict_import_validation`.
/// Check after computation: `contract_inv_strict_import_validation!(result_expr)`
macro_rules! contract_inv_strict_import_validation {
    () => {{}};
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

/// Invariants for equation `truncation_detection`.
/// Check after computation: `contract_inv_truncation_detection!(result_expr)`
macro_rules! contract_inv_truncation_detection {
    () => {{}};
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

// Auto-generated from contracts/apr-gpu-diagnostics-v1.yaml — DO NOT EDIT
// Contract: apr-gpu-diagnostics-v1

/// Preconditions for equation `cbtop_measurement_accuracy`.
/// Call at function entry: `contract_pre_cbtop_measurement_accuracy!(input_expr)`
macro_rules! contract_pre_cbtop_measurement_accuracy {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `cbtop_measurement_accuracy`.
/// Call before return: `contract_post_cbtop_measurement_accuracy!(result_expr)`
macro_rules! contract_post_cbtop_measurement_accuracy {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Invariants for equation `cbtop_measurement_accuracy`.
/// Check after computation: `contract_inv_cbtop_measurement_accuracy!(result_expr)`
macro_rules! contract_inv_cbtop_measurement_accuracy {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `cbtop_measurement_accuracy`.
macro_rules! contract_cbtop_measurement_accuracy {
    ($input:expr, $body:expr) => {{
        contract_pre_cbtop_measurement_accuracy!($input);
        let _contract_result = $body;
        contract_post_cbtop_measurement_accuracy!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `cbtop_monitoring`.
/// Call at function entry: `contract_pre_cbtop_monitoring!(input_expr)`
macro_rules! contract_pre_cbtop_monitoring {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `cbtop_monitoring`.
/// Call before return: `contract_post_cbtop_monitoring!(result_expr)`
macro_rules! contract_post_cbtop_monitoring {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Invariants for equation `cbtop_monitoring`.
/// Check after computation: `contract_inv_cbtop_monitoring!(result_expr)`
macro_rules! contract_inv_cbtop_monitoring {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `cbtop_monitoring`.
macro_rules! contract_cbtop_monitoring {
    ($input:expr, $body:expr) => {{
        contract_pre_cbtop_monitoring!($input);
        let _contract_result = $body;
        contract_post_cbtop_monitoring!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `ptx_code_generation`.
/// Domain-specific. Call: `contract_pre_ptx_code_generation!(slice_expr)`
macro_rules! contract_pre_ptx_code_generation {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
    }};
}

/// Postconditions for equation `ptx_code_generation`.
/// Call before return: `contract_post_ptx_code_generation!(result_expr)`
macro_rules! contract_post_ptx_code_generation {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Invariants for equation `ptx_code_generation`.
/// Check after computation: `contract_inv_ptx_code_generation!(result_expr)`
macro_rules! contract_inv_ptx_code_generation {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `ptx_code_generation`.
macro_rules! contract_ptx_code_generation {
    ($input:expr, $body:expr) => {{
        contract_pre_ptx_code_generation!($input);
        let _contract_result = $body;
        contract_post_ptx_code_generation!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `ptx_kernel_mapping`.
/// Domain-specific. Call: `contract_pre_ptx_kernel_mapping!(slice_expr)`
macro_rules! contract_pre_ptx_kernel_mapping {
    () => {{}};
    ($input:expr) => {{
        let _pv_model = &$input;
    }};
}

/// Postconditions for equation `ptx_kernel_mapping`.
/// Call before return: `contract_post_ptx_kernel_mapping!(result_expr)`
macro_rules! contract_post_ptx_kernel_mapping {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Invariants for equation `ptx_kernel_mapping`.
/// Check after computation: `contract_inv_ptx_kernel_mapping!(result_expr)`
macro_rules! contract_inv_ptx_kernel_mapping {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `ptx_kernel_mapping`.
macro_rules! contract_ptx_kernel_mapping {
    ($input:expr, $body:expr) => {{
        contract_pre_ptx_kernel_mapping!($input);
        let _contract_result = $body;
        contract_post_ptx_kernel_mapping!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/apr-model-diagnostics-v1.yaml — DO NOT EDIT
// Contract: apr-model-diagnostics-v1

/// Preconditions for equation `diagnose_fault_isolation`.
/// Call at function entry: `contract_pre_diagnose_fault_isolation!(input_expr)`
macro_rules! contract_pre_diagnose_fault_isolation {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `diagnose_fault_isolation`.
/// Call before return: `contract_post_diagnose_fault_isolation!(result_expr)`
macro_rules! contract_post_diagnose_fault_isolation {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Invariants for equation `diagnose_fault_isolation`.
/// Check after computation: `contract_inv_diagnose_fault_isolation!(result_expr)`
macro_rules! contract_inv_diagnose_fault_isolation {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `diagnose_fault_isolation`.
macro_rules! contract_diagnose_fault_isolation {
    ($input:expr, $body:expr) => {{
        contract_pre_diagnose_fault_isolation!($input);
        let _contract_result = $body;
        contract_post_diagnose_fault_isolation!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `hex_display_fidelity`.
/// Domain-specific. Call: `contract_pre_hex_display_fidelity!(slice_expr)`
macro_rules! contract_pre_hex_display_fidelity {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
    }};
}

/// Postconditions for equation `hex_display_fidelity`.
/// Call before return: `contract_post_hex_display_fidelity!(result_expr)`
macro_rules! contract_post_hex_display_fidelity {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Invariants for equation `hex_display_fidelity`.
/// Check after computation: `contract_inv_hex_display_fidelity!(result_expr)`
macro_rules! contract_inv_hex_display_fidelity {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `hex_display_fidelity`.
macro_rules! contract_hex_display_fidelity {
    ($input:expr, $body:expr) => {{
        contract_pre_hex_display_fidelity!($input);
        let _contract_result = $body;
        contract_post_hex_display_fidelity!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `oracle_compatibility_matrix`.
/// Call at function entry: `contract_pre_oracle_compatibility_matrix!(input_expr)`
macro_rules! contract_pre_oracle_compatibility_matrix {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `oracle_compatibility_matrix`.
/// Call before return: `contract_post_oracle_compatibility_matrix!(result_expr)`
macro_rules! contract_post_oracle_compatibility_matrix {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Invariants for equation `oracle_compatibility_matrix`.
/// Check after computation: `contract_inv_oracle_compatibility_matrix!(result_expr)`
macro_rules! contract_inv_oracle_compatibility_matrix {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `oracle_compatibility_matrix`.
macro_rules! contract_oracle_compatibility_matrix {
    ($input:expr, $body:expr) => {{
        contract_pre_oracle_compatibility_matrix!($input);
        let _contract_result = $body;
        contract_post_oracle_compatibility_matrix!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `oracle_family_detection`.
/// Call at function entry: `contract_pre_oracle_family_detection!(input_expr)`
macro_rules! contract_pre_oracle_family_detection {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `oracle_family_detection`.
/// Call before return: `contract_post_oracle_family_detection!(result_expr)`
macro_rules! contract_post_oracle_family_detection {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Invariants for equation `oracle_family_detection`.
/// Check after computation: `contract_inv_oracle_family_detection!(result_expr)`
macro_rules! contract_inv_oracle_family_detection {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `oracle_family_detection`.
macro_rules! contract_oracle_family_detection {
    ($input:expr, $body:expr) => {{
        contract_pre_oracle_family_detection!($input);
        let _contract_result = $body;
        contract_post_oracle_family_detection!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `rosetta_fingerprint_determinism`.
/// Call at function entry: `contract_pre_rosetta_fingerprint_determinism!(input_expr)`
macro_rules! contract_pre_rosetta_fingerprint_determinism {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `rosetta_fingerprint_determinism`.
/// Call before return: `contract_post_rosetta_fingerprint_determinism!(result_expr)`
macro_rules! contract_post_rosetta_fingerprint_determinism {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Invariants for equation `rosetta_fingerprint_determinism`.
/// Check after computation: `contract_inv_rosetta_fingerprint_determinism!(result_expr)`
macro_rules! contract_inv_rosetta_fingerprint_determinism {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `rosetta_fingerprint_determinism`.
macro_rules! contract_rosetta_fingerprint_determinism {
    ($input:expr, $body:expr) => {{
        contract_pre_rosetta_fingerprint_determinism!($input);
        let _contract_result = $body;
        contract_post_rosetta_fingerprint_determinism!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/apr-model-graph-v1.yaml — DO NOT EDIT
// Contract: apr-model-graph-v1

/// Preconditions for equation `attention_mechanism`.
/// Call at function entry: `contract_pre_attention_mechanism!(input_expr)`
macro_rules! contract_pre_attention_mechanism {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `attention_mechanism`.
/// Call before return: `contract_post_attention_mechanism!(result_expr)`
macro_rules! contract_post_attention_mechanism {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Invariants for equation `attention_mechanism`.
/// Check after computation: `contract_inv_attention_mechanism!(result_expr)`
macro_rules! contract_inv_attention_mechanism {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `attention_mechanism`.
macro_rules! contract_attention_mechanism {
    ($input:expr, $body:expr) => {{
        contract_pre_attention_mechanism!($input);
        let _contract_result = $body;
        contract_post_attention_mechanism!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `ffn_computation`.
/// Domain-specific. Call: `contract_pre_ffn_computation!(slice_expr)`
macro_rules! contract_pre_ffn_computation {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
    }};
}

/// Postconditions for equation `ffn_computation`.
/// Call before return: `contract_post_ffn_computation!(result_expr)`
macro_rules! contract_post_ffn_computation {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Invariants for equation `ffn_computation`.
/// Check after computation: `contract_inv_ffn_computation!(result_expr)`
macro_rules! contract_inv_ffn_computation {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `ffn_computation`.
macro_rules! contract_ffn_computation {
    ($input:expr, $body:expr) => {{
        contract_pre_ffn_computation!($input);
        let _contract_result = $body;
        contract_post_ffn_computation!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `forward_pass_completeness`.
/// Domain-specific. Call: `contract_pre_forward_pass_completeness!(slice_expr)`
macro_rules! contract_pre_forward_pass_completeness {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
    }};
}

/// Postconditions for equation `forward_pass_completeness`.
/// Call before return: `contract_post_forward_pass_completeness!(result_expr)`
macro_rules! contract_post_forward_pass_completeness {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Invariants for equation `forward_pass_completeness`.
/// Check after computation: `contract_inv_forward_pass_completeness!(result_expr)`
macro_rules! contract_inv_forward_pass_completeness {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `forward_pass_completeness`.
macro_rules! contract_forward_pass_completeness {
    ($input:expr, $body:expr) => {{
        contract_pre_forward_pass_completeness!($input);
        let _contract_result = $body;
        contract_post_forward_pass_completeness!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `kv_cache_management`.
/// Call at function entry: `contract_pre_kv_cache_management!(input_expr)`
macro_rules! contract_pre_kv_cache_management {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `kv_cache_management`.
/// Call before return: `contract_post_kv_cache_management!(result_expr)`
macro_rules! contract_post_kv_cache_management {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Invariants for equation `kv_cache_management`.
/// Check after computation: `contract_inv_kv_cache_management!(result_expr)`
macro_rules! contract_inv_kv_cache_management {
    () => {{}};
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

/// Preconditions for equation `quantization_precision`.
/// Call at function entry: `contract_pre_quantization_precision!(input_expr)`
macro_rules! contract_pre_quantization_precision {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `quantization_precision`.
/// Call before return: `contract_post_quantization_precision!(result_expr)`
macro_rules! contract_post_quantization_precision {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Invariants for equation `quantization_precision`.
/// Check after computation: `contract_inv_quantization_precision!(result_expr)`
macro_rules! contract_inv_quantization_precision {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `quantization_precision`.
macro_rules! contract_quantization_precision {
    ($input:expr, $body:expr) => {{
        contract_pre_quantization_precision!($input);
        let _contract_result = $body;
        contract_post_quantization_precision!(_contract_result);
        _contract_result
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

/// Postconditions for equation `residual_stream`.
/// Call before return: `contract_post_residual_stream!(result_expr)`
macro_rules! contract_post_residual_stream {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Invariants for equation `residual_stream`.
/// Check after computation: `contract_inv_residual_stream!(result_expr)`
macro_rules! contract_inv_residual_stream {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `residual_stream`.
macro_rules! contract_residual_stream {
    ($input:expr, $body:expr) => {{
        contract_pre_residual_stream!($input);
        let _contract_result = $body;
        contract_post_residual_stream!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `tensor_name_resolution`.
/// Domain-specific. Call: `contract_pre_tensor_name_resolution!(slice_expr)`
macro_rules! contract_pre_tensor_name_resolution {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
    }};
}

/// Postconditions for equation `tensor_name_resolution`.
/// Call before return: `contract_post_tensor_name_resolution!(result_expr)`
macro_rules! contract_post_tensor_name_resolution {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Invariants for equation `tensor_name_resolution`.
/// Check after computation: `contract_inv_tensor_name_resolution!(result_expr)`
macro_rules! contract_inv_tensor_name_resolution {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `tensor_name_resolution`.
macro_rules! contract_tensor_name_resolution {
    ($input:expr, $body:expr) => {{
        contract_pre_tensor_name_resolution!($input);
        let _contract_result = $body;
        contract_post_tensor_name_resolution!(_contract_result);
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

/// Invariants for equation `export_roundtrip`.
/// Check after computation: `contract_inv_export_roundtrip!(result_expr)`
macro_rules! contract_inv_export_roundtrip {
    () => {{}};
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

/// Invariants for equation `import_format_detection`.
/// Check after computation: `contract_inv_import_format_detection!(result_expr)`
macro_rules! contract_inv_import_format_detection {
    () => {{}};
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
        debug_assert!(
            _pv_models.len() >= 2,
            "Contract merge_weight_conservation: precondition violated — models.len() >= 2"
        );
    }};
}

/// Postconditions for equation `merge_weight_conservation`.
/// Call before return: `contract_post_merge_weight_conservation!(result_expr)`
macro_rules! contract_post_merge_weight_conservation {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Invariants for equation `merge_weight_conservation`.
/// Check after computation: `contract_inv_merge_weight_conservation!(result_expr)`
macro_rules! contract_inv_merge_weight_conservation {
    () => {{}};
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

/// Invariants for equation `pull_cache_integrity`.
/// Check after computation: `contract_inv_pull_cache_integrity!(result_expr)`
macro_rules! contract_inv_pull_cache_integrity {
    () => {{}};
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

/// Invariants for equation `quantize_precision_bound`.
/// Check after computation: `contract_inv_quantize_precision_bound!(result_expr)`
macro_rules! contract_inv_quantize_precision_bound {
    () => {{}};
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

// Auto-generated from contracts/apr-model-optimization-v1.yaml — DO NOT EDIT
// Contract: apr-model-optimization-v1

/// Preconditions for equation `distill_knowledge_transfer`.
/// Domain-specific. Call: `contract_pre_distill_knowledge_transfer!(slice_expr)`
macro_rules! contract_pre_distill_knowledge_transfer {
    () => {{}};
    ($input:expr) => {{
        let _pv_teacher = &$input;
        debug_assert!(_pv_teacher.is_frozen() == true,
            "Contract distill_knowledge_transfer: precondition violated — teacher.is_frozen() == true");
    }};
}

/// Postconditions for equation `distill_knowledge_transfer`.
/// Call before return: `contract_post_distill_knowledge_transfer!(result_expr)`
macro_rules! contract_post_distill_knowledge_transfer {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Invariants for equation `distill_knowledge_transfer`.
/// Check after computation: `contract_inv_distill_knowledge_transfer!(result_expr)`
macro_rules! contract_inv_distill_knowledge_transfer {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `distill_knowledge_transfer`.
macro_rules! contract_distill_knowledge_transfer {
    ($input:expr, $body:expr) => {{
        contract_pre_distill_knowledge_transfer!($input);
        let _contract_result = $body;
        contract_post_distill_knowledge_transfer!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `finetune_checkpoint_determinism`.
/// Domain-specific. Call: `contract_pre_finetune_checkpoint_determinism!(slice_expr)`
macro_rules! contract_pre_finetune_checkpoint_determinism {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
    }};
}

/// Postconditions for equation `finetune_checkpoint_determinism`.
/// Call before return: `contract_post_finetune_checkpoint_determinism!(result_expr)`
macro_rules! contract_post_finetune_checkpoint_determinism {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Invariants for equation `finetune_checkpoint_determinism`.
/// Check after computation: `contract_inv_finetune_checkpoint_determinism!(result_expr)`
macro_rules! contract_inv_finetune_checkpoint_determinism {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `finetune_checkpoint_determinism`.
macro_rules! contract_finetune_checkpoint_determinism {
    ($input:expr, $body:expr) => {{
        contract_pre_finetune_checkpoint_determinism!($input);
        let _contract_result = $body;
        contract_post_finetune_checkpoint_determinism!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `finetune_lora_rank_correctness`.
/// Domain-specific. Call: `contract_pre_finetune_lora_rank_correctness!(slice_expr)`
macro_rules! contract_pre_finetune_lora_rank_correctness {
    () => {{}};
    ($input:expr) => {{
        let _pv_x = &$input;
    }};
}

/// Postconditions for equation `finetune_lora_rank_correctness`.
/// Call before return: `contract_post_finetune_lora_rank_correctness!(result_expr)`
macro_rules! contract_post_finetune_lora_rank_correctness {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Invariants for equation `finetune_lora_rank_correctness`.
/// Check after computation: `contract_inv_finetune_lora_rank_correctness!(result_expr)`
macro_rules! contract_inv_finetune_lora_rank_correctness {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `finetune_lora_rank_correctness`.
macro_rules! contract_finetune_lora_rank_correctness {
    ($input:expr, $body:expr) => {{
        contract_pre_finetune_lora_rank_correctness!($input);
        let _contract_result = $body;
        contract_post_finetune_lora_rank_correctness!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `prune_architecture_preservation`.
/// Domain-specific. Call: `contract_pre_prune_architecture_preservation!(slice_expr)`
macro_rules! contract_pre_prune_architecture_preservation {
    () => {{}};
    ($input:expr) => {{
        let _pv_model = &$input;
    }};
}

/// Postconditions for equation `prune_architecture_preservation`.
/// Call before return: `contract_post_prune_architecture_preservation!(result_expr)`
macro_rules! contract_post_prune_architecture_preservation {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Invariants for equation `prune_architecture_preservation`.
/// Check after computation: `contract_inv_prune_architecture_preservation!(result_expr)`
macro_rules! contract_inv_prune_architecture_preservation {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `prune_architecture_preservation`.
macro_rules! contract_prune_architecture_preservation {
    ($input:expr, $body:expr) => {{
        contract_pre_prune_architecture_preservation!($input);
        let _contract_result = $body;
        contract_post_prune_architecture_preservation!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `prune_sparsity_target`.
/// Domain-specific. Call: `contract_pre_prune_sparsity_target!(slice_expr)`
macro_rules! contract_pre_prune_sparsity_target {
    () => {{}};
    ($input:expr) => {{
        let _pv_model = &$input;
    }};
}

/// Postconditions for equation `prune_sparsity_target`.
/// Call before return: `contract_post_prune_sparsity_target!(result_expr)`
macro_rules! contract_post_prune_sparsity_target {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Invariants for equation `prune_sparsity_target`.
/// Check after computation: `contract_inv_prune_sparsity_target!(result_expr)`
macro_rules! contract_inv_prune_sparsity_target {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `prune_sparsity_target`.
macro_rules! contract_prune_sparsity_target {
    ($input:expr, $body:expr) => {{
        contract_pre_prune_sparsity_target!($input);
        let _contract_result = $body;
        contract_post_prune_sparsity_target!(_contract_result);
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

/// Invariants for equation `canary_regression_detection`.
/// Check after computation: `contract_inv_canary_regression_detection!(result_expr)`
macro_rules! contract_inv_canary_regression_detection {
    () => {{}};
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

/// Invariants for equation `lint_model_conventions`.
/// Check after computation: `contract_inv_lint_model_conventions!(result_expr)`
macro_rules! contract_inv_lint_model_conventions {
    () => {{}};
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

/// Invariants for equation `model_integrity_check`.
/// Check after computation: `contract_inv_model_integrity_check!(result_expr)`
macro_rules! contract_inv_model_integrity_check {
    () => {{}};
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
        debug_assert!(
            _pv_properties.len() > 0,
            "Contract probar_property_tests: precondition violated — properties.len() > 0"
        );
    }};
}

/// Postconditions for equation `probar_property_tests`.
/// Call before return: `contract_post_probar_property_tests!(result_expr)`
macro_rules! contract_post_probar_property_tests {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Invariants for equation `probar_property_tests`.
/// Check after computation: `contract_inv_probar_property_tests!(result_expr)`
macro_rules! contract_inv_probar_property_tests {
    () => {{}};
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

/// Invariants for equation `qa_gate_composition`.
/// Check after computation: `contract_inv_qa_gate_composition!(result_expr)`
macro_rules! contract_inv_qa_gate_composition {
    () => {{}};
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

// Auto-generated from contracts/apr-model-security-v1.yaml — DO NOT EDIT
// Contract: apr-model-security-v1

/// Preconditions for equation `authentication_integrity`.
/// Call at function entry: `contract_pre_authentication_integrity!(input_expr)`
macro_rules! contract_pre_authentication_integrity {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `authentication_integrity`.
/// Call before return: `contract_post_authentication_integrity!(result_expr)`
macro_rules! contract_post_authentication_integrity {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Invariants for equation `authentication_integrity`.
/// Check after computation: `contract_inv_authentication_integrity!(result_expr)`
macro_rules! contract_inv_authentication_integrity {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `authentication_integrity`.
macro_rules! contract_authentication_integrity {
    ($input:expr, $body:expr) => {{
        contract_pre_authentication_integrity!($input);
        let _contract_result = $body;
        contract_post_authentication_integrity!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `encryption_roundtrip`.
/// Call at function entry: `contract_pre_encryption_roundtrip!(input_expr)`
macro_rules! contract_pre_encryption_roundtrip {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `encryption_roundtrip`.
/// Call before return: `contract_post_encryption_roundtrip!(result_expr)`
macro_rules! contract_post_encryption_roundtrip {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Invariants for equation `encryption_roundtrip`.
/// Check after computation: `contract_inv_encryption_roundtrip!(result_expr)`
macro_rules! contract_inv_encryption_roundtrip {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `encryption_roundtrip`.
macro_rules! contract_encryption_roundtrip {
    ($input:expr, $body:expr) => {{
        contract_pre_encryption_roundtrip!($input);
        let _contract_result = $body;
        contract_post_encryption_roundtrip!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `key_derivation_correctness`.
/// Call at function entry: `contract_pre_key_derivation_correctness!(input_expr)`
macro_rules! contract_pre_key_derivation_correctness {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `key_derivation_correctness`.
/// Call before return: `contract_post_key_derivation_correctness!(result_expr)`
macro_rules! contract_post_key_derivation_correctness {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Invariants for equation `key_derivation_correctness`.
/// Check after computation: `contract_inv_key_derivation_correctness!(result_expr)`
macro_rules! contract_inv_key_derivation_correctness {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `key_derivation_correctness`.
macro_rules! contract_key_derivation_correctness {
    ($input:expr, $body:expr) => {{
        contract_pre_key_derivation_correctness!($input);
        let _contract_result = $body;
        contract_post_key_derivation_correctness!(_contract_result);
        _contract_result
    }};
}

/// Preconditions for equation `publish_manifest_integrity`.
/// Call at function entry: `contract_pre_publish_manifest_integrity!(input_expr)`
macro_rules! contract_pre_publish_manifest_integrity {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
    }};
}

/// Postconditions for equation `publish_manifest_integrity`.
/// Call before return: `contract_post_publish_manifest_integrity!(result_expr)`
macro_rules! contract_post_publish_manifest_integrity {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Invariants for equation `publish_manifest_integrity`.
/// Check after computation: `contract_inv_publish_manifest_integrity!(result_expr)`
macro_rules! contract_inv_publish_manifest_integrity {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Combined pre+post contract for equation `publish_manifest_integrity`.
macro_rules! contract_publish_manifest_integrity {
    ($input:expr, $body:expr) => {{
        contract_pre_publish_manifest_integrity!($input);
        let _contract_result = $body;
        contract_post_publish_manifest_integrity!(_contract_result);
        _contract_result
    }};
}

// Auto-generated from contracts/apr-serve-v1.yaml — DO NOT EDIT
// Contract: apr-serve-v1

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

/// Invariants for equation `concurrent_inference_isolation`.
/// Check after computation: `contract_inv_concurrent_inference_isolation!(result_expr)`
macro_rules! contract_inv_concurrent_inference_isolation {
    () => {{}};
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

/// Invariants for equation `graceful_shutdown`.
/// Check after computation: `contract_inv_graceful_shutdown!(result_expr)`
macro_rules! contract_inv_graceful_shutdown {
    () => {{}};
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

/// Invariants for equation `request_routing`.
/// Check after computation: `contract_inv_request_routing!(result_expr)`
macro_rules! contract_inv_request_routing {
    () => {{}};
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

/// Invariants for equation `server_lifecycle`.
/// Check after computation: `contract_inv_server_lifecycle!(result_expr)`
macro_rules! contract_inv_server_lifecycle {
    () => {{}};
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

// Auto-generated from contracts/batch-training-v1.yaml — DO NOT EDIT
// Contract: batch-training-v1

/// Preconditions for equation `batch_loss`.
/// Domain-specific. Call: `contract_pre_batch_loss!(slice_expr)`
macro_rules! contract_pre_batch_loss {
    () => {{}};
    ($input:expr) => {{
        let _pv_predicted = &$input;
        debug_assert!(
            _pv_predicted.len() > 0,
            "Contract batch_loss: precondition violated — predicted.len() > 0"
        );
    }};
}

/// Invariants for equation `batch_loss`.
/// Check after computation: `contract_inv_batch_loss!(result_expr)`
macro_rules! contract_inv_batch_loss {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Preconditions for equation `gradient_accumulation`.
/// Domain-specific. Call: `contract_pre_gradient_accumulation!(slice_expr)`
macro_rules! contract_pre_gradient_accumulation {
    () => {{}};
    ($input:expr) => {{
        let _pv_params = &$input;
        debug_assert!(
            _pv_params.len() > 0,
            "Contract gradient_accumulation: precondition violated — params.len() > 0"
        );
    }};
}

/// Invariants for equation `gradient_accumulation`.
/// Check after computation: `contract_inv_gradient_accumulation!(result_expr)`
macro_rules! contract_inv_gradient_accumulation {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Preconditions for equation `gradient_clipping`.
/// Domain-specific. Call: `contract_pre_gradient_clipping!(slice_expr)`
macro_rules! contract_pre_gradient_clipping {
    () => {{}};
    ($input:expr) => {{
        let _pv_params = &$input;
        debug_assert!(
            _pv_params.len() > 0,
            "Contract gradient_clipping: precondition violated — params.len() > 0"
        );
    }};
}

/// Invariants for equation `gradient_clipping`.
/// Check after computation: `contract_inv_gradient_clipping!(result_expr)`
macro_rules! contract_inv_gradient_clipping {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
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

/// Invariants for equation `dispatch_completeness`.
/// Check after computation: `contract_inv_dispatch_completeness!(result_expr)`
macro_rules! contract_inv_dispatch_completeness {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
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

/// Invariants for equation `exit_code_semantics`.
/// Check after computation: `contract_inv_exit_code_semantics!(result_expr)`
macro_rules! contract_inv_exit_code_semantics {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
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

/// Invariants for equation `idempotent_inspection`.
/// Check after computation: `contract_inv_idempotent_inspection!(result_expr)`
macro_rules! contract_inv_idempotent_inspection {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
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

/// Invariants for equation `output_format_fidelity`.
/// Check after computation: `contract_inv_output_format_fidelity!(result_expr)`
macro_rules! contract_inv_output_format_fidelity {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
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

/// Invariants for equation `cors_negotiation`.
/// Check after computation: `contract_inv_cors_negotiation!(result_expr)`
macro_rules! contract_inv_cors_negotiation {
    () => {{}};
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

/// Invariants for equation `error_envelope_preservation`.
/// Check after computation: `contract_inv_error_envelope_preservation!(result_expr)`
macro_rules! contract_inv_error_envelope_preservation {
    () => {{}};
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

/// Invariants for equation `request_response_schema`.
/// Check after computation: `contract_inv_request_response_schema!(result_expr)`
macro_rules! contract_inv_request_response_schema {
    () => {{}};
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

/// Invariants for equation `timeout_honoring`.
/// Check after computation: `contract_inv_timeout_honoring!(result_expr)`
macro_rules! contract_inv_timeout_honoring {
    () => {{}};
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

/// Invariants for equation `fusion_decision_registry`.
/// Check after computation: `contract_inv_fusion_decision_registry!(result_expr)`
macro_rules! contract_inv_fusion_decision_registry {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
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

/// Invariants for equation `fusion_performance`.
/// Check after computation: `contract_inv_fusion_performance!(result_expr)`
macro_rules! contract_inv_fusion_performance {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Preconditions for equation `identity`.
/// Domain-specific. Call: `contract_pre_identity!(slice_expr)`
macro_rules! contract_pre_identity {
    () => {{}};
    ($input:expr) => {{
        let _pv_q = &$input;
        debug_assert!(
            _pv_q.len() > 0,
            "Contract identity: precondition violated — q.len() > 0"
        );
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
        debug_assert!(
            _pv_cpu_logits.len() > 0,
            "Contract cosine_parity_gate: precondition violated — cpu_logits.len() > 0"
        );
    }};
}

/// Invariants for equation `cosine_parity_gate`.
/// Check after computation: `contract_inv_cosine_parity_gate!(result_expr)`
macro_rules! contract_inv_cosine_parity_gate {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
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

/// Invariants for equation `layer_parity`.
/// Check after computation: `contract_inv_layer_parity!(result_expr)`
macro_rules! contract_inv_layer_parity {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
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

/// Invariants for equation `error_mapping`.
/// Check after computation: `contract_inv_error_mapping!(result_expr)`
macro_rules! contract_inv_error_mapping {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
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

/// Invariants for equation `idempotency_classification`.
/// Check after computation: `contract_inv_idempotency_classification!(result_expr)`
macro_rules! contract_inv_idempotency_classification {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
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

/// Invariants for equation `session_state_machine`.
/// Check after computation: `contract_inv_session_state_machine!(result_expr)`
macro_rules! contract_inv_session_state_machine {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
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

/// Invariants for equation `tool_schema_fidelity`.
/// Check after computation: `contract_inv_tool_schema_fidelity!(result_expr)`
macro_rules! contract_inv_tool_schema_fidelity {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
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

/// Invariants for equation `apr_tokenizer_embedding`.
/// Check after computation: `contract_inv_apr_tokenizer_embedding!(result_expr)`
macro_rules! contract_inv_apr_tokenizer_embedding {
    () => {{}};
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

/// Invariants for equation `export_fidelity`.
/// Check after computation: `contract_inv_export_fidelity!(result_expr)`
macro_rules! contract_inv_export_fidelity {
    () => {{}};
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

/// Invariants for equation `format_conversion_roundtrip`.
/// Check after computation: `contract_inv_format_conversion_roundtrip!(result_expr)`
macro_rules! contract_inv_format_conversion_roundtrip {
    () => {{}};
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

/// Invariants for equation `import_integrity`.
/// Check after computation: `contract_inv_import_integrity!(result_expr)`
macro_rules! contract_inv_import_integrity {
    () => {{}};
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
        debug_assert!(
            _pv_models.len() >= 2,
            "Contract merge_weight_algebra: precondition violated — models.len() >= 2"
        );
    }};
}

/// Postconditions for equation `merge_weight_algebra`.
/// Call before return: `contract_post_merge_weight_algebra!(result_expr)`
macro_rules! contract_post_merge_weight_algebra {
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Invariants for equation `merge_weight_algebra`.
/// Check after computation: `contract_inv_merge_weight_algebra!(result_expr)`
macro_rules! contract_inv_merge_weight_algebra {
    () => {{}};
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

/// Invariants for equation `quantization_bounds`.
/// Check after computation: `contract_inv_quantization_bounds!(result_expr)`
macro_rules! contract_inv_quantization_bounds {
    () => {{}};
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

/// Invariants for equation `gqa_ratio`.
/// Check after computation: `contract_inv_gqa_ratio!(result_expr)`
macro_rules! contract_inv_gqa_ratio {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Invariants for equation `head_dim`.
/// Check after computation: `contract_inv_head_dim!(result_expr)`
macro_rules! contract_inv_head_dim {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
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

/// Invariants for equation `bsum_decomposition`.
/// Check after computation: `contract_inv_bsum_decomposition!(result_expr)`
macro_rules! contract_inv_bsum_decomposition {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
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

/// Invariants for equation `format_isolation`.
/// Check after computation: `contract_inv_format_isolation!(result_expr)`
macro_rules! contract_inv_format_isolation {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Preconditions for equation `identity`.
/// Domain-specific. Call: `contract_pre_identity!(slice_expr)`
macro_rules! contract_pre_identity {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(
            _pv_input.len() > 0,
            "Contract identity: precondition violated — input.len() > 0"
        );
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

/// Invariants for equation `simd_scalar_equivalence`.
/// Check after computation: `contract_inv_simd_scalar_equivalence!(result_expr)`
macro_rules! contract_inv_simd_scalar_equivalence {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
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

/// Invariants for equation `kv_projection`.
/// Check after computation: `contract_inv_kv_projection!(result_expr)`
macro_rules! contract_inv_kv_projection {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
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

/// Invariants for equation `q_projection`.
/// Check after computation: `contract_inv_q_projection!(result_expr)`
macro_rules! contract_inv_q_projection {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
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

/// Invariants for equation `swiglu_expansion`.
/// Check after computation: `contract_inv_swiglu_expansion!(result_expr)`
macro_rules! contract_inv_swiglu_expansion {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
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

// Auto-generated from contracts/special-tokens-registry-v1.yaml — DO NOT EDIT
// Contract: special-tokens-registry-v1

/// Invariants for equation `token_id_bound`.
/// Check after computation: `contract_inv_token_id_bound!(result_expr)`
macro_rules! contract_inv_token_id_bound {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
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
        debug_assert!(
            _pv_a.len() > 0,
            "Contract identity: precondition violated — a.len() > 0"
        );
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

/// Invariants for equation `quant_dispatch_exhaustiveness`.
/// Check after computation: `contract_inv_quant_dispatch_exhaustiveness!(result_expr)`
macro_rules! contract_inv_quant_dispatch_exhaustiveness {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
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

/// Invariants for equation `transpose_invariant`.
/// Check after computation: `contract_inv_transpose_invariant!(result_expr)`
macro_rules! contract_inv_transpose_invariant {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Preconditions for equation `validated_tensor_construction`.
/// Domain-specific. Call: `contract_pre_validated_tensor_construction!(slice_expr)`
macro_rules! contract_pre_validated_tensor_construction {
    () => {{}};
    ($input:expr) => {{
        let _pv_data = &$input;
        debug_assert!(
            _pv_data.len() > 0,
            "Contract validated_tensor_construction: precondition violated — data.len() > 0"
        );
    }};
}

/// Invariants for equation `validated_tensor_construction`.
/// Check after computation: `contract_inv_validated_tensor_construction!(result_expr)`
macro_rules! contract_inv_validated_tensor_construction {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
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

/// Invariants for equation `byte_encoder_coverage`.
/// Check after computation: `contract_inv_byte_encoder_coverage!(result_expr)`
macro_rules! contract_inv_byte_encoder_coverage {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Preconditions for equation `identity`.
/// Call at function entry: `contract_pre_identity!(input_expr)`
macro_rules! contract_pre_identity {
    () => {{}};
    ($input:expr) => {{
        let _contract_input = &$input;
        debug_assert!(
            !_contract_input.is_empty(),
            "Contract identity: precondition violated — !input.is_empty()"
        );
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

/// Invariants for equation `roundtrip_encoding`.
/// Check after computation: `contract_inv_roundtrip_encoding!(result_expr)`
macro_rules! contract_inv_roundtrip_encoding {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

// Auto-generated from contracts/tokenizer-vocab-v1.yaml — DO NOT EDIT
// Contract: tokenizer-vocab-v1

/// Invariants for equation `vocab_size_consistency`.
/// Check after computation: `contract_inv_vocab_size_consistency!(result_expr)`
macro_rules! contract_inv_vocab_size_consistency {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
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
        debug_assert!(
            _pv_predicted.len() > 0,
            "Contract ema_loss: precondition violated — predicted.len() > 0"
        );
    }};
}

/// Invariants for equation `ema_loss`.
/// Check after computation: `contract_inv_ema_loss!(result_expr)`
macro_rules! contract_inv_ema_loss {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Preconditions for equation `val_split`.
/// Domain-specific. Call: `contract_pre_val_split!(slice_expr)`
macro_rules! contract_pre_val_split {
    () => {{}};
    ($input:expr) => {{
        let _pv_input = &$input;
        debug_assert!(
            _pv_input.len() > 0,
            "Contract val_split: precondition violated — input.len() > 0"
        );
        debug_assert!(
            _pv_input.iter().all(|v| v.is_finite()),
            "Contract val_split: precondition violated — input.iter().all(|v| v.is_finite())"
        );
    }};
}

/// Invariants for equation `val_split`.
/// Check after computation: `contract_inv_val_split!(result_expr)`
macro_rules! contract_inv_val_split {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

/// Preconditions for equation `warmup_lr`.
/// Domain-specific. Call: `contract_pre_warmup_lr!(slice_expr)`
macro_rules! contract_pre_warmup_lr {
    () => {{}};
    ($input:expr) => {{
        let _pv_params = &$input;
        debug_assert!(
            _pv_params.len() > 0,
            "Contract warmup_lr: precondition violated — params.len() > 0"
        );
    }};
}

/// Invariants for equation `warmup_lr`.
/// Check after computation: `contract_inv_warmup_lr!(result_expr)`
macro_rules! contract_inv_warmup_lr {
    () => {{}};
    ($result:expr) => {{
        let _contract_result = &$result;
    }};
}

// Total: 22 preconditions, 0 postconditions, 0 invariants from 30 contracts

// ── Stub macros for PMAT-493/541 call-site annotations (contracts pending) ──
// These will be replaced when the corresponding YAML contracts are created.

macro_rules! contract_pre_dispatch_core_command { () => {{}}; ($($t:tt)*) => {{}}; }
macro_rules! contract_pre_execute_command { () => {{}}; ($($t:tt)*) => {{}}; }
macro_rules! contract_pre_with_stdin_support { () => {{}}; ($($t:tt)*) => {{}}; }
macro_rules! contract_pre_resolve_model_path { () => {{}}; ($($t:tt)*) => {{}}; }
macro_rules! contract_pre_merge_tensor_shape { () => {{}}; ($($t:tt)*) => {{}}; }
macro_rules! contract_pre_validate_exit_code_consistency { () => {{}}; ($($t:tt)*) => {{}}; }

macro_rules! contract_pre_validate_exit_code_consistency { () => {{}}; ($($t:tt)*) => {{}}; }
macro_rules! contract_pre_format_conversion_roundtrip { () => {{}}; ($($t:tt)*) => {{}}; }
macro_rules! contract_pre_encryption_idempotency { () => {{}}; ($($t:tt)*) => {{}}; }

