//! Banco route wiring.

use axum::{
    middleware,
    routing::{get, post},
    Router,
};

use super::audit::{audit_layer, AuditLog};
use super::auth::auth_layer;
use axum::routing::delete;

use super::compat_ollama::{
    ollama_chat_handler, ollama_delete_handler, ollama_generate_handler, ollama_pull_handler,
    ollama_show_handler, ollama_tags_handler,
};
use axum::routing::put;

use super::handlers::{
    chat_completions_handler, create_conversation_handler, delete_conversation_handler,
    delete_prompt_handler, detokenize_handler, embeddings_handler, export_conversations_handler,
    get_conversation_handler, get_parameters_handler, get_prompt_handler, health_handler,
    import_conversations_handler, list_conversations_handler, list_prompts_handler,
    liveness_handler, models_handler, readiness_handler, rename_conversation_handler,
    save_prompt_handler, search_conversations_handler, system_handler, tokenize_handler,
    update_parameters_handler,
};
use super::handlers_audio::{audio_formats_handler, transcribe_handler};
use super::handlers_audit::audit_query_handler;
use super::handlers_batch::{get_batch_handler, list_batches_handler, submit_batch_handler};
use super::handlers_completions::{completions_handler, model_detail_handler};
use super::handlers_config::{get_config_handler, update_config_handler};
use super::handlers_data::{
    delete_file_handler, file_info_handler, list_files_handler, upload_handler, upload_json_handler,
};
use super::handlers_eval::{eval_perplexity_handler, get_eval_run_handler, list_eval_runs_handler};
use super::handlers_experiment::{
    add_run_to_experiment_handler, compare_experiment_handler, create_experiment_handler,
    list_experiments_handler,
};
use super::handlers_mcp::{mcp_handler, mcp_info_handler};
use super::handlers_merge::{list_merge_strategies_handler, merge_models_handler};
use super::handlers_metrics::metrics_handler;
use super::handlers_models::{model_load_handler, model_status_handler, model_unload_handler};
use super::handlers_rag::{
    rag_clear_handler, rag_index_handler, rag_search_handler, rag_status_handler,
};
use super::handlers_recipes::{
    create_recipe_handler, get_recipe_handler, list_datasets_handler, list_recipes_handler,
    preview_dataset_handler, run_recipe_handler,
};
use super::handlers_registry::{
    list_registry_handler, pull_model_handler, remove_cached_model_handler,
};
use super::handlers_tools::{
    configure_tool_handler, execute_tool_handler, list_tools_handler, register_tool_handler,
};
use super::handlers_train::{
    delete_training_run_handler, export_training_handler, get_training_run_handler,
    list_presets_handler, list_training_runs_handler, start_training_handler,
    stop_training_handler, training_metrics_handler,
};
use super::handlers_ui::{assets_handler, chat_form_handler, index_handler};
use super::handlers_ws::ws_handler;
use super::middleware::privacy_layer;
use super::state::BancoState;

/// Build the full Banco router with all endpoints and middleware.
///
/// Mounts endpoints at both `/api/v1/` (Banco canonical) and `/v1/` (OpenAI SDK compat).
pub fn create_banco_router(state: BancoState) -> Router {
    let tier = state.privacy_tier;
    let auth = state.auth.clone();
    let log = state.audit_log.clone();
    create_banco_router_inner(state, tier, auth, log)
}

/// Build router with an explicit audit log (for testing).
pub fn create_banco_router_with_audit(state: BancoState, audit_log: AuditLog) -> Router {
    let tier = state.privacy_tier;
    let auth = state.auth.clone();
    create_banco_router_inner(state, tier, auth, audit_log)
}

fn create_banco_router_inner(
    state: BancoState,
    tier: crate::serve::backends::PrivacyTier,
    auth: super::auth::AuthStore,
    log: AuditLog,
) -> Router {
    Router::new()
        // Browser UI
        .route("/", get(index_handler))
        .route("/ui/chat", post(chat_form_handler))
        .route("/assets/*path", get(assets_handler))
        .route("/health", get(health_handler))
        .route("/health/ready", get(readiness_handler))
        .route("/health/live", get(liveness_handler))
        // Banco canonical paths
        .route("/api/v1/models", get(models_handler))
        .route("/api/v1/chat/completions", post(chat_completions_handler))
        .route("/api/v1/system", get(system_handler))
        .route("/api/v1/tokenize", post(tokenize_handler))
        .route("/api/v1/detokenize", post(detokenize_handler))
        .route("/api/v1/embeddings", post(embeddings_handler))
        // Model management
        .route("/api/v1/models/load", post(model_load_handler))
        .route("/api/v1/models/unload", post(model_unload_handler))
        .route("/api/v1/models/status", get(model_status_handler))
        // Inference parameters
        .route(
            "/api/v1/chat/parameters",
            get(get_parameters_handler).put(update_parameters_handler),
        )
        // Conversation endpoints
        .route(
            "/api/v1/conversations",
            get(list_conversations_handler).post(create_conversation_handler),
        )
        .route("/api/v1/conversations/search", get(search_conversations_handler))
        .route("/api/v1/conversations/export", get(export_conversations_handler))
        .route("/api/v1/conversations/import", post(import_conversations_handler))
        .route(
            "/api/v1/conversations/:id",
            get(get_conversation_handler)
                .patch(rename_conversation_handler)
                .delete(delete_conversation_handler),
        )
        // Prompt presets
        .route("/api/v1/prompts", get(list_prompts_handler).post(save_prompt_handler))
        .route("/api/v1/prompts/:id", get(get_prompt_handler).delete(delete_prompt_handler))
        // OpenAI SDK compat paths (/v1/ prefix)
        .route("/v1/models", get(models_handler))
        .route("/v1/models/:id", get(model_detail_handler))
        .route("/v1/completions", post(completions_handler))
        .route("/v1/chat/completions", post(chat_completions_handler))
        .route("/v1/embeddings", post(embeddings_handler))
        .route("/v1/audio/transcriptions", post(transcribe_handler))
        // Data management (Phase 3)
        .route("/api/v1/data/upload", post(upload_handler))
        .route("/api/v1/data/upload/json", post(upload_json_handler))
        .route("/api/v1/data/files", get(list_files_handler))
        .route("/api/v1/data/files/:id", delete(delete_file_handler))
        .route("/api/v1/data/files/:id/info", get(file_info_handler))
        // Data recipes
        .route("/api/v1/data/recipes", get(list_recipes_handler).post(create_recipe_handler))
        .route("/api/v1/data/recipes/:id", get(get_recipe_handler))
        .route("/api/v1/data/recipes/:id/run", post(run_recipe_handler))
        .route("/api/v1/data/datasets", get(list_datasets_handler))
        .route("/api/v1/data/datasets/:id/preview", get(preview_dataset_handler))
        // RAG (retrieval-augmented generation)
        .route("/api/v1/rag/index", post(rag_index_handler).delete(rag_clear_handler))
        .route("/api/v1/rag/status", get(rag_status_handler))
        .route("/api/v1/rag/search", get(rag_search_handler))
        // Eval
        .route("/api/v1/eval/perplexity", post(eval_perplexity_handler))
        .route("/api/v1/eval/runs", get(list_eval_runs_handler))
        .route("/api/v1/eval/runs/:id", get(get_eval_run_handler))
        // Training
        .route("/api/v1/train/start", post(start_training_handler))
        .route("/api/v1/train/runs", get(list_training_runs_handler))
        .route(
            "/api/v1/train/runs/:id",
            get(get_training_run_handler).delete(delete_training_run_handler),
        )
        .route("/api/v1/train/runs/:id/stop", post(stop_training_handler))
        .route("/api/v1/train/runs/:id/metrics", get(training_metrics_handler))
        .route("/api/v1/train/runs/:id/export", post(export_training_handler))
        .route("/api/v1/train/presets", get(list_presets_handler))
        // Model merge
        .route("/api/v1/models/merge", post(merge_models_handler))
        .route("/api/v1/models/merge/strategies", get(list_merge_strategies_handler))
        // Model registry (pacha)
        .route("/api/v1/models/pull", post(pull_model_handler))
        .route("/api/v1/models/registry", get(list_registry_handler))
        .route("/api/v1/models/registry/:name", delete(remove_cached_model_handler))
        // Experiments
        .route("/api/v1/experiments", get(list_experiments_handler).post(create_experiment_handler))
        .route("/api/v1/experiments/:id/runs", post(add_run_to_experiment_handler))
        .route("/api/v1/experiments/:id/compare", get(compare_experiment_handler))
        // Batch inference
        .route("/api/v1/batch", post(submit_batch_handler).get(list_batches_handler))
        .route("/api/v1/batch/:id", get(get_batch_handler))
        // Config + Audit
        .route("/api/v1/config", get(get_config_handler).put(update_config_handler))
        .route("/api/v1/audit", get(audit_query_handler))
        .route("/api/v1/metrics", get(metrics_handler))
        // Audio transcription (whisper-apr)
        .route("/api/v1/audio/transcriptions", post(transcribe_handler))
        .route("/api/v1/audio/formats", get(audio_formats_handler))
        // Tool calling
        .route("/api/v1/tools", get(list_tools_handler).post(register_tool_handler))
        .route("/api/v1/tools/execute", post(execute_tool_handler))
        .route("/api/v1/tools/:name/config", put(configure_tool_handler))
        // MCP (Model Context Protocol)
        .route("/api/v1/mcp", post(mcp_handler))
        .route("/api/v1/mcp/info", get(mcp_info_handler))
        // WebSocket for real-time events
        .route("/api/v1/ws", get(ws_handler))
        // Ollama compat paths (/api/ prefix — Ollama protocol)
        .route("/api/generate", post(ollama_generate_handler))
        .route("/api/chat", post(ollama_chat_handler))
        .route("/api/tags", get(ollama_tags_handler))
        .route("/api/show", post(ollama_show_handler))
        .route("/api/pull", post(ollama_pull_handler))
        .route("/api/delete", delete(ollama_delete_handler))
        // Middleware stack (outermost first):
        // 1. Audit logging
        .layer(middleware::from_fn(move |req, next| audit_layer(log.clone(), req, next)))
        // 2. Authentication (API key check)
        .layer(middleware::from_fn(move |req, next| auth_layer(auth.clone(), req, next)))
        // 3. Privacy tier header + sovereign gate
        .layer(middleware::from_fn(move |req, next| privacy_layer(tier, req, next)))
        .with_state(state)
}
