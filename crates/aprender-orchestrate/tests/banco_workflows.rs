//! Banco L2 Integration Tests — training, merge, eval, and experiment workflows.
//!
//! Tests multi-step workflows (start training, list runs, merge models, etc.)
//! against a real Banco TCP server.

#![cfg(feature = "banco")]

use std::time::Duration;

/// Start a Banco server on a random port. Returns (base_url, abort_handle).
async fn start_server() -> (String, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let base = format!("http://127.0.0.1:{port}");

    let state = batuta::serve::banco::state::BancoStateInner::with_defaults();
    let app = batuta::serve::banco::router::create_banco_router(state);

    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    tokio::time::sleep(Duration::from_millis(100)).await;
    (base, handle)
}

// ============================================================================
// Training workflow
// ============================================================================

#[tokio::test]
async fn l2_training_presets() {
    let (base, handle) = start_server().await;
    let resp = reqwest::get(format!("{base}/api/v1/train/presets")).await.unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert!(json["presets"].is_array());
    let presets = json["presets"].as_array().unwrap();
    assert!(presets.len() >= 3, "Should have at least 3 presets");
    handle.abort();
}

#[tokio::test]
async fn l2_training_start_and_list() {
    let (base, handle) = start_server().await;
    let client = reqwest::Client::new();

    // Start training
    let resp = client
        .post(format!("{base}/api/v1/train/start"))
        .json(&serde_json::json!({
            "dataset_id": "inline-test",
            "preset": "quick-lora"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert!(json["id"].is_string());
    assert!(json["status"].is_string());
    // Training is simulated (no real gradient-based training yet)
    assert_eq!(json["simulated"], true);

    // List runs
    let resp = reqwest::get(format!("{base}/api/v1/train/runs")).await.unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert!(json["runs"].is_array());

    handle.abort();
}

// ============================================================================
// Merge workflow
// ============================================================================

#[tokio::test]
async fn l2_merge_strategies() {
    let (base, handle) = start_server().await;
    let resp = reqwest::get(format!("{base}/api/v1/models/merge/strategies")).await.unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    let strategies = json["strategies"].as_array().unwrap();
    assert!(strategies.len() >= 4);
    let names: Vec<&str> = strategies.iter().filter_map(|s| s["name"].as_str()).collect();
    assert!(names.contains(&"weighted_average"));
    assert!(names.contains(&"ties"));
    assert!(names.contains(&"dare"));
    assert!(names.contains(&"slerp"));
    handle.abort();
}

#[tokio::test]
async fn l2_merge_weighted_average() {
    let (base, handle) = start_server().await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{base}/api/v1/models/merge"))
        .json(&serde_json::json!({
            "models": ["model-a", "model-b"],
            "strategy": "weighted_average",
            "weights": [0.7, 0.3]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert!(json["merge_id"].is_string());
    assert_eq!(json["strategy"], "weighted_average");
    assert_eq!(json["simulated"], true, "Merge on placeholders should be marked simulated");
    handle.abort();
}

#[tokio::test]
async fn l2_merge_slerp() {
    let (base, handle) = start_server().await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{base}/api/v1/models/merge"))
        .json(&serde_json::json!({
            "models": ["model-a", "model-b"],
            "strategy": "slerp",
            "interpolation_t": 0.3
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(json["strategy"], "slerp");
    handle.abort();
}

#[tokio::test]
async fn l2_merge_slerp_rejects_three_models() {
    let (base, handle) = start_server().await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{base}/api/v1/models/merge"))
        .json(&serde_json::json!({
            "models": ["a", "b", "c"],
            "strategy": "slerp"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
    handle.abort();
}

// ============================================================================
// Eval workflow
// ============================================================================

#[tokio::test]
async fn l2_eval_no_model() {
    let (base, handle) = start_server().await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{base}/api/v1/eval/perplexity"))
        .json(&serde_json::json!({
            "text": "The quick brown fox jumps over the lazy dog."
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(json["status"], "no_model");
    handle.abort();
}

#[tokio::test]
async fn l2_eval_runs_list() {
    let (base, handle) = start_server().await;
    let resp = reqwest::get(format!("{base}/api/v1/eval/runs")).await.unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert!(json["runs"].is_array());
    handle.abort();
}

// ============================================================================
// Experiment workflow
// ============================================================================

#[tokio::test]
async fn l2_experiments_create_and_list() {
    let (base, handle) = start_server().await;
    let client = reqwest::Client::new();

    // Create experiment
    let resp = client
        .post(format!("{base}/api/v1/experiments"))
        .json(&serde_json::json!({
            "name": "test-exp",
            "description": "L2 test experiment"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    // List experiments
    let resp = reqwest::get(format!("{base}/api/v1/experiments")).await.unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert!(json["experiments"].is_array());

    handle.abort();
}

// ============================================================================
// File operations
// ============================================================================

#[tokio::test]
async fn l2_file_list_empty() {
    let (base, handle) = start_server().await;
    let resp = reqwest::get(format!("{base}/api/v1/data/files")).await.unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert!(json["files"].is_array());
    handle.abort();
}

#[tokio::test]
async fn l2_rag_status() {
    let (base, handle) = start_server().await;
    let resp = reqwest::get(format!("{base}/api/v1/rag/status")).await.unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert!(json.is_object());
    handle.abort();
}

#[tokio::test]
async fn l2_csv_upload_schema_detection() {
    let (base, handle) = start_server().await;
    let client = reqwest::Client::new();

    // Upload CSV with schema
    let resp = client
        .post(format!("{base}/api/v1/data/upload/json"))
        .json(&serde_json::json!({
            "name": "data.csv",
            "content": "name,age,score\nAlice,30,95.5\nBob,25,88.0",
            "content_type": "text/csv"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    // Schema should be detected
    if let Some(schema) = json["schema"].as_array() {
        assert!(!schema.is_empty(), "Schema should detect CSV columns");
    }
    handle.abort();
}

#[tokio::test]
async fn l2_tools_list() {
    let (base, handle) = start_server().await;
    let resp = reqwest::get(format!("{base}/api/v1/tools")).await.unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert!(json["tools"].is_array());
    let tools = json["tools"].as_array().unwrap();
    // Should have at least calculator and code_execution
    let names: Vec<&str> = tools.iter().filter_map(|t| t["name"].as_str()).collect();
    assert!(names.contains(&"calculator"), "Should have calculator tool");
    handle.abort();
}

#[tokio::test]
async fn l2_recipes_crud() {
    let (base, handle) = start_server().await;
    let client = reqwest::Client::new();

    // List recipes (initially empty)
    let resp = reqwest::get(format!("{base}/api/v1/data/recipes")).await.unwrap();
    assert_eq!(resp.status(), 200);

    // Create recipe
    let resp = client
        .post(format!("{base}/api/v1/data/recipes"))
        .json(&serde_json::json!({
            "name": "test-recipe",
            "source_files": [],
            "steps": [{"type": "extract_text", "config": {}}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    handle.abort();
}

#[tokio::test]
async fn l2_conversation_search() {
    let (base, handle) = start_server().await;

    let resp = reqwest::get(format!("{base}/api/v1/conversations/search?q=test")).await.unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert!(json["conversations"].is_array());

    handle.abort();
}

#[tokio::test]
async fn l2_model_registry_list() {
    let (base, handle) = start_server().await;

    let resp = reqwest::get(format!("{base}/api/v1/models/registry")).await.unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert!(json["models"].is_array());

    handle.abort();
}

#[tokio::test]
async fn l2_ollama_generate() {
    let (base, handle) = start_server().await;
    let client = reqwest::Client::new();

    let resp = client
        .post(format!("{base}/api/generate"))
        .json(&serde_json::json!({
            "model": "local",
            "prompt": "Hello",
            "stream": false
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert!(json["response"].is_string());

    handle.abort();
}

#[tokio::test]
async fn l2_ollama_chat() {
    let (base, handle) = start_server().await;
    let client = reqwest::Client::new();

    let resp = client
        .post(format!("{base}/api/chat"))
        .json(&serde_json::json!({
            "model": "local",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": false
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert!(json["message"].is_object());

    handle.abort();
}

#[tokio::test]
async fn l2_recipe_upload_run_workflow() {
    let (base, handle) = start_server().await;
    let client = reqwest::Client::new();

    // 1. Upload a text file
    let resp = client
        .post(format!("{base}/api/v1/data/upload/json"))
        .json(&serde_json::json!({
            "name": "training.txt",
            "content": "Hello world\nThis is training data\nFor a small model"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let upload: serde_json::Value = resp.json().await.unwrap();
    let file_id = upload["id"].as_str().unwrap().to_string();

    // 2. Create recipe referencing the file
    let resp = client
        .post(format!("{base}/api/v1/data/recipes"))
        .json(&serde_json::json!({
            "name": "test-pipeline",
            "source_files": [file_id],
            "steps": [
                {"type": "extract_text", "config": {}},
                {"type": "chunk", "config": {"max_tokens": 64}}
            ]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let recipe: serde_json::Value = resp.json().await.unwrap();
    let recipe_id = recipe["id"].as_str().unwrap().to_string();

    // 3. Run the recipe
    let resp =
        client.post(format!("{base}/api/v1/data/recipes/{recipe_id}/run")).send().await.unwrap();
    assert_eq!(resp.status(), 200);
    let result: serde_json::Value = resp.json().await.unwrap();
    assert!(result["dataset_id"].is_string());
    assert!(result["record_count"].as_u64().unwrap() > 0);

    // 4. List datasets — should have one
    let resp = reqwest::get(format!("{base}/api/v1/data/datasets")).await.unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert!(!json["datasets"].as_array().unwrap().is_empty());

    // 5. Train on the dataset
    let dataset_id = result["dataset_id"].as_str().unwrap();
    let resp = client
        .post(format!("{base}/api/v1/train/start"))
        .json(&serde_json::json!({
            "dataset_id": dataset_id,
            "preset": "quick-lora"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let train: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(train["status"], "complete");
    assert_eq!(train["simulated"], true);
    // Metrics should exist
    assert!(!train["metrics"].as_array().unwrap().is_empty());

    handle.abort();
}

#[tokio::test]
async fn l2_prompts_crud() {
    let (base, handle) = start_server().await;
    let client = reqwest::Client::new();

    // Create prompt preset
    let resp = client
        .post(format!("{base}/api/v1/prompts"))
        .json(&serde_json::json!({
            "name": "test-prompt",
            "content": "You are a helpful assistant. {{input}}"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    // List prompts
    let resp = reqwest::get(format!("{base}/api/v1/prompts")).await.unwrap();
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert!(json["presets"].is_array());
    handle.abort();
}

// Fixes #53: training export/stop workflow
#[tokio::test]
async fn l2_training_export_and_stop() {
    let (base, handle) = start_server().await;
    let client = reqwest::Client::new();
    // Start training
    let resp = client
        .post(format!("{base}/api/v1/train/start"))
        .json(&serde_json::json!({"dataset_id": "test", "preset": "quick-lora"}))
        .send()
        .await
        .unwrap();
    let run: serde_json::Value = resp.json().await.unwrap();
    let id = run["id"].as_str().unwrap();
    // Export
    let resp = client
        .post(format!("{base}/api/v1/train/runs/{id}/export"))
        .json(&serde_json::json!({"format": "safetensors", "merge": false}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    // Stop (already complete, should handle gracefully)
    let resp = client.post(format!("{base}/api/v1/train/runs/{id}/stop")).send().await.unwrap();
    assert!(resp.status().is_success() || resp.status() == 400);
    handle.abort();
}
