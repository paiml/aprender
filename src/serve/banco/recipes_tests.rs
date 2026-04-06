//! Recipe engine + endpoint tests.

use super::recipes::{RecipeStatus, RecipeStep, StepType};

// ============================================================================
// RecipeStore unit tests
// ============================================================================

#[test]
#[allow(non_snake_case)]
fn test_RECIPE_001_create_recipe() {
    let store = super::recipes::RecipeStore::new();
    let recipe = store.create(
        "test-recipe",
        vec!["file-1".to_string()],
        vec![RecipeStep { step_type: StepType::ExtractText, config: serde_json::json!({}) }],
        "jsonl",
    );
    assert!(recipe.id.starts_with("recipe-"));
    assert_eq!(recipe.name, "test-recipe");
    assert_eq!(recipe.status, RecipeStatus::Created);
}

#[test]
#[allow(non_snake_case)]
fn test_RECIPE_002_list_recipes() {
    let store = super::recipes::RecipeStore::new();
    store.create("a", vec![], vec![], "jsonl");
    store.create("b", vec![], vec![], "jsonl");
    assert_eq!(store.list().len(), 2);
}

#[test]
#[allow(non_snake_case)]
fn test_RECIPE_003_run_extract_text() {
    let store = super::recipes::RecipeStore::new();
    let recipe = store.create(
        "extract",
        vec![],
        vec![RecipeStep { step_type: StepType::ExtractText, config: serde_json::json!({}) }],
        "jsonl",
    );
    let result = store.run(&recipe.id, &[("doc.txt", "Hello world")]).expect("run");
    assert_eq!(result.record_count, 1);
    assert_eq!(result.records[0].text, "Hello world");
}

#[test]
#[allow(non_snake_case)]
fn test_RECIPE_004_run_chunk() {
    let store = super::recipes::RecipeStore::new();
    let recipe = store.create(
        "chunker",
        vec![],
        vec![RecipeStep {
            step_type: StepType::Chunk,
            config: serde_json::json!({"max_tokens": 5, "overlap": 1}),
        }],
        "jsonl",
    );
    // 5 tokens * 4 chars = 20 chars max per chunk. Input is 26 chars.
    let result = store.run(&recipe.id, &[("doc.txt", "abcdefghijklmnopqrstuvwxyz")]).expect("run");
    assert!(result.record_count > 1, "should split into multiple chunks");
    assert!(result.records[0].metadata.contains_key("chunk_index"));
}

#[test]
#[allow(non_snake_case)]
fn test_RECIPE_005_run_filter() {
    let store = super::recipes::RecipeStore::new();
    let recipe = store.create(
        "filter",
        vec![],
        vec![RecipeStep {
            step_type: StepType::Filter,
            config: serde_json::json!({"min_length": 5}),
        }],
        "jsonl",
    );
    let result = store.run(&recipe.id, &[("a.txt", "hi"), ("b.txt", "hello world")]).expect("run");
    assert_eq!(result.record_count, 1);
    assert_eq!(result.records[0].text, "hello world");
}

#[test]
#[allow(non_snake_case)]
fn test_RECIPE_006_run_format_chatml() {
    let store = super::recipes::RecipeStore::new();
    let recipe = store.create(
        "formatter",
        vec![],
        vec![RecipeStep {
            step_type: StepType::Format,
            config: serde_json::json!({"template": "chatml"}),
        }],
        "jsonl",
    );
    let result = store.run(&recipe.id, &[("doc.txt", "What is Rust?")]).expect("run");
    assert!(result.records[0].text.contains("<|im_start|>user"));
    assert!(result.records[0].text.contains("What is Rust?"));
}

#[test]
#[allow(non_snake_case)]
fn test_RECIPE_007_run_dedup() {
    let store = super::recipes::RecipeStore::new();
    let recipe = store.create(
        "dedup",
        vec![],
        vec![RecipeStep { step_type: StepType::Deduplicate, config: serde_json::json!({}) }],
        "jsonl",
    );
    let result = store
        .run(&recipe.id, &[("a.txt", "same text"), ("b.txt", "same text"), ("c.txt", "different")])
        .expect("run");
    assert_eq!(result.record_count, 2);
}

#[test]
#[allow(non_snake_case)]
fn test_RECIPE_008_pipeline_chain() {
    let store = super::recipes::RecipeStore::new();
    let recipe = store.create(
        "full-pipeline",
        vec![],
        vec![
            RecipeStep { step_type: StepType::ExtractText, config: serde_json::json!({}) },
            RecipeStep {
                step_type: StepType::Filter,
                config: serde_json::json!({"min_length": 3}),
            },
            RecipeStep {
                step_type: StepType::Format,
                config: serde_json::json!({"template": "alpaca"}),
            },
        ],
        "jsonl",
    );
    let result =
        store.run(&recipe.id, &[("a.txt", "hi"), ("b.txt", "longer text here")]).expect("run");
    assert_eq!(result.record_count, 1);
    assert!(result.records[0].text.contains("### Instruction:"));
}

#[test]
#[allow(non_snake_case)]
fn test_RECIPE_009_status_transitions() {
    let store = super::recipes::RecipeStore::new();
    let recipe = store.create("test", vec![], vec![], "jsonl");
    assert_eq!(store.get(&recipe.id).expect("get").status, RecipeStatus::Created);

    store.run(&recipe.id, &[("a.txt", "data")]).expect("run");
    assert_eq!(store.get(&recipe.id).expect("get").status, RecipeStatus::Complete);
}

// ============================================================================
// CSV/JSONL parsing tests
// ============================================================================

#[test]
#[allow(non_snake_case)]
fn test_RECIPE_010_parse_csv() {
    let store = super::recipes::RecipeStore::new();
    let recipe = store.create(
        "csv-parse",
        vec![],
        vec![RecipeStep { step_type: StepType::ParseCsv, config: serde_json::json!({}) }],
        "jsonl",
    );
    let csv_data = "name,text,label\nAlice,Hello world,1\nBob,Goodbye,0\n";
    let result = store.run(&recipe.id, &[("data.csv", csv_data)]).expect("run");
    assert_eq!(result.record_count, 2);
    assert!(result.records[0].text.contains("Alice"));
    assert!(result.records[0].metadata.contains_key("row_index"));
}

#[test]
#[allow(non_snake_case)]
fn test_RECIPE_011_parse_csv_with_text_column() {
    let store = super::recipes::RecipeStore::new();
    let recipe = store.create(
        "csv-col",
        vec![],
        vec![RecipeStep {
            step_type: StepType::ParseCsv,
            config: serde_json::json!({"text_column": "text"}),
        }],
        "jsonl",
    );
    let csv_data = "name,text,label\nAlice,Hello world,1\nBob,Goodbye,0\n";
    let result = store.run(&recipe.id, &[("data.csv", csv_data)]).expect("run");
    assert_eq!(result.record_count, 2);
    assert_eq!(result.records[0].text, "Hello world");
    assert_eq!(result.records[1].text, "Goodbye");
}

#[test]
#[allow(non_snake_case)]
fn test_RECIPE_012_parse_jsonl() {
    let store = super::recipes::RecipeStore::new();
    let recipe = store.create(
        "jsonl-parse",
        vec![],
        vec![RecipeStep { step_type: StepType::ParseJsonl, config: serde_json::json!({}) }],
        "jsonl",
    );
    let jsonl = r#"{"text": "Hello", "label": 1}
{"text": "World", "label": 0}
"#;
    let result = store.run(&recipe.id, &[("data.jsonl", jsonl)]).expect("run");
    assert_eq!(result.record_count, 2);
    // Without text_field, uses first string field
    assert!(
        result.records[0].text == "Hello" || result.records[0].text.contains("Hello"),
        "got: {}",
        result.records[0].text
    );
}

#[test]
#[allow(non_snake_case)]
fn test_RECIPE_013_parse_jsonl_with_text_field() {
    let store = super::recipes::RecipeStore::new();
    let recipe = store.create(
        "jsonl-field",
        vec![],
        vec![RecipeStep {
            step_type: StepType::ParseJsonl,
            config: serde_json::json!({"text_field": "content"}),
        }],
        "jsonl",
    );
    let jsonl = r#"{"id": 1, "content": "Training example one"}
{"id": 2, "content": "Training example two"}
"#;
    let result = store.run(&recipe.id, &[("data.jsonl", jsonl)]).expect("run");
    assert_eq!(result.record_count, 2);
    assert_eq!(result.records[0].text, "Training example one");
    assert_eq!(result.records[1].text, "Training example two");
}

#[test]
#[allow(non_snake_case)]
fn test_RECIPE_014_csv_then_chunk_pipeline() {
    let store = super::recipes::RecipeStore::new();
    let recipe = store.create(
        "csv-chunk",
        vec![],
        vec![
            RecipeStep {
                step_type: StepType::ParseCsv,
                config: serde_json::json!({"text_column": "text"}),
            },
            RecipeStep { step_type: StepType::Chunk, config: serde_json::json!({"max_tokens": 3}) },
        ],
        "jsonl",
    );
    let csv_data = "id,text\n1,This is a very long text that should be chunked\n";
    let result = store.run(&recipe.id, &[("data.csv", csv_data)]).expect("run");
    assert!(result.record_count > 1, "long CSV text should be chunked");
}

// ============================================================================
// Recipe endpoint tests
// ============================================================================

#[tokio::test]
#[allow(non_snake_case)]
async fn test_RECIPE_HDL_001_create_recipe() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let body = serde_json::json!({
        "name": "test-recipe",
        "steps": [{"type": "extract_text", "config": {}}]
    });
    let response = app
        .oneshot(
            Request::post("/api/v1/data/recipes")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert!(json["id"].as_str().expect("id").starts_with("recipe-"));
    assert_eq!(json["status"], "created");
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_RECIPE_HDL_002_list_recipes() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let state = super::state::BancoStateInner::with_defaults();
    state.recipes.create("a", vec![], vec![], "jsonl");
    state.recipes.create("b", vec![], vec![], "jsonl");

    let app = super::router::create_banco_router(state);
    let response = app
        .oneshot(Request::get("/api/v1/data/recipes").body(Body::empty()).expect("req"))
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert_eq!(json["recipes"].as_array().expect("recipes").len(), 2);
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_RECIPE_HDL_003_run_recipe_produces_dataset() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let state = super::state::BancoStateInner::with_defaults();

    // Upload a file first
    let file_info = state.files.store("test.txt", b"Some training data content here");

    // Create recipe referencing the file
    let recipe = state.recipes.create(
        "run-test",
        vec![file_info.id],
        vec![RecipeStep { step_type: StepType::ExtractText, config: serde_json::json!({}) }],
        "jsonl",
    );

    // Run — note: in-memory FileStore doesn't persist content, so source will be empty
    // This tests the endpoint plumbing, not file content
    let app = super::router::create_banco_router(state);
    let response = app
        .oneshot(
            Request::post(&format!("/api/v1/data/recipes/{}/run", recipe.id))
                .body(Body::empty())
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert!(json["dataset_id"].as_str().expect("id").starts_with("ds-"));
}
