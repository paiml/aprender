//! Recipe endpoint handlers — create, list, get, run recipes.

use axum::{extract::State, http::StatusCode, response::Json};
use serde::Deserialize;

use super::recipes::{DatasetResult, Recipe, RecipeStep};
use super::state::BancoState;
use super::types::ErrorResponse;

/// POST /api/v1/data/recipes — create a recipe.
pub async fn create_recipe_handler(
    State(state): State<BancoState>,
    Json(request): Json<CreateRecipeRequest>,
) -> Json<Recipe> {
    let recipe = state.recipes.create(
        &request.name,
        request.source_files,
        request.steps,
        &request.output_format.unwrap_or_else(|| "jsonl".to_string()),
    );
    Json(recipe)
}

/// GET /api/v1/data/recipes — list recipes.
pub async fn list_recipes_handler(State(state): State<BancoState>) -> Json<RecipesListResponse> {
    Json(RecipesListResponse { recipes: state.recipes.list() })
}

/// GET /api/v1/data/recipes/:id — get recipe by ID.
pub async fn get_recipe_handler(
    State(state): State<BancoState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<Json<Recipe>, (StatusCode, Json<ErrorResponse>)> {
    state.recipes.get(&id).map(Json).ok_or((
        StatusCode::NOT_FOUND,
        Json(ErrorResponse::new(format!("Recipe {id} not found"), "not_found", 404)),
    ))
}

/// POST /api/v1/data/recipes/:id/run — execute a recipe.
pub async fn run_recipe_handler(
    State(state): State<BancoState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<Json<DatasetResult>, (StatusCode, Json<ErrorResponse>)> {
    let recipe = state.recipes.get(&id).ok_or((
        StatusCode::NOT_FOUND,
        Json(ErrorResponse::new(format!("Recipe {id} not found"), "not_found", 404)),
    ))?;

    // Gather source texts from uploaded files
    let source_texts: Vec<(String, String)> = recipe
        .source_files
        .iter()
        .filter_map(|file_id| {
            let info = state.files.get(file_id)?;
            // For in-memory store, read content; for disk, read from file
            let content = state
                .files
                .read_content(file_id)
                .map(|bytes| String::from_utf8_lossy(&bytes).to_string())
                .unwrap_or_default();
            Some((info.name, content))
        })
        .collect();

    let source_refs: Vec<(&str, &str)> =
        source_texts.iter().map(|(n, c)| (n.as_str(), c.as_str())).collect();

    state.recipes.run(&id, &source_refs).map(Json).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse::new(e.to_string(), "recipe_error", 500)),
        )
    })
}

/// GET /api/v1/data/datasets — list datasets.
pub async fn list_datasets_handler(State(state): State<BancoState>) -> Json<DatasetsListResponse> {
    Json(DatasetsListResponse { datasets: state.recipes.list_datasets() })
}

/// GET /api/v1/data/datasets/:id/preview — preview dataset rows.
pub async fn preview_dataset_handler(
    State(state): State<BancoState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<Json<DatasetPreview>, (StatusCode, Json<ErrorResponse>)> {
    let dataset = state.recipes.get_dataset(&id).ok_or((
        StatusCode::NOT_FOUND,
        Json(ErrorResponse::new(format!("Dataset {id} not found"), "not_found", 404)),
    ))?;

    let preview_records: Vec<_> = dataset.records.iter().take(10).cloned().collect();
    Ok(Json(DatasetPreview {
        dataset_id: dataset.dataset_id,
        total_records: dataset.record_count,
        preview: preview_records,
    }))
}

// ============================================================================
// Request/Response types
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct CreateRecipeRequest {
    pub name: String,
    #[serde(default)]
    pub source_files: Vec<String>,
    pub steps: Vec<RecipeStep>,
    #[serde(default)]
    pub output_format: Option<String>,
}

#[derive(Debug, serde::Serialize)]
pub struct RecipesListResponse {
    pub recipes: Vec<Recipe>,
}

#[derive(Debug, serde::Serialize)]
pub struct DatasetsListResponse {
    pub datasets: Vec<DatasetResult>,
}

#[derive(Debug, serde::Serialize)]
pub struct DatasetPreview {
    pub dataset_id: String,
    pub total_records: usize,
    pub preview: Vec<super::recipes::Record>,
}
