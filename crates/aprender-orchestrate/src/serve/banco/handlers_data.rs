//! Data management handlers — file upload, list, delete.

use axum::{
    extract::{Multipart, State},
    http::StatusCode,
    response::Json,
};

use super::state::BancoState;
use super::storage::FileInfo;
use super::types::ErrorResponse;

/// POST /api/v1/data/upload — upload files via multipart form.
pub async fn upload_handler(
    State(state): State<BancoState>,
    mut multipart: Multipart,
) -> Result<Json<Vec<FileInfo>>, (StatusCode, Json<ErrorResponse>)> {
    let mut uploaded = Vec::new();

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.file_name().unwrap_or("unnamed").to_string();
        let data = field.bytes().await.map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse::new(format!("Failed to read field: {e}"), "upload_error", 400)),
            )
        })?;

        if data.is_empty() {
            continue;
        }

        let info = state.files.store(&name, &data);
        // Auto-index for RAG
        let text = String::from_utf8_lossy(&data);
        state.rag.index_document(&info.id, &info.name, &text);
        // Emit event
        state.events.emit(&super::events::BancoEvent::FileUploaded {
            file_id: info.id.clone(),
            name: info.name.clone(),
        });
        uploaded.push(info);
    }

    if uploaded.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new("No files uploaded", "no_files", 400)),
        ));
    }

    Ok(Json(uploaded))
}

/// POST /api/v1/data/upload/json — upload a file via JSON body (simpler for testing).
pub async fn upload_json_handler(
    State(state): State<BancoState>,
    Json(request): Json<UploadJsonRequest>,
) -> Json<FileInfo> {
    let info = state.files.store(&request.name, request.content.as_bytes());
    // Auto-index for RAG
    state.rag.index_document(&info.id, &info.name, &request.content);
    // Emit event
    state.events.emit(&super::events::BancoEvent::FileUploaded {
        file_id: info.id.clone(),
        name: info.name.clone(),
    });
    Json(info)
}

/// JSON upload request (alternative to multipart).
#[derive(Debug, serde::Deserialize)]
pub struct UploadJsonRequest {
    pub name: String,
    pub content: String,
}

/// GET /api/v1/data/files — list all uploaded files.
pub async fn list_files_handler(State(state): State<BancoState>) -> Json<FilesListResponse> {
    Json(FilesListResponse { files: state.files.list() })
}

/// DELETE /api/v1/data/files/:id — delete an uploaded file.
pub async fn delete_file_handler(
    State(state): State<BancoState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<StatusCode, (StatusCode, Json<ErrorResponse>)> {
    state.files.delete(&id).map(|()| StatusCode::NO_CONTENT).map_err(|_| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse::new(format!("File {id} not found"), "not_found", 404)),
        )
    })
}

/// GET /api/v1/data/files/:id/info — file details + schema (for structured files).
pub async fn file_info_handler(
    State(state): State<BancoState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<Json<FileInfoDetail>, (StatusCode, Json<ErrorResponse>)> {
    let info = state.files.get(&id).ok_or((
        StatusCode::NOT_FOUND,
        Json(ErrorResponse::new(format!("File {id} not found"), "not_found", 404)),
    ))?;

    let content = state.files.read_content(&id);
    let preview_lines: Vec<String> = content
        .as_ref()
        .map(|bytes| String::from_utf8_lossy(bytes).lines().take(5).map(String::from).collect())
        .unwrap_or_default();

    let schema = detect_schema(&info.content_type, content.as_deref());

    Ok(Json(FileInfoDetail { info, preview_lines, schema }))
}

/// Detect schema for structured file types.
#[cfg(feature = "alimentar")]
fn detect_schema(content_type: &str, content: Option<&[u8]>) -> Option<Vec<SchemaField>> {
    use alimentar::{ArrowDataset, Dataset};

    let bytes = content?;
    let text = std::str::from_utf8(bytes).ok()?;

    let dataset = match content_type {
        "text/csv" => ArrowDataset::from_csv_str(text).ok()?,
        "application/json" | "application/jsonl" => ArrowDataset::from_json_str(text).ok()?,
        _ => return None,
    };

    let schema = dataset.schema();
    Some(
        schema
            .fields()
            .iter()
            .map(|f| SchemaField {
                name: f.name().clone(),
                data_type: format!("{:?}", f.data_type()),
                nullable: f.is_nullable(),
            })
            .collect(),
    )
}

/// Schema detection fallback (no alimentar).
#[cfg(not(feature = "alimentar"))]
fn detect_schema(content_type: &str, content: Option<&[u8]>) -> Option<Vec<SchemaField>> {
    let bytes = content?;
    let text = std::str::from_utf8(bytes).ok()?;

    match content_type {
        "text/csv" => {
            let header = text.lines().next()?;
            Some(
                header
                    .split(',')
                    .map(|col| SchemaField {
                        name: col.trim().to_string(),
                        data_type: "Utf8".to_string(),
                        nullable: true,
                    })
                    .collect(),
            )
        }
        _ => None,
    }
}

/// File list response.
#[derive(Debug, serde::Serialize)]
pub struct FilesListResponse {
    pub files: Vec<FileInfo>,
}

/// Detailed file info with preview and schema.
#[derive(Debug, serde::Serialize)]
pub struct FileInfoDetail {
    #[serde(flatten)]
    pub info: FileInfo,
    pub preview_lines: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema: Option<Vec<SchemaField>>,
}

/// Schema field descriptor.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SchemaField {
    pub name: String,
    pub data_type: String,
    pub nullable: bool,
}
