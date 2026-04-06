//! Runtime config endpoint handlers.

use axum::{extract::State, response::Json};

use super::config::BancoConfig;
use super::state::BancoState;

/// GET /api/v1/config — read current configuration.
pub async fn get_config_handler(State(_state): State<BancoState>) -> Json<BancoConfig> {
    Json(BancoConfig::load())
}

/// PUT /api/v1/config — update configuration and persist to disk.
pub async fn update_config_handler(
    State(_state): State<BancoState>,
    Json(config): Json<BancoConfig>,
) -> Json<ConfigUpdateResponse> {
    let saved = config.save().is_ok();
    Json(ConfigUpdateResponse { saved, config })
}

/// Response from config update.
#[derive(Debug, serde::Serialize)]
pub struct ConfigUpdateResponse {
    pub saved: bool,
    pub config: BancoConfig,
}
