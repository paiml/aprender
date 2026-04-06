//! HuggingFace leaderboard HTTP client
//!
//! Fetches leaderboard data from the HuggingFace datasets-server JSON API.
//! Uses the rows endpoint to avoid Parquet parsing.

use super::types::{HfLeaderboard, LeaderboardEntry, LeaderboardKind};
use crate::hf_pipeline::error::FetchError;
use crate::hf_pipeline::HfModelFetcher;

/// HTTP client for fetching HuggingFace leaderboard data
pub struct LeaderboardClient {
    token: Option<String>,
    client: reqwest::blocking::Client,
}

impl LeaderboardClient {
    /// Create a new leaderboard client with automatic token resolution
    pub fn new() -> Result<Self, FetchError> {
        let token = HfModelFetcher::resolve_token();
        let client =
            reqwest::blocking::Client::builder().user_agent("entrenar/0.5").build().map_err(
                |e| FetchError::HttpError { message: format!("Failed to create HTTP client: {e}") },
            )?;

        Ok(Self { token, client })
    }

    /// Create a client with an explicit token
    pub fn with_token(token: impl Into<String>) -> Result<Self, FetchError> {
        let client =
            reqwest::blocking::Client::builder().user_agent("entrenar/0.5").build().map_err(
                |e| FetchError::HttpError { message: format!("Failed to create HTTP client: {e}") },
            )?;

        Ok(Self { token: Some(token.into()), client })
    }

    /// Fetch leaderboard data (first page)
    pub fn fetch(&self, kind: LeaderboardKind) -> Result<HfLeaderboard, FetchError> {
        self.fetch_paginated(kind, 0, 100)
    }

    /// Fetch leaderboard data with pagination
    pub fn fetch_paginated(
        &self,
        kind: LeaderboardKind,
        offset: usize,
        limit: usize,
    ) -> Result<HfLeaderboard, FetchError> {
        let repo_id = kind.dataset_repo_id();
        let url = format!(
            "https://datasets-server.huggingface.co/rows?dataset={repo_id}&config=default&split=train&offset={offset}&length={limit}"
        );

        let mut request = self.client.get(&url);
        if let Some(token) = &self.token {
            request = request.bearer_auth(token);
        }

        let response = request.send().map_err(|e| FetchError::HttpError {
            message: format!("Leaderboard request failed: {e}"),
        })?;

        if !response.status().is_success() {
            let status = response.status();
            if status.as_u16() == 404 {
                return Err(FetchError::LeaderboardNotFound { kind: kind.to_string() });
            }
            return Err(FetchError::HttpError {
                message: format!("Leaderboard API returned {status} for {repo_id}"),
            });
        }

        let body: serde_json::Value = response.json().map_err(|e| FetchError::HttpError {
            message: format!("Failed to parse leaderboard JSON: {e}"),
        })?;
        parse_response(kind, &body)
    }

    /// Find a specific model in a leaderboard
    pub fn find_model(
        &self,
        kind: LeaderboardKind,
        model_repo_id: &str,
    ) -> Result<Option<LeaderboardEntry>, FetchError> {
        // Fetch the full leaderboard and search locally
        // (HF datasets-server doesn't support server-side filtering by row content)
        let leaderboard = self.fetch(kind)?;
        Ok(leaderboard.find_model(model_repo_id).cloned())
    }
}

/// Parse HF datasets-server JSON response into our types
fn parse_response(
    kind: LeaderboardKind,
    body: &serde_json::Value,
) -> Result<HfLeaderboard, FetchError> {
    let mut leaderboard = HfLeaderboard::new(kind);

    // Extract total count from "num_rows_total"
    leaderboard.total_count =
        body.get("num_rows_total").and_then(serde_json::Value::as_u64).unwrap_or(0) as usize;

    // Extract rows
    let rows = body.get("rows").and_then(|v| v.as_array()).ok_or_else(|| {
        FetchError::LeaderboardParseError {
            message: "Missing 'rows' array in response".to_string(),
        }
    })?;

    for row in rows {
        let row_data = row.get("row").unwrap_or(row);

        // Try to extract model ID from common column names
        let model_id = row_data
            .get("model")
            .or_else(|| row_data.get("model_id"))
            .or_else(|| row_data.get("model_name"))
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();

        let mut entry = LeaderboardEntry::new(model_id);

        // Extract all numeric values as scores
        if let Some(obj) = row_data.as_object() {
            for (key, value) in obj {
                if let Some(num) = value.as_f64() {
                    entry.scores.insert(key.clone(), num);
                } else if let Some(s) = value.as_str() {
                    entry.metadata.insert(key.clone(), s.to_string());
                }
            }
        }

        leaderboard.entries.push(entry);
    }

    Ok(leaderboard)
}

impl std::fmt::Debug for LeaderboardClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LeaderboardClient")
            .field("has_token", &self.token.is_some())
            .finish_non_exhaustive()
    }
}
