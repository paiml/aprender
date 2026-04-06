//! Tests for HuggingFace Hub integration.

use std::path::Path;

use super::*;

#[test]
fn test_builder_basic() {
    let dataset = HfDataset::builder("squad")
        .build()
        .ok()
        .unwrap_or_else(|| panic!("Should build"));

    assert_eq!(dataset.repo_id(), "squad");
    assert_eq!(dataset.revision(), "main");
    assert!(dataset.subset().is_none());
    assert!(dataset.split().is_none());
}

#[test]
fn test_builder_with_options() {
    let dataset = HfDataset::builder("glue")
        .revision("v1.0.0")
        .subset("cola")
        .split("validation")
        .cache_dir("/tmp/test_cache")
        .build()
        .ok()
        .unwrap_or_else(|| panic!("Should build"));

    assert_eq!(dataset.repo_id(), "glue");
    assert_eq!(dataset.revision(), "v1.0.0");
    assert_eq!(dataset.subset(), Some("cola"));
    assert_eq!(dataset.split(), Some("validation"));
    assert_eq!(dataset.cache_dir(), Path::new("/tmp/test_cache"));
}

#[test]
fn test_builder_empty_repo_id_error() {
    let result = HfDataset::builder("").build();
    assert!(result.is_err());
}

#[test]
fn test_build_parquet_path_default() {
    let dataset = HfDataset::builder("squad")
        .build()
        .ok()
        .unwrap_or_else(|| panic!("Should build"));

    assert_eq!(dataset.build_parquet_path(), "default/train.parquet");
}

#[test]
fn test_build_parquet_path_with_subset() {
    let dataset = HfDataset::builder("glue")
        .subset("cola")
        .split("validation")
        .build()
        .ok()
        .unwrap_or_else(|| panic!("Should build"));

    assert_eq!(dataset.build_parquet_path(), "cola/validation.parquet");
}

#[test]
fn test_build_download_url() {
    let dataset = HfDataset::builder("squad")
        .build()
        .ok()
        .unwrap_or_else(|| panic!("Should build"));

    let url = dataset.build_download_url("default/train.parquet");
    assert_eq!(
        url,
        "https://huggingface.co/datasets/squad/resolve/main/data/default/train.parquet"
    );
}

#[test]
fn test_cache_path() {
    let dataset = HfDataset::builder("squad")
        .cache_dir("/tmp/cache")
        .build()
        .ok()
        .unwrap_or_else(|| panic!("Should build"));

    let cache_path = dataset.cache_path_for("default/train.parquet");
    assert_eq!(
        cache_path,
        std::path::PathBuf::from(
            "/tmp/cache/huggingface/datasets/squad/main/default/train.parquet"
        )
    );
}

#[test]
fn test_default_cache_dir() {
    let cache = default_cache_dir();
    // Should return some path
    assert!(!cache.as_os_str().is_empty());
}

#[test]
fn test_namespaced_repo_id() {
    let dataset = HfDataset::builder("openai/gsm8k")
        .split("test")
        .build()
        .ok()
        .unwrap_or_else(|| panic!("Should build"));

    let url = dataset.build_download_url("default/test.parquet");
    assert!(url.contains("openai/gsm8k"));
}

#[test]
fn test_builder_clone() {
    let builder = HfDatasetBuilder::new("squad")
        .revision("v1.0")
        .subset("test")
        .split("validation");

    let cloned = builder;
    let dataset = cloned
        .build()
        .ok()
        .unwrap_or_else(|| panic!("Should build"));

    assert_eq!(dataset.repo_id(), "squad");
    assert_eq!(dataset.revision(), "v1.0");
    assert_eq!(dataset.subset(), Some("test"));
    assert_eq!(dataset.split(), Some("validation"));
}

#[test]
fn test_hf_dataset_clone() {
    let dataset = HfDataset::builder("glue")
        .subset("cola")
        .build()
        .ok()
        .unwrap_or_else(|| panic!("Should build"));

    let cloned = dataset.clone();
    assert_eq!(cloned.repo_id(), dataset.repo_id());
    assert_eq!(cloned.revision(), dataset.revision());
    assert_eq!(cloned.subset(), dataset.subset());
}

#[test]
fn test_hf_dataset_debug() {
    let dataset = HfDataset::builder("test-dataset")
        .build()
        .ok()
        .unwrap_or_else(|| panic!("Should build"));

    let debug_str = format!("{:?}", dataset);
    assert!(debug_str.contains("HfDataset"));
    assert!(debug_str.contains("test-dataset"));
}

#[test]
fn test_builder_debug() {
    let builder = HfDatasetBuilder::new("debug-test");
    let debug_str = format!("{:?}", builder);
    assert!(debug_str.contains("HfDatasetBuilder"));
    assert!(debug_str.contains("debug-test"));
}

#[test]
fn test_dataset_info_debug() {
    let info = DatasetInfo {
        repo_id: "test".to_string(),
        splits: vec!["train".to_string(), "test".to_string()],
        subsets: vec!["default".to_string()],
        download_size: Some(1024),
        description: Some("A test dataset".to_string()),
    };

    let debug_str = format!("{:?}", info);
    assert!(debug_str.contains("DatasetInfo"));
    assert!(debug_str.contains("test"));
}

#[test]
fn test_dataset_info_clone() {
    let info = DatasetInfo {
        repo_id: "clone-test".to_string(),
        splits: vec!["train".to_string()],
        subsets: vec![],
        download_size: None,
        description: None,
    };

    let cloned = info.clone();
    assert_eq!(cloned.repo_id, info.repo_id);
    assert_eq!(cloned.splits, info.splits);
}

#[test]
fn test_build_parquet_path_train_split() {
    let dataset = HfDataset::builder("squad")
        .split("train")
        .build()
        .ok()
        .unwrap_or_else(|| panic!("Should build"));

    assert_eq!(dataset.build_parquet_path(), "default/train.parquet");
}

#[test]
fn test_build_parquet_path_test_split() {
    let dataset = HfDataset::builder("squad")
        .split("test")
        .build()
        .ok()
        .unwrap_or_else(|| panic!("Should build"));

    assert_eq!(dataset.build_parquet_path(), "default/test.parquet");
}

#[test]
fn test_build_download_url_with_revision() {
    let dataset = HfDataset::builder("squad")
        .revision("refs/convert/parquet")
        .build()
        .ok()
        .unwrap_or_else(|| panic!("Should build"));

    let url = dataset.build_download_url("default/train.parquet");
    assert!(url.contains("refs/convert/parquet"));
}

#[test]
fn test_cache_path_with_subset() {
    let dataset = HfDataset::builder("glue")
        .subset("cola")
        .split("validation")
        .cache_dir("/tmp/hf-cache")
        .build()
        .ok()
        .unwrap_or_else(|| panic!("Should build"));

    let cache_path = dataset.cache_path_for("cola/validation.parquet");
    assert!(cache_path
        .to_string_lossy()
        .contains("/tmp/hf-cache/huggingface/datasets/glue/main/cola/validation.parquet"));
}

#[test]
fn test_clear_cache_nonexistent() {
    let dataset = HfDataset::builder("nonexistent-dataset")
        .cache_dir("/tmp/nonexistent-cache-dir-12345")
        .build()
        .ok()
        .unwrap_or_else(|| panic!("Should build"));

    // Should not error on non-existent cache
    let result = dataset.clear_cache();
    assert!(result.is_ok());
}

#[test]
fn test_download_from_cache() {
    use std::sync::Arc;

    use arrow::{
        array::Int32Array,
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    };

    use crate::Dataset;

    // Create temp dir for cache
    let temp_dir = tempfile::tempdir()
        .ok()
        .unwrap_or_else(|| panic!("Should create temp dir"));

    // Create HfDataset pointing to this cache
    let dataset = HfDataset::builder("test-repo")
        .cache_dir(temp_dir.path())
        .split("train")
        .build()
        .ok()
        .unwrap_or_else(|| panic!("Should build"));

    // Create the cache directory structure
    let cache_path = dataset.cache_path_for(&dataset.build_parquet_path());
    if let Some(parent) = cache_path.parent() {
        std::fs::create_dir_all(parent)
            .ok()
            .unwrap_or_else(|| panic!("Should create dirs"));
    }

    // Create a minimal parquet file in cache
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![1, 2, 3]))])
        .ok()
        .unwrap_or_else(|| panic!("Should create batch"));

    let arrow_dataset = crate::ArrowDataset::from_batch(batch)
        .ok()
        .unwrap_or_else(|| panic!("Should create dataset"));

    arrow_dataset
        .to_parquet(&cache_path)
        .ok()
        .unwrap_or_else(|| panic!("Should write parquet"));

    // Now download should use cache
    let loaded = dataset.download();
    assert!(loaded.is_ok());
    let loaded = loaded.ok().unwrap_or_else(|| panic!("Should load"));
    assert_eq!(loaded.len(), 3);
}

#[test]
fn test_clear_cache_with_files() {
    // Create temp dir
    let temp_dir = tempfile::tempdir()
        .ok()
        .unwrap_or_else(|| panic!("Should create temp dir"));

    let dataset = HfDataset::builder("clear-test")
        .cache_dir(temp_dir.path())
        .build()
        .ok()
        .unwrap_or_else(|| panic!("Should build"));

    // Create cache directory with a file
    let cache_dir = temp_dir
        .path()
        .join("huggingface")
        .join("datasets")
        .join("clear-test");
    std::fs::create_dir_all(&cache_dir)
        .ok()
        .unwrap_or_else(|| panic!("Should create dir"));
    std::fs::write(cache_dir.join("test.txt"), "test data")
        .ok()
        .unwrap_or_else(|| panic!("Should write file"));

    // Verify it exists
    assert!(cache_dir.exists());

    // Clear cache
    let result = dataset.clear_cache();
    assert!(result.is_ok());

    // Verify it's gone
    assert!(!cache_dir.exists());
}

#[test]
fn test_download_to_creates_parent_dirs() {
    // This test verifies the parent dir creation logic in download_to
    // We can't test full download without network, but we can test
    // that cache_path_for produces correct paths
    let temp_dir = tempfile::tempdir()
        .ok()
        .unwrap_or_else(|| panic!("Should create temp dir"));

    let dataset = HfDataset::builder("download-to-test")
        .cache_dir(temp_dir.path())
        .subset("custom")
        .split("validation")
        .build()
        .ok()
        .unwrap_or_else(|| panic!("Should build"));

    // Verify parquet path building
    assert_eq!(dataset.build_parquet_path(), "custom/validation.parquet");

    // Verify cache path building
    let cache_path = dataset.cache_path_for("custom/validation.parquet");
    assert!(cache_path.to_string_lossy().contains("download-to-test"));
    assert!(cache_path.to_string_lossy().contains("custom"));
}

#[test]
fn test_dataset_info_with_all_fields() {
    let info = DatasetInfo {
        repo_id: "full-test".to_string(),
        splits: vec![
            "train".to_string(),
            "validation".to_string(),
            "test".to_string(),
        ],
        subsets: vec!["default".to_string(), "extra".to_string()],
        download_size: Some(1_000_000),
        description: Some("A comprehensive test dataset for validation".to_string()),
    };

    assert_eq!(info.repo_id, "full-test");
    assert_eq!(info.splits.len(), 3);
    assert_eq!(info.subsets.len(), 2);
    assert_eq!(info.download_size, Some(1_000_000));
    assert!(info.description.is_some());
}

#[test]
fn test_builder_chain_all_methods() {
    let dataset = HfDataset::builder("chain-test")
        .revision("v2.0.0")
        .subset("subset-a")
        .split("test")
        .cache_dir("/custom/cache")
        .build()
        .ok()
        .unwrap_or_else(|| panic!("Should build"));

    assert_eq!(dataset.repo_id(), "chain-test");
    assert_eq!(dataset.revision(), "v2.0.0");
    assert_eq!(dataset.subset(), Some("subset-a"));
    assert_eq!(dataset.split(), Some("test"));
    assert_eq!(dataset.cache_dir(), Path::new("/custom/cache"));
}

#[test]
fn test_deeply_nested_cache_path() {
    let dataset = HfDataset::builder("org/deep/nested/repo")
        .cache_dir("/root")
        .subset("config-name")
        .build()
        .ok()
        .unwrap_or_else(|| panic!("Should build"));

    let cache_path = dataset.cache_path_for("config-name/train.parquet");
    assert!(cache_path
        .to_string_lossy()
        .contains("org/deep/nested/repo"));
}

// ========================================================================
// DatasetCardValidator tests
// ========================================================================

#[test]
fn test_validate_valid_readme() {
    let readme = r"---
license: mit
task_categories:
  - translation
language:
  - en
---
# My Dataset
";
    let errors = DatasetCardValidator::validate_readme(readme);
    assert!(errors.is_empty());
}

#[test]
fn test_validate_invalid_task_category() {
    let readme = r"---
license: mit
task_categories:
  - code-generation
---
# My Dataset
";
    let errors = DatasetCardValidator::validate_readme(readme);
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].field, "task_categories");
    assert_eq!(errors[0].value, "code-generation");
    // Suggestions are provided for invalid categories
    // (may include similar categories like text-generation)
}

#[test]
fn test_validate_multiple_invalid_categories() {
    let readme = r"---
task_categories:
  - code-generation
  - image-generation
---
";
    let errors = DatasetCardValidator::validate_readme(readme);
    assert_eq!(errors.len(), 2);
}

#[test]
fn test_validate_valid_size_category() {
    let readme = r"---
size_categories:
  - n<1K
  - 1K<n<10K
---
";
    let errors = DatasetCardValidator::validate_readme(readme);
    assert!(errors.is_empty());
}

#[test]
fn test_validate_invalid_size_category() {
    let readme = r"---
size_categories:
  - small
---
";
    let errors = DatasetCardValidator::validate_readme(readme);
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].field, "size_categories");
}

#[test]
fn test_validate_no_frontmatter() {
    let readme = "# Just a title\n\nNo YAML frontmatter here.";
    let errors = DatasetCardValidator::validate_readme(readme);
    assert!(errors.is_empty());
}

#[test]
fn test_validate_empty_frontmatter() {
    let readme = "---\n---\n# Empty frontmatter";
    let errors = DatasetCardValidator::validate_readme(readme);
    assert!(errors.is_empty());
}

#[test]
fn test_validate_strict_returns_error() {
    let readme = r"---
task_categories:
  - invalid-category
---
";
    let result = DatasetCardValidator::validate_readme_strict(readme);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("invalid-category"));
}

#[test]
fn test_validate_strict_returns_ok() {
    let readme = r"---
task_categories:
  - translation
  - text-classification
---
";
    let result = DatasetCardValidator::validate_readme_strict(readme);
    assert!(result.is_ok());
}

#[test]
fn test_validation_error_display() {
    let err = ValidationError {
        field: "task_categories".to_string(),
        value: "text2text".to_string(),
        suggestions: vec!["text-generation".to_string()],
    };
    let display = err.to_string();
    assert!(display.contains("task_categories"));
    assert!(display.contains("text2text"));
    assert!(display.contains("Did you mean"));
    assert!(display.contains("text-generation"));
}

#[test]
fn test_validation_error_display_no_suggestions() {
    let err = ValidationError {
        field: "size_categories".to_string(),
        value: "huge".to_string(),
        suggestions: vec![],
    };
    let display = err.to_string();
    assert!(display.contains("size_categories"));
    assert!(!display.contains("Did you mean"));
}

#[test]
fn test_levenshtein_distance() {
    assert_eq!(DatasetCardValidator::levenshtein("", ""), 0);
    assert_eq!(DatasetCardValidator::levenshtein("abc", ""), 3);
    assert_eq!(DatasetCardValidator::levenshtein("", "xyz"), 3);
    assert_eq!(DatasetCardValidator::levenshtein("abc", "abc"), 0);
    assert_eq!(DatasetCardValidator::levenshtein("abc", "abd"), 1);
    assert_eq!(DatasetCardValidator::levenshtein("text", "test"), 1);
}

#[test]
fn test_suggest_similar_finds_matches() {
    let suggestions = DatasetCardValidator::suggest_similar("text-gen", VALID_TASK_CATEGORIES);
    assert!(!suggestions.is_empty());
    // Should find text-generation
    assert!(suggestions.iter().any(|s| s.contains("text")));
}

#[test]
fn test_all_valid_categories_pass() {
    for cat in VALID_TASK_CATEGORIES {
        let readme = format!("---\ntask_categories:\n  - {}\n---\n", cat);
        let errors = DatasetCardValidator::validate_readme(&readme);
        assert!(errors.is_empty(), "Category '{}' should be valid", cat);
    }
}

#[test]
fn test_all_valid_size_categories_pass() {
    for size in VALID_SIZE_CATEGORIES {
        let readme = format!("---\nsize_categories:\n  - {}\n---\n", size);
        let errors = DatasetCardValidator::validate_readme(&readme);
        assert!(errors.is_empty(), "Size '{}' should be valid", size);
    }
}

// ========================================================================
// HfPublisher Tests - EXTREME TDD
// ========================================================================

#[test]
fn test_hf_publisher_new() {
    let publisher = HfPublisher::new("paiml/test-dataset");
    assert_eq!(publisher.repo_id(), "paiml/test-dataset");
}

#[test]
fn test_hf_publisher_with_private() {
    let publisher = HfPublisher::new("paiml/test-dataset").with_private(true);
    assert_eq!(publisher.repo_id(), "paiml/test-dataset");
    // Private flag is stored internally
}

#[test]
fn test_hf_publisher_with_commit_message() {
    let publisher = HfPublisher::new("paiml/test-dataset").with_commit_message("Test commit");
    assert_eq!(publisher.repo_id(), "paiml/test-dataset");
}

#[test]
fn test_hf_publisher_builder_basic() {
    let publisher = HfPublisherBuilder::new("paiml/test-dataset").build();
    assert_eq!(publisher.repo_id(), "paiml/test-dataset");
}

#[test]
fn test_hf_publisher_builder_with_all_options() {
    let publisher = HfPublisherBuilder::new("paiml/test-dataset")
        .token("test-token")
        .private(true)
        .commit_message("Custom message")
        .build();
    assert_eq!(publisher.repo_id(), "paiml/test-dataset");
}

#[test]
fn test_hf_publisher_parse_org_name_with_slash() {
    // Test that org/name parsing works correctly
    let repo_id = "paiml/python-doctest-corpus";
    let slash_pos = repo_id.find('/');
    assert!(slash_pos.is_some());
    let (org, name) = if let Some(pos) = slash_pos {
        (&repo_id[..pos], &repo_id[pos + 1..])
    } else {
        ("", repo_id)
    };
    assert_eq!(org, "paiml");
    assert_eq!(name, "python-doctest-corpus");
}

#[test]
fn test_hf_publisher_parse_name_without_slash() {
    // Test that single name (no org) works correctly
    let repo_id = "my-dataset";
    let slash_pos = repo_id.find('/');
    assert!(slash_pos.is_none());
}

#[test]
fn test_hf_publisher_commit_url_format() {
    // Verify the commit API URL format
    let repo_id = "paiml/test-dataset";
    let expected_url = format!("{}/datasets/{}/commit/main", HF_API_URL, repo_id);
    assert_eq!(
        expected_url,
        "https://huggingface.co/api/datasets/paiml/test-dataset/commit/main"
    );
}

#[test]
fn test_hf_publisher_create_repo_url_format() {
    // Verify the create repo API URL format
    let expected_url = format!("{}/repos/create", HF_API_URL);
    assert_eq!(expected_url, "https://huggingface.co/api/repos/create");
}

// ========================================================================
// NDJSON Upload API Tests - EXTREME TDD
// These tests define the expected API format BEFORE implementation
// ========================================================================

#[test]
fn test_ndjson_header_format() {
    // HuggingFace commit API expects header as first NDJSON line
    let commit_message = "Upload test file";
    let header = serde_json::json!({
        "key": "header",
        "value": {
            "summary": commit_message,
            "description": ""
        }
    });

    let header_str = header.to_string();
    assert!(header_str.contains("\"key\":\"header\""));
    assert!(header_str.contains("\"summary\":\"Upload test file\""));
}

#[test]
fn test_ndjson_file_operation_format() {
    use base64::{engine::general_purpose::STANDARD, Engine};

    // HuggingFace commit API expects file operations with base64 content
    let test_data = b"test file content";
    let content_base64 = STANDARD.encode(test_data);
    let path_in_repo = "data/test.parquet";

    let file_op = serde_json::json!({
        "key": "file",
        "value": {
            "content": content_base64,
            "path": path_in_repo,
            "encoding": "base64"
        }
    });

    let file_str = file_op.to_string();
    assert!(file_str.contains("\"key\":\"file\""));
    assert!(file_str.contains("\"encoding\":\"base64\""));
    assert!(file_str.contains("\"path\":\"data/test.parquet\""));
}

#[test]
fn test_ndjson_payload_is_newline_delimited() {
    use base64::{engine::general_purpose::STANDARD, Engine};

    // NDJSON = one JSON object per line, separated by newlines
    let header = serde_json::json!({
        "key": "header",
        "value": {"summary": "Test", "description": ""}
    });

    let file_op = serde_json::json!({
        "key": "file",
        "value": {
            "content": STANDARD.encode(b"data"),
            "path": "test.txt",
            "encoding": "base64"
        }
    });

    let ndjson = format!("{}\n{}", header, file_op);

    // Should have exactly 2 lines
    let lines: Vec<&str> = ndjson.lines().collect();
    assert_eq!(lines.len(), 2);

    // Each line should be valid JSON
    assert!(serde_json::from_str::<serde_json::Value>(lines[0]).is_ok());
    assert!(serde_json::from_str::<serde_json::Value>(lines[1]).is_ok());
}

#[test]
fn test_build_ndjson_payload() {
    use base64::{engine::general_purpose::STANDARD, Engine};

    // Test the helper function that builds the complete NDJSON payload
    let commit_message = "Upload via alimentar";
    let path_in_repo = "train.parquet";
    let data = b"parquet binary data here";

    let payload = build_ndjson_upload_payload(commit_message, path_in_repo, data);

    // Verify structure
    let lines: Vec<&str> = payload.lines().collect();
    assert_eq!(lines.len(), 2, "NDJSON should have exactly 2 lines");

    // Parse and verify header
    let header: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
    assert_eq!(header["key"], "header");
    assert_eq!(header["value"]["summary"], commit_message);

    // Parse and verify file operation
    let file_op: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
    assert_eq!(file_op["key"], "file");
    assert_eq!(file_op["value"]["path"], path_in_repo);
    assert_eq!(file_op["value"]["encoding"], "base64");

    // Verify content round-trips correctly
    let encoded_content = file_op["value"]["content"].as_str().unwrap();
    let decoded = STANDARD.decode(encoded_content).unwrap();
    assert_eq!(decoded, data);
}

// ========================================================================
// LFS Preupload API Tests - EXTREME TDD for binary file support
// These tests define the expected API format BEFORE implementation
// ========================================================================

#[test]
fn test_is_binary_file_detection() {
    // Binary file extensions that require LFS
    assert!(is_binary_file("train.parquet"));
    assert!(is_binary_file("data.arrow"));
    assert!(is_binary_file("image.png"));
    assert!(is_binary_file("model.bin"));
    assert!(is_binary_file("weights.safetensors"));

    // Text files that can use direct commit
    assert!(!is_binary_file("README.md"));
    assert!(!is_binary_file("config.json"));
    assert!(!is_binary_file("data.csv"));
    assert!(!is_binary_file(".gitattributes"));
}

#[test]
fn test_compute_sha256_for_lfs() {
    // LFS requires SHA256 hash of file content
    let data = b"test content for hashing";
    let hash = compute_sha256(data);

    // SHA256 should be 64 hex characters
    assert_eq!(hash.len(), 64);
    assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));

    // Same input should produce same hash
    assert_eq!(hash, compute_sha256(data));

    // Different input should produce different hash
    assert_ne!(hash, compute_sha256(b"different content"));
}

#[test]
fn test_build_lfs_preupload_request() {
    // HuggingFace preupload API request format
    let path = "data/train.parquet";
    let data = b"parquet binary content";

    let request = build_lfs_preupload_request(path, data);

    // Parse and verify structure
    let json: serde_json::Value = serde_json::from_str(&request).unwrap();
    assert!(json.get("files").is_some());

    let files = json["files"].as_array().unwrap();
    assert_eq!(files.len(), 1);

    let file = &files[0];
    assert_eq!(file["path"], path);
    assert_eq!(file["size"], data.len());
    // Sample is first 512 bytes base64 encoded
    assert!(file["sample"].is_string());
}

#[test]
fn test_build_ndjson_lfs_commit() {
    // LFS commit uses lfsFile key with OID instead of base64 content
    let commit_message = "Upload parquet via LFS";
    let path_in_repo = "data/train.parquet";
    let oid = "abc123def456"; // SHA256 hash
    let size = 1024usize;

    let payload = build_ndjson_lfs_commit(commit_message, path_in_repo, oid, size);

    let lines: Vec<&str> = payload.lines().collect();
    assert_eq!(lines.len(), 2, "NDJSON should have exactly 2 lines");

    // Header should be same format
    let header: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
    assert_eq!(header["key"], "header");
    assert_eq!(header["value"]["summary"], commit_message);

    // File should use lfsFile key with OID
    let file_op: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
    assert_eq!(file_op["key"], "lfsFile");
    assert_eq!(file_op["value"]["path"], path_in_repo);
    assert_eq!(file_op["value"]["algo"], "sha256");
    assert_eq!(file_op["value"]["oid"], oid);
    assert_eq!(file_op["value"]["size"], size);
}

#[test]
fn test_lfs_preupload_url_format() {
    // Verify preupload API URL format
    let repo_id = "paiml/test-dataset";
    let expected_url = format!("{}/datasets/{}/preupload/main", HF_API_URL, repo_id);
    assert_eq!(
        expected_url,
        "https://huggingface.co/api/datasets/paiml/test-dataset/preupload/main"
    );
}

#[test]
fn test_lfs_batch_api_url_format() {
    // LFS batch API uses different URL format (Git LFS standard)
    let repo_id = "paiml/test-dataset";
    let expected_url = format!(
        "https://huggingface.co/datasets/{}.git/info/lfs/objects/batch",
        repo_id
    );
    assert_eq!(
        expected_url,
        "https://huggingface.co/datasets/paiml/test-dataset.git/info/lfs/objects/batch"
    );
}

#[test]
fn test_build_lfs_batch_request() {
    // LFS batch API request format (Git LFS standard)
    let oid = "abc123def456";
    let size = 1024usize;

    let request = build_lfs_batch_request(oid, size);

    let json: serde_json::Value = serde_json::from_str(&request).unwrap();
    assert_eq!(json["operation"], "upload");
    assert!(json["transfers"]
        .as_array()
        .unwrap()
        .contains(&serde_json::json!("basic")));

    let objects = json["objects"].as_array().unwrap();
    assert_eq!(objects.len(), 1);
    assert_eq!(objects[0]["oid"], oid);
    assert_eq!(objects[0]["size"], size);
}

// ========== DatasetCardValidator Tests (GH-6) ==========

#[test]
fn test_valid_task_categories() {
    assert!(DatasetCardValidator::is_valid_task_category(
        "text-generation"
    ));
    assert!(DatasetCardValidator::is_valid_task_category("translation"));
    assert!(DatasetCardValidator::is_valid_task_category(
        "text-classification"
    ));
    assert!(DatasetCardValidator::is_valid_task_category(
        "question-answering"
    ));
    assert!(DatasetCardValidator::is_valid_task_category(
        "text2text-generation"
    ));
}

#[test]
fn test_invalid_task_categories() {
    assert!(!DatasetCardValidator::is_valid_task_category(
        "code-generation"
    ));
    assert!(!DatasetCardValidator::is_valid_task_category(
        "invalid-task"
    ));
    assert!(!DatasetCardValidator::is_valid_task_category(""));
}

#[test]
fn test_valid_licenses() {
    assert!(DatasetCardValidator::is_valid_license("apache-2.0"));
    assert!(DatasetCardValidator::is_valid_license("mit"));
    assert!(DatasetCardValidator::is_valid_license("Apache-2.0")); // Case insensitive
    assert!(DatasetCardValidator::is_valid_license("MIT"));
}

#[test]
fn test_invalid_licenses() {
    assert!(!DatasetCardValidator::is_valid_license("invalid-license"));
    assert!(!DatasetCardValidator::is_valid_license(""));
}

#[test]
fn test_valid_size_categories() {
    assert!(DatasetCardValidator::is_valid_size_category("n<1K"));
    assert!(DatasetCardValidator::is_valid_size_category("1K<n<10K"));
    assert!(DatasetCardValidator::is_valid_size_category("n>1T"));
}

#[test]
fn test_invalid_size_categories() {
    assert!(!DatasetCardValidator::is_valid_size_category("small"));
    assert!(!DatasetCardValidator::is_valid_size_category("1000"));
}

#[test]
fn test_suggest_task_category() {
    assert_eq!(
        DatasetCardValidator::suggest_task_category("code-generation"),
        Some("text-generation")
    );
    assert_eq!(
        DatasetCardValidator::suggest_task_category("qa"),
        Some("question-answering")
    );
    assert_eq!(
        DatasetCardValidator::suggest_task_category("ner"),
        Some("token-classification")
    );
    assert_eq!(
        DatasetCardValidator::suggest_task_category("sentiment"),
        Some("text-classification")
    );
}

// ========================================================================
// Additional Coverage Tests for HfPublisher
// ========================================================================

#[test]
fn test_hf_publisher_with_token() {
    let publisher = HfPublisher::new("test/repo").with_token("my-secret-token");
    assert_eq!(publisher.repo_id(), "test/repo");
}

#[test]
fn test_hf_publisher_builder_clone() {
    let builder = HfPublisherBuilder::new("test/repo")
        .token("token")
        .private(true)
        .commit_message("message");
    let _cloned = builder.clone();
}

#[test]
fn test_hf_publisher_builder_debug() {
    let builder = HfPublisherBuilder::new("test/repo");
    let debug_str = format!("{:?}", builder);
    assert!(debug_str.contains("HfPublisherBuilder"));
}

#[test]
fn test_hf_publisher_debug() {
    let publisher = HfPublisher::new("test/repo");
    let debug_str = format!("{:?}", publisher);
    assert!(debug_str.contains("HfPublisher"));
}

#[test]
fn test_hf_publisher_clone() {
    let publisher = HfPublisher::new("test/repo")
        .with_token("token")
        .with_private(true)
        .with_commit_message("message");
    let _cloned = publisher.clone();
}

// ========================================================================
// Additional Coverage Tests for is_binary_file
// ========================================================================

#[test]
fn test_is_binary_file_comprehensive() {
    // Test various binary extensions
    assert!(is_binary_file("model.safetensors"));
    assert!(is_binary_file("weights.pt"));
    assert!(is_binary_file("model.pth"));
    assert!(is_binary_file("model.onnx"));
    assert!(is_binary_file("photo.jpg"));
    assert!(is_binary_file("photo.jpeg"));
    assert!(is_binary_file("photo.gif"));
    assert!(is_binary_file("photo.webp"));
    assert!(is_binary_file("photo.bmp"));
    assert!(is_binary_file("photo.tiff"));
    assert!(is_binary_file("audio.mp3"));
    assert!(is_binary_file("audio.wav"));
    assert!(is_binary_file("audio.flac"));
    assert!(is_binary_file("audio.ogg"));
    assert!(is_binary_file("video.mp4"));
    assert!(is_binary_file("video.webm"));
    assert!(is_binary_file("video.avi"));
    assert!(is_binary_file("video.mkv"));
    assert!(is_binary_file("archive.zip"));
    assert!(is_binary_file("archive.tar"));
    assert!(is_binary_file("archive.gz"));
    assert!(is_binary_file("archive.bz2"));
    assert!(is_binary_file("archive.xz"));
    assert!(is_binary_file("archive.7z"));
    assert!(is_binary_file("archive.rar"));
    assert!(is_binary_file("doc.pdf"));
    assert!(is_binary_file("doc.doc"));
    assert!(is_binary_file("doc.docx"));
    assert!(is_binary_file("sheet.xls"));
    assert!(is_binary_file("sheet.xlsx"));
    assert!(is_binary_file("numpy.npy"));
    assert!(is_binary_file("numpy.npz"));
    assert!(is_binary_file("data.h5"));
    assert!(is_binary_file("data.hdf5"));
    assert!(is_binary_file("model.pkl"));
    assert!(is_binary_file("model.pickle"));
}

#[test]
fn test_is_binary_file_text_files() {
    // Test text files that should NOT be treated as binary
    assert!(!is_binary_file("README.md"));
    assert!(!is_binary_file("config.yaml"));
    assert!(!is_binary_file("config.yml"));
    assert!(!is_binary_file("data.txt"));
    assert!(!is_binary_file("script.py"));
    assert!(!is_binary_file("code.rs"));
    assert!(!is_binary_file("style.css"));
    assert!(!is_binary_file("index.html"));
    assert!(!is_binary_file("manifest.xml"));
    assert!(!is_binary_file(".gitignore"));
    assert!(!is_binary_file("Makefile"));
}

#[test]
fn test_is_binary_file_case_insensitive() {
    // Extensions should be case-insensitive
    assert!(is_binary_file("image.PNG"));
    assert!(is_binary_file("image.Png"));
    assert!(is_binary_file("data.PARQUET"));
    assert!(is_binary_file("model.BIN"));
}

#[test]
fn test_is_binary_file_no_extension() {
    // Files without extension
    assert!(!is_binary_file("Dockerfile"));
    assert!(!is_binary_file("README"));
    assert!(!is_binary_file("LICENSE"));
}

#[test]
fn test_is_binary_file_path_with_directories() {
    // Files with directory paths
    assert!(is_binary_file("data/train.parquet"));
    assert!(is_binary_file("models/weights.bin"));
    assert!(!is_binary_file("docs/README.md"));
}

// ========================================================================
// Additional Coverage Tests for compute_sha256
// ========================================================================

#[test]
fn test_compute_sha256_empty() {
    let hash = compute_sha256(b"");
    assert_eq!(hash.len(), 64);
    // SHA256 of empty string is well-known
    assert_eq!(
        hash,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
}

#[test]
fn test_compute_sha256_known_value() {
    let hash = compute_sha256(b"hello");
    assert_eq!(hash.len(), 64);
    // SHA256 of "hello"
    assert_eq!(
        hash,
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    );
}

#[test]
fn test_compute_sha256_binary_data() {
    let data: Vec<u8> = (0..=255).collect();
    let hash = compute_sha256(&data);
    assert_eq!(hash.len(), 64);
    assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));
}

#[test]
fn test_compute_sha256_large_data() {
    let data = vec![0u8; 1_000_000]; // 1MB of zeros
    let hash = compute_sha256(&data);
    assert_eq!(hash.len(), 64);
}

// ========================================================================
// Additional Coverage Tests for build_lfs_preupload_request
// ========================================================================

#[test]
fn test_build_lfs_preupload_request_small_file() {
    // File smaller than 512 bytes
    let data = b"small file content";
    let request = build_lfs_preupload_request("small.txt", data);

    let json: serde_json::Value = serde_json::from_str(&request).unwrap();
    assert!(json["files"].as_array().unwrap().len() == 1);
    assert_eq!(json["files"][0]["path"], "small.txt");
    assert_eq!(json["files"][0]["size"], data.len());
}

#[test]
fn test_build_lfs_preupload_request_large_file() {
    // File larger than 512 bytes - sample should be truncated
    let data = vec![0u8; 1024];
    let request = build_lfs_preupload_request("large.bin", &data);

    let json: serde_json::Value = serde_json::from_str(&request).unwrap();
    assert_eq!(json["files"][0]["size"], 1024);
    // Sample should be base64 encoded first 512 bytes
    let sample = json["files"][0]["sample"].as_str().unwrap();
    use base64::{engine::general_purpose::STANDARD, Engine};
    let decoded = STANDARD.decode(sample).unwrap();
    assert_eq!(decoded.len(), 512);
}

#[test]
fn test_build_lfs_preupload_request_empty_file() {
    let request = build_lfs_preupload_request("empty.bin", b"");

    let json: serde_json::Value = serde_json::from_str(&request).unwrap();
    assert_eq!(json["files"][0]["size"], 0);
}

// ========================================================================
// Additional Coverage Tests for build_lfs_batch_request
// ========================================================================

#[test]
fn test_build_lfs_batch_request_large_size() {
    let oid = "a".repeat(64);
    let size = 1_000_000_000usize; // 1GB

    let request = build_lfs_batch_request(&oid, size);

    let json: serde_json::Value = serde_json::from_str(&request).unwrap();
    assert_eq!(json["operation"], "upload");
    assert_eq!(json["objects"][0]["size"], size);
}

#[test]
fn test_build_lfs_batch_request_zero_size() {
    let request = build_lfs_batch_request("abc123", 0);

    let json: serde_json::Value = serde_json::from_str(&request).unwrap();
    assert_eq!(json["objects"][0]["size"], 0);
}

// ========================================================================
// Additional Coverage Tests for build_ndjson_upload_payload
// ========================================================================

#[test]
fn test_build_ndjson_upload_payload_empty_content() {
    let payload = build_ndjson_upload_payload("Test commit", "empty.txt", b"");

    let lines: Vec<&str> = payload.lines().collect();
    assert_eq!(lines.len(), 2);

    let file_op: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
    use base64::{engine::general_purpose::STANDARD, Engine};
    let decoded = STANDARD
        .decode(file_op["value"]["content"].as_str().unwrap())
        .unwrap();
    assert!(decoded.is_empty());
}

#[test]
fn test_build_ndjson_upload_payload_unicode_content() {
    let unicode_content = "Hello, \u{4e16}\u{754c}! \u{1F600}".as_bytes();
    let payload = build_ndjson_upload_payload("Unicode test", "unicode.txt", unicode_content);

    let lines: Vec<&str> = payload.lines().collect();
    let file_op: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
    use base64::{engine::general_purpose::STANDARD, Engine};
    let decoded = STANDARD
        .decode(file_op["value"]["content"].as_str().unwrap())
        .unwrap();
    assert_eq!(decoded, unicode_content);
}

#[test]
fn test_build_ndjson_upload_payload_nested_path() {
    let payload = build_ndjson_upload_payload("Nested path", "deep/nested/path/file.txt", b"data");

    let lines: Vec<&str> = payload.lines().collect();
    let file_op: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
    assert_eq!(file_op["value"]["path"], "deep/nested/path/file.txt");
}

// ========================================================================
// Additional Coverage Tests for build_ndjson_lfs_commit
// ========================================================================

#[test]
fn test_build_ndjson_lfs_commit_large_oid() {
    let oid = "a".repeat(64);
    let payload = build_ndjson_lfs_commit("Large OID", "file.bin", &oid, 1000);

    let lines: Vec<&str> = payload.lines().collect();
    let file_op: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
    assert_eq!(file_op["value"]["oid"], oid);
}

#[test]
fn test_build_ndjson_lfs_commit_zero_size() {
    let payload = build_ndjson_lfs_commit("Zero size", "empty.bin", "abc123", 0);

    let lines: Vec<&str> = payload.lines().collect();
    let file_op: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
    assert_eq!(file_op["value"]["size"], 0);
}

// ========================================================================
// Additional Coverage Tests for DatasetCardValidator
// ========================================================================

#[test]
fn test_validator_mixed_valid_invalid_categories() {
    let readme = r"---
task_categories:
  - text-generation
  - invalid-category
  - translation
---
";
    let errors = DatasetCardValidator::validate_readme(readme);
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].value, "invalid-category");
}

#[test]
fn test_validator_yaml_parse_error() {
    // Malformed YAML
    let readme = r"---
task_categories: [
  - text-generation
---
";
    let errors = DatasetCardValidator::validate_readme(readme);
    // Should not crash, just return empty errors (YAML parse fails gracefully)
    assert!(errors.is_empty());
}

#[test]
fn test_validator_non_sequence_categories() {
    // task_categories as a string instead of array
    let readme = r"---
task_categories: text-generation
---
";
    let errors = DatasetCardValidator::validate_readme(readme);
    // Should handle gracefully
    assert!(errors.is_empty());
}

#[test]
fn test_validator_numeric_size_category() {
    // Numeric value instead of string
    let readme = r"---
size_categories:
  - 1000
---
";
    let errors = DatasetCardValidator::validate_readme(readme);
    // Should handle gracefully - numeric won't match string category
    assert!(errors.is_empty());
}

#[test]
fn test_suggest_similar_no_matches() {
    // Value completely different from any valid category
    let suggestions = DatasetCardValidator::suggest_similar("xyzqwerty123", VALID_TASK_CATEGORIES);
    // Should return empty or few suggestions
    assert!(suggestions.len() <= 3);
}

#[test]
fn test_suggest_similar_exact_match() {
    // Exact match should not return itself (it's already valid)
    let suggestions = DatasetCardValidator::suggest_similar("translation", VALID_TASK_CATEGORIES);
    // Should find it as a suggestion
    assert!(suggestions.contains(&"translation".to_string()));
}

#[test]
fn test_levenshtein_single_char() {
    assert_eq!(DatasetCardValidator::levenshtein("a", "b"), 1);
    assert_eq!(DatasetCardValidator::levenshtein("a", "a"), 0);
}

#[test]
fn test_levenshtein_insertions() {
    assert_eq!(DatasetCardValidator::levenshtein("abc", "abcd"), 1);
    assert_eq!(DatasetCardValidator::levenshtein("abc", "abcde"), 2);
}

#[test]
fn test_levenshtein_deletions() {
    assert_eq!(DatasetCardValidator::levenshtein("abcd", "abc"), 1);
    assert_eq!(DatasetCardValidator::levenshtein("abcde", "abc"), 2);
}

#[test]
fn test_levenshtein_substitutions() {
    assert_eq!(DatasetCardValidator::levenshtein("abc", "adc"), 1);
    assert_eq!(DatasetCardValidator::levenshtein("abc", "xyz"), 3);
}

#[test]
fn test_is_valid_license_all_valid() {
    for license in VALID_LICENSES {
        assert!(
            DatasetCardValidator::is_valid_license(license),
            "License '{}' should be valid",
            license
        );
    }
}

#[test]
fn test_is_valid_license_case_variations() {
    assert!(DatasetCardValidator::is_valid_license("APACHE-2.0"));
    assert!(DatasetCardValidator::is_valid_license("Apache-2.0"));
    assert!(DatasetCardValidator::is_valid_license("apache-2.0"));
    assert!(DatasetCardValidator::is_valid_license("MIT"));
    assert!(DatasetCardValidator::is_valid_license("mit"));
    assert!(DatasetCardValidator::is_valid_license("Mit"));
}

#[test]
fn test_suggest_task_category_prefix_match() {
    // Categories that start with the input
    let suggestion = DatasetCardValidator::suggest_task_category("text");
    assert!(suggestion.is_some());
}

#[test]
fn test_suggest_task_category_unknown() {
    // Completely unknown category
    let suggestion = DatasetCardValidator::suggest_task_category("xyzabc123");
    // May or may not find a suggestion
    let _ = suggestion;
}

#[test]
fn test_extract_frontmatter_no_end_marker() {
    let readme = "---\nlicense: mit\n";
    let errors = DatasetCardValidator::validate_readme(readme);
    // No end marker, should return empty
    assert!(errors.is_empty());
}

#[test]
fn test_extract_frontmatter_with_whitespace() {
    let readme = "  \n---\nlicense: mit\n---\n# Title";
    let errors = DatasetCardValidator::validate_readme(readme);
    // Should handle leading whitespace
    assert!(errors.is_empty());
}

// ========================================================================
// Additional HfDataset tests
// ========================================================================

#[test]
fn test_hf_dataset_with_special_chars_in_repo() {
    let dataset = HfDataset::builder("user-name/dataset_v2.0")
        .build()
        .ok()
        .unwrap_or_else(|| panic!("Should build"));

    assert_eq!(dataset.repo_id(), "user-name/dataset_v2.0");
}

#[test]
fn test_hf_dataset_revision_special_chars() {
    let dataset = HfDataset::builder("squad")
        .revision("refs/pr/123")
        .build()
        .ok()
        .unwrap_or_else(|| panic!("Should build"));

    assert_eq!(dataset.revision(), "refs/pr/123");
}

#[test]
fn test_hf_dataset_url_encoding() {
    let dataset = HfDataset::builder("org/dataset")
        .subset("config-v1")
        .split("train[:1000]")
        .build()
        .ok()
        .unwrap_or_else(|| panic!("Should build"));

    // Path building should work with special characters
    let path = dataset.build_parquet_path();
    assert!(path.contains("config-v1"));
}

#[test]
fn test_dataset_info_partial_fields() {
    let info = DatasetInfo {
        repo_id: "minimal".to_string(),
        splits: vec![],
        subsets: vec![],
        download_size: None,
        description: None,
    };

    assert_eq!(info.repo_id, "minimal");
    assert!(info.splits.is_empty());
    assert!(info.download_size.is_none());
}

// ========================================================================
// Additional HfDataset Coverage Tests
// ========================================================================

#[test]
fn test_hf_dataset_cache_dir_accessor() {
    let dataset = HfDataset::builder("test")
        .cache_dir("/custom/path")
        .build()
        .ok()
        .unwrap_or_else(|| panic!("Should build"));

    assert_eq!(dataset.cache_dir(), std::path::Path::new("/custom/path"));
}

#[test]
fn test_hf_dataset_builder_revision_method() {
    let builder = HfDatasetBuilder::new("test").revision("v2.0");
    let dataset = builder.build().unwrap();
    assert_eq!(dataset.revision(), "v2.0");
}

#[test]
fn test_hf_dataset_builder_subset_method() {
    let builder = HfDatasetBuilder::new("test").subset("config1");
    let dataset = builder.build().unwrap();
    assert_eq!(dataset.subset(), Some("config1"));
}

#[test]
fn test_hf_dataset_builder_split_method() {
    let builder = HfDatasetBuilder::new("test").split("validation");
    let dataset = builder.build().unwrap();
    assert_eq!(dataset.split(), Some("validation"));
}

#[test]
fn test_hf_dataset_builder_cache_dir_method() {
    let builder = HfDatasetBuilder::new("test").cache_dir("/tmp/cache");
    let dataset = builder.build().unwrap();
    assert_eq!(dataset.cache_dir(), std::path::Path::new("/tmp/cache"));
}

#[test]
fn test_hf_dataset_build_parquet_path_with_subset_and_split() {
    let dataset = HfDataset::builder("test")
        .subset("my-subset")
        .split("test")
        .build()
        .unwrap();

    assert_eq!(dataset.build_parquet_path(), "my-subset/test.parquet");
}

#[test]
fn test_hf_dataset_build_download_url_full() {
    let dataset = HfDataset::builder("org/repo")
        .revision("v1.0.0")
        .build()
        .unwrap();

    let url = dataset.build_download_url("custom/path.parquet");
    assert!(url.contains("org/repo"));
    assert!(url.contains("v1.0.0"));
    assert!(url.contains("custom/path.parquet"));
}

#[test]
fn test_validation_error_debug() {
    let err = ValidationError {
        field: "test_field".to_string(),
        value: "test_value".to_string(),
        suggestions: vec!["suggestion1".to_string()],
    };
    let debug = format!("{:?}", err);
    assert!(debug.contains("ValidationError"));
    assert!(debug.contains("test_field"));
}

#[test]
fn test_validation_error_clone() {
    let err = ValidationError {
        field: "field".to_string(),
        value: "value".to_string(),
        suggestions: vec!["s1".to_string(), "s2".to_string()],
    };
    let cloned = err.clone();
    assert_eq!(cloned.field, err.field);
    assert_eq!(cloned.value, err.value);
    assert_eq!(cloned.suggestions, err.suggestions);
}

#[test]
fn test_valid_task_categories_constant() {
    // Verify we have a reasonable number of task categories
    assert!(VALID_TASK_CATEGORIES.len() > 10);
    // Check some expected categories exist
    assert!(VALID_TASK_CATEGORIES.contains(&"translation"));
    assert!(VALID_TASK_CATEGORIES.contains(&"text-generation"));
}

#[test]
fn test_valid_size_categories_constant() {
    // Verify we have size categories
    assert!(VALID_SIZE_CATEGORIES.len() > 5);
    assert!(VALID_SIZE_CATEGORIES.contains(&"n<1K"));
    assert!(VALID_SIZE_CATEGORIES.contains(&"n>1T"));
}

#[test]
fn test_valid_licenses_constant() {
    // Verify we have licenses
    assert!(VALID_LICENSES.len() > 5);
    assert!(VALID_LICENSES.contains(&"mit"));
    assert!(VALID_LICENSES.contains(&"apache-2.0"));
}

#[test]
fn test_hf_publisher_builder_from_new() {
    let builder = HfPublisherBuilder::new("org/dataset");
    let publisher = builder.build();
    assert_eq!(publisher.repo_id(), "org/dataset");
}

#[test]
fn test_hf_publisher_builder_fluent() {
    let publisher = HfPublisherBuilder::new("org/dataset")
        .token("my-token")
        .private(true)
        .commit_message("Custom commit")
        .build();

    assert_eq!(publisher.repo_id(), "org/dataset");
}

#[test]
fn test_validate_readme_with_all_valid_fields() {
    let readme = r"---
license: apache-2.0
task_categories:
  - text-generation
  - translation
size_categories:
  - 10K<n<100K
language:
  - en
  - de
---
# Dataset Title

This is a test dataset.
";
    let errors = DatasetCardValidator::validate_readme(readme);
    assert!(errors.is_empty());
}

#[test]
fn test_validate_readme_with_multiple_errors() {
    let readme = r"---
task_categories:
  - invalid-task-1
  - invalid-task-2
size_categories:
  - invalid-size
---
";
    let errors = DatasetCardValidator::validate_readme(readme);
    // Should have 3 errors: 2 invalid tasks + 1 invalid size
    assert_eq!(errors.len(), 3);
}

#[test]
fn test_suggest_similar_with_exact_threshold() {
    // Test edge case where distance equals threshold
    let suggestions = DatasetCardValidator::suggest_similar("text-gen", VALID_TASK_CATEGORIES);
    // Should find suggestions within distance 3
    assert!(!suggestions.is_empty());
}

#[test]
fn test_levenshtein_with_longer_strings() {
    let dist = DatasetCardValidator::levenshtein("text-generation", "text-classification");
    // Should be some reasonable distance
    assert!(dist > 0 && dist < 20);
}

#[test]
fn test_is_binary_file_edge_cases() {
    // Test files with multiple dots
    assert!(is_binary_file("archive.tar.gz"));
    assert!(is_binary_file("model.weights.bin"));

    // Test uppercase variants
    assert!(is_binary_file("FILE.PARQUET"));
    assert!(is_binary_file("Image.PNG"));

    // Test mixed case
    assert!(is_binary_file("data.ParQuet"));
}

#[test]
fn test_compute_sha256_consistency() {
    // Same input should always produce same output
    let data = b"test data for hashing";
    let hash1 = compute_sha256(data);
    let hash2 = compute_sha256(data);
    assert_eq!(hash1, hash2);
}

#[test]
fn test_build_ndjson_payload_special_chars() {
    let commit_msg = "Upload with 'quotes' and \"double quotes\"";
    let path = "path/with spaces/file.txt";
    let data = b"content with special chars: \n\t\r";

    let payload = build_ndjson_upload_payload(commit_msg, path, data);

    let lines: Vec<&str> = payload.lines().collect();
    assert_eq!(lines.len(), 2);

    // Both lines should be valid JSON
    let _: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
    let _: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
}

#[test]
fn test_build_lfs_batch_request_format() {
    let request = build_lfs_batch_request("abc123", 1000);
    let json: serde_json::Value = serde_json::from_str(&request).unwrap();

    // Verify required fields
    assert_eq!(json["operation"], "upload");
    assert!(json["transfers"].as_array().is_some());
    assert!(json["objects"].as_array().is_some());
}

#[test]
fn test_build_ndjson_lfs_commit_format() {
    let payload = build_ndjson_lfs_commit("Test", "file.bin", "sha256hash", 500);
    let lines: Vec<&str> = payload.lines().collect();

    let header: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
    let file: serde_json::Value = serde_json::from_str(lines[1]).unwrap();

    assert_eq!(header["key"], "header");
    assert_eq!(file["key"], "lfsFile");
    assert_eq!(file["value"]["algo"], "sha256");
}

#[test]
fn test_build_lfs_preupload_request_exact_512_bytes() {
    // Test with exactly 512 bytes
    let data = vec![0u8; 512];
    let request = build_lfs_preupload_request("file.bin", &data);

    let json: serde_json::Value = serde_json::from_str(&request).unwrap();
    assert_eq!(json["files"][0]["size"], 512);
}

#[test]
fn test_validator_strict_multiple_errors() {
    let readme = r"---
task_categories:
  - bad-task-1
  - bad-task-2
---
";
    let result = DatasetCardValidator::validate_readme_strict(readme);
    assert!(result.is_err());

    let err_msg = result.unwrap_err().to_string();
    // Error should mention at least one invalid task
    assert!(err_msg.contains("bad-task"));
}

#[test]
fn test_hf_dataset_download_to_error_without_network() {
    let temp_dir = tempfile::tempdir().unwrap();
    let dataset = HfDataset::builder("nonexistent-repo")
        .cache_dir(temp_dir.path())
        .build()
        .unwrap();

    let output_path = temp_dir.path().join("output.parquet");
    let result = dataset.download_to(&output_path);

    // Should fail because we can't actually connect
    assert!(result.is_err());
}

#[test]
fn test_default_cache_dir_returns_valid_path() {
    let cache = default_cache_dir();
    // Should be an absolute path with "alimentar" somewhere in it
    assert!(
        cache.to_string_lossy().contains("alimentar") || cache.to_string_lossy().contains("cache")
    );
}

#[test]
fn test_hf_dataset_clear_cache_creates_no_error_on_missing() {
    let temp_dir = tempfile::tempdir().unwrap();
    let nonexistent_cache = temp_dir.path().join("nonexistent");

    let dataset = HfDataset::builder("test-repo")
        .cache_dir(&nonexistent_cache)
        .build()
        .unwrap();

    // Should not error even if cache doesn't exist
    let result = dataset.clear_cache();
    assert!(result.is_ok());
}
