//! RO-Crate package for bundling research data.

use super::descriptor::RoCrateDescriptor;
use super::entity::{EntityType, RoCrateEntity};
use crate::research::artifact::ResearchArtifact;
use serde_json::json;
use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::io::Write;
use std::path::{Path, PathBuf};

/// RO-Crate package
#[derive(Debug, Clone)]
pub struct RoCrate {
    /// Root directory path
    pub root: PathBuf,
    /// Metadata descriptor
    pub descriptor: RoCrateDescriptor,
    /// Data files to include (relative path -> content)
    pub data_files: HashMap<String, Vec<u8>>,
}

impl RoCrate {
    /// Create a new RO-Crate
    pub fn new(root: impl Into<PathBuf>) -> Self {
        let mut descriptor = RoCrateDescriptor::new();

        // Add root dataset
        let root_entity = RoCrateEntity::root_dataset()
            .with_property("datePublished", chrono::Utc::now().format("%Y-%m-%d").to_string());
        descriptor.add_entity(root_entity);

        Self { root: root.into(), descriptor, data_files: HashMap::new() }
    }

    /// Create from a research artifact
    pub fn from_artifact(artifact: &ResearchArtifact, root: impl Into<PathBuf>) -> Self {
        let mut crate_pkg = Self::new(root);

        // Update root dataset with artifact metadata
        if let Some(root_entity) = crate_pkg.descriptor.root_dataset_mut() {
            root_entity.properties.insert("name".to_string(), json!(artifact.title));
            if let Some(desc) = &artifact.description {
                root_entity.properties.insert("description".to_string(), json!(desc));
            }
            root_entity.properties.insert("version".to_string(), json!(artifact.version));
            root_entity
                .properties
                .insert("license".to_string(), json!(artifact.license.to_string()));

            if let Some(doi) = &artifact.doi {
                root_entity.properties.insert("identifier".to_string(), json!(doi));
            }

            if !artifact.keywords.is_empty() {
                root_entity
                    .properties
                    .insert("keywords".to_string(), json!(artifact.keywords.join(", ")));
            }
        }

        // Add author entities
        let mut author_ids = Vec::new();
        for (i, author) in artifact.authors.iter().enumerate() {
            let author_id = format!("#author-{}", i + 1);
            author_ids.push(author_id.clone());

            let mut person_entity = RoCrateEntity::person(&author_id, &author.name);

            if let Some(orcid) = &author.orcid {
                person_entity =
                    person_entity.with_property("identifier", format!("https://orcid.org/{orcid}"));
            }

            if let Some(affiliation) = author.affiliations.first() {
                let org_id = format!("#org-{}", i + 1);
                let org_entity = RoCrateEntity::new(&org_id, EntityType::Organization)
                    .with_name(&affiliation.name);
                crate_pkg.descriptor.add_entity(org_entity);
                person_entity = person_entity.with_reference("affiliation", &org_id);
            }

            crate_pkg.descriptor.add_entity(person_entity);
        }

        // Link authors to root dataset
        if !author_ids.is_empty() {
            if let Some(root_entity) = crate_pkg.descriptor.root_dataset_mut() {
                let author_refs: Vec<serde_json::Value> =
                    author_ids.iter().map(|id| json!({ "@id": id })).collect();
                root_entity.properties.insert("author".to_string(), json!(author_refs));
            }
        }

        crate_pkg
    }

    /// Add a data file
    pub fn add_file(&mut self, path: impl Into<String>, content: Vec<u8>) {
        let path_str = path.into();

        // Add file entity to descriptor
        let file_entity = RoCrateEntity::file(&path_str)
            .with_property("contentSize", content.len().to_string())
            .with_property("encodingFormat", guess_mime_type(&path_str));

        self.descriptor.add_entity(file_entity);
        self.data_files.insert(path_str, content);
    }

    /// Add a text file
    pub fn add_text_file(&mut self, path: impl Into<String>, content: impl Into<String>) {
        self.add_file(path, content.into().into_bytes());
    }

    /// Write to a directory
    pub fn to_directory(&self) -> std::io::Result<PathBuf> {
        // Create root directory
        std::fs::create_dir_all(&self.root)?;

        // Write ro-crate-metadata.json
        let metadata_path = self.root.join("ro-crate-metadata.json");
        std::fs::write(&metadata_path, self.descriptor.to_json())?;

        // Write data files
        for (path, content) in &self.data_files {
            let file_path = self.root.join(path);
            if let Some(parent) = file_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&file_path, content)?;
        }

        Ok(self.root.clone())
    }

    /// Create a ZIP archive
    #[cfg(not(target_arch = "wasm32"))]
    pub fn to_zip(&self) -> std::io::Result<Vec<u8>> {
        let mut buffer = std::io::Cursor::new(Vec::new());

        {
            let mut zip = zip::ZipWriter::new(&mut buffer);
            let options = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated);

            // Write ro-crate-metadata.json
            zip.start_file("ro-crate-metadata.json", options)?;
            zip.write_all(self.descriptor.to_json().as_bytes())?;

            // Write data files
            for (path, content) in &self.data_files {
                zip.start_file(path, options)?;
                zip.write_all(content)?;
            }

            zip.finish()?;
        }

        Ok(buffer.into_inner())
    }

    /// Get entity count
    pub fn entity_count(&self) -> usize {
        self.descriptor.graph.len()
    }

    /// Get file count
    pub fn file_count(&self) -> usize {
        self.data_files.len()
    }
}

/// Guess MIME type from file extension
pub fn guess_mime_type(path: &str) -> &'static str {
    let ext = Path::new(path).extension().and_then(|e| e.to_str()).unwrap_or("");

    match ext.to_lowercase().as_str() {
        "json" => "application/json",
        "yaml" | "yml" => "application/x-yaml",
        "csv" => "text/csv",
        "txt" => "text/plain",
        "md" => "text/markdown",
        "py" => "text/x-python",
        "rs" => "text/x-rust",
        "pdf" => "application/pdf",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "parquet" => "application/vnd.apache.parquet",
        "safetensors" => "application/octet-stream",
        other => {
            eprintln!(
                "Warning: unknown file extension '{other}', defaulting to application/octet-stream"
            );
            "application/octet-stream"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_guess_mime_type_all_extension_variants() {
        let cases: &[(&str, &str)] = &[
            ("data.json", "application/json"),
            ("config.yaml", "application/x-yaml"),
            ("config.yml", "application/x-yaml"),
            ("data.csv", "text/csv"),
            ("readme.txt", "text/plain"),
            ("notes.md", "text/markdown"),
            ("script.py", "text/x-python"),
            ("main.rs", "text/x-rust"),
            ("paper.pdf", "application/pdf"),
            ("image.png", "image/png"),
            ("photo.jpg", "image/jpeg"),
            ("photo.jpeg", "image/jpeg"),
            ("data.parquet", "application/vnd.apache.parquet"),
            ("model.safetensors", "application/octet-stream"),
            ("archive.xyz", "application/octet-stream"),
        ];

        for &(path, expected) in cases {
            let result = guess_mime_type(path);

            // Syntactic match covering all arms from guess_mime_type
            let ext = Path::new(path).extension().and_then(|e| e.to_str()).unwrap_or("");

            let matched = match ext.to_lowercase().as_str() {
                "json" => "application/json",
                "yaml" | "yml" => "application/x-yaml",
                "csv" => "text/csv",
                "txt" => "text/plain",
                "md" => "text/markdown",
                "py" => "text/x-python",
                "rs" => "text/x-rust",
                "pdf" => "application/pdf",
                "png" => "image/png",
                "jpg" | "jpeg" => "image/jpeg",
                "parquet" => "application/vnd.apache.parquet",
                "safetensors" => "application/octet-stream",
                _other => "application/octet-stream",
            };

            assert_eq!(result, expected, "MIME mismatch for {path}");
            assert_eq!(matched, expected, "match mismatch for {path}");
        }
    }
}
