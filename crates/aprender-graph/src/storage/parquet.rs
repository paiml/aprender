//! Parquet I/O for graph persistence
//!
//! Based on `DuckDB` (Raasveldt et al., SIGMOD 2019) columnar storage patterns.
//!
//! # Format
//!
//! Graphs are stored as two Parquet files:
//! - `{path}_edges.parquet`: (source, target, weight)
//! - `{path}_nodes.parquet`: (`node_id`, name)

use super::{CsrGraph, NodeId};
use anyhow::{Context, Result};
use arrow::array::{Float32Array, StringArray, UInt32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::file::properties::WriterProperties;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

impl CsrGraph {
    /// Write graph to Parquet files
    ///
    /// Creates two files:
    /// - `{path}_edges.parquet`: Edge list (source, target, weight)
    /// - `{path}_nodes.parquet`: Node metadata (`node_id`, name)
    ///
    /// # Errors
    ///
    /// Returns error if file I/O fails or Arrow conversion fails
    #[allow(clippy::unused_async)] // Async API for future I/O operations
    pub async fn write_parquet<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let base_path = path.as_ref();

        // Write edges
        self.write_edges_parquet(base_path)?;

        // Write nodes (metadata)
        self.write_nodes_parquet(base_path)?;

        Ok(())
    }

    /// Read graph from Parquet files
    ///
    /// # Errors
    ///
    /// Returns error if files don't exist or Arrow conversion fails
    #[allow(clippy::unused_async)] // Async API for future I/O operations
    pub async fn read_parquet<P: AsRef<Path>>(path: P) -> Result<Self> {
        let base_path = path.as_ref();

        // Read edges
        let edges = Self::read_edges_parquet(base_path)?;

        // Build graph from edge list
        let mut graph = Self::from_edge_list(&edges)?;

        // Read node names
        let node_names = Self::read_nodes_parquet(base_path)?;
        for (node_id, name) in node_names {
            graph.set_node_name(node_id, name);
        }

        Ok(graph)
    }

    fn write_edges_parquet(&self, base_path: &Path) -> Result<()> {
        let edges_path = format!("{}_edges.parquet", base_path.display());

        // Convert CSR to edge list arrays
        let mut sources = Vec::new();
        let mut targets = Vec::new();
        let mut weights = Vec::new();

        for (src, target_nodes, edge_weights) in self.iter_adjacency() {
            for (dst, weight) in target_nodes.iter().zip(edge_weights.iter()) {
                sources.push(src.0);
                targets.push(*dst);
                weights.push(*weight);
            }
        }

        // Create Arrow schema
        let schema = Arc::new(Schema::new(vec![
            Field::new("source", DataType::UInt32, false),
            Field::new("target", DataType::UInt32, false),
            Field::new("weight", DataType::Float32, false),
        ]));

        // Create Arrow arrays
        let source_array = Arc::new(UInt32Array::from(sources));
        let target_array = Arc::new(UInt32Array::from(targets));
        let weight_array = Arc::new(Float32Array::from(weights));

        // Create RecordBatch
        let batch =
            RecordBatch::try_new(schema.clone(), vec![source_array, target_array, weight_array])
                .context("Failed to create RecordBatch")?;

        // Write to Parquet
        let file =
            File::create(&edges_path).with_context(|| format!("Failed to create {edges_path}"))?;

        let props = WriterProperties::builder()
            .set_compression(parquet::basic::Compression::ZSTD(parquet::basic::ZstdLevel::try_new(
                3,
            )?))
            .build();

        let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
        writer.write(&batch)?;
        writer.close()?;

        Ok(())
    }

    fn write_nodes_parquet(&self, base_path: &Path) -> Result<()> {
        let nodes_path = format!("{}_nodes.parquet", base_path.display());

        // Collect node IDs and names
        let mut node_ids = Vec::new();
        let mut names = Vec::new();

        for node_id in 0..self.num_nodes() {
            #[allow(clippy::cast_possible_truncation)] // Graphs >4B nodes not supported yet
            let node_u32 = node_id as u32;
            node_ids.push(node_u32);
            let name = self
                .get_node_name(NodeId(node_u32))
                .unwrap_or(&format!("node_{node_id}"))
                .to_string();
            names.push(name);
        }

        // Create Arrow schema
        let schema = Arc::new(Schema::new(vec![
            Field::new("node_id", DataType::UInt32, false),
            Field::new("name", DataType::Utf8, false),
        ]));

        // Create Arrow arrays
        let node_id_array = Arc::new(UInt32Array::from(node_ids));
        let name_array = Arc::new(StringArray::from(names));

        // Create RecordBatch
        let batch = RecordBatch::try_new(schema.clone(), vec![node_id_array, name_array])
            .context("Failed to create nodes RecordBatch")?;

        // Write to Parquet
        let file =
            File::create(&nodes_path).with_context(|| format!("Failed to create {nodes_path}"))?;

        let props = WriterProperties::builder()
            .set_compression(parquet::basic::Compression::ZSTD(parquet::basic::ZstdLevel::try_new(
                3,
            )?))
            .build();

        let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
        writer.write(&batch)?;
        writer.close()?;

        Ok(())
    }

    fn read_edges_parquet(base_path: &Path) -> Result<Vec<(NodeId, NodeId, f32)>> {
        let edges_path = format!("{}_edges.parquet", base_path.display());

        let file =
            File::open(&edges_path).with_context(|| format!("Failed to open {edges_path}"))?;

        let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;

        let mut edges = Vec::new();

        for batch_result in reader {
            let batch: RecordBatch = batch_result?;

            let sources = batch
                .column(0)
                .as_any()
                .downcast_ref::<UInt32Array>()
                .context("Invalid source column type")?;

            let targets = batch
                .column(1)
                .as_any()
                .downcast_ref::<UInt32Array>()
                .context("Invalid target column type")?;

            let weights = batch
                .column(2)
                .as_any()
                .downcast_ref::<Float32Array>()
                .context("Invalid weight column type")?;

            for i in 0..batch.num_rows() {
                edges.push((NodeId(sources.value(i)), NodeId(targets.value(i)), weights.value(i)));
            }
        }

        Ok(edges)
    }

    fn read_nodes_parquet(base_path: &Path) -> Result<Vec<(NodeId, String)>> {
        let nodes_path = format!("{}_nodes.parquet", base_path.display());

        let file =
            File::open(&nodes_path).with_context(|| format!("Failed to open {nodes_path}"))?;

        let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;

        let mut nodes = Vec::new();

        for batch_result in reader {
            let batch: RecordBatch = batch_result?;

            let node_ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<UInt32Array>()
                .context("Invalid node_id column type")?;

            let names = batch
                .column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .context("Invalid name column type")?;

            for i in 0..batch.num_rows() {
                nodes.push((NodeId(node_ids.value(i)), names.value(i).to_string()));
            }
        }

        Ok(nodes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_parquet_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_graph");

        // Create graph
        let mut graph = CsrGraph::new();
        graph.add_edge(NodeId(0), NodeId(1), 1.0).unwrap();
        graph.add_edge(NodeId(0), NodeId(2), 2.0).unwrap();
        graph.add_edge(NodeId(1), NodeId(2), 3.0).unwrap();

        graph.set_node_name(NodeId(0), "main".to_string());
        graph.set_node_name(NodeId(1), "parse_args".to_string());
        graph.set_node_name(NodeId(2), "validate".to_string());

        // Write to Parquet
        graph.write_parquet(&path).await.unwrap();

        // Read back
        let loaded = CsrGraph::read_parquet(&path).await.unwrap();

        // Verify structure
        assert_eq!(loaded.num_nodes(), graph.num_nodes());
        assert_eq!(loaded.num_edges(), graph.num_edges());

        // Verify edges
        assert_eq!(loaded.outgoing_neighbors(NodeId(0)).unwrap(), &[1, 2]);

        // Verify node names
        assert_eq!(loaded.get_node_name(NodeId(0)), Some("main"));
        assert_eq!(loaded.get_node_name(NodeId(1)), Some("parse_args"));
        assert_eq!(loaded.get_node_name(NodeId(2)), Some("validate"));
    }

    #[tokio::test]
    async fn test_empty_graph_parquet() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("empty_graph");

        let graph = CsrGraph::new();
        graph.write_parquet(&path).await.unwrap();

        let loaded = CsrGraph::read_parquet(&path).await.unwrap();
        assert_eq!(loaded.num_nodes(), 0);
        assert_eq!(loaded.num_edges(), 0);
    }
}
