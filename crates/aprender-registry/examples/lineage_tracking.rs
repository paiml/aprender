//! Lineage Tracking Example
//!
//! Demonstrates model lineage tracking:
//! - Fine-tuning relationships
//! - Quantization tracking
//! - Model merging
//!
//! Run with: cargo run --example lineage_tracking

use pacha::prelude::*;

fn main() {
    println!("=== Lineage Tracking Example ===\n");

    // Create a lineage graph
    let mut graph = LineageGraph::new();

    // Add base model
    println!("1. Creating base model node...");
    let base_id = ModelId::new();
    let base_idx = graph.add_node(pacha::lineage::LineageNode {
        model_id: base_id.clone(),
        model_name: "llama-7b".to_string(),
        model_version: "1.0.0".to_string(),
    });
    println!("   ✓ Base model: llama-7b:1.0.0");

    // Add fine-tuned model
    println!("\n2. Creating fine-tuned model node...");
    let finetuned_id = ModelId::new();
    let finetuned_idx = graph.add_node(pacha::lineage::LineageNode {
        model_id: finetuned_id.clone(),
        model_name: "llama-7b-fraud".to_string(),
        model_version: "1.0.0".to_string(),
    });

    // Add fine-tuning edge
    graph.add_edge(
        base_idx,
        finetuned_idx,
        ModelLineageEdge::FineTuned {
            parent: base_id.clone(),
            recipe: pacha::recipe::RecipeId::new(),
        },
    );
    println!("   ✓ Fine-tuned: llama-7b-fraud:1.0.0");
    println!("   ✓ Edge: llama-7b → llama-7b-fraud (fine-tuned)");

    // Add quantized model
    println!("\n3. Creating quantized model node...");
    let quantized_id = ModelId::new();
    let quantized_idx = graph.add_node(pacha::lineage::LineageNode {
        model_id: quantized_id.clone(),
        model_name: "llama-7b-fraud-int8".to_string(),
        model_version: "1.0.0".to_string(),
    });

    // Add quantization edge
    graph.add_edge(
        finetuned_idx,
        quantized_idx,
        ModelLineageEdge::Quantized {
            source: finetuned_id.clone(),
            quantization: QuantizationType::Int8,
        },
    );
    println!("   ✓ Quantized: llama-7b-fraud-int8:1.0.0");
    println!("   ✓ Edge: llama-7b-fraud → llama-7b-fraud-int8 (quantized INT8)");

    // Add another fine-tuned variant for merging
    println!("\n4. Creating another fine-tuned variant...");
    let variant_id = ModelId::new();
    let variant_idx = graph.add_node(pacha::lineage::LineageNode {
        model_id: variant_id.clone(),
        model_name: "llama-7b-code".to_string(),
        model_version: "1.0.0".to_string(),
    });

    graph.add_edge(
        base_idx,
        variant_idx,
        ModelLineageEdge::FineTuned {
            parent: base_id.clone(),
            recipe: pacha::recipe::RecipeId::new(),
        },
    );
    println!("   ✓ Fine-tuned: llama-7b-code:1.0.0");

    // Create merged model
    println!("\n5. Creating merged model...");
    let merged_id = ModelId::new();
    let merged_idx = graph.add_node(pacha::lineage::LineageNode {
        model_id: merged_id,
        model_name: "llama-7b-fraud-code".to_string(),
        model_version: "1.0.0".to_string(),
    });

    graph.add_edge(
        finetuned_idx,
        merged_idx,
        ModelLineageEdge::Merged {
            sources: vec![finetuned_id, variant_id],
            weights: vec![0.7, 0.3],
        },
    );
    println!("   ✓ Merged: llama-7b-fraud-code:1.0.0");
    println!("   ✓ Edge: llama-7b-fraud + llama-7b-code → merged (0.7/0.3)");

    // Print graph statistics
    println!("\n6. Lineage graph statistics:");
    println!("   Nodes: {}", graph.node_count());
    println!("   Edges: {}", graph.edge_count());

    // Show relationships
    println!("\n7. Model relationships:");
    println!("   Base model (idx={base_idx}):");
    println!("     Descendants: {:?}", graph.descendants(base_idx));

    println!("   Fine-tuned model (idx={finetuned_idx}):");
    println!("     Ancestors: {:?}", graph.ancestors(finetuned_idx));
    println!("     Descendants: {:?}", graph.descendants(finetuned_idx));

    println!("   Quantized model (idx={quantized_idx}):");
    println!("     Ancestors: {:?}", graph.ancestors(quantized_idx));

    // Demonstrate quantization types
    println!("\n8. Supported quantization types:");
    let quant_types = [
        QuantizationType::Int8,
        QuantizationType::Int4,
        QuantizationType::Fp16,
        QuantizationType::Bf16,
        QuantizationType::Dynamic,
    ];
    for qt in quant_types {
        println!("   - {qt}");
    }

    println!("\n✅ Lineage tracking example complete!");
}
