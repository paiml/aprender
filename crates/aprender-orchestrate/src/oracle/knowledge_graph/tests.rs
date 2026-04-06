//! Tests for the knowledge graph.

use super::super::types::*;
use super::types::KnowledgeGraph;

// =========================================================================
// Basic Knowledge Graph Tests
// =========================================================================

#[test]
fn test_knowledge_graph_new() {
    let graph = KnowledgeGraph::new();
    assert_eq!(graph.component_count(), 0);
    assert_eq!(graph.integration_count(), 0);
}

#[test]
fn test_knowledge_graph_default() {
    let graph = KnowledgeGraph::default();
    assert_eq!(graph.component_count(), 0);
}

#[test]
fn test_knowledge_graph_sovereign_stack() {
    let graph = KnowledgeGraph::sovereign_stack();
    assert!(graph.component_count() >= 15);
    assert!(graph.integration_count() > 0);
}

// =========================================================================
// Component Registration Tests
// =========================================================================

#[test]
fn test_register_component() {
    let mut graph = KnowledgeGraph::new();
    let comp = StackComponent::new("test", "1.0.0", StackLayer::Primitives, "Test component")
        .with_capability(Capability::new("test_cap", CapabilityCategory::Compute));

    graph.register_component(comp);

    assert_eq!(graph.component_count(), 1);
    assert!(graph.get_component("test").is_some());
}

#[test]
fn test_get_component() {
    let graph = KnowledgeGraph::sovereign_stack();

    let trueno = graph.get_component("trueno");
    assert!(trueno.is_some());
    let trueno = trueno.unwrap();
    assert_eq!(trueno.name, "trueno");
    assert_eq!(trueno.version, "0.11.0");
    assert_eq!(trueno.layer, StackLayer::Primitives);
}

#[test]
fn test_get_component_not_found() {
    let graph = KnowledgeGraph::sovereign_stack();
    assert!(graph.get_component("nonexistent").is_none());
}

#[test]
fn test_component_names() {
    let graph = KnowledgeGraph::sovereign_stack();
    let names: Vec<_> = graph.component_names().collect();
    assert!(names.contains(&&"trueno".to_string()));
    assert!(names.contains(&&"aprender".to_string()));
    assert!(names.contains(&&"repartir".to_string()));
}

// =========================================================================
// Layer Query Tests
// =========================================================================

#[test]
fn test_components_in_layer_primitives() {
    let graph = KnowledgeGraph::sovereign_stack();
    let primitives = graph.components_in_layer(StackLayer::Primitives);

    assert!(primitives.len() >= 3);
    assert!(primitives.iter().any(|c| c.name == "trueno"));
    assert!(primitives.iter().any(|c| c.name == "trueno-db"));
    assert!(primitives.iter().any(|c| c.name == "trueno-graph"));
}

#[test]
fn test_components_in_layer_ml_algorithms() {
    let graph = KnowledgeGraph::sovereign_stack();
    let ml = graph.components_in_layer(StackLayer::MlAlgorithms);

    assert_eq!(ml.len(), 1);
    assert_eq!(ml[0].name, "aprender");
}

#[test]
fn test_components_in_layer_transpilers() {
    let graph = KnowledgeGraph::sovereign_stack();
    let transpilers = graph.components_in_layer(StackLayer::Transpilers);

    assert_eq!(transpilers.len(), 4);
    let names: Vec<_> = transpilers.iter().map(|c| &c.name).collect();
    assert!(names.contains(&&"depyler".to_string()));
    assert!(names.contains(&&"decy".to_string()));
    assert!(names.contains(&&"bashrs".to_string()));
    assert!(names.contains(&&"ruchy".to_string()));
}

// =========================================================================
// Capability Query Tests
// =========================================================================

#[test]
fn test_find_by_capability_simd() {
    let graph = KnowledgeGraph::sovereign_stack();
    let simd_components = graph.find_by_capability("simd");

    assert!(!simd_components.is_empty());
    assert!(simd_components.iter().any(|c| c.name == "trueno"));
}

#[test]
fn test_find_by_capability_random_forest() {
    let graph = KnowledgeGraph::sovereign_stack();
    let rf_components = graph.find_by_capability("random_forest");

    assert!(!rf_components.is_empty());
    assert!(rf_components.iter().any(|c| c.name == "aprender"));
}

#[test]
fn test_find_by_capability_model_serving() {
    let graph = KnowledgeGraph::sovereign_stack();
    let serving = graph.find_by_capability("model_serving");

    assert!(!serving.is_empty());
    assert!(serving.iter().any(|c| c.name == "realizar"));
}

#[test]
fn test_find_by_capability_not_found() {
    let graph = KnowledgeGraph::sovereign_stack();
    let result = graph.find_by_capability("nonexistent_capability");
    assert!(result.is_empty());
}

// =========================================================================
// Domain Query Tests
// =========================================================================

#[test]
fn test_find_by_domain_supervised_learning() {
    let graph = KnowledgeGraph::sovereign_stack();
    let components = graph.find_by_domain(ProblemDomain::SupervisedLearning);

    assert!(!components.is_empty());
    assert!(components.iter().any(|c| c.name == "aprender"));
}

#[test]
fn test_find_by_domain_linear_algebra() {
    let graph = KnowledgeGraph::sovereign_stack();
    let components = graph.find_by_domain(ProblemDomain::LinearAlgebra);

    assert!(!components.is_empty());
    assert!(components.iter().any(|c| c.name == "trueno"));
}

#[test]
fn test_find_by_domain_python_migration() {
    let graph = KnowledgeGraph::sovereign_stack();
    let components = graph.find_by_domain(ProblemDomain::PythonMigration);

    assert!(!components.is_empty());
    assert!(components.iter().any(|c| c.name == "depyler"));
}

#[test]
fn test_find_by_domain_distributed_compute() {
    let graph = KnowledgeGraph::sovereign_stack();
    let components = graph.find_by_domain(ProblemDomain::DistributedCompute);

    assert!(!components.is_empty());
    assert!(components.iter().any(|c| c.name == "repartir"));
}

// =========================================================================
// Integration Pattern Tests
// =========================================================================

#[test]
fn test_integrations_from() {
    let graph = KnowledgeGraph::sovereign_stack();
    let patterns = graph.integrations_from("aprender");

    assert!(!patterns.is_empty());
    assert!(patterns.iter().any(|p| p.to == "realizar"));
}

#[test]
fn test_integrations_to() {
    let graph = KnowledgeGraph::sovereign_stack();
    let patterns = graph.integrations_to("aprender");

    assert!(!patterns.is_empty());
    assert!(patterns.iter().any(|p| p.from == "depyler"));
}

#[test]
fn test_get_integration() {
    let graph = KnowledgeGraph::sovereign_stack();
    let pattern = graph.get_integration("aprender", "realizar");

    assert!(pattern.is_some());
    let pattern = pattern.unwrap();
    assert_eq!(pattern.pattern_name, "model_export");
}

#[test]
fn test_get_integration_not_found() {
    let graph = KnowledgeGraph::sovereign_stack();
    let pattern = graph.get_integration("trueno", "bashrs");
    assert!(pattern.is_none());
}

#[test]
fn test_integration_has_code_template() {
    let graph = KnowledgeGraph::sovereign_stack();
    let pattern = graph.get_integration("aprender", "realizar").unwrap();
    assert!(pattern.code_template.is_some());
}

// =========================================================================
// Component Detail Tests
// =========================================================================

#[test]
fn test_trueno_capabilities() {
    let graph = KnowledgeGraph::sovereign_stack();
    let trueno = graph.get_component("trueno").unwrap();

    assert!(trueno.has_capability("simd"));
    assert!(trueno.has_capability("gpu"));
    assert!(trueno.has_capability("vector_ops"));
    assert!(trueno.has_capability("matrix_ops"));
}

#[test]
fn test_aprender_capabilities() {
    let graph = KnowledgeGraph::sovereign_stack();
    let aprender = graph.get_component("aprender").unwrap();

    assert!(aprender.has_capability("random_forest"));
    assert!(aprender.has_capability("linear_regression"));
    assert!(aprender.has_capability("gbm"));
    assert!(aprender.has_capability("kmeans"));
    assert!(aprender.has_capability("pca"));
    assert!(aprender.has_capability("standard_scaler"));
}

#[test]
fn test_repartir_capabilities() {
    let graph = KnowledgeGraph::sovereign_stack();
    let repartir = graph.get_component("repartir").unwrap();

    assert!(repartir.has_capability("work_stealing"));
    assert!(repartir.has_capability("cpu_executor"));
    assert!(repartir.has_capability("gpu_executor"));
}

#[test]
fn test_realizar_capabilities() {
    let graph = KnowledgeGraph::sovereign_stack();
    let realizar = graph.get_component("realizar").unwrap();

    assert!(realizar.has_capability("model_serving"));
    assert!(realizar.has_capability("gguf"));
    assert!(realizar.has_capability("safetensors"));
    assert!(realizar.has_capability("transformer_serving"));
    assert!(realizar.has_capability("continuous_batching"));
    assert!(realizar.has_capability("lambda"));
}

#[test]
fn test_apr_qa_capabilities() {
    let graph = KnowledgeGraph::sovereign_stack();
    let apr_qa = graph.get_component("apr-qa").unwrap();

    assert_eq!(apr_qa.layer, StackLayer::Quality);
    assert!(apr_qa.has_capability("qa_test_generation"));
    assert!(apr_qa.has_capability("model_validation"));
    assert!(apr_qa.has_capability("qa_runner"));
    assert!(apr_qa.has_capability("benchmark_runner"));
    assert!(apr_qa.has_capability("qa_report"));
    assert!(apr_qa.has_capability("coverage_report"));
}

#[test]
fn test_apr_qa_integrations() {
    let graph = KnowledgeGraph::sovereign_stack();

    let to_aprender = graph.get_integration("apr-qa", "aprender");
    assert!(to_aprender.is_some());
    assert_eq!(to_aprender.unwrap().pattern_name, "model_validation");

    let to_realizar = graph.get_integration("apr-qa", "realizar");
    assert!(to_realizar.is_some());

    let to_certeza = graph.get_integration("apr-qa", "certeza");
    assert!(to_certeza.is_some());
}

// =========================================================================
// Statistics Tests
// =========================================================================

#[test]
fn test_all_capabilities() {
    let graph = KnowledgeGraph::sovereign_stack();
    let caps: Vec<_> = graph.all_capabilities().collect();

    assert!(caps.len() > 30);
    assert!(caps.contains(&&"simd".to_string()));
    assert!(caps.contains(&&"random_forest".to_string()));
}

#[test]
fn test_capability_count() {
    let graph = KnowledgeGraph::sovereign_stack();
    assert!(graph.capability_count() > 30);
}

// =========================================================================
// Version Tests
// =========================================================================

#[test]
fn test_component_versions() {
    let graph = KnowledgeGraph::sovereign_stack();

    assert_eq!(graph.get_component("trueno").unwrap().version, "0.11.0");
    assert_eq!(graph.get_component("trueno-db").unwrap().version, "0.3.3");
    assert_eq!(graph.get_component("trueno-graph").unwrap().version, "0.1.1");
    assert_eq!(graph.get_component("aprender").unwrap().version, "0.21.0");
    assert_eq!(graph.get_component("repartir").unwrap().version, "2.0.0");
    assert_eq!(graph.get_component("pepita").unwrap().version, "0.1.0");
    assert_eq!(graph.get_component("realizar").unwrap().version, "0.4.0");
    assert_eq!(graph.get_component("renacer").unwrap().version, "0.7.0");
}
