//! Problem domain to capability mappings.

use super::super::types::ProblemDomain;
use super::types::KnowledgeGraph;

impl KnowledgeGraph {
    /// Initialize problem domain to capability mappings
    pub(crate) fn initialize_domain_mappings(&mut self) {
        use ProblemDomain::{
            CMigration, DataPipeline, DeepLearning, DistributedCompute, GraphAnalytics, Inference,
            LinearAlgebra, MediaProduction, ModelServing, Profiling, PythonMigration,
            ShellMigration, SpeechRecognition, SupervisedLearning, Testing, UnsupervisedLearning,
            Validation, VectorSearch,
        };

        self.domain_capabilities.insert(
            SupervisedLearning,
            vec![
                "linear_regression".into(),
                "logistic_regression".into(),
                "decision_tree".into(),
                "random_forest".into(),
                "gbm".into(),
                "naive_bayes".into(),
                "knn".into(),
                "svm".into(),
            ],
        );

        self.domain_capabilities.insert(
            UnsupervisedLearning,
            vec!["kmeans".into(), "pca".into(), "dbscan".into(), "hierarchical".into()],
        );

        self.domain_capabilities.insert(
            DeepLearning,
            vec!["autograd".into(), "lora".into(), "qlora".into(), "quantization".into()],
        );

        self.domain_capabilities.insert(
            Inference,
            vec!["model_serving".into(), "batching".into(), "moe_routing".into()],
        );

        self.domain_capabilities.insert(
            SpeechRecognition,
            vec![
                "speech_recognition".into(),
                "streaming_transcription".into(),
                "multilingual".into(),
                "whisper_quantization".into(),
            ],
        );

        self.domain_capabilities.insert(
            LinearAlgebra,
            vec!["vector_ops".into(), "matrix_ops".into(), "simd".into(), "gpu".into()],
        );

        self.domain_capabilities.insert(
            VectorSearch,
            vec!["vector_store".into(), "similarity_search".into(), "knn_search".into()],
        );

        self.domain_capabilities.insert(
            GraphAnalytics,
            vec!["pathfinding".into(), "centrality".into(), "community_detection".into()],
        );

        self.domain_capabilities.insert(
            PythonMigration,
            vec!["type_inference".into(), "sklearn_to_aprender".into(), "numpy_to_trueno".into()],
        );

        self.domain_capabilities
            .insert(CMigration, vec!["ownership_inference".into(), "unsafe_elimination".into()]);

        self.domain_capabilities
            .insert(ShellMigration, vec!["script_conversion".into(), "cli_generation".into()]);

        self.domain_capabilities.insert(
            DistributedCompute,
            vec![
                "work_stealing".into(),
                "cpu_executor".into(),
                "gpu_executor".into(),
                "remote_executor".into(),
            ],
        );

        self.domain_capabilities.insert(
            DataPipeline,
            vec!["csv".into(), "parquet".into(), "json".into(), "streaming".into()],
        );

        self.domain_capabilities.insert(
            ModelServing,
            vec!["model_serving".into(), "lambda".into(), "container".into(), "edge".into()],
        );

        self.domain_capabilities.insert(
            Testing,
            vec![
                "coverage_check".into(),
                "mutation_testing".into(),
                "tdg_scoring".into(),
                "parity_checking".into(),
                "oracle_generation".into(),
                "falsification_testing".into(),
                "quantization_drift".into(),
                "roundtrip_validation".into(),
            ],
        );

        self.domain_capabilities.insert(
            Profiling,
            vec!["syscall_trace".into(), "flamegraph".into(), "golden_trace_comparison".into()],
        );

        self.domain_capabilities.insert(
            Validation,
            vec![
                "privacy_audit".into(),
                "quality_gates".into(),
                "complexity_analysis".into(),
                "contract_parsing".into(),
                "scaffold_generation".into(),
                "kani_verification".into(),
                "probar_generation".into(),
                "traceability_audit".into(),
                "binding_registry".into(),
            ],
        );

        self.domain_capabilities.insert(
            MediaProduction,
            vec![
                "video_rendering".into(),
                "mlt_xml".into(),
                "ffmpeg_encode".into(),
                "transition_blend".into(),
                "course_production".into(),
                "audio_processing".into(),
                "subtitle_burn_in".into(),
                "media_concat".into(),
                "transcription_qa".into(),
                "vocabulary_enrichment".into(),
                "content_generation".into(),
                "svg_asset_pipeline".into(),
                "course_quality_scoring".into(),
                "course_publishing".into(),
            ],
        );
    }
}
