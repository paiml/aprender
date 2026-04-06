//! Fine-tuning pipeline for code generation models
//!
//! This module implements SPEC-FT-001: Rust Test Generation Fine-Tuning Pipeline.
//!
//! # Features
//!
//! - CUDA auto-detection with fallback to CPU
//! - QLoRA fine-tuning with 4-bit quantization
//! - Popperian 100-point falsification QA
//! - Scientific reproducibility protocol
//!
//! # References
//!
//! - Hu et al. (2021) "LoRA: Low-Rank Adaptation of Large Language Models"
//! - Dettmers et al. (2023) "QLoRA: Efficient Finetuning of Quantized LLMs"
//! - Popper (1959) "The Logic of Scientific Discovery"

mod backward_graph;
pub mod classification;
pub mod classify_eval_report;
pub mod classify_pipeline;
pub mod classify_trainer;
pub mod classify_tuner;
mod corpus;
pub mod data_parallel;
mod device;
pub mod distributed;
mod eval;
mod fused_lora_clip;
pub mod gpu_backward_fallback;
pub mod gradient_server;
pub mod instruct_corpus;
pub mod instruct_pipeline;
pub mod instruct_trainer;
pub mod linear_probe;
pub mod multi_adapter_pipeline;
mod popperian;
mod reproducibility;
pub mod ring_allreduce;
pub mod training_plan;
pub mod tune_searchers;
pub mod wgpu_pipeline;
pub mod worker_client;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_classification_contract_falsify;
#[cfg(test)]
mod tests_ssc_contract_falsify;

pub use classification::{
    bce_with_logits_loss, compute_class_weights, corpus_stats, cross_entropy_loss,
    load_multi_label_corpus, load_safety_corpus, ClassWeightStrategy, ClassificationHead,
    MultiLabelSafetySample, SafetyCorpusStats, SafetySample, TokenizedSample,
};
pub use classify_eval_report::{evaluate_checkpoint, ClassifyEvalReport, SSC_LABELS};
pub use classify_pipeline::{
    BatchResult, ClassifyConfig, ClassifyPipeline, DataStats, DiagSeverity, HyperparamDiagnostic,
    HyperparamDiagnostics,
};
pub use classify_trainer::{ClassifyTrainer, EpochMetrics, TrainResult, TrainingConfig};
pub use classify_tuner::{
    default_classify_search_space, extract_trial_params, ClassifyTuner, SchedulerKind,
    TrialSummary, TuneConfig, TuneResult, TuneScheduler, TuneSearcher, TuneStrategy,
};
pub use corpus::{CorpusStats, SampleMetadata, TestGenCorpus, TestGenSample};
pub use data_parallel::{
    average_gradients, has_non_finite, shard_samples, DataParallelCoordinator,
};
pub use device::{ComputeDevice, DeviceInfo};
pub use distributed::{DistributedConfig, NodeRole, WireMessage};
pub use eval::{
    contains_tautology, count_test_functions, has_edge_case_tests, has_meaningful_assertions,
    EvalMetrics, EvalResult, TestEvaluator,
};
pub use gradient_server::{
    AllReduceResult, BlockAllReduceResult, GradientServer, NonBlockAllReduceResult,
};
pub use instruct_corpus::{
    format_chat_prompt, instruct_corpus_stats, load_instruct_corpus, InstructCorpusStats,
    InstructMetadata, InstructSample,
};
pub use instruct_pipeline::{
    GenerateConfig, InstructBatchResult, InstructConfig, InstructPipeline, InstructStepResult,
};
pub use instruct_trainer::{
    InstructEpochMetrics, InstructTrainResult, InstructTrainer, InstructTrainingConfig,
};
pub use linear_probe::{
    binary_mcc, bootstrap_mcc_ci, check_ship_gate, compare_baselines, compute_confidence_scores,
    evaluate as evaluate_classification, generalization_test, should_escalate, BaselineComparison,
    BootstrapCI, ClassificationMetrics, ConfidenceScore, EscalationLevel, GeneralizationResult,
    LinearProbe, MlpProbe, ShipGateResult,
};
pub use multi_adapter_pipeline::{
    AdapterConfig, AdapterSchedule, AdapterSlot, MultiAdapterPipeline,
};
pub use popperian::{PopperianQA, QAGrade};
pub use reproducibility::{ExperimentLock, ReproducibilityConfig};
pub use ring_allreduce::{allreduce_pair, RingAllReduceWorker};
pub use training_plan::{
    execute_plan, plan as training_plan, ApplyConfig, CheckStatus, DataAudit, HyperparameterPlan,
    ManualConfig, ModelInfo, PlanConfig, PlanIssue, PlanVerdict, PreFlightCheck, ResourceEstimate,
    TrainingPlan, TrialPreview,
};
pub use worker_client::{
    AveragedBlockResult, AveragedNonBlockResult, AveragedResult, ShardAssignment, WorkerClient,
};
