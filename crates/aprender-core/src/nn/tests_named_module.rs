//! Named-parameter traversal (ENC-04) and mode-propagation (ENC-05) tests.
//!
//! Proves, for every module on the BERT encoder path — `Linear`, `LayerNorm`,
//! `Dropout`, `Sequential`, `MultiHeadAttention`:
//!
//! 1. `named_parameters()` has the same LENGTH and ORDER as `parameters()`, and
//!    element `i` of each is literally the same tensor (compared by `TensorId`).
//! 2. Names are SEMANTIC (`"weight"`, `"q_proj.bias"`), not positional fallbacks.
//! 3. Names are unique within one implementor's output — a duplicate key would let
//!    prefix-matched freeze groups address the wrong tensor, or zero tensors.
//! 4. `set_training` recurses through composites into every `Dropout` child, and
//!    propagates through the `set_training` channel (not only via `train`/`eval`).
//! 5. Flipping `train -> eval -> train` leaves every registered parameter
//!    byte-identical (`f32::to_bits`) and the name SEQUENCE unchanged.
//!
//! This module also hosts [`snapshot_named`], the shared byte-identity helper. It is
//! `pub(crate)` on purpose: the encoder conformance tests (plan 01-06) live in a
//! different module and must reuse this helper rather than duplicate it. A private
//! `fn` inside a `mod tests` block would not be reachable from `crate::setfit::…`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::autograd::Tensor;
use crate::nn::{Dropout, LayerNorm, Linear, Module, MultiHeadAttention, Sequential};

// ---------------------------------------------------------------------------
// Shared helpers (reused by plan 01-06's encoder conformance tests)
// ---------------------------------------------------------------------------

/// Snapshot `(name, per-element f32 bit pattern)` for every named parameter.
///
/// Bit patterns rather than `f32` comparison: this is the ENC-05 proof that a
/// mode switch mutates nothing. `to_bits` distinguishes `-0.0` from `0.0` and
/// makes `NaN` comparable, so a snapshot equality is an exact-bytes claim.
pub(crate) fn snapshot_named(m: &dyn Module) -> Vec<(String, Vec<u32>)> {
    m.named_parameters()
        .into_iter()
        .map(|(name, tensor)| {
            let bits = tensor.data().iter().map(|v| v.to_bits()).collect();
            (name, bits)
        })
        .collect()
}

/// Collect just the name sequence, in traversal order.
fn names_of(m: &dyn Module) -> Vec<String> {
    m.named_parameters()
        .into_iter()
        .map(|(name, _)| name)
        .collect()
}

/// Assert the universal invariant: named and positional agree in arity, order, and
/// tensor identity, and every name is unique.
fn assert_named_agrees_with_positional(label: &str, m: &dyn Module) {
    let positional = m.parameters();
    let named = m.named_parameters();

    assert_eq!(
        named.len(),
        positional.len(),
        "{label}: named_parameters() arity ({}) must equal parameters() arity ({})",
        named.len(),
        positional.len()
    );

    for (i, (name, tensor)) in named.iter().enumerate() {
        assert_eq!(
            tensor.id(),
            positional[i].id(),
            "{label}: named[{i}] (\"{name}\") is a different tensor than parameters()[{i}]"
        );
    }

    let mut sorted: Vec<&String> = named.iter().map(|(n, _)| n).collect();
    let total = sorted.len();
    sorted.sort();
    sorted.dedup();
    assert_eq!(
        sorted.len(),
        total,
        "{label}: duplicate parameter name in {:?}",
        names_of(m)
    );
}

/// Assert that a `train -> eval -> train` cycle changes no registered parameter.
fn assert_mode_flip_is_byte_identical(label: &str, m: &mut dyn Module) {
    let before = snapshot_named(&*m);

    m.set_training(true);
    m.set_training(false);
    m.set_training(true);

    let after = snapshot_named(&*m);

    let before_names: Vec<&String> = before.iter().map(|(n, _)| n).collect();
    let after_names: Vec<&String> = after.iter().map(|(n, _)| n).collect();
    assert_eq!(
        before_names, after_names,
        "{label}: mode flip changed the name SEQUENCE"
    );
    assert_eq!(
        before, after,
        "{label}: mode flip mutated parameter bytes (ENC-05 violation)"
    );
}

// ---------------------------------------------------------------------------
// Leaf: Linear
// ---------------------------------------------------------------------------

#[test]
fn named_module_linear_with_bias_has_semantic_names() {
    let layer = Linear::with_seed(4, 3, Some(7));
    assert_eq!(names_of(&layer), vec!["weight", "bias"]);
    assert_named_agrees_with_positional("Linear(with bias)", &layer);
}

#[test]
fn named_module_linear_without_bias_has_weight_only() {
    let layer = Linear::without_bias_with_seed(4, 3, Some(7));
    assert_eq!(names_of(&layer), vec!["weight"]);
    assert_named_agrees_with_positional("Linear(no bias)", &layer);
}

#[test]
fn named_module_linear_named_mut_mirrors_positional() {
    let mut layer = Linear::with_seed(4, 3, Some(7));
    let positional_len = layer.parameters_mut().len();
    let named = layer.named_parameters_mut();
    let names: Vec<String> = named.iter().map(|(n, _)| n.clone()).collect();
    assert_eq!(names, vec!["weight", "bias"]);
    assert_eq!(named.len(), positional_len);
}

#[test]
fn named_module_linear_without_bias_named_mut_mirrors_positional() {
    let mut layer = Linear::without_bias_with_seed(4, 3, Some(7));
    let positional_len = layer.parameters_mut().len();
    let named = layer.named_parameters_mut();
    let names: Vec<String> = named.iter().map(|(n, _)| n.clone()).collect();
    assert_eq!(names, vec!["weight"]);
    assert_eq!(named.len(), positional_len);
}

// ---------------------------------------------------------------------------
// Leaf: LayerNorm
// ---------------------------------------------------------------------------

#[test]
fn named_module_layernorm_has_weight_and_bias() {
    let norm = LayerNorm::new(&[4]);
    assert_eq!(names_of(&norm), vec!["weight", "bias"]);
    assert_named_agrees_with_positional("LayerNorm", &norm);
}

#[test]
fn named_module_layernorm_without_affine_has_no_names() {
    let norm = LayerNorm::without_affine(&[4]);
    assert!(
        names_of(&norm).is_empty(),
        "non-affine LayerNorm registers no learnable parameters"
    );
    assert_named_agrees_with_positional("LayerNorm(no affine)", &norm);
}

#[test]
fn named_module_layernorm_named_mut_mirrors_positional() {
    let mut norm = LayerNorm::new(&[4]);
    let positional_len = norm.parameters_mut().len();
    let named = norm.named_parameters_mut();
    let names: Vec<String> = named.iter().map(|(n, _)| n.clone()).collect();
    assert_eq!(names, vec!["weight", "bias"]);
    assert_eq!(named.len(), positional_len);
}

// ---------------------------------------------------------------------------
// Leaf: Dropout — RNG / seed / mode state must NEVER surface as a parameter
// ---------------------------------------------------------------------------

#[test]
fn named_module_dropout_has_no_named_parameters() {
    let drop = Dropout::new(0.5);
    assert!(
        names_of(&drop).is_empty(),
        "Dropout has no learnable parameters"
    );
    assert_named_agrees_with_positional("Dropout", &drop);
}

#[test]
fn named_module_seeded_dropout_does_not_expose_rng_state() {
    let drop = Dropout::with_seed(0.5, 1234);
    assert!(
        names_of(&drop).is_empty(),
        "seed/RNG state is not a learnable parameter and must not be named"
    );
    assert_named_agrees_with_positional("Dropout(seeded)", &drop);
}

#[test]
fn named_module_dropout_mode_flip_adds_no_parameters() {
    let mut drop = Dropout::with_seed(0.5, 1234);
    assert_mode_flip_is_byte_identical("Dropout", &mut drop);
    assert!(names_of(&drop).is_empty());
}

// ---------------------------------------------------------------------------
// Composite: MultiHeadAttention (BERT encoder path)
// ---------------------------------------------------------------------------

/// The exact q/k/v/out sequence a freeze group addresses. Order mirrors the
/// existing `MultiHeadAttention::parameters()`.
const MHA_EXPECTED_NAMES: [&str; 8] = [
    "q_proj.weight",
    "q_proj.bias",
    "k_proj.weight",
    "k_proj.bias",
    "v_proj.weight",
    "v_proj.bias",
    "out_proj.weight",
    "out_proj.bias",
];

#[test]
fn named_module_mha_emits_exact_semantic_name_sequence() {
    let mha = MultiHeadAttention::new(4, 2);
    // Vec<String> equality: order matters, a HashSet comparison would not catch a swap.
    assert_eq!(names_of(&mha), MHA_EXPECTED_NAMES.to_vec());
}

#[test]
fn named_module_mha_named_arity_equals_positional() {
    let mha = MultiHeadAttention::new(4, 2);
    assert_eq!(mha.named_parameters().len(), mha.parameters().len());
    assert_named_agrees_with_positional("MultiHeadAttention", &mha);
}

#[test]
fn named_module_mha_named_mut_mirrors_positional() {
    let mut mha = MultiHeadAttention::new(4, 2);
    let positional_len = mha.parameters_mut().len();
    let named = mha.named_parameters_mut();
    let names: Vec<String> = named.iter().map(|(n, _)| n.clone()).collect();
    assert_eq!(names, MHA_EXPECTED_NAMES.to_vec());
    assert_eq!(named.len(), positional_len);
}

#[test]
fn named_module_mha_mode_flip_is_byte_identical() {
    let mut mha = MultiHeadAttention::new(4, 2);
    assert_mode_flip_is_byte_identical("MultiHeadAttention", &mut mha);
}

#[test]
fn named_module_mha_set_training_flips_mode() {
    let mut mha = MultiHeadAttention::new(4, 2);
    mha.set_training(false);
    assert!(!mha.training());
    mha.set_training(true);
    assert!(mha.training());
}

// ---------------------------------------------------------------------------
// Composite: Sequential — index-dot prefixing and recursion
// ---------------------------------------------------------------------------

fn linear_dropout_linear() -> Sequential {
    Sequential::new()
        .add(Linear::with_seed(4, 3, Some(7)))
        .add(Dropout::with_seed(0.5, 1234))
        .add(Linear::with_seed(3, 2, Some(11)))
}

#[test]
fn named_module_sequential_prefixes_child_names_with_index() {
    let seq = linear_dropout_linear();
    // Index 1 is the Dropout: it contributes no parameters, so no "1.*" name
    // exists — and the surviving indices stay 0 and 2, never renumbered.
    assert_eq!(
        names_of(&seq),
        vec!["0.weight", "0.bias", "2.weight", "2.bias"]
    );
}

#[test]
fn named_module_sequential_named_agrees_with_positional() {
    let seq = linear_dropout_linear();
    assert_named_agrees_with_positional("Sequential", &seq);
}

#[test]
fn named_module_sequential_named_mut_mirrors_positional() {
    let mut seq = linear_dropout_linear();
    let positional_len = seq.parameters_mut().len();
    let named = seq.named_parameters_mut();
    let names: Vec<String> = named.iter().map(|(n, _)| n.clone()).collect();
    assert_eq!(
        names,
        vec!["0.weight", "0.bias", "2.weight", "2.bias"],
        "named_parameters_mut must mirror named_parameters exactly"
    );
    assert_eq!(named.len(), positional_len);
}

#[test]
fn named_module_nested_sequential_composes_dotted_paths() {
    let inner = Sequential::new().add(Linear::with_seed(4, 3, Some(7)));
    let outer = Sequential::new()
        .add(inner)
        .add(Linear::with_seed(3, 2, Some(11)));

    assert_eq!(
        names_of(&outer),
        vec!["0.0.weight", "0.0.bias", "1.weight", "1.bias"],
        "nested composites must compose prefixes, not flatten them"
    );
    assert_named_agrees_with_positional("Sequential(nested)", &outer);
}

#[test]
fn named_module_sequential_names_are_unique_across_identical_children() {
    // Two structurally identical Linears: only the index prefix distinguishes
    // them. A missing prefix would produce duplicate "weight"/"bias" keys.
    let seq = Sequential::new()
        .add(Linear::with_seed(4, 4, Some(7)))
        .add(Linear::with_seed(4, 4, Some(7)));

    assert_eq!(
        names_of(&seq),
        vec!["0.weight", "0.bias", "1.weight", "1.bias"]
    );
    assert_named_agrees_with_positional("Sequential(identical children)", &seq);
}

#[test]
fn named_module_sequential_set_training_recurses_into_dropout_children() {
    let mut seq = linear_dropout_linear();

    seq.set_training(false);
    assert!(!seq.training(), "Sequential must record its own mode");
    let dropout_child = seq.get(1).expect("index 1 is the Dropout child");
    assert!(
        !dropout_child.training(),
        "set_training(false) must recurse into every Dropout child"
    );

    seq.set_training(true);
    let dropout_child = seq.get(1).expect("index 1 is the Dropout child");
    assert!(
        dropout_child.training(),
        "set_training(true) must recurse into every Dropout child"
    );
}

/// Child that implements ONLY `set_training` — `train`/`eval` stay the trait
/// no-op defaults. A composite that propagates mode by calling `child.eval()`
/// would silently leave this child unflipped; propagating via `set_training`
/// reaches it. This pins the propagation CHANNEL, not just the outcome.
struct SetTrainingOnlyProbe {
    flag: Arc<AtomicBool>,
}

impl Module for SetTrainingOnlyProbe {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.clone()
    }

    fn set_training(&mut self, training: bool) {
        self.flag.store(training, Ordering::SeqCst);
    }

    fn training(&self) -> bool {
        self.flag.load(Ordering::SeqCst)
    }
}

#[test]
fn named_module_sequential_propagates_through_set_training_channel() {
    let flag = Arc::new(AtomicBool::new(true));
    let mut seq =
        Sequential::new()
            .add(Linear::with_seed(4, 3, Some(7)))
            .add(SetTrainingOnlyProbe {
                flag: Arc::clone(&flag),
            });

    seq.set_training(false);
    assert!(
        !flag.load(Ordering::SeqCst),
        "Sequential::set_training must call set_training on children, \
         not only train()/eval() — a child overriding set_training alone was skipped"
    );

    seq.set_training(true);
    assert!(flag.load(Ordering::SeqCst));
}

// ---------------------------------------------------------------------------
// ENC-05: mode flips never touch parameters
// ---------------------------------------------------------------------------

#[test]
fn named_module_composite_mode_flip_is_byte_identical() {
    let mut seq = linear_dropout_linear();
    assert_mode_flip_is_byte_identical("Sequential(with Dropout)", &mut seq);
}

#[test]
fn named_module_mode_flip_preserves_name_sequence_and_arity() {
    let mut seq = linear_dropout_linear();
    let before = names_of(&seq);
    let before_len = seq.parameters().len();

    seq.set_training(false);
    seq.set_training(true);

    assert_eq!(
        names_of(&seq),
        before,
        "mode flip must not add, remove, or reorder names"
    );
    assert_eq!(
        seq.parameters().len(),
        before_len,
        "mode flip must not change positional arity"
    );
}

#[test]
fn named_module_leaf_mode_flips_are_byte_identical() {
    let mut linear = Linear::with_seed(4, 3, Some(7));
    assert_mode_flip_is_byte_identical("Linear", &mut linear);

    let mut norm = LayerNorm::new(&[4]);
    assert_mode_flip_is_byte_identical("LayerNorm", &mut norm);
}

// ---------------------------------------------------------------------------
// Cross-cutting: every BERT-path module returns semantic, non-fallback names
// ---------------------------------------------------------------------------

#[test]
fn named_module_bert_path_modules_never_use_positional_fallback_names() {
    // A purely numeric name means the implementor fell back to the trait default.
    // Freeze groups and optimizer partitions address these by semantic prefix, so
    // a fallback here would match zero tensors — silently, and with no error.
    let linear = Linear::with_seed(4, 3, Some(7));
    let norm = LayerNorm::new(&[4]);
    let mha = MultiHeadAttention::new(4, 2);

    for (label, names) in [
        ("Linear", names_of(&linear)),
        ("LayerNorm", names_of(&norm)),
        ("MultiHeadAttention", names_of(&mha)),
    ] {
        assert!(
            !names.is_empty(),
            "{label}: expected named parameters, found none"
        );
        for name in &names {
            assert!(
                name.parse::<usize>().is_err(),
                "{label}: name \"{name}\" is a positional fallback, not a semantic name"
            );
        }
    }
}
