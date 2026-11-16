//! `apr-recipe-v1` schema tests (contracts/apr-recipe-v1.yaml, PMAT-5316).
//!
//! Every refusal must name the recipe field at fault, and a valid recipe must
//! pass: a validator that refuses everything satisfies the negative cases.

use super::*;

const VALID: &str = "\
recipe: apr-recipe-v1
name: qwen-lora-demo
base: models/base.apr
method: lora
data:
  train: data/train.jsonl
eval:
  heldout: data/heldout.jsonl
seed: 42
";

fn parse(src: &str) -> Result<TrainRecipe> {
    TrainRecipe::parse(src, Path::new("/recipes"), "r.yaml")
}

/// The refusal text for `src`; panics if the recipe was accepted.
fn refusal(src: &str) -> String {
    match parse(src) {
        Ok(r) => panic!("recipe must be refused, but parsed to {r:?}"),
        Err(e) => e.to_string(),
    }
}

/// `VALID` with one line removed (and any indented lines under it).
fn without(key: &str) -> String {
    let mut out = String::new();
    let mut skipping = false;
    for line in VALID.lines() {
        if line.starts_with(key) {
            skipping = true;
            continue;
        }
        if skipping && line.starts_with(' ') {
            continue;
        }
        skipping = false;
        out.push_str(line);
        out.push('\n');
    }
    assert_ne!(out, VALID, "`{key}` was not in the fixture");
    out
}

#[test]
fn valid_minimal_recipe_parses_with_paths_anchored_at_the_recipe_dir() {
    let r = parse(VALID).expect("the positive control must parse");
    assert_eq!(r.name, "qwen-lora-demo");
    assert_eq!(r.method, RecipeMethod::Lora);
    assert_eq!(r.base, Path::new("/recipes/models/base.apr"));
    assert_eq!(r.train, Path::new("/recipes/data/train.jsonl"));
    assert_eq!(r.heldout, Path::new("/recipes/data/heldout.jsonl"));
    assert_eq!(r.seed, 42);
    assert_eq!(r.teacher, None);
    assert_eq!(
        (r.rank, r.epochs, r.learning_rate, r.max_seq_len),
        (None, None, None, None)
    );
    assert_eq!(r.output_dir, None);
    assert_eq!(r.benchmark, None);
    assert_eq!(r.model_size, None);
    assert_eq!(r.data_sha256_pin, None);
    assert_eq!(r.recipe_sha256, sha256_bytes(VALID.as_bytes()));
}

#[test]
fn absolute_paths_are_kept() {
    let r = parse(&VALID.replace("models/base.apr", "/abs/base.apr")).expect("parses");
    assert_eq!(r.base, Path::new("/abs/base.apr"));
}

/// FALSIFY-APR-RECIPE-001: the R-1 first-green proof. A recipe with no `eval`
/// block is refused, and the text names the field.
#[test]
fn missing_eval_is_refused_and_the_error_names_the_field() {
    let msg = refusal(&without("eval:"));
    assert!(msg.contains("field `eval` is required"), "{msg}");
    assert!(
        msg.contains("r.yaml"),
        "the refusal names the recipe: {msg}"
    );
    assert!(msg.contains(RECIPE_SCHEMA), "{msg}");
}

#[test]
fn each_missing_required_field_is_named() {
    for (key, field) in [
        ("recipe:", "`recipe`"),
        ("name:", "`name`"),
        ("base:", "`base`"),
        ("method:", "`method`"),
        ("data:", "`data`"),
        ("seed:", "`seed`"),
    ] {
        let msg = refusal(&without(key));
        assert!(
            msg.contains(&format!("field {field} is required")),
            "dropping {key} must name {field}: {msg}"
        );
    }
}

#[test]
fn missing_nested_required_fields_are_named() {
    let msg = refusal(&VALID.replace("  train: data/train.jsonl\n", "  sha256: null\n"));
    assert!(msg.contains("field `data.train` is required"), "{msg}");
    let msg = refusal(&VALID.replace("  heldout: data/heldout.jsonl\n", "  benchmark: x\n"));
    assert!(msg.contains("field `eval.heldout` is required"), "{msg}");
}

#[test]
fn empty_strings_are_refused_by_field() {
    let msg = refusal(&VALID.replace("base: models/base.apr", "base: \"  \""));
    assert!(msg.contains("field `base` is empty"), "{msg}");
    let msg = refusal(&VALID.replace(
        "  heldout: data/heldout.jsonl\n",
        "  heldout: data/heldout.jsonl\n  benchmark: \"\"\n",
    ));
    assert!(msg.contains("field `eval.benchmark` is empty"), "{msg}");
    let msg = refusal(&format!("{VALID}model_size: \"\"\n"));
    assert!(msg.contains("field `model_size` is empty"), "{msg}");
    let msg = refusal(&format!("{VALID}output:\n  dir: \"\"\n"));
    assert!(msg.contains("field `output.dir` is empty"), "{msg}");
    let msg = refusal(&format!("{VALID}output: {{}}\n"));
    assert!(msg.contains("field `output.dir` is required"), "{msg}");
}

#[test]
fn every_violation_is_reported_at_once() {
    let msg = refusal("recipe: apr-recipe-v1\nname: x\n");
    for field in ["`base`", "`method`", "`data`", "`eval`", "`seed`"] {
        assert!(msg.contains(field), "{field} missing from: {msg}");
    }
}

#[test]
fn wrong_schema_id_and_unknown_method_are_refused() {
    let msg = refusal(&VALID.replace("apr-recipe-v1", "apr-recipe-v0"));
    assert!(
        msg.contains("field `recipe` must be `apr-recipe-v1`"),
        "{msg}"
    );
    assert!(msg.contains("apr-recipe-v0"), "{msg}");
    let msg = refusal(&VALID.replace("method: lora", "method: dora"));
    assert!(msg.contains("field `method` must be one of"), "{msg}");
    assert!(msg.contains("dora"), "{msg}");
}

#[test]
fn unknown_keys_are_refused_by_name_at_every_level() {
    let msg = refusal(&format!("{VALID}evall:\n  heldout: x\n"));
    assert!(msg.contains("evall"), "{msg}");
    let msg = refusal(&VALID.replace("  train:", "  trian:"));
    assert!(msg.contains("trian"), "{msg}");
    let msg = refusal(&format!("{VALID}training:\n  lr: 0.1\n"));
    assert!(msg.contains("lr"), "{msg}");
    let msg = refusal(&VALID.replace("  heldout:", "  heldot:"));
    assert!(msg.contains("heldot"), "{msg}");
    let msg = refusal(&format!("{VALID}teacher:\n  modle: t.apr\n"));
    assert!(msg.contains("modle"), "{msg}");
    let msg = refusal(&format!("{VALID}output:\n  dri: out\n"));
    assert!(msg.contains("dri"), "{msg}");
}

/// A wrong type or an unknown key is refused by the YAML reader, which stops
/// at the first one: the fields missing beside it are NOT listed. This pins
/// the limit the contract states, so widening it later is a visible change.
#[test]
fn a_reader_error_is_reported_alone() {
    let msg = refusal("recipe: apr-recipe-v1\nname: x\nseed: forty-two\n");
    assert!(msg.contains("r.yaml") && msg.contains("seed"), "{msg}");
    assert!(!msg.contains("field `base`"), "{msg}");
}

#[test]
fn wrong_type_is_refused() {
    let msg = refusal(&VALID.replace("seed: 42", "seed: forty-two"));
    assert!(msg.contains("r.yaml") && msg.contains("seed"), "{msg}");
    let msg = refusal(&VALID.replace("seed: 42", "seed: -1"));
    assert!(msg.contains("r.yaml") && msg.contains("seed"), "{msg}");
}

/// PMAT-5316 review: a negative `seed` (a value fault, not a type fault) is
/// listed beside the missing fields in one message.
#[test]
fn bad_seed_does_not_hide_the_other_faulty_fields() {
    let msg = refusal("recipe: apr-recipe-v1\nname: x\nseed: -1\n");
    for field in ["`base`", "`method`", "`data`", "`eval`", "`seed`"] {
        assert!(msg.contains(&format!("field {field}")), "{field}: {msg}");
    }
}

/// A `training:` key whose sub-keys are all commented out is YAML null, like
/// an empty `output:` or `teacher:`; it is the defaults, not a type error.
#[test]
fn empty_training_block_is_the_defaults() {
    let r = parse(&format!("{VALID}training:\n  # rank: 8\n")).expect("null training is ok");
    assert_eq!(r.rank, None);
}

/// PMAT-5316 review: `eval.heldout` spelled differently from `data.train` but
/// naming the same file is still the training file.
#[test]
fn heldout_alias_of_the_training_file_is_refused() {
    for alias in [
        "./data/train.jsonl",
        "data/../data/train.jsonl",
        "/recipes/data/train.jsonl",
        "data//train.jsonl",
        "../../recipes/data/train.jsonl",
        "/../recipes/data/train.jsonl",
    ] {
        let msg = refusal(&VALID.replace("data/heldout.jsonl", alias));
        assert!(
            msg.contains("field `eval.heldout` must not be the training file"),
            "{alias}: {msg}"
        );
    }
}

/// A `..` with no name left to remove is kept, so two different files under
/// a relative recipe directory are not mistaken for one.
#[test]
fn leading_parent_dirs_are_not_dropped() {
    assert_eq!(
        lexical_normalize(Path::new("../../a/./b/../c")),
        Path::new("../../a/c")
    );
    assert_eq!(lexical_normalize(Path::new("/../a")), Path::new("/a"));
    let src = VALID.replace("data/heldout.jsonl", "../../data/train.jsonl");
    let r = TrainRecipe::parse(&src, Path::new(""), "r.yaml").expect("a different file");
    assert_eq!(r.heldout, Path::new("../../data/train.jsonl"));
}

/// With `data.train` missing, the same-file rule has nothing to compare and
/// must not add a second message.
#[test]
fn same_file_rule_is_silent_without_a_training_file() {
    let src = VALID
        .replace("  train: data/train.jsonl\n", "  sha256: null\n")
        .replace("data/heldout.jsonl", ".");
    let msg = refusal(&src);
    assert!(msg.contains("field `data.train` is required"), "{msg}");
    assert!(!msg.contains("must not be the training file"), "{msg}");
}

#[test]
fn heldout_may_not_be_the_training_file() {
    let msg = refusal(&VALID.replace("data/heldout.jsonl", "data/train.jsonl"));
    assert!(
        msg.contains("field `eval.heldout` must not be the training file"),
        "{msg}"
    );
}

#[test]
fn data_pin_must_be_a_sha256() {
    for bad in [
        "abc".to_string(),
        "AB".repeat(32),
        "zz".repeat(32),
        "a1".repeat(32)[1..].to_string(),
        "a1".repeat(32) + "0",
    ] {
        let msg = refusal(&VALID.replace(
            "  train: data/train.jsonl\n",
            &format!("  train: data/train.jsonl\n  sha256: \"{bad}\"\n"),
        ));
        assert!(msg.contains("field `data.sha256`"), "{bad}: {msg}");
    }
}

#[test]
fn training_numbers_must_be_positive() {
    for (line, field) in [
        ("rank: 0", "`training.rank`"),
        ("epochs: 0", "`training.epochs`"),
        ("max_seq_len: 0", "`training.max_seq_len`"),
        ("learning_rate: 0.0", "`training.learning_rate`"),
        ("learning_rate: -0.5", "`training.learning_rate`"),
        ("learning_rate: .nan", "`training.learning_rate`"),
        ("learning_rate: .inf", "`training.learning_rate`"),
    ] {
        let msg = refusal(&format!("{VALID}training:\n  {line}\n"));
        assert!(msg.contains(field), "{line}: {msg}");
    }
}

const DISTILL: &str = "\
recipe: apr-recipe-v1
name: qwen-distill-demo
base: models/student.apr
method: distill
teacher:
  model: models/teacher.apr
data:
  train: data/train.jsonl
eval:
  heldout: data/heldout.jsonl
seed: 7
";

#[test]
fn distill_recipe_parses_with_teacher_defaults() {
    let r = parse(DISTILL).expect("distill recipe parses");
    assert_eq!(r.method, RecipeMethod::Distill);
    assert_eq!(
        r.teacher,
        Some(RecipeTeacher {
            model: PathBuf::from("/recipes/models/teacher.apr"),
            temperature: 3.0,
            alpha: 0.7,
        })
    );
    let r = parse(&DISTILL.replace(
        "  model: models/teacher.apr\n",
        "  model: models/teacher.apr\n  temperature: 2.0\n  alpha: 0.5\n",
    ))
    .expect("explicit teacher numbers parse");
    let t = r.teacher.expect("teacher");
    assert_eq!((t.temperature, t.alpha), (2.0, 0.5));
}

#[test]
fn teacher_rules_are_enforced_by_field() {
    let msg = refusal(&DISTILL.replace("teacher:\n  model: models/teacher.apr\n", ""));
    assert!(
        msg.contains("field `teacher` is required when `method` is distill"),
        "{msg}"
    );
    let msg = refusal(&DISTILL.replace("method: distill", "method: lora"));
    assert!(
        msg.contains("field `teacher` is only valid when `method` is distill"),
        "{msg}"
    );
    let msg = refusal(&DISTILL.replace("  model: models/teacher.apr\n", "  alpha: 0.5\n"));
    assert!(msg.contains("field `teacher.model` is required"), "{msg}");
    for (line, field) in [
        ("temperature: 0.0", "`teacher.temperature`"),
        ("temperature: .inf", "`teacher.temperature`"),
        ("alpha: 1.5", "`teacher.alpha`"),
        ("alpha: -0.1", "`teacher.alpha`"),
        ("temperature: -1.0", "`teacher.temperature`"),
        ("temperature: .nan", "`teacher.temperature`"),
        ("alpha: .nan", "`teacher.alpha`"),
    ] {
        let msg = refusal(&DISTILL.replace(
            "  model: models/teacher.apr\n",
            &format!("  model: models/teacher.apr\n  {line}\n"),
        ));
        assert!(msg.contains(field), "{line}: {msg}");
    }
}

/// The positive control for the optional keys: a value that passes its rule
/// reaches the parsed recipe, so a refusal test cannot pass by dropping it.
#[test]
fn valid_optional_fields_are_kept() {
    let pin = "0f3a9c5e".repeat(8);
    let src = VALID.replace(
        "  train: data/train.jsonl\n",
        &format!("  train: data/train.jsonl\n  sha256: {pin}\n"),
    )
        + "model_size: \" 0.8B \"\ntraining:\n  rank: 8\n  epochs: 3\n  learning_rate: 0.0002\n  \
         max_seq_len: 512\noutput:\n  dir: out\n";
    let src = src.replace(
        "  heldout: data/heldout.jsonl\n",
        "  heldout: data/heldout.jsonl\n  benchmark: humaneval\n",
    );
    let r = parse(&src).expect("every optional value here is valid");
    assert_eq!(r.data_sha256_pin.as_deref(), Some(pin.as_str()));
    assert_eq!(r.model_size.as_deref(), Some("0.8B"));
    assert_eq!(r.benchmark.as_deref(), Some("humaneval"));
    assert_eq!(
        (r.rank, r.epochs, r.learning_rate, r.max_seq_len),
        (Some(8), Some(3), Some(0.0002), Some(512))
    );
    assert_eq!(r.output_dir.as_deref(), Some(Path::new("/recipes/out")));
}

#[test]
fn method_spellings_round_trip() {
    for m in [
        RecipeMethod::Lora,
        RecipeMethod::Qlora,
        RecipeMethod::Distill,
    ] {
        assert_eq!(RecipeMethod::parse(m.as_str()), Some(m));
    }
    assert_eq!(
        RecipeMethod::parse("LoRA"),
        None,
        "the schema is lower-case"
    );
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("apr-recipe-{tag}-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(dir.join("data")).expect("scratch data dir");
    std::fs::create_dir_all(dir.join("models")).expect("scratch models dir");
    dir
}

#[test]
fn teacher_alpha_bounds_are_inclusive() {
    for alpha in ["0.0", "1.0"] {
        let r = parse(&DISTILL.replace(
            "  model: models/teacher.apr\n",
            &format!("  model: models/teacher.apr\n  alpha: {alpha}\n"),
        ))
        .expect("alpha at a bound parses");
        let want: f64 = alpha.parse().expect("number");
        assert_eq!(r.teacher.expect("teacher").alpha, want);
    }
}

/// An unknown method is one fault, not two: the teacher rule waits until the
/// method is known.
#[test]
fn teacher_rule_is_silent_when_the_method_is_unknown() {
    let msg = refusal(&DISTILL.replace("method: distill", "method: dora"));
    assert!(msg.contains("field `method` must be one of"), "{msg}");
    assert!(!msg.contains("field `teacher`"), "{msg}");
}

/// An empty heldout is reported as empty, not also as the training file.
#[test]
fn same_file_rule_is_silent_without_a_heldout_file() {
    let src = VALID
        .replace("data/train.jsonl", ".")
        .replace("data/heldout.jsonl", "\"\"");
    let msg = refusal(&src);
    assert!(msg.contains("field `eval.heldout`"), "{msg}");
    assert!(!msg.contains("must not be the training file"), "{msg}");
}

#[test]
fn load_names_the_file_in_a_refusal() {
    let dir = scratch("load-refusal");
    let path = dir.join("bad.yaml");
    std::fs::write(&path, without("eval:")).expect("write recipe");
    let msg = TrainRecipe::load(&path).expect_err("refused").to_string();
    assert!(msg.contains("field `eval` is required"), "{msg}");
    assert!(msg.contains(&path.display().to_string()), "{msg}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn load_reads_a_file_and_names_it_when_it_cannot() {
    let dir = scratch("load");
    let path = dir.join("r.yaml");
    std::fs::write(&path, VALID).expect("write recipe");
    let r = TrainRecipe::load(&path).expect("load");
    assert_eq!(r.base, dir.join("models/base.apr"));

    let missing = dir.join("absent.yaml");
    let msg = TrainRecipe::load(&missing)
        .expect_err("absent recipe")
        .to_string();
    assert!(msg.contains("cannot read recipe"), "{msg}");
    assert!(msg.contains("absent.yaml"), "{msg}");

    std::fs::write(&path, [0xff_u8, 0xfe, 0x00]).expect("write bytes");
    let msg = TrainRecipe::load(&path).expect_err("not utf-8").to_string();
    assert!(msg.contains("is not UTF-8"), "{msg}");
    std::fs::remove_dir_all(&dir).ok();
}
