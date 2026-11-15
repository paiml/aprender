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
fn full_recipe_parses_every_optional_field() {
    let src = format!(
        "{VALID}training:\n  rank: 16\n  epochs: 2\n  learning_rate: 0.0001\n  max_seq_len: 256\n\
         output:\n  dir: out/run1\n"
    )
    .replace(
        "  train: data/train.jsonl\n",
        &format!("  train: data/train.jsonl\n  sha256: {}\n", "ab".repeat(32)),
    )
    .replace(
        "  heldout: data/heldout.jsonl\n",
        "  heldout: data/heldout.jsonl\n  benchmark: humaneval-subset\n",
    );
    let src = src.replace("method: lora\n", "method: lora\nmodel_size: \" 0.8B \"\n");
    let r = parse(&src).expect("full recipe parses");
    assert_eq!(r.model_size.as_deref(), Some("0.8B"));
    assert_eq!(r.rank, Some(16));
    assert_eq!(r.epochs, Some(2));
    assert_eq!(r.learning_rate, Some(0.0001));
    assert_eq!(r.max_seq_len, Some(256));
    assert_eq!(
        r.output_dir.as_deref(),
        Some(Path::new("/recipes/out/run1"))
    );
    assert_eq!(r.benchmark.as_deref(), Some("humaneval-subset"));
    assert_eq!(r.data_sha256_pin, Some("ab".repeat(32)));
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
}

#[test]
fn wrong_type_is_refused() {
    let msg = refusal(&VALID.replace("seed: 42", "seed: forty-two"));
    assert!(msg.contains("r.yaml") && msg.contains("seed"), "{msg}");
    let msg = refusal(&VALID.replace("seed: 42", "seed: -1"));
    assert!(msg.contains("r.yaml") && msg.contains("seed"), "{msg}");
}

/// PMAT-5316 review: the refusal names EVERY field at fault in one message,
/// a bad `seed` included, rather than aborting on the first type error.
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
    ] {
        let msg = refusal(&VALID.replace("data/heldout.jsonl", alias));
        assert!(
            msg.contains("field `eval.heldout` must not be the training file"),
            "{alias}: {msg}"
        );
    }
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
    for bad in ["abc".to_string(), "AB".repeat(32), "zz".repeat(32)] {
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
    ] {
        let msg = refusal(&DISTILL.replace(
            "  model: models/teacher.apr\n",
            &format!("  model: models/teacher.apr\n  {line}\n"),
        ));
        assert!(msg.contains(field), "{line}: {msg}");
    }
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

#[test]
fn require_inputs_names_each_missing_file_then_passes_and_checks_the_pin() {
    let dir = scratch("inputs");
    let recipe = dir.join("r.yaml");
    std::fs::write(&recipe, DISTILL).expect("write recipe");
    let r = TrainRecipe::load(&recipe).expect("load");
    let msg = r
        .require_inputs()
        .expect_err("nothing exists yet")
        .to_string();
    for field in [
        "`base`",
        "`data.train`",
        "`eval.heldout`",
        "`teacher.model`",
    ] {
        assert!(
            msg.contains(&format!("field {field}: no such file")),
            "{msg}"
        );
    }

    std::fs::write(dir.join("models/student.apr"), b"s").expect("student");
    std::fs::write(dir.join("models/teacher.apr"), b"t").expect("teacher");
    std::fs::write(dir.join("data/train.jsonl"), b"{\"a\":1}\n").expect("train");
    std::fs::write(dir.join("data/heldout.jsonl"), b"{\"a\":2}\n").expect("heldout");
    r.require_inputs().expect("every input exists");

    // The pin: right hash passes, a planted-wrong hash is refused by field.
    let actual = sha256_file(&dir.join("data/train.jsonl")).expect("hash");
    assert_eq!(actual, sha256_bytes(b"{\"a\":1}\n"));
    let pinned = |hash: &str| {
        DISTILL.replace(
            "  train: data/train.jsonl\n",
            &format!("  train: data/train.jsonl\n  sha256: {hash}\n"),
        )
    };
    std::fs::write(&recipe, pinned(&actual)).expect("write pinned");
    TrainRecipe::load(&recipe)
        .expect("load pinned")
        .require_inputs()
        .expect("the true hash passes");
    std::fs::write(&recipe, pinned(&"0".repeat(64))).expect("write wrong pin");
    let msg = TrainRecipe::load(&recipe)
        .expect("load wrong pin")
        .require_inputs()
        .expect_err("a planted-wrong data hash is refused")
        .to_string();
    assert!(msg.contains("field `data.sha256` pins"), "{msg}");
    assert!(msg.contains(&actual), "{msg}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn sha256_matches_the_published_test_vector() {
    assert_eq!(
        sha256_bytes(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(sha256_file(Path::new("/nonexistent/recipe/input")), None);
}
