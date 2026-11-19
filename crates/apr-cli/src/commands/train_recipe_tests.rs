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

/// A sample list: the receipt carries whatever the caller passes.
const NOT_APPLIED_SAMPLE: &[&str] = &["eval"];

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

/// A symlink or hard link is the same file on disk: `require_inputs` refuses
/// it even though the two paths differ lexically.
#[cfg(unix)]
#[test]
fn heldout_linked_to_the_training_file_is_refused_at_require_inputs() {
    let dir = std::env::temp_dir().join(format!("apr-recipe-alias-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    for f in ["base.apr", "train.jsonl"] {
        std::fs::write(dir.join(f), b"x").expect("write");
    }
    std::fs::remove_file(dir.join("held.jsonl")).ok();
    std::os::unix::fs::symlink(dir.join("train.jsonl"), dir.join("held.jsonl")).expect("link");
    let src = "recipe: apr-recipe-v1\nname: x\nbase: base.apr\nmethod: lora\n\
               data:\n  train: train.jsonl\neval:\n  heldout: held.jsonl\nseed: 1\n";
    let r = TrainRecipe::parse(src, &dir, "r.yaml").expect("lexically distinct");
    let msg = r.require_inputs().expect_err("same file").to_string();
    assert!(
        msg.contains("field `eval.heldout`") && msg.contains("training file"),
        "{msg}"
    );
    std::fs::remove_dir_all(&dir).ok();
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
    let msg = refusal(&DISTILL.replace("method: distill", "method: qlora"));
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

/// Any temperature above 0 is accepted: the bound is 0, not 1.
#[test]
fn teacher_temperature_just_above_zero_is_accepted() {
    let r = parse(&DISTILL.replace(
        "  model: models/teacher.apr\n",
        "  model: models/teacher.apr\n  temperature: 0.001\n",
    ))
    .expect("a small positive temperature parses");
    assert_eq!(r.teacher.expect("teacher").temperature, 0.001);
}

/// With an unknown method only the presence rule waits; the values inside the
/// teacher block are still checked.
#[test]
fn teacher_values_are_checked_when_the_method_is_unknown() {
    let msg = refusal(&DISTILL.replace("method: distill", "method: dora").replace(
        "  model: models/teacher.apr\n",
        "  model: models/teacher.apr\n  alpha: 1.5\n",
    ));
    assert!(msg.contains("field `method` must be one of"), "{msg}");
    assert!(msg.contains("field `teacher.alpha`"), "{msg}");
}

/// An unknown method is one fault, not two: the teacher rule waits until the
/// method is known.
#[test]
fn teacher_rule_is_silent_when_the_method_is_unknown() {
    let msg = refusal(&DISTILL.replace("method: distill", "method: dora"));
    assert!(msg.contains("field `method` must be one of"), "{msg}");
    assert!(!msg.contains("field `teacher` is"), "{msg}");
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
    assert!(msg.contains(&path.display().to_string()), "{msg}");
    std::fs::remove_dir_all(&dir).ok();
}

/// A recipe path that is not a regular file is refused without being opened:
/// `/dev/zero` never ends, and opening a pipe waits for a writer.
#[cfg(unix)]
#[test]
fn load_refuses_a_recipe_path_that_is_not_a_regular_file() {
    let dir = scratch("load-notfile");
    let pipe = dir.join("r.pipe");
    let made = std::process::Command::new("mkfifo")
        .arg(&pipe)
        .status()
        .expect("mkfifo runs");
    assert!(made.success(), "mkfifo");
    for path in [dir.clone(), PathBuf::from("/dev/zero"), pipe] {
        let shown = path.display().to_string();
        let msg = within_20s(move || TrainRecipe::load(&path).expect_err("refused").to_string());
        assert!(msg.contains("is not a regular file"), "{shown}: {msg}");
        assert!(msg.contains(&shown), "{msg}");
    }
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

/// A hard link has its own path but the same inode: canonicalize does not
/// fold it, so the inode comparison must.
#[cfg(unix)]
#[test]
fn heldout_hard_linked_to_the_training_file_is_refused_at_require_inputs() {
    let dir = scratch("hardlink");
    std::fs::write(dir.join("models/base.apr"), b"b").expect("base");
    std::fs::write(dir.join("data/train.jsonl"), b"same\n").expect("train");
    std::fs::hard_link(dir.join("data/train.jsonl"), dir.join("data/heldout.jsonl")).expect("link");
    let r = TrainRecipe::parse(VALID, &dir, "r.yaml").expect("lexically distinct");
    let msg = r.require_inputs().expect_err("one inode").to_string();
    assert!(msg.contains("recipe `qwen-lora-demo` cannot run"), "{msg}");
    assert!(msg.contains("they are one file on disk"), "{msg}");
    assert!(!msg.contains("no such file"), "{msg}");

    // Two files with the same bytes are still two files.
    std::fs::remove_file(dir.join("data/heldout.jsonl")).expect("unlink");
    std::fs::write(dir.join("data/heldout.jsonl"), b"same\n").expect("heldout");
    r.require_inputs()
        .expect("equal content is not the same file; a lora recipe needs no teacher file");
    std::fs::remove_dir_all(&dir).ok();
}

/// A directory where a file is expected is refused: the input must be a file.
#[test]
fn require_inputs_refuses_a_directory_and_skips_the_pin_of_an_absent_file() {
    let dir = scratch("inputs-dir");
    std::fs::create_dir_all(dir.join("models/base.apr")).expect("a directory named like the model");
    std::fs::write(dir.join("data/heldout.jsonl"), b"h\n").expect("heldout");
    let src = VALID.replace(
        "  train: data/train.jsonl\n",
        &format!("  train: data/train.jsonl\n  sha256: {}\n", "0".repeat(64)),
    );
    let r = TrainRecipe::parse(&src, &dir, "r.yaml").expect("parses");
    let msg = r.require_inputs().expect_err("refused").to_string();
    assert!(msg.contains("field `base`: no such file"), "{msg}");
    assert!(msg.contains("field `data.train`: no such file"), "{msg}");
    assert!(!msg.contains("field `eval.heldout`"), "{msg}");
    assert!(!msg.contains("field `teacher.model`"), "{msg}");
    assert!(
        !msg.contains("field `data.sha256`"),
        "an absent file has no hash to compare: {msg}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// The file hash is streamed in 64 KiB reads; a file longer than one read
/// must hash the same as its bytes.
#[test]
fn sha256_file_hashes_past_one_buffer() {
    let dir = scratch("bigfile");
    let bytes: Vec<u8> = (0..200_000_u32).map(|i| (i % 251) as u8).collect();
    let path = dir.join("data/big.bin");
    std::fs::write(&path, &bytes).expect("write");
    assert_eq!(sha256_file(&path), Some(sha256_bytes(&bytes)));
    // The published vector for one million `a` bytes (FIPS 180 examples).
    let million = dir.join("data/million.bin");
    std::fs::write(&million, vec![b'a'; 1_000_000]).expect("write");
    assert_eq!(
        sha256_file(&million).as_deref(),
        Some("cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0")
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// `f` on its own thread; panics if it has not answered in 20 seconds.
#[cfg(unix)]
fn within_20s<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || tx.send(f()).ok());
    rx.recv_timeout(std::time::Duration::from_secs(20))
        .expect("the call must return: it is reading a file that has no end")
}

/// Review of a163e8112 and a31785740: a path that is not a regular file has no
/// hash and is never opened. `/dev/zero` never ends, and opening a pipe waits
/// for a writer, so either would hang the run.
#[cfg(unix)]
#[test]
fn a_path_that_is_not_a_regular_file_has_no_hash_and_is_not_read() {
    let dir = scratch("notfile");
    assert_eq!(sha256_file(&dir.join("data")), None, "a directory");
    assert_eq!(within_20s(|| sha256_file(Path::new("/dev/zero"))), None);
    let src = VALID.replace("data/train.jsonl", "/dev/zero");
    let r = TrainRecipe::parse(&src, &dir, "r.yaml").expect("parses");
    let (refused, receipt) = within_20s(move || {
        let refused = r.require_inputs().expect_err("refused").to_string();
        (refused, r.receipt(Vec::new(), Vec::new(), &[]))
    });
    assert!(
        refused.contains("field `data.train`: no such file /dev/zero"),
        "{refused}"
    );
    assert_eq!(receipt.data_sha256, None);

    for name in ["train.pipe", "heldout.pipe"] {
        let made = std::process::Command::new("mkfifo")
            .arg(dir.join("data").join(name))
            .status()
            .expect("mkfifo runs");
        assert!(made.success(), "mkfifo {name}");
    }
    let pipe = dir.join("data/train.pipe");
    assert_eq!(within_20s(move || sha256_file(&pipe)), None);
    let src = VALID
        .replace("data/train.jsonl", "data/train.pipe")
        .replace("data/heldout.jsonl", "data/heldout.pipe");
    let pinned = src.replace(
        "data/train.pipe\n",
        &format!("data/train.pipe\n  sha256: {}\n", sha256_bytes(b"x")),
    );
    assert_ne!(pinned, src, "the pin line was added");
    for (what, src) in [("no pin", src), ("pinned", pinned)] {
        let r = TrainRecipe::parse(&src, &dir, "r.yaml").expect("parses");
        let (refused, receipt) = within_20s(move || {
            let refused = r.require_inputs().expect_err("refused").to_string();
            (refused, r.receipt(Vec::new(), Vec::new(), &[]))
        });
        for field in ["`data.train`", "`eval.heldout`"] {
            assert!(
                refused.contains(&format!("field {field}: no such file")),
                "{what}: {refused}"
            );
        }
        assert!(
            !refused.contains("field `data.sha256`"),
            "{what}: {refused}"
        );
        assert_eq!((receipt.data_sha256, receipt.eval_sha256), (None, None));
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// Contract invariant: the eval block is part of the recipe hash.
#[test]
fn recipe_hash_changes_with_the_eval_block() {
    let a = parse(VALID).expect("parses");
    let b = parse(&VALID.replace("data/heldout.jsonl", "data/other.jsonl")).expect("parses");
    assert_ne!(a.recipe_sha256, b.recipe_sha256);
}

#[test]
fn receipt_carries_the_contract_fields() {
    let dir = scratch("receipt");
    let path = dir.join("r.yaml");
    std::fs::write(&path, VALID).expect("write recipe");
    let r = TrainRecipe::load(&path).expect("load");

    // Nothing to hash yet: the hashes are absent, never invented.
    let receipt = r.receipt(vec![apr_engine()], Vec::new(), NOT_APPLIED_SAMPLE);
    assert_eq!(receipt.data_sha256, None);
    assert_eq!(receipt.eval_sha256, None);

    std::fs::write(dir.join("data/train.jsonl"), b"train\n").expect("train");
    std::fs::write(dir.join("data/heldout.jsonl"), b"heldout\n").expect("heldout");
    let receipt = r.receipt(
        vec![apr_engine()],
        vec!["--vram 8".to_string()],
        NOT_APPLIED_SAMPLE,
    );
    assert_eq!(receipt.schema, RECEIPT_SCHEMA);
    assert_eq!(receipt.recipe_sha256, sha256_bytes(VALID.as_bytes()));
    assert_eq!(receipt.data_sha256, Some(sha256_bytes(b"train\n")));
    assert_eq!(receipt.eval_sha256, Some(sha256_bytes(b"heldout\n")));
    assert_eq!(receipt.seed, 42);
    assert_eq!(receipt.recipe_name, "qwen-lora-demo");
    assert_eq!(receipt.method, RecipeMethod::Lora);
    assert_eq!(receipt.validated_not_applied, ["eval"]);
    assert_eq!(receipt.engines.len(), 1);
    assert_eq!(receipt.engines[0].name, "apr");
    assert_eq!(receipt.engines[0].version, env!("CARGO_PKG_VERSION"));
    assert!(!receipt.engines[0].sha.is_empty());

    let json = serde_json::to_value(&receipt).expect("receipt serializes");
    for key in [
        "schema",
        "recipe_name",
        "recipe_sha256",
        "data_sha256",
        "eval_sha256",
        "method",
        "seed",
        "engines",
        "overrides",
        "validated_not_applied",
    ] {
        assert!(json.get(key).is_some(), "receipt lacks `{key}`: {json}");
    }
    assert_eq!(json["method"], "lora");
    assert_eq!(json["overrides"][0], "--vram 8");
    assert_eq!(json["validated_not_applied"][0], "eval");
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
