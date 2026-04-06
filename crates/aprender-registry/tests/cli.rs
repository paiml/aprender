//! CLI integration tests.

use std::process::Command;
use tempfile::TempDir;

fn pacha_cmd() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pacha"))
}

fn setup_registry() -> TempDir {
    let dir = TempDir::new().expect("temp dir");
    pacha_cmd().args(["--registry", dir.path().to_str().unwrap(), "init"]).output().expect("init");
    dir
}

#[test]
fn test_cli_init() {
    let dir = TempDir::new().expect("temp dir");
    let output = pacha_cmd()
        .args(["--registry", dir.path().to_str().unwrap(), "init"])
        .output()
        .expect("run");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Registry initialized"));
}

#[test]
fn test_cli_stats_empty() {
    let dir = TempDir::new().expect("temp dir");

    // Init first
    pacha_cmd().args(["--registry", dir.path().to_str().unwrap(), "init"]).output().expect("init");

    let output = pacha_cmd()
        .args(["--registry", dir.path().to_str().unwrap(), "stats"])
        .output()
        .expect("stats");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Models:   0"));
    assert!(stdout.contains("Datasets: 0"));
}

#[test]
fn test_cli_model_list_empty() {
    let dir = TempDir::new().expect("temp dir");

    pacha_cmd().args(["--registry", dir.path().to_str().unwrap(), "init"]).output().expect("init");

    let output = pacha_cmd()
        .args(["--registry", dir.path().to_str().unwrap(), "model", "list"])
        .output()
        .expect("list");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Models:"));
}

#[test]
fn test_cli_model_register_and_get() {
    let dir = TempDir::new().expect("temp dir");
    let model_file = dir.path().join("test.model");
    std::fs::write(&model_file, b"test model data").expect("write model");

    pacha_cmd().args(["--registry", dir.path().to_str().unwrap(), "init"]).output().expect("init");

    // Register
    let output = pacha_cmd()
        .args([
            "--registry",
            dir.path().to_str().unwrap(),
            "model",
            "register",
            "test-model",
            model_file.to_str().unwrap(),
            "-v",
            "1.0.0",
            "-d",
            "Test model",
        ])
        .output()
        .expect("register");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Registered model: test-model:1.0.0"));

    // Get
    let output = pacha_cmd()
        .args([
            "--registry",
            dir.path().to_str().unwrap(),
            "model",
            "get",
            "test-model",
            "-v",
            "1.0.0",
        ])
        .output()
        .expect("get");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Model: test-model:1.0.0"));
    assert!(stdout.contains("Stage:       development"));
}

#[test]
fn test_cli_model_stage_transition() {
    let dir = TempDir::new().expect("temp dir");
    let model_file = dir.path().join("test.model");
    std::fs::write(&model_file, b"test model data").expect("write model");

    pacha_cmd().args(["--registry", dir.path().to_str().unwrap(), "init"]).output().expect("init");

    pacha_cmd()
        .args([
            "--registry",
            dir.path().to_str().unwrap(),
            "model",
            "register",
            "test-model",
            model_file.to_str().unwrap(),
            "-v",
            "1.0.0",
        ])
        .output()
        .expect("register");

    let output = pacha_cmd()
        .args([
            "--registry",
            dir.path().to_str().unwrap(),
            "model",
            "stage",
            "test-model",
            "-v",
            "1.0.0",
            "-t",
            "staging",
        ])
        .output()
        .expect("stage");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Transitioned test-model:1.0.0 to staging"));
}

#[test]
fn test_cli_data_register_and_get() {
    let dir = TempDir::new().expect("temp dir");
    let data_file = dir.path().join("data.csv");
    std::fs::write(&data_file, b"id,value\n1,100\n2,200").expect("write data");

    pacha_cmd().args(["--registry", dir.path().to_str().unwrap(), "init"]).output().expect("init");

    let output = pacha_cmd()
        .args([
            "--registry",
            dir.path().to_str().unwrap(),
            "data",
            "register",
            "test-data",
            data_file.to_str().unwrap(),
            "-v",
            "1.0.0",
            "-p",
            "Test dataset",
        ])
        .output()
        .expect("register");

    assert!(output.status.success());

    let output = pacha_cmd()
        .args([
            "--registry",
            dir.path().to_str().unwrap(),
            "data",
            "get",
            "test-data",
            "-v",
            "1.0.0",
        ])
        .output()
        .expect("get");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Dataset: test-data:1.0.0"));
}

#[test]
fn test_cli_model_download() {
    let dir = setup_registry();
    let model_file = dir.path().join("model.bin");
    std::fs::write(&model_file, b"model weights").expect("write");

    pacha_cmd()
        .args([
            "--registry",
            dir.path().to_str().unwrap(),
            "model",
            "register",
            "dl-model",
            model_file.to_str().unwrap(),
            "-v",
            "1.0.0",
        ])
        .output()
        .expect("register");

    let output_path = dir.path().join("downloaded.bin");
    let output = pacha_cmd()
        .args([
            "--registry",
            dir.path().to_str().unwrap(),
            "model",
            "download",
            "dl-model",
            "-v",
            "1.0.0",
            "-o",
            output_path.to_str().unwrap(),
        ])
        .output()
        .expect("download");

    assert!(output.status.success());
    assert!(output_path.exists());
    assert_eq!(std::fs::read(&output_path).unwrap(), b"model weights");
}

#[test]
fn test_cli_data_download() {
    let dir = setup_registry();
    let data_file = dir.path().join("data.csv");
    std::fs::write(&data_file, b"a,b,c").expect("write");

    pacha_cmd()
        .args([
            "--registry",
            dir.path().to_str().unwrap(),
            "data",
            "register",
            "dl-data",
            data_file.to_str().unwrap(),
            "-v",
            "1.0.0",
        ])
        .output()
        .expect("register");

    let output_path = dir.path().join("downloaded.csv");
    let output = pacha_cmd()
        .args([
            "--registry",
            dir.path().to_str().unwrap(),
            "data",
            "download",
            "dl-data",
            "-v",
            "1.0.0",
            "-o",
            output_path.to_str().unwrap(),
        ])
        .output()
        .expect("download");

    assert!(output.status.success());
    assert!(output_path.exists());
}

#[test]
fn test_cli_model_lineage() {
    let dir = setup_registry();
    let model_file = dir.path().join("model.bin");
    std::fs::write(&model_file, b"weights").expect("write");

    pacha_cmd()
        .args([
            "--registry",
            dir.path().to_str().unwrap(),
            "model",
            "register",
            "lin-model",
            model_file.to_str().unwrap(),
            "-v",
            "1.0.0",
        ])
        .output()
        .expect("register");

    let output = pacha_cmd()
        .args([
            "--registry",
            dir.path().to_str().unwrap(),
            "model",
            "lineage",
            "lin-model",
            "-v",
            "1.0.0",
        ])
        .output()
        .expect("lineage");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Lineage"));
}

// Recipe register requires generated ID which is not supported via TOML file
// The recipe functionality is tested via library integration tests

#[test]
fn test_cli_recipe_list() {
    let dir = setup_registry();

    let output = pacha_cmd()
        .args(["--registry", dir.path().to_str().unwrap(), "recipe", "list"])
        .output()
        .expect("list");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Recipes:"));
}

#[test]
fn test_cli_data_list() {
    let dir = setup_registry();

    let output = pacha_cmd()
        .args(["--registry", dir.path().to_str().unwrap(), "data", "list"])
        .output()
        .expect("list");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Datasets:"));
}

#[test]
fn test_cli_model_list_versions() {
    let dir = setup_registry();
    let model_file = dir.path().join("m.bin");
    std::fs::write(&model_file, b"data").expect("write");

    // Register multiple versions
    for v in ["1.0.0", "1.1.0", "2.0.0"] {
        pacha_cmd()
            .args([
                "--registry",
                dir.path().to_str().unwrap(),
                "model",
                "register",
                "multi-ver",
                model_file.to_str().unwrap(),
                "-v",
                v,
            ])
            .output()
            .expect("register");
    }

    let output = pacha_cmd()
        .args(["--registry", dir.path().to_str().unwrap(), "model", "list", "multi-ver"])
        .output()
        .expect("list versions");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("1.0.0"));
}

#[test]
fn test_cli_run_list_empty() {
    let dir = setup_registry();

    // Need a recipe first - create via library
    let config = pacha::RegistryConfig::new(dir.path());
    let registry = pacha::Registry::open(config).expect("registry");
    let recipe = pacha::recipe::TrainingRecipe::builder()
        .name("test-recipe")
        .version(pacha::recipe::RecipeVersion::new(1, 0, 0))
        .description("Test")
        .hyperparameters(pacha::recipe::Hyperparameters::default())
        .build();
    registry.register_recipe(&recipe).expect("register");

    let output = pacha_cmd()
        .args([
            "--registry",
            dir.path().to_str().unwrap(),
            "run",
            "list",
            "test-recipe",
            "-v",
            "1.0.0",
        ])
        .output()
        .expect("run list");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Runs for"));
}

#[test]
fn test_cli_recipe_get() {
    let dir = setup_registry();

    let config = pacha::RegistryConfig::new(dir.path());
    let registry = pacha::Registry::open(config).expect("registry");
    let recipe = pacha::recipe::TrainingRecipe::builder()
        .name("get-recipe")
        .version(pacha::recipe::RecipeVersion::new(1, 0, 0))
        .description("Recipe for get test")
        .hyperparameters(
            pacha::recipe::Hyperparameters::builder()
                .learning_rate(0.001)
                .batch_size(32)
                .epochs(10)
                .build(),
        )
        .build();
    registry.register_recipe(&recipe).expect("register");

    let output = pacha_cmd()
        .args([
            "--registry",
            dir.path().to_str().unwrap(),
            "recipe",
            "get",
            "get-recipe",
            "-v",
            "1.0.0",
        ])
        .output()
        .expect("recipe get");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Recipe: get-recipe:1.0.0"));
    assert!(stdout.contains("Learning rate:"));
}

#[test]
fn test_cli_recipe_validate() {
    let dir = setup_registry();

    let config = pacha::RegistryConfig::new(dir.path());
    let registry = pacha::Registry::open(config).expect("registry");
    let recipe = pacha::recipe::TrainingRecipe::builder()
        .name("val-recipe")
        .version(pacha::recipe::RecipeVersion::new(1, 0, 0))
        .description("Recipe for validation")
        .hyperparameters(pacha::recipe::Hyperparameters::default())
        .build();
    registry.register_recipe(&recipe).expect("register");

    let output = pacha_cmd()
        .args([
            "--registry",
            dir.path().to_str().unwrap(),
            "recipe",
            "validate",
            "val-recipe",
            "-v",
            "1.0.0",
        ])
        .output()
        .expect("recipe validate");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Validating recipe"));
}

#[test]
fn test_cli_run_get() {
    let dir = setup_registry();

    let config = pacha::RegistryConfig::new(dir.path());
    let registry = pacha::Registry::open(config).expect("registry");

    let recipe = pacha::recipe::TrainingRecipe::builder()
        .name("run-recipe")
        .version(pacha::recipe::RecipeVersion::new(1, 0, 0))
        .description("Recipe")
        .hyperparameters(pacha::recipe::Hyperparameters::default())
        .build();
    registry.register_recipe(&recipe).expect("register recipe");

    let mut run = pacha::experiment::ExperimentRun::from_recipe(
        recipe.reference(),
        pacha::recipe::Hyperparameters::default(),
    );
    run.log_metric("loss", 0.5, 0);
    let run_id = registry.start_run(run).expect("start run");

    let output = pacha_cmd()
        .args(["--registry", dir.path().to_str().unwrap(), "run", "get", &run_id.to_string()])
        .output()
        .expect("run get");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Run:"));
    assert!(stdout.contains("Status:"));
}

#[test]
fn test_cli_run_compare() {
    let dir = setup_registry();

    let config = pacha::RegistryConfig::new(dir.path());
    let registry = pacha::Registry::open(config).expect("registry");

    let recipe = pacha::recipe::TrainingRecipe::builder()
        .name("cmp-recipe")
        .version(pacha::recipe::RecipeVersion::new(1, 0, 0))
        .description("Recipe")
        .hyperparameters(pacha::recipe::Hyperparameters::default())
        .build();
    registry.register_recipe(&recipe).expect("register recipe");

    let run1 = pacha::experiment::ExperimentRun::from_recipe(
        recipe.reference(),
        pacha::recipe::Hyperparameters::default(),
    );
    let run2 = pacha::experiment::ExperimentRun::from_recipe(
        recipe.reference(),
        pacha::recipe::Hyperparameters::default(),
    );
    let id1 = registry.start_run(run1).expect("start run1");
    let id2 = registry.start_run(run2).expect("start run2");

    let output = pacha_cmd()
        .args([
            "--registry",
            dir.path().to_str().unwrap(),
            "run",
            "compare",
            &id1.to_string(),
            &id2.to_string(),
        ])
        .output()
        .expect("run compare");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Run Comparison"));
}

#[test]
fn test_cli_run_best() {
    let dir = setup_registry();

    let config = pacha::RegistryConfig::new(dir.path());
    let registry = pacha::Registry::open(config).expect("registry");

    let recipe = pacha::recipe::TrainingRecipe::builder()
        .name("best-recipe")
        .version(pacha::recipe::RecipeVersion::new(1, 0, 0))
        .description("Recipe")
        .hyperparameters(pacha::recipe::Hyperparameters::default())
        .build();
    registry.register_recipe(&recipe).expect("register recipe");

    // Create runs with different metrics
    for auc in [0.8, 0.95, 0.85] {
        let mut run = pacha::experiment::ExperimentRun::from_recipe(
            recipe.reference(),
            pacha::recipe::Hyperparameters::default(),
        );
        run.log_metric("auc", auc, 0);
        run.complete();
        let id = registry.start_run(run).expect("start run");
        let mut r = registry.get_run(&id).expect("get");
        r.complete();
        registry.update_run(&r).expect("update");
    }

    let output = pacha_cmd()
        .args([
            "--registry",
            dir.path().to_str().unwrap(),
            "run",
            "best",
            "best-recipe",
            "-v",
            "1.0.0",
            "-m",
            "auc",
        ])
        .output()
        .expect("run best");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Best run") || stdout.contains("No completed"));
}

#[test]
fn test_cli_run_best_minimize() {
    let dir = setup_registry();

    let config = pacha::RegistryConfig::new(dir.path());
    let registry = pacha::Registry::open(config).expect("registry");

    let recipe = pacha::recipe::TrainingRecipe::builder()
        .name("min-recipe")
        .version(pacha::recipe::RecipeVersion::new(1, 0, 0))
        .description("Recipe")
        .hyperparameters(pacha::recipe::Hyperparameters::default())
        .build();
    registry.register_recipe(&recipe).expect("register recipe");

    for loss in [0.5, 0.1, 0.3] {
        let mut run = pacha::experiment::ExperimentRun::from_recipe(
            recipe.reference(),
            pacha::recipe::Hyperparameters::default(),
        );
        run.log_metric("loss", loss, 0);
        let id = registry.start_run(run).expect("start run");
        let mut r = registry.get_run(&id).expect("get");
        r.complete();
        registry.update_run(&r).expect("update");
    }

    let output = pacha_cmd()
        .args([
            "--registry",
            dir.path().to_str().unwrap(),
            "run",
            "best",
            "min-recipe",
            "-v",
            "1.0.0",
            "-m",
            "loss",
            "--minimize",
        ])
        .output()
        .expect("run best minimize");

    assert!(output.status.success());
}

#[test]
fn test_cli_model_get_with_metrics() {
    let dir = setup_registry();

    let config = pacha::RegistryConfig::new(dir.path());
    let registry = pacha::Registry::open(config).expect("registry");

    let card = pacha::model::ModelCard::builder()
        .description("Model with metrics")
        .metrics([("accuracy", 0.95), ("f1", 0.88)])
        .build();

    registry
        .register_model("metric-model", &pacha::model::ModelVersion::new(1, 0, 0), b"weights", card)
        .expect("register");

    let output = pacha_cmd()
        .args([
            "--registry",
            dir.path().to_str().unwrap(),
            "model",
            "get",
            "metric-model",
            "-v",
            "1.0.0",
        ])
        .output()
        .expect("model get");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Metrics:"));
}

#[test]
fn test_cli_error_handling() {
    let dir = setup_registry();

    // Try to get non-existent model
    let output = pacha_cmd()
        .args([
            "--registry",
            dir.path().to_str().unwrap(),
            "model",
            "get",
            "nonexistent",
            "-v",
            "1.0.0",
        ])
        .output()
        .expect("model get");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Error"));
}

#[test]
fn test_cli_run_get_with_finished() {
    let dir = setup_registry();

    let config = pacha::RegistryConfig::new(dir.path());
    let registry = pacha::Registry::open(config).expect("registry");

    let recipe = pacha::recipe::TrainingRecipe::builder()
        .name("fin-recipe")
        .version(pacha::recipe::RecipeVersion::new(1, 0, 0))
        .description("Recipe")
        .hyperparameters(pacha::recipe::Hyperparameters::default())
        .build();
    registry.register_recipe(&recipe).expect("register recipe");

    let mut run = pacha::experiment::ExperimentRun::from_recipe(
        recipe.reference(),
        pacha::recipe::Hyperparameters::default(),
    );
    run.log_metric("loss", 0.5, 0);
    run.log_metric("loss", 0.2, 100);
    let run_id = registry.start_run(run).expect("start run");

    let mut r = registry.get_run(&run_id).expect("get");
    r.complete();
    registry.update_run(&r).expect("update");

    let output = pacha_cmd()
        .args(["--registry", dir.path().to_str().unwrap(), "run", "get", &run_id.to_string()])
        .output()
        .expect("run get");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Finished:"));
    assert!(stdout.contains("Final metrics:"));
}

#[test]
fn test_cli_run_best_no_metric() {
    let dir = setup_registry();

    let config = pacha::RegistryConfig::new(dir.path());
    let registry = pacha::Registry::open(config).expect("registry");

    let recipe = pacha::recipe::TrainingRecipe::builder()
        .name("no-metric-recipe")
        .version(pacha::recipe::RecipeVersion::new(1, 0, 0))
        .description("Recipe")
        .hyperparameters(pacha::recipe::Hyperparameters::default())
        .build();
    registry.register_recipe(&recipe).expect("register recipe");

    // Create a completed run without the metric we're searching for
    let mut run = pacha::experiment::ExperimentRun::from_recipe(
        recipe.reference(),
        pacha::recipe::Hyperparameters::default(),
    );
    run.log_metric("other", 0.5, 0);
    let id = registry.start_run(run).expect("start run");
    let mut r = registry.get_run(&id).expect("get");
    r.complete();
    registry.update_run(&r).expect("update");

    let output = pacha_cmd()
        .args([
            "--registry",
            dir.path().to_str().unwrap(),
            "run",
            "best",
            "no-metric-recipe",
            "-v",
            "1.0.0",
            "-m",
            "nonexistent",
        ])
        .output()
        .expect("run best");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("No completed runs"));
}

#[test]
fn test_cli_data_list_with_name() {
    let dir = setup_registry();
    let data_file = dir.path().join("data.csv");
    std::fs::write(&data_file, b"a,b,c").expect("write");

    pacha_cmd()
        .args([
            "--registry",
            dir.path().to_str().unwrap(),
            "data",
            "register",
            "list-data",
            data_file.to_str().unwrap(),
            "-v",
            "1.0.0",
        ])
        .output()
        .expect("register");

    let output = pacha_cmd()
        .args(["--registry", dir.path().to_str().unwrap(), "data", "list", "list-data"])
        .output()
        .expect("list");

    assert!(output.status.success());
}

#[test]
fn test_cli_recipe_list_with_name() {
    let dir = setup_registry();

    let config = pacha::RegistryConfig::new(dir.path());
    let registry = pacha::Registry::open(config).expect("registry");

    let recipe = pacha::recipe::TrainingRecipe::builder()
        .name("list-recipe")
        .version(pacha::recipe::RecipeVersion::new(1, 0, 0))
        .description("Recipe")
        .hyperparameters(pacha::recipe::Hyperparameters::default())
        .build();
    registry.register_recipe(&recipe).expect("register recipe");

    let output = pacha_cmd()
        .args(["--registry", dir.path().to_str().unwrap(), "recipe", "list", "list-recipe"])
        .output()
        .expect("list");

    assert!(output.status.success());
}

#[test]
fn test_cli_recipe_validate_warnings() {
    let dir = setup_registry();

    let config = pacha::RegistryConfig::new(dir.path());
    let registry = pacha::Registry::open(config).expect("registry");

    // Recipe with invalid hyperparameters to trigger warnings
    let recipe = pacha::recipe::TrainingRecipe::builder()
        .name("warn-recipe")
        .version(pacha::recipe::RecipeVersion::new(1, 0, 0))
        .description("Recipe with bad params")
        .hyperparameters(
            pacha::recipe::Hyperparameters::builder()
                .learning_rate(0.0) // Should warn
                .batch_size(0) // Should warn
                .epochs(10)
                .build(),
        )
        .build();
    registry.register_recipe(&recipe).expect("register recipe");

    let output = pacha_cmd()
        .args([
            "--registry",
            dir.path().to_str().unwrap(),
            "recipe",
            "validate",
            "warn-recipe",
            "-v",
            "1.0.0",
        ])
        .output()
        .expect("recipe validate");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Warning"));
}

#[test]
fn test_cli_run_list_with_runs() {
    let dir = setup_registry();

    let config = pacha::RegistryConfig::new(dir.path());
    let registry = pacha::Registry::open(config).expect("registry");

    let recipe = pacha::recipe::TrainingRecipe::builder()
        .name("runs-recipe")
        .version(pacha::recipe::RecipeVersion::new(1, 0, 0))
        .description("Recipe")
        .hyperparameters(pacha::recipe::Hyperparameters::default())
        .build();
    registry.register_recipe(&recipe).expect("register recipe");

    // Create some runs
    for _ in 0..3 {
        let run = pacha::experiment::ExperimentRun::from_recipe(
            recipe.reference(),
            pacha::recipe::Hyperparameters::default(),
        );
        registry.start_run(run).expect("start run");
    }

    let output = pacha_cmd()
        .args([
            "--registry",
            dir.path().to_str().unwrap(),
            "run",
            "list",
            "runs-recipe",
            "-v",
            "1.0.0",
        ])
        .output()
        .expect("run list");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Runs for"));
}
