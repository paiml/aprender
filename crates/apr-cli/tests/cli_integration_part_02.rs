
#[test]
fn test_gh122_hex_list_tensors() {
    let file = create_apr1_test_file();

    apr()
        .args(["hex", file.path().to_str().unwrap(), "--list"])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "encoder.layers.0.self_attn.q_proj.weight",
        ))
        .stdout(predicate::str::contains("decoder.layers.0.cross_attn"));
}

#[test]
fn test_gh122_hex_with_filter() {
    let file = create_apr1_test_file();

    apr()
        .args([
            "hex",
            file.path().to_str().unwrap(),
            "--tensor",
            "cross_attn",
            "--list",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("cross_attn"))
        .stdout(predicate::str::contains("2 tensors").or(predicate::str::contains("tensors")));
}

#[test]
fn test_gh122_hex_with_stats() {
    let file = create_apr1_test_file();

    apr()
        .args([
            "hex",
            file.path().to_str().unwrap(),
            "--tensor",
            "encoder.layers.0.self_attn.q_proj.weight",
            "--stats",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("min="))
        .stdout(predicate::str::contains("max="))
        .stdout(predicate::str::contains("mean="))
        .stdout(predicate::str::contains("std="));
}

#[test]
fn test_gh122_hex_json_output() {
    let file = create_apr1_test_file();

    apr()
        .args([
            "hex",
            file.path().to_str().unwrap(),
            "--tensor",
            "encoder",
            "--json",
            "--stats",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"name\""))
        .stdout(predicate::str::contains("\"shape\""))
        .stdout(predicate::str::contains("\"stats\""));
}

#[test]
fn test_gh122_hex_dump_display() {
    let file = create_apr1_test_file();

    apr()
        .args([
            "hex",
            file.path().to_str().unwrap(),
            "--tensor",
            "encoder.layers.0.self_attn.q_proj.weight",
            "--limit",
            "8",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("Tensor"))
        .stdout(predicate::str::contains("00000000:")); // Hex offset
}

#[test]
fn test_gh122_hex_no_match() {
    let file = create_apr1_test_file();

    apr()
        .args([
            "hex",
            file.path().to_str().unwrap(),
            "--tensor",
            "nonexistent_tensor",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("No tensors match"));
}

// ============================================================================
// GH-122: Tree Command Tests
// ============================================================================

#[test]
fn test_gh122_tree_help() {
    apr()
        .args(["tree", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("tree"))
        .stdout(predicate::str::contains("format"));
}

#[test]
fn test_gh122_tree_ascii_default() {
    let file = create_apr1_test_file();

    apr()
        .args(["tree", file.path().to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("encoder"))
        .stdout(predicate::str::contains("decoder"))
        .stdout(predicate::str::contains("tensors"));
}

#[test]
fn test_gh122_tree_with_sizes() {
    let file = create_apr1_test_file();

    apr()
        .args(["tree", file.path().to_str().unwrap(), "--sizes"])
        .assert()
        .success()
        .stdout(predicate::str::contains("KB").or(predicate::str::contains("MB")));
}

#[test]
fn test_gh122_tree_with_filter() {
    let file = create_apr1_test_file();

    apr()
        .args(["tree", file.path().to_str().unwrap(), "--filter", "encoder"])
        .assert()
        .success()
        .stdout(predicate::str::contains("encoder"));
}

#[test]
fn test_gh122_tree_depth_limit() {
    let file = create_apr1_test_file();

    apr()
        .args(["tree", file.path().to_str().unwrap(), "--depth", "2"])
        .assert()
        .success()
        .stdout(predicate::str::contains("encoder"))
        .stdout(predicate::str::contains("layers"));
}

#[test]
fn test_gh122_tree_mermaid_format() {
    let file = create_apr1_test_file();

    apr()
        .args(["tree", file.path().to_str().unwrap(), "--format", "mermaid"])
        .assert()
        .success()
        .stdout(predicate::str::contains("```mermaid"))
        .stdout(predicate::str::contains("graph TD"));
}

#[test]
fn test_gh122_tree_dot_format() {
    let file = create_apr1_test_file();

    apr()
        .args(["tree", file.path().to_str().unwrap(), "--format", "dot"])
        .assert()
        .success()
        .stdout(predicate::str::contains("digraph"))
        .stdout(predicate::str::contains("rankdir"));
}

#[test]
fn test_gh122_tree_json_format() {
    let file = create_apr1_test_file();

    apr()
        .args(["tree", file.path().to_str().unwrap(), "--format", "json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"name\""))
        .stdout(predicate::str::contains("\"children\""));
}

// ============================================================================
// GH-122: Flow Command Tests
// ============================================================================

#[test]
fn test_gh122_flow_help() {
    apr()
        .args(["flow", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("flow"))
        .stdout(predicate::str::contains("component"));
}

#[test]
fn test_gh122_flow_full_model() {
    let file = create_apr1_test_file();

    apr()
        .args(["flow", file.path().to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("encoder-decoder").or(predicate::str::contains("Model")));
}

#[test]
fn test_gh122_flow_cross_attn() {
    let file = create_apr1_test_file();

    apr()
        .args([
            "flow",
            file.path().to_str().unwrap(),
            "--component",
            "cross_attn",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("CROSS-ATTENTION"))
        .stdout(predicate::str::contains("encoder_output"))
        .stdout(predicate::str::contains("softmax"));
}

#[test]
fn test_gh122_flow_self_attn() {
    let file = create_apr1_test_file();

    apr()
        .args([
            "flow",
            file.path().to_str().unwrap(),
            "--component",
            "self_attn",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("SELF-ATTENTION"));
}

#[test]
fn test_gh122_flow_encoder() {
    let file = create_apr1_test_file();

    apr()
        .args([
            "flow",
            file.path().to_str().unwrap(),
            "--component",
            "encoder",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("ENCODER"));
}

#[test]
fn test_gh122_flow_decoder() {
    let file = create_apr1_test_file();

    apr()
        .args([
            "flow",
            file.path().to_str().unwrap(),
            "--component",
            "decoder",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("DECODER"));
}

#[test]
fn test_gh122_flow_verbose() {
    let file = create_apr1_test_file();

    apr()
        .args(["flow", file.path().to_str().unwrap(), "--verbose"])
        .assert()
        .success();
}

#[test]
fn test_gh122_flow_with_layer_filter() {
    let file = create_apr1_test_file();

    apr()
        .args([
            "flow",
            file.path().to_str().unwrap(),
            "--layer",
            "decoder.layers.0",
        ])
        .assert()
        .success();
}

// ============================================================================
// GH-179 / PMAT-191: Missing Tool Tests (Tool Coverage Gap)
// ============================================================================

// F-RUN-001: apr run help works
#[test]
fn test_f_run_001_help() {
    apr()
        .args(["run", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Run").or(predicate::str::contains("inference")));
}

// F-RUN-002: apr run with missing model shows error
#[test]
fn test_f_run_002_missing_model_error() {
    apr()
        .args(["run", "/nonexistent/model.gguf", "--prompt", "test"])
        .assert()
        .failure()
        .stderr(
            predicate::str::contains("not found")
                .or(predicate::str::contains("No such file"))
                .or(predicate::str::contains("Failed")),
        );
}

// F-CHAT-001: apr chat help works
#[test]
fn test_f_chat_001_help() {
    apr()
        .args(["chat", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("chat").or(predicate::str::contains("Chat")));
}

// F-CHAT-002: apr chat with missing model shows error
#[test]
fn test_f_chat_002_missing_model_error() {
    apr()
        .args(["chat", "/nonexistent/model.gguf"])
        .assert()
        .failure()
        .stderr(
            predicate::str::contains("not found")
                .or(predicate::str::contains("No such file"))
                .or(predicate::str::contains("Failed")),
        );
}

// F-SERVE-001: apr serve help works
#[test]
fn test_f_serve_001_help() {
    apr()
        .args(["serve", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("serve").or(predicate::str::contains("Serve")))
        .stdout(predicate::str::contains("port").or(predicate::str::contains("PORT")));
}

// F-SERVE-002: apr serve with missing model shows error
#[test]
fn test_f_serve_002_missing_model_error() {
    apr()
        .args(["serve", "/nonexistent/model.gguf"])
        .assert()
        .failure()
        .stderr(
            predicate::str::contains("not found")
                .or(predicate::str::contains("No such file"))
                .or(predicate::str::contains("Failed")),
        );
}

// F-CANARY-001: apr canary help works
#[test]
fn test_f_canary_001_help() {
    apr()
        .args(["canary", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("canary").or(predicate::str::contains("Canary")))
        .stdout(predicate::str::contains("create").or(predicate::str::contains("check")));
}

// F-CANARY-002: apr canary create with missing model shows error
#[test]
fn test_f_canary_002_create_missing_model() {
    apr()
        .args([
            "canary",
            "create",
            "--input",
            "/tmp/test.wav",
            "--output",
            "/tmp/canary.json",
            "/nonexistent/model.gguf",
        ])
        .assert()
        .failure()
        .stderr(
            predicate::str::contains("not found")
                .or(predicate::str::contains("No such file"))
                .or(predicate::str::contains("Failed"))
                .or(predicate::str::contains("does not exist")),
        );
}

// F-TUNE-001: apr tune help works
#[test]
fn test_f_tune_001_help() {
    apr()
        .args(["tune", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("tune").or(predicate::str::contains("Tune")))
        .stdout(predicate::str::contains("plan").or(predicate::str::contains("lora")));
}
