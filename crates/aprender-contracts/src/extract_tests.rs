use super::*;

#[test]
fn extract_function_basic() {
    let src =
        "def relu(x: Tensor) -> Tensor:\n    \"\"\"Apply ReLU.\"\"\"\n    return x.clamp(0)\n";
    let k = extract_function(src, "relu", "test.py").unwrap();
    assert_eq!(k.function_name, "relu");
    assert_eq!(k.module_path, "test.py");
    assert_eq!(k.return_type, "Tensor");
    assert!(!k.equations.is_empty());
}

#[test]
fn extract_function_not_found() {
    let src = "def foo(x):\n    pass\n";
    let err = extract_function(src, "bar", "f.py").unwrap_err();
    assert!(err.contains("bar"));
    assert!(err.contains("f.py"));
}

#[test]
fn extract_arguments_basic() {
    // Use a multiline def to test argument extraction cleanly
    let lines = vec!["def foo(", "    a: int,", "    b: float", "):", "    pass"];
    let args = extract_arguments(&lines, 0);
    assert!(args.iter().any(|(n, _)| n == "a"));
    assert!(args.iter().any(|(n, _)| n == "b"));
}

#[test]
fn extract_arguments_skips_self() {
    let lines = vec!["def forward(self, x: Tensor):", "    pass"];
    let args = extract_arguments(&lines, 0);
    assert_eq!(args.len(), 1);
    assert_eq!(args[0].0, "x");
}

#[test]
fn extract_arguments_skips_underscore() {
    let lines = vec!["def foo(_unused: int, x: float):", "    pass"];
    let args = extract_arguments(&lines, 0);
    // _unused should be skipped (starts with _)
    assert!(args.iter().all(|(n, _)| !n.starts_with('_')));
    assert!(args.iter().any(|(n, _)| n == "x"));
}

#[test]
fn extract_arguments_multiline() {
    let lines = vec!["def foo(", "    a: int,", "    b: float", "):", "    pass"];
    let args = extract_arguments(&lines, 0);
    assert!(args.iter().any(|(n, _)| n == "a"));
    assert!(args.iter().any(|(n, _)| n == "b"));
}

#[test]
fn extract_docstring_triple_quotes() {
    let lines = vec![
        "def f():",
        "    \"\"\"This is a docstring.\"\"\"",
        "    pass",
    ];
    let doc = extract_docstring(&lines, 0);
    assert_eq!(doc, "This is a docstring.");
}

#[test]
fn extract_docstring_multiline() {
    let lines = vec![
        "def f():",
        "    \"\"\"",
        "    Line one.",
        "    Line two.",
        "    \"\"\"",
    ];
    let doc = extract_docstring(&lines, 0);
    assert!(doc.contains("Line one."));
    assert!(doc.contains("Line two."));
}

#[test]
fn extract_docstring_raw_string() {
    let lines = vec!["def f():", "    r\"\"\"Raw docstring.\"\"\"", "    pass"];
    let doc = extract_docstring(&lines, 0);
    assert_eq!(doc, "Raw docstring.");
}

#[test]
fn extract_docstring_none() {
    let lines = vec!["def f():", "    return 42"];
    let doc = extract_docstring(&lines, 0);
    assert!(doc.is_empty());
}

#[test]
fn extract_equations_with_math() {
    let doc = "Computes :math:`\\exp(x)` for all elements.";
    let eqs = extract_equations_from_docstring(doc, "exp_fn", "test.py", 1);
    assert_eq!(eqs.len(), 1);
    assert!(eqs[0].formula.contains("exp"));
    assert_eq!(eqs[0].name, "exp_fn");
    assert_eq!(eqs[0].source_file, "test.py");
    assert_eq!(eqs[0].source_line, 1);
}

#[test]
fn extract_equations_multiple_math() {
    let doc = "Uses :math:`\\mu` and :math:`\\sigma` for normalization.";
    let eqs = extract_equations_from_docstring(doc, "norm", "f.py", 5);
    assert_eq!(eqs.len(), 2);
    assert!(eqs[0].formula.contains("\u{03bc}"));
    assert!(eqs[1].formula.contains("\u{03c3}"));
}

#[test]
fn extract_equations_no_math_fallback() {
    let doc = "A simple function with no LaTeX.";
    let eqs = extract_equations_from_docstring(doc, "simple", "f.py", 0);
    assert_eq!(eqs.len(), 1);
    assert!(eqs[0].formula.contains("simple"));
    assert!(eqs[0].formula.contains("output"));
}

#[test]
fn extract_return_type_arrow() {
    let lines = vec!["def f(x: int) -> float:", "    return 1.0"];
    let ret = extract_return_type(&lines, 0);
    assert_eq!(ret, "float");
}

#[test]
fn extract_return_type_no_annotation() {
    let lines = vec!["def f(x):", "    return x"];
    let ret = extract_return_type(&lines, 0);
    assert_eq!(ret, "Tensor");
}

#[test]
fn infer_preconditions_with_dim() {
    let doc = "Applies along a given dim.";
    let pres = infer_preconditions(doc, "fn");
    assert!(pres.iter().any(|p| p.contains("dim")));
}

#[test]
fn infer_preconditions_positive() {
    let doc = "Requires values > 0 for log.";
    let pres = infer_preconditions(doc, "fn");
    assert!(pres.iter().any(|p| p.contains("> 0.0")));
}

#[test]
fn infer_preconditions_basic() {
    let doc = "A plain docstring.";
    let pres = infer_preconditions(doc, "fn");
    assert_eq!(pres.len(), 1);
    assert!(pres[0].contains("is_empty"));
}

#[test]
fn infer_postconditions_range() {
    let doc = "Output is in [0, 1].";
    let posts = infer_postconditions(doc, "fn");
    assert!(posts.iter().any(|p| p.contains(">= 0.0")));
}

#[test]
fn infer_postconditions_sum_to_one() {
    let doc = "Values sum to 1 across the dimension.";
    let posts = infer_postconditions(doc, "fn");
    assert!(posts.iter().any(|p| p.contains("sum")));
}

#[test]
fn infer_postconditions_normalized() {
    let doc = "Returns normalized output.";
    let posts = infer_postconditions(doc, "fn");
    assert!(posts.iter().any(|p| p.contains("is_finite")));
}

#[test]
fn infer_postconditions_fallback() {
    let doc = "A basic operation.";
    let posts = infer_postconditions(doc, "fn");
    assert_eq!(posts.len(), 1);
    assert!(posts[0].contains("is_finite"));
}

#[test]
fn capitalize_basic() {
    assert_eq!(capitalize("hello"), "Hello");
    assert_eq!(capitalize(""), "");
    assert_eq!(capitalize("a"), "A");
}

#[test]
fn kernel_to_yaml_output() {
    let kernel = ExtractedKernel {
        function_name: "relu".into(),
        module_path: "torch/nn.py".into(),
        docstring: String::new(),
        equations: vec![ExtractedEquation {
            name: "relu".into(),
            formula: "max(0, x)".into(),
            preconditions: vec!["!input.is_empty()".into()],
            postconditions: vec!["ret >= 0".into()],
            source_file: "torch/nn.py".into(),
            source_line: 10,
        }],
        arguments: vec![("x".into(), "Tensor".into())],
        return_type: "Tensor".into(),
    };
    let yaml = kernel_to_yaml(&kernel);
    assert!(yaml.contains("Auto-extracted from torch/nn.py"));
    assert!(yaml.contains("Function: relu"));
    assert!(yaml.contains("metadata:"));
    assert!(yaml.contains("equations:"));
    assert!(yaml.contains("max(0, x)"));
    assert!(yaml.contains("falsification_tests:"));
    assert!(yaml.contains("FALSIFY-RELU-001"));
    assert!(yaml.contains("preconditions:"));
    assert!(yaml.contains("postconditions:"));
    assert!(yaml.contains("lean_theorem:"));
    assert!(yaml.contains("Relu"));
}

#[test]
fn extract_from_pytorch_file_not_found() {
    let err = extract_from_pytorch("/nonexistent/file.py").unwrap_err();
    assert!(err.contains("Failed to read"));
}

#[test]
fn extract_from_pytorch_with_target_separator() {
    let err = extract_from_pytorch("/nonexistent/file.py::func").unwrap_err();
    assert!(err.contains("Failed to read"));
}

#[test]
fn infer_postconditions_positive_keyword() {
    let doc = "Values must be positive and bounded.";
    let posts = infer_postconditions(doc, "fn");
    // No explicit [0,1] / sum to 1 / normalized triggers, so fallback
    assert!(posts.iter().any(|p| p.contains("is_finite")));
}

#[test]
fn infer_preconditions_dim_and_positive() {
    let doc = "Along dim with positive values > 0.";
    let pres = infer_preconditions(doc, "fn");
    assert!(pres.len() >= 3);
    assert!(pres.iter().any(|p| p.contains("dim")));
    assert!(pres.iter().any(|p| p.contains("> 0.0")));
}

#[test]
fn kernel_to_yaml_multiple_equations() {
    let kernel = ExtractedKernel {
        function_name: "norm".into(),
        module_path: "torch/norm.py".into(),
        docstring: String::new(),
        equations: vec![
            ExtractedEquation {
                name: "mean".into(),
                formula: "E[x]".into(),
                preconditions: vec![],
                postconditions: vec!["finite".into()],
                source_file: "torch/norm.py".into(),
                source_line: 1,
            },
            ExtractedEquation {
                name: "var".into(),
                formula: "Var(x)".into(),
                preconditions: vec!["len > 1".into()],
                postconditions: vec![],
                source_file: "torch/norm.py".into(),
                source_line: 2,
            },
        ],
        arguments: vec![],
        return_type: "Tensor".into(),
    };
    let yaml = kernel_to_yaml(&kernel);
    assert!(yaml.contains("mean:"));
    assert!(yaml.contains("var:"));
    assert!(yaml.contains("E[x]"));
    assert!(yaml.contains("Var(x)"));
}

#[test]
fn latex_to_readable_sqrt_and_log() {
    let result = latex_to_readable("\\sqrt{\\log(x)}");
    assert!(result.contains("\u{221a}"));
    assert!(result.contains("log"));
}

#[test]
fn latex_to_readable_epsilon() {
    let result = latex_to_readable("x + \\epsilon");
    assert!(result.contains("\u{03b5}"));
}

#[test]
fn extract_return_type_multiline_arrow() {
    // Arrow on second line within the 5-line scan window
    let lines = vec!["def f(", "    x: int", ") -> List[float]:", "    pass"];
    let ret = extract_return_type(&lines, 0);
    assert_eq!(ret, "List[float]");
}
