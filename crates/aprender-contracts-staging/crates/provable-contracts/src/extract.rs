//! `PyTorch` kernel extraction — reads Python source, extracts equations.
//!
//! Parses docstrings for LaTeX math, extracts preconditions from type hints
//! and assertions, and generates YAML contract skeletons.

use std::path::Path;

/// Extracted equation from `PyTorch` source.
#[derive(Debug, Clone)]
pub struct ExtractedEquation {
    pub name: String,
    pub formula: String,
    pub preconditions: Vec<String>,
    pub postconditions: Vec<String>,
    pub source_file: String,
    pub source_line: usize,
}

/// Extracted kernel from a `PyTorch` source file.
#[derive(Debug, Clone)]
pub struct ExtractedKernel {
    pub function_name: String,
    pub module_path: String,
    pub docstring: String,
    pub equations: Vec<ExtractedEquation>,
    pub arguments: Vec<(String, String)>, // (name, type)
    pub return_type: String,
}

/// Extract a kernel from a `PyTorch` Python source file.
///
/// Parses the function definition, docstring, and LaTeX math.
/// `target` is either a file path or `file.py::function_name`.
pub fn extract_from_pytorch(target: &str) -> Result<ExtractedKernel, String> {
    let (file_path, fn_name) = if target.contains("::") {
        let parts: Vec<&str> = target.splitn(2, "::").collect();
        (parts[0], Some(parts[1]))
    } else {
        (target, None)
    };

    let content = std::fs::read_to_string(file_path)
        .map_err(|e| format!("Failed to read {file_path}: {e}"))?;

    let fn_name = fn_name.unwrap_or_else(|| {
        // Guess from filename
        Path::new(file_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
    });

    extract_function(&content, fn_name, file_path)
}

fn extract_function(
    content: &str,
    fn_name: &str,
    file_path: &str,
) -> Result<ExtractedKernel, String> {
    let lines: Vec<&str> = content.lines().collect();

    // Find the function definition
    let def_pattern = format!("def {fn_name}(");
    let def_line = lines
        .iter()
        .enumerate()
        .find(|(_, line)| line.trim().starts_with(&def_pattern))
        .map(|(i, _)| i)
        .ok_or_else(|| format!("Function `{fn_name}` not found in {file_path}"))?;

    // Extract arguments from the def line
    let args = extract_arguments(&lines, def_line);

    // Extract docstring
    let docstring = extract_docstring(&lines, def_line);

    // Extract equations from LaTeX in docstring
    let equations = extract_equations_from_docstring(&docstring, fn_name, file_path, def_line);

    // Infer return type from type hints
    let return_type = extract_return_type(&lines, def_line);

    Ok(ExtractedKernel {
        function_name: fn_name.to_string(),
        module_path: file_path.to_string(),
        docstring,
        equations,
        arguments: args,
        return_type,
    })
}

fn extract_arguments(lines: &[&str], def_line: usize) -> Vec<(String, String)> {
    let mut args = Vec::new();
    let mut i = def_line;
    let mut in_def = true;

    while i < lines.len() && in_def {
        let line = lines[i].trim();
        // Parse "name: Type" patterns
        for part in line.split(',') {
            let part = part
                .trim()
                .trim_start_matches("def ")
                .trim_start_matches('(');
            if let Some(colon) = part.find(':') {
                let name = part[..colon].trim().to_string();
                let typ = part[colon + 1..]
                    .trim()
                    .trim_end_matches(')')
                    .trim_end_matches(',')
                    .to_string();
                if !name.is_empty() && name != "self" && !name.starts_with('_') {
                    args.push((name, typ));
                }
            }
        }
        if line.contains("):") || line.ends_with("):") || line.ends_with(") ->") {
            in_def = false;
        }
        i += 1;
    }
    args
}

fn extract_docstring(lines: &[&str], def_line: usize) -> String {
    let mut doc = String::new();
    let mut i = def_line + 1;
    let mut in_docstring = false;

    while i < lines.len() {
        let trimmed = lines[i].trim();
        if in_docstring {
            if trimmed.contains("\"\"\"") {
                let before = trimmed.trim_end_matches("\"\"\"");
                doc.push_str(before);
                break;
            }
            doc.push_str(trimmed);
            doc.push('\n');
        } else if trimmed.starts_with("r\"\"\"") || trimmed.starts_with("\"\"\"") {
            in_docstring = true;
            let after = trimmed
                .trim_start_matches("r\"\"\"")
                .trim_start_matches("\"\"\"");
            if after.ends_with("\"\"\"") {
                doc.push_str(after.trim_end_matches("\"\"\""));
                break;
            }
            doc.push_str(after);
            doc.push('\n');
        }
        i += 1;
    }
    doc
}

fn extract_equations_from_docstring(
    docstring: &str,
    fn_name: &str,
    file_path: &str,
    line: usize,
) -> Vec<ExtractedEquation> {
    let mut equations = Vec::new();

    // Extract LaTeX math from :math:`...`
    let mut pos = 0;
    while let Some(start) = docstring[pos..].find(":math:`") {
        let abs_start = pos + start + 7; // skip ":math:`"
        if let Some(end) = docstring[abs_start..].find('`') {
            let formula = &docstring[abs_start..abs_start + end];

            // Convert LaTeX to readable math
            let readable = latex_to_readable(formula);

            // Infer preconditions from argument types and docstring
            let preconditions = infer_preconditions(docstring, fn_name);

            // Infer postconditions from docstring descriptions
            let postconditions = infer_postconditions(docstring, fn_name);

            equations.push(ExtractedEquation {
                name: fn_name.to_string(),
                formula: readable,
                preconditions,
                postconditions,
                source_file: file_path.to_string(),
                source_line: line,
            });

            pos = abs_start + end + 1;
        } else {
            break;
        }
    }

    if equations.is_empty() {
        // No LaTeX found — create a basic equation from function signature
        equations.push(ExtractedEquation {
            name: fn_name.to_string(),
            formula: format!("{fn_name}(input) → output"),
            preconditions: vec!["!input.is_empty()".to_string()],
            postconditions: vec!["ret.iter().all(|x| x.is_finite())".to_string()],
            source_file: file_path.to_string(),
            source_line: line,
        });
    }

    equations
}

fn extract_return_type(lines: &[&str], def_line: usize) -> String {
    for line in lines.iter().skip(def_line).take(5) {
        if let Some(arrow) = line.find("->") {
            let ret = line[arrow + 2..].trim().trim_end_matches(':').trim();
            return ret.to_string();
        }
    }
    "Tensor".to_string()
}

fn latex_to_readable(latex: &str) -> String {
    latex
        .replace("\\text{", "")
        .replace("\\frac{", "(")
        .replace("}{", ") / (")
        .replace("\\exp", "exp")
        .replace("\\sum", "Σ")
        .replace("\\log", "log")
        .replace("\\max", "max")
        .replace("\\sqrt", "√")
        .replace("\\sigma", "σ")
        .replace("\\mu", "μ")
        .replace("\\epsilon", "ε")
        .replace('}', ")")
        .replace('{', "(")
        .replace("_((", "_(")
}

fn infer_preconditions(docstring: &str, _fn_name: &str) -> Vec<String> {
    let mut pres = vec!["!input.is_empty()".to_string()];

    if docstring.contains("dim") {
        pres.push("dim < input.ndim()".to_string());
    }
    if docstring.contains("positive") || docstring.contains("> 0") {
        pres.push("input.iter().all(|x| *x > 0.0)".to_string());
    }

    pres
}

fn infer_postconditions(docstring: &str, _fn_name: &str) -> Vec<String> {
    let mut posts = Vec::new();

    if docstring.contains("[0, 1]") || docstring.contains("range `[0, 1]`") {
        posts.push("ret.iter().all(|&v| v >= 0.0 && v <= 1.0)".to_string());
    }
    if docstring.contains("sum to 1") || docstring.contains("sum to one") {
        posts.push("(ret.iter().sum::<f32>() - 1.0).abs() < 1e-6".to_string());
    }
    if docstring.contains("normalized") || docstring.contains("unit") {
        posts.push("ret.iter().all(|x| x.is_finite())".to_string());
    }

    if posts.is_empty() {
        posts.push("ret.iter().all(|x| x.is_finite())".to_string());
    }

    posts
}

/// Generate YAML contract from extracted kernel.
pub fn kernel_to_yaml(kernel: &ExtractedKernel) -> String {
    let mut yaml = String::new();

    yaml.push_str(&format!("# Auto-extracted from {}\n", kernel.module_path));
    yaml.push_str(&format!("# Function: {}\n\n", kernel.function_name));

    yaml.push_str("metadata:\n");
    yaml.push_str("  version: \"1.0.0\"\n");
    yaml.push_str("  created: \"2026-03-21\"\n");
    yaml.push_str("  author: \"pv extract-pytorch\"\n");
    yaml.push_str(&format!(
        "  description: \"Contract for {} extracted from PyTorch\"\n",
        kernel.function_name
    ));
    yaml.push_str("  references:\n");
    yaml.push_str(&format!("    - \"{}\"\n\n", kernel.module_path));

    yaml.push_str("equations:\n");
    for eq in &kernel.equations {
        yaml.push_str(&format!("  {}:\n", eq.name));
        yaml.push_str(&format!(
            "    formula: \"{}\"\n",
            eq.formula.replace('"', "'")
        ));
        if !eq.preconditions.is_empty() {
            yaml.push_str("    preconditions:\n");
            for pre in &eq.preconditions {
                yaml.push_str(&format!("      - \"{pre}\"\n"));
            }
        }
        if !eq.postconditions.is_empty() {
            yaml.push_str("    postconditions:\n");
            for post in &eq.postconditions {
                yaml.push_str(&format!("      - \"{post}\"\n"));
            }
        }
        yaml.push_str(&format!(
            "    lean_theorem: \"ProvableContracts.Theorems.{}.Correctness\"\n\n",
            capitalize(&eq.name)
        ));
    }

    yaml.push_str("falsification_tests:\n");
    yaml.push_str(&format!(
        "  - id: FALSIFY-{}-001\n",
        kernel.function_name.to_uppercase()
    ));
    yaml.push_str(&format!(
        "    rule: \"{} correctness\"\n",
        kernel.function_name
    ));
    yaml.push_str(&format!(
        "    test: \"test_{}_basic\"\n",
        kernel.function_name
    ));
    yaml.push_str(&format!(
        "    prediction: \"{} output matches PyTorch reference\"\n",
        kernel.function_name
    ));
    yaml.push_str(&format!(
        "    if_fails: \"{} implementation diverges from PyTorch\"\n",
        kernel.function_name
    ));

    yaml
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().to_string() + c.as_str(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latex_to_readable() {
        assert_eq!(
            latex_to_readable("\\frac{\\exp(x_i)}{\\sum_j \\exp(x_j)}"),
            "(exp(x_i)) / (Σ_j exp(x_j))"
        );
    }

    #[test]
    fn test_extract_softmax() {
        let pytorch_path = "/home/noah/src/pytorch/torch/nn/functional.py";
        if std::path::Path::new(pytorch_path).exists() {
            let kernel = extract_from_pytorch(&format!("{pytorch_path}::softmax")).unwrap();
            assert_eq!(kernel.function_name, "softmax");
            assert!(!kernel.equations.is_empty());
            assert!(kernel.equations[0].formula.contains("exp"));
        }
    }
}

#[cfg(test)]
#[path = "extract_tests.rs"]
mod extract_tests;
