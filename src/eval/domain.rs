//! Domain classification for course directories

/// Classify a course directory name into a domain.
///
/// Returns one of: `aws`, `ml`, `k8s`, `lang`, `devops`, `data`, `cloud`, `other`
pub fn classify_domain(course_dir: &str) -> &'static str {
    let lower = course_dir.to_lowercase();

    // AWS
    if matches_any(
        &lower,
        &[
            "52-weeks-aws",
            "aws-cloud-practitioner",
            "aws-certified",
            "aws-lambda",
            "aws-llmops",
            "bedrock",
            "building-genai-aws",
            "amazon-",
            "sagemaker",
        ],
    ) {
        return "aws";
    }

    // ML / AI
    if matches_any(
        &lower,
        &[
            "machine-learning",
            "applied-ml",
            "pytorch",
            "mlops",
            "hugging-face",
            "data-science",
            "deep-learning",
            "neural",
            "llm",
            "advanced-rag",
            "genai",
            "ai-",
            "openai",
        ],
    ) {
        return "ml";
    }

    // Kubernetes / Containers
    if matches_any(
        &lower,
        &["kubernetes", "docker", "assimilate-containers", "container", "minikube", "k8s"],
    ) {
        return "k8s";
    }

    // Programming Languages
    if matches_any(
        &lower,
        &[
            "rust",
            "python",
            "assimilate-go",
            "golang",
            "assimilate-java",
            "java-",
            "swift",
            "julia",
            "c-programming",
            "cpp",
            "javascript",
            "typescript",
        ],
    ) {
        return "lang";
    }

    // DevOps
    if matches_any(
        &lower,
        &[
            "devops",
            "github-actions",
            "makefile",
            "linux",
            "terraform",
            "ci-cd",
            "ansible",
            "jenkins",
        ],
    ) {
        return "devops";
    }

    // Data Engineering
    if matches_any(
        &lower,
        &["data-engineering", "databricks", "pandas", "sql", "snowflake", "spark", "etl", "dbt"],
    ) {
        return "data";
    }

    // Cloud (non-AWS)
    if matches_any(
        &lower,
        &[
            "going-pro-cloud-computing",
            "duke-cloud",
            "assimilate-azure",
            "assimilate-google",
            "gcp",
            "azure",
        ],
    ) {
        return "cloud";
    }

    "other"
}

fn matches_any(haystack: &str, patterns: &[&str]) -> bool {
    patterns.iter().any(|p| haystack.contains(p))
}

/// Extract course directory name from a source path.
///
/// Example: `/data/courses/52-weeks-aws/build/week1.srt` → `52-weeks-aws`
pub fn extract_course_dir(source_path: &str) -> &str {
    let parts: Vec<&str> = source_path.split('/').collect();
    // Find "courses" in path and take the next component
    for (i, p) in parts.iter().enumerate() {
        if *p == "courses" && i + 1 < parts.len() {
            return parts[i + 1];
        }
    }
    // Fallback: use the component before the filename
    if parts.len() >= 3 {
        return parts[parts.len() - 3];
    }
    "unknown"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_aws() {
        assert_eq!(classify_domain("52-weeks-aws"), "aws");
        assert_eq!(classify_domain("aws-cloud-practitioner"), "aws");
        assert_eq!(classify_domain("aws-certified-solutions-architect"), "aws");
        assert_eq!(classify_domain("bedrock-genai"), "aws");
    }

    #[test]
    fn test_classify_ml() {
        assert_eq!(classify_domain("machine-learning"), "ml");
        assert_eq!(classify_domain("Applied-ML"), "ml");
        assert_eq!(classify_domain("pytorch"), "ml");
        assert_eq!(classify_domain("hugging-face"), "ml");
    }

    #[test]
    fn test_classify_k8s() {
        assert_eq!(classify_domain("kubernetes-basics"), "k8s");
        assert_eq!(classify_domain("docker"), "k8s");
        assert_eq!(classify_domain("assimilate-containers"), "k8s");
    }

    #[test]
    fn test_classify_lang() {
        assert_eq!(classify_domain("rust-fundamentals"), "lang");
        assert_eq!(classify_domain("python-for-devops"), "lang");
        assert_eq!(classify_domain("assimilate-go"), "lang");
    }

    #[test]
    fn test_classify_devops() {
        assert_eq!(classify_domain("github-actions"), "devops");
        assert_eq!(classify_domain("terraform-iac"), "devops");
        assert_eq!(classify_domain("linux-fundamentals"), "devops");
    }

    #[test]
    fn test_classify_data() {
        assert_eq!(classify_domain("data-engineering"), "data");
        assert_eq!(classify_domain("databricks-ml"), "data");
        assert_eq!(classify_domain("pandas-intro"), "data");
    }

    #[test]
    fn test_classify_cloud() {
        assert_eq!(classify_domain("going-pro-cloud-computing"), "cloud");
        assert_eq!(classify_domain("assimilate-azure-basics"), "cloud");
    }

    #[test]
    fn test_classify_other() {
        assert_eq!(classify_domain("random-course"), "other");
        assert_eq!(classify_domain("cooking-101"), "other");
    }

    #[test]
    fn test_extract_course_dir() {
        assert_eq!(
            extract_course_dir("/data/courses/52-weeks-aws/build/week1.srt"),
            "52-weeks-aws"
        );
        assert_eq!(extract_course_dir("/data/courses/pytorch/build/lesson1.srt"), "pytorch");
    }
}
