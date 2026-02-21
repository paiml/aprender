#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::super::vae_conditional::*;
    use super::super::*;

    include!("vae_tests_core.rs");
    include!("vae_tests_beta.rs");
}
