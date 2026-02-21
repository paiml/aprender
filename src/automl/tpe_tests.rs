
#[cfg(test)]
mod tests {
    use super::*;
    use crate::automl::params::RandomForestParam as RF;
    use crate::automl::search::ParamValue;
    use crate::automl::SearchSpace;
include!("tpe_tests_config.rs");
include!("tpe_tests_log_scale.rs");
}
