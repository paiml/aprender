// Batuta-specific contract macros (analyze, transpile)
// These are batuta pipeline contracts not in the provable-contracts catalog.

macro_rules! contract_pre_analyze {
    () => {{}};
    ($input:expr) => {{
        let _input = &$input;
    }};
}

macro_rules! contract_pre_transpile {
    () => {{}};
    ($input:expr) => {{
        let _input = &$input;
    }};
}
