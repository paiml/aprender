//! `pv` binary. The command surface lives in the library
//! (`aprender_contracts_cli`) so `apr pv` can reach the same code.

/// Update identity (EPIC #4232). `update` is dispatched here, not in the
/// library, so `apr pv update` can never install pv over the apr executable.
const PRODUCT: sovereign_update::Product = sovereign_update::Product {
    bin: "pv",
    repo: "paiml/aprender",
    version: env!("CARGO_PKG_VERSION"),
    build_sha: None,
    release_asset: Some("{bin}-{tag}-{target}.tar.gz"),
    nightly: true,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).is_some_and(|a| a == "update") {
        std::process::exit(sovereign_update::update_main(&PRODUCT, &args[2..]));
    }
    sovereign_update::startup(&PRODUCT, &args);
    aprender_contracts_cli::run();
}
