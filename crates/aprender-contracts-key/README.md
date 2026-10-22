# aprender-contracts-key

The one `CONTRACT_*` env-var key builder shared by `build.rs` scripts and `aprender-contracts`.

Part of the [Aprender](https://github.com/paiml/aprender) monorepo (`paiml/aprender`). It has no dependencies on purpose: a crate that build-depends on it does not mark the contract engine affected.

```rust
let key = aprender_contracts_key::env_key("my-contract", "eq.1");
assert_eq!(key, "CONTRACT_MY_CONTRACT_EQ_1");
```

## License

MIT
