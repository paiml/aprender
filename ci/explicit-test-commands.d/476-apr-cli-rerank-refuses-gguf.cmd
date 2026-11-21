# #7294: apr rerank refuses a GGUF by name in a no-inference build; compiles to 0 tests under default features, runs under the no-default-features lane.
cargo test -p apr-cli --test rerank_refuses_gguf_without_inference
