# #3762: LLAMA_FTYPES (the scheme a GGUF declares in general.file_type) equals the
# upstream-extracted fixture, row for row.
cargo test -p aprender-quant --test llama_ftypes_fixture
