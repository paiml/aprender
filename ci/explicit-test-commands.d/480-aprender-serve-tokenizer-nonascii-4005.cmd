# #4054: the #4005 non-ASCII tokenizer parity target. Unignored here: violations() (pure).
# The model-bound test stays #[ignore]d; the release gate scripts/check_tokenizer_nonascii_parity.sh runs it.
cargo test -p aprender-serve --test tokenizer_nonascii_4005
