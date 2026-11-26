Fixtures for #7330 (offline, no weights).

`config.json` and `model.safetensors.index.json` are the published files of
`Qwen/Qwen3.8-2.4T-A95B`, verbatim, fetched 2026-11-24 from
`https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B/raw/main/`:

    4e3819548967e319ab435d044a3a331dbe3b078590ce822e9d74b79430533987  config.json
    e36c40d4e99b2714fff821218a0433bda2dec46afdb1ebee8ce96ced997928ee  model.safetensors.index.json

`inspect_families_tests.rs` asserts both hashes. The index names 1609 tensors over 92
language layers and one MTP layer; the 512 experts of a layer are packed into two tensors
(`mlp.experts.gate_up_proj`, `mlp.experts.down_proj`), so the expert count comes from
`config.json`, not from the index.

`unnamed.index.json` is synthetic (hand-written, layers 0-3): it adds one tensor family the
classifier has no name for.
