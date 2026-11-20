Synthetic fixtures for #7330 (offline, no weights): `config.json` carries the
model_type (`qwen3_5_moe_text`), 92 layers and 512 experts of Qwen3.8-2.4T-A95B;
`model.safetensors.index.json` names tensors with the published Qwen3.5-MoE
spellings but only for layers 0-3, one MTP layer and a few vision tensors.
`unnamed.index.json` adds one tensor family the classifier has no name for.
