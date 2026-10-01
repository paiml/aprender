# API Coverage — AWS (S3, IAM, Lambda, CloudWatch Logs, STS), pmcp.run via cargo-pmcp, Hugging Face Hub

> Full coverage by default. Opt-outs are explicit, reasoned decisions.
> Scope: the external services Phase 8 integrates — the Lambda crate's S3 cold-start loader
> (`aws-sdk-s3`), the deploy recipes (`aws` CLI, `cargo pmcp`), and the trainer's pinned base-weight
> download (Laya's `huggingface_hub` path). The `classify` MCP server is PROVIDED by this phase, not
> consumed, so it has no row here; its surface is contracted in `contracts/decide-tool-boundary-v1.yaml`.

| capability | decision | reason |
|---|---|---|
| s3.head_object (runtime loader: object length before allocation) | INTEGRATE | |
| s3.get_object_ranged (runtime loader: 16-way 64 MiB parts into memory) | INTEGRATE | |
| s3.put_object (runtime) | OPT-OUT | the served function is read-only by design; uploads happen once, from the laptop recipe (`aws s3 cp`) |
| s3.list_objects (runtime) | OPT-OUT | not needed — the key is content-addressed and pinned by sha256. The stack grants s3:GetObject on `decide/<server>/*` only and no s3:ListBucket, so a missing key reads 403 AccessDenied, not 404 |
| s3.delete_object | OPT-OUT | explicitly out of scope for automation — `laya-teardown` prints the `aws s3 rm` command for the human |
| s3.multipart_upload / presigned_urls / select / versioning / lifecycle_rules | OPT-OUT | not needed — immutable content-addressed objects, no client-side downloads, no expiry by design |
| s3.create_bucket + public_access_block + default_encryption + tagging (recipe) | INTEGRATE | |
| s3.cp upload + head-object size check (recipe) | INTEGRATE | |
| iam.list_role_policies / get_role_policy / list_attached_role_policies / get_policy / get_policy_version (`laya-grant`: READ-ONLY check of the stack-declared grant, every read's status checked) | INTEGRATE | |
| iam.delete_role_policy (`laya-teardown`: removes the LEGACY pre-08-17 out-of-band policy by name; NoSuchEntity reads as absent, any other failure exits 1) | INTEGRATE | |
| iam.put_role_policy | OPT-OUT | no longer used — since 08-17 option 1 the weights read is stack-declared (`[[iam.statements]]` in the deploy config, rendered by cargo-pmcp into the role's default policy); `laya-grant` only reads |
| iam.create_role / attach_managed_policy | OPT-OUT | the function role and its policies are owned by pmcp.run's stack; the phase writes no IAM, and `laya-grant` refuses any attached managed policy that reaches S3 |
| lambda.get_function (role discovery) | INTEGRATE | |
| lambda.update_function_configuration (per-sample cold-start forcing) | INTEGRATE | |
| lambda.put_function_concurrency / get_function_concurrency (containment) | INTEGRATE | |
| lambda.delete_function | OPT-OUT | deployment lifecycle belongs to pmcp.run; removal is printed (`cargo pmcp deploy destroy`) for the human, never automated |
| logs.filter_log_events (cold-start evidence correlation) | INTEGRATE | |
| sts.get_caller_identity (account check and evidence scrub) | INTEGRATE | |
| cargo-pmcp.deploy | INTEGRATE | |
| cargo-pmcp.login | INTEGRATE | human precondition of plan 08-11 (user_setup) |
| cargo-pmcp.oauth enable | INTEGRATE | used only when the 08-11 checkpoint chooses deploy-auth-on |
| cargo-pmcp.destroy | OPT-OUT | outward-facing removal is the human's call; printed by `laya-teardown`, containment uses reserved concurrency 0 instead |
| cargo-pmcp.secrets | OPT-OUT | not needed — the S3 URI and sha256 pin are non-secret configuration |
| cargo-pmcp.logs / metrics / outputs | OPT-OUT | not needed — CloudWatch is read directly for cold-start correlation |
| cargo-pmcp.rollback | OPT-OUT | not needed yet — a new model is a new content-addressed artifact and a redeploy |
| cargo-pmcp.test | OPT-OUT | not needed — the in-repo probe (`aprender-mcp-decide-lambda/examples/probe.rs`) asserts identity and timings |
| cargo-pmcp.package import / workbook / loadtest / pentest | OPT-OUT | explicitly out of scope for this phase |
| hf_hub.snapshot_download (pinned revision + sha256 mapping, via Laya) | INTEGRATE | |
| hf_hub.upload / create_repo | OPT-OUT | explicitly out of scope — no model is published to the Hub in this phase |
