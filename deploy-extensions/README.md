# SetFit training: the infrastructure pmcp.run does not provide

The SetFit **training** MCP server is two Lambda functions, deployed by two
different tools, and the order matters. This file is the runbook; the reasons
live next to the code they explain.

```
        client
          │  MCP over streamable HTTP
          ▼
  ┌───────────────────┐   mint task, write envelope   ┌──────────────┐
  │  request function │──────────────────────────────▶│  DynamoDB    │
  │  bootstrap        │                                │  task table  │
  │  1 GB · 30 s      │                                └──────┬───────┘
  │  deployed by      │   async invoke (task_id, owner)       │ read envelope
  │  cargo-pmcp       │────────────────┐                      │ terminal write
  └───────────────────┘                ▼                      │
                            ┌──────────────────────┐          │
                            │  training worker     │◀─────────┘
                            │  6 GB · 900 s        │
                            │  apr + dataset       │──▶ S3 artifact bucket
                            │  + encoder (159 MB)  │
                            │  deployed by CDK     │
                            └──────────────────────┘
```

**Why two functions.** pmcp.run fronts its Lambdas with an API Gateway HTTP
API, whose 30-second integration timeout cannot be raised, and Lambda freezes
the execution environment the moment a handler returns. The measured 8-shot
reference train is 127 seconds at 4.0 GB peak RSS. So training cannot happen
inside the request — not inline, and not as a background task the handler
leaves running.

**Why two CDK apps.** `deploy/lib/stack.ts` is rendered by cargo-pmcp. Hand
edits flip it to `cdk synth`, whose hashed logical IDs read as resource
replacements and collide with the platform's fixed names — that is what
produced an `UPDATE_ROLLBACK_COMPLETE` on the predict server. Extensions live
in their own app, the same separation chess-mcp uses.

## Deploying

Every step is a `just` recipe from the repo root. `dev` and `prod` are the only
environments; the app refuses anything else rather than synthesizing a stack
named after a typo.

```bash
# 1. Build the worker package: apr (arm64) + trainer + dataset + encoder.
#    Checks the glibc floor against provided.al2023's 2.34 — a mismatch there
#    otherwise fails at INVOCATION, long after a successful deploy.
just build-trainer-asset

# 2. Review. Prints every IAM change; this is the gate `deploy-training` skips.
just diff-training dev

# 3. Create the table, the bucket and the worker.
just deploy-training dev
#
#    If this fails with `'MemorySize' value failed to satisfy constraint:
#    Member must have value less than or equal to 3008`, the account's Lambda
#    memory ceiling has never been raised. That limit is NOT in Service Quotas —
#    it is an AWS Support case. Until it is raised you can deploy under it and
#    measure, which synth then warns about:
#
#        just deploy-training dev 3008
#
#    The plumbing works at 3008 MB; the reference train is expected to OOM,
#    because its measured peak is 4.0 GB. CloudWatch's "Max Memory Used" on the
#    worker's log group is what settles whether it actually does.

# 4. Point the request function at what step 3 created, from SSM. Writes the
#    gitignored .pmcp/deploy.toml from the tracked .pmcp/deploy.toml.template.
just pmcp-train-config dev

# 5. Deploy the request function. The recipe carries --manifest-path and
#    raises the fd limit; both are required and both fail confusingly.
just pmcp-train-deploy

# 6. Grant it access to the table, the worker and the two S3 prefixes it
#    presigns for. AFTER the deploy, because pmcp.run creates the execution
#    role — there is nothing to attach to until it has. Re-run it after any
#    `deploy-training` too: the policy document is a stack output, and this is
#    what carries a changed one across to the platform-owned role.
just pmcp-train-grant dev
```

Step 4 is not a convenience. The artifact bucket carries the account id for
global uniqueness, so its name does not exist until step 3 has run — and that
is also why its output is generated and gitignored rather than committed: this
tree is destined for a public upstream repo, and an account id has no business
travelling there.

Step 5 is a recipe because getting it wrong deploys the **predict** server under
the training server's name, and looks like a success while doing it.

cargo-pmcp picks the package to build in `find_lambda_package_dir`: first a
directory `<deploy-root>/{server_name}-lambda`, then the FIRST `*-lambda`
workspace package exposing a `bootstrap` binary. This workspace has two such
packages and the predict one sorts first, so any deploy root that does not
satisfy the first branch silently builds `aprender-mcp-setfit-lambda`.

That is why the deploy root is `crates/` and the crate is named
`aprender-setfit-train-lambda`: together they make
`crates/aprender-setfit-train-lambda/` match the first branch by construction.
`[server] binary` exists in cargo-pmcp's schema but the Lambda path never reads
it, so the directory name is the only lever.

**The tell:** `aprender-setfit-train-lambda` in the compile log. If you see
`aprender-mcp-setfit-lambda`, it is building the wrong server and nothing else
matters until that is fixed — the endpoint will come up healthy and answer every
MCP call with `no embedded model in this build`, which is the predict binary's
error.

It also raises `ulimit -n`. Linking the aarch64 bootstrap opens ~245 object
files through cargo-zigbuild's wrapper, and under macOS's default soft limit
the link dies with `ProcessFdQuotaExceeded` — which reads like a toolchain
fault rather than a shell setting.

Step 6 comes last for a reason worth stating, because the intuitive order is
the reverse: the policy names resources this stack owns, but it attaches to a
role that **pmcp.run** creates, whose name carries a random suffix
(`pmcp-<hash>-<server>-ExecutionRole-<id>`). So it cannot be attached before
the deploy, and the recipe discovers the role from the function rather than
asking anyone to copy it. Between steps 5 and 6 the server is live and every
`train` call compensates to `failed` with an AccessDenied — a clear error
rather than a hang, but not a working server.

Re-run step 6 after any pmcp.run redeploy. It is an out-of-band change to a
role that a platform-owned CloudFormation stack manages, and a stack update may
drop it; `put-role-policy` replaces by name, so re-running is free.

## Training on your own dataset

The `train` tool takes an optional `dataset_uri`. Absent, the run uses the
benchmark packaged with the worker; present, the worker fetches what it names.
MCP has no file-upload primitive, so the upload is out of band:

```bash
# 1. Pack an attested benchmark directory (what `apr data tweet-eval-stance`
#    and `apr data select` write). Flat, with selection-manifest.json at the root.
just dataset-pack path/to/my-attested-dir /tmp/mine.tar.gz

# 2. From an MCP client, call `dataset_upload_url` (no arguments). It returns
#    an `upload_url` (PUT, valid 15 minutes) and a `dataset_uri`.

# 3. Upload.
curl -X PUT --upload-file /tmp/mine.tar.gz "<upload_url>"

# 4. Call `train` with {"config": ..., "dataset_uri": "<dataset_uri>"}.
```

`train` checks the URI synchronously — that it is one this deployment issued,
and that something was actually uploaded to it — so a forgotten step 3 is a
tool error, not a task that fails a minute later. What the archive CONTAINS is
judged by the CLI's own pre-flight in the worker, exactly as the packaged
dataset is: this deployment looks for one file by name and validates nothing.

Producing an attested directory from arbitrary labeled data is `apr data` work
that does not exist yet; today the only producer is `apr data tweet-eval-stance`.

## Getting the trained artifact

`train_status` on a completed task carries `artifact_url`: a presigned GET,
valid one hour, minted at read time. It is never stored — the verdict in the
task store must stay true for the task's whole TTL, and a signed URL does not.
`tasks/result` serves the stored verdict and so has no link; poll `train_status`
for the download.

## Verifying it

```bash
# The TaskBackend contract against the real table — conditional writes,
# owner scoping, expiry filtering, the guarded terminal write.
AWS_PROFILE=ze-kasher-dev AWS_REGION=us-east-1 \
APRENDER_SETFIT_E2E_TASKS_TABLE=aprender-setfit-training-tasks-dev \
  cargo test -p aprender-setfit-train-lambda --test dynamodb_contract
```

The same suite runs against DynamoDB Local with no account at all — see the
module doc in `tests/dynamodb_contract.rs`.

## What differs between dev and prod

Environment is the only input, and it decides exactly one thing: how much the
stack is willing to lose.

| | dev | prod |
|---|---|---|
| Table / bucket on destroy | deleted | **retained** |
| Point-in-time recovery | off | on |
| Bucket versioning | off | on |
| Artifact lifecycle | 7 days | 90 days |
| Log retention | 1 week | 3 months |

The worker is identical in both: 6144 MB (unless overridden — see step 3),
900 s, arm64, 2 GB of `/tmp`, and **zero retries**. A retry re-runs a 127-second CPU-saturating job, and a failure
here is a bad config or a broken package rather than a transient — the guarded
terminal write makes a retry safe, but it cannot make one useful.

## Tearing down

```bash
just destroy-training dev
```

`dev` destroys its data by design; `prod` retains the table and the bucket, so a
`destroy` there leaves them behind for you to remove deliberately.
