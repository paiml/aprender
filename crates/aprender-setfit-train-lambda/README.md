# aprender-setfit-train-lambda

The AWS deployment of the SetFit training MCP server: two Lambda entry points
and the AWS ends of the seams `aprender-mcp-setfit-train` leaves open.

Part of the [aprender monorepo](https://github.com/paiml/aprender).

## What it is

`aprender-mcp-setfit-train` defines three seams and fills each with a
process-local implementation, so the whole server runs on a laptop over stdio or
HTTP with no cloud account. This crate fills the same three with AWS ones and
adds nothing else:

| Seam | Local | Here |
|------|-------|------|
| `TaskBackend` | `InMemoryTaskBackend` | `DynamoDbTaskBackend` |
| `Dispatcher` | `LocalDispatcher` (spawn a child) | `LambdaDispatcher` (async invoke) |
| terminal write | the spawning process | `aprender-setfit-trainer` |
| dataset in | a directory on the machine | an S3 object the client PUT to a presigned URL |
| artifact out | a file on the machine | a presigned GET, minted by `train_status` on read |

The tool surface, the task store, the mint handoff, the status payload and the
one implementation of "run `apr setfit train`" all stay upstream.

## Two binaries

`bootstrap` is the **request** function: it mints an MCP task, writes the run's
config into the task's envelope, asynchronously invokes the worker, and answers
`working`. Every one of those is milliseconds. It carries no `apr` binary, no
dataset and no encoder — which is what keeps it inside the deployment package
ceiling.

`aprender-setfit-trainer` is the **worker**: 6 GB, 15 minutes, invoked with
nothing but a task id and an owner. It reads the envelope, runs the pinned
`apr`, uploads the artifact to S3 and performs the guarded terminal write.

They are one crate because they are the two halves of one distributed
transaction and share its vocabulary — the task backend, the run envelope, the
status payload. Cargo builds each binary separately, so one package does not
mean one deployment package.

## Datasets and artifacts cross the boundary as S3 objects

MCP has no file-upload primitive, so `dataset_upload_url` hands the client a
presigned PUT and the `dataset_uri` to name the result by; `train` verifies the
URI is one this deployment issued and that the upload happened, and the worker
fetches and unpacks it before the CLI's pre-flight judges it. A completed
artifact comes back the same way in reverse: `train_status` mints a presigned
GET at read time, never stored, so the verdict in the task store stays true for
the task's whole TTL. Both grants are prefix-scoped (`datasets/`, `tasks/`).

## Why the split exists at all

pmcp.run fronts its Lambdas with an API Gateway HTTP API, whose 30-second
integration timeout cannot be raised, and Lambda freezes the execution
environment the moment a handler returns. The measured 8-shot reference train is
127 seconds at 4.0 GB peak RSS. Training therefore cannot happen inside the
request — not inline, and not as a background task the handler leaves running.

## Deploying

See [`deploy-extensions/README.md`](../../deploy-extensions/README.md) for the
runbook. The order matters, and two of the orderings are not guessable.

## Testing

Unit tests cover the item mapping and the S3 URI parsing — the parts checkable
without a network. The parts that carry correctness are DynamoDB's own
semantics, and `tests/dynamodb_contract.rs` drives them against a real table
(DynamoDB Local needs no AWS account). See that file's module doc for both
commands.
