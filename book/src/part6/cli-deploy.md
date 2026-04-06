# `batuta deploy`

Generate production deployment configurations for ML models across multiple platforms.

## Synopsis

```bash
batuta deploy <COMMAND> [OPTIONS]
```

## Description

The deploy command generates deployment artifacts (Dockerfiles, Lambda handlers, Kubernetes manifests, etc.) for serving ML models in production. Each target platform has its own subcommand with platform-specific options.

## Subcommands

| Command | Description |
|---------|-------------|
| `docker` | Generate Dockerfile for containerized deployment |
| `lambda` | Generate AWS Lambda deployment package |
| `k8s` | Generate Kubernetes manifests (Deployment, Service, HPA) |
| `fly` | Generate Fly.io configuration (`fly.toml`) |
| `cloudflare` | Generate Cloudflare Workers deployment |

## Examples

### Docker Deployment

```bash
$ batuta deploy docker pacha://llama3:8b
```

### AWS Lambda

```bash
$ batuta deploy lambda my-model:v1.0
```

### Kubernetes with Scaling

```bash
$ batuta deploy k8s --replicas 3
```

### Fly.io

```bash
$ batuta deploy fly --region iad
```

### Cloudflare Workers

```bash
$ batuta deploy cloudflare --wasm
```

## See Also

- [`batuta serve`](./cli-serve.md) - Local model serving
- [Phase 5: Deployment](../part2/phase5-deployment.md)
- [Docker Containerization](../part2/docker.md)

---

**Previous:** [`batuta serve`](./cli-serve.md)
**Next:** [`batuta pacha`](./cli-stack.md)
