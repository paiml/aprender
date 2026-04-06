# Docker Bake configuration for reproducible builds
# B1/B2: Hermetic build infrastructure

variable "RUST_VERSION" {
  default = "1.83.0"
}

variable "RANDOM_SEED" {
  default = "42"
}

group "default" {
  targets = ["presentar", "ptop"]
}

target "base" {
  dockerfile = "Dockerfile"
  context = "."
  args = {
    RUST_VERSION = "${RUST_VERSION}"
    RANDOM_SEED = "${RANDOM_SEED}"
  }
}

target "presentar" {
  inherits = ["base"]
  target = "runtime"
  tags = ["presentar:latest", "presentar:${RUST_VERSION}"]
  platforms = ["linux/amd64", "linux/arm64"]
  output = ["type=docker"]
}

target "ptop" {
  inherits = ["base"]
  target = "ptop"
  tags = ["ptop:latest"]
  platforms = ["linux/amd64"]
  output = ["type=docker"]
}

target "dev" {
  inherits = ["base"]
  target = "development"
  tags = ["presentar-dev:latest"]
  cache-from = ["type=local,src=/tmp/buildx-cache"]
  cache-to = ["type=local,dest=/tmp/buildx-cache,mode=max"]
}

target "test" {
  inherits = ["base"]
  target = "test"
  args = {
    PRESENTAR_TEST_SEED = "${RANDOM_SEED}"
  }
}

target "bench" {
  inherits = ["base"]
  target = "bench"
  args = {
    PRESENTAR_BENCH_SEED = "12345"
    CRITERION_SAMPLE_SIZE = "1000"
  }
}
