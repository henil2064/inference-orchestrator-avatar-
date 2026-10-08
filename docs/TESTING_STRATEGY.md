# Testing Strategy

This repository has multiple runtime services, languages, and dependency
stacks. The test organization should preserve service ownership while giving
developers one clear way to run the right level of checks.

The goal is not to merge everything into one large test script. A single script
would become brittle because `media-edge` uses Go, the Python services have
separate environments, and some checks require real models or infrastructure.
Instead, keep tests close to the service they validate and add a repo-level test
entrypoint that dispatches to the correct suite.

## Current Shape

The repository currently has service-local tests:

| Area | Current tests |
| --- | --- |
| `adaptive_llm` | Python tests under `adaptive_llm/tests` |
| `orchestrator` | Python tests under `orchestrator/tests` |
| `stt` | Python tests under `stt/tests` |
| `tts` | Python tests under `tts/tests` |
| `media-edge` | Go tests under `media-edge/cmd/media-edge` and `media-edge/internal` |
| `scripts/orchestration` | Python self-checks for host-process planning and supervision |
| `pipeline_test` | End-to-end WAV/WebRTC pipeline harness |

Many Python tests are written as pytest-compatible modules but also retain
standalone `if __name__ == "__main__"` entrypoints. That is useful for ad hoc
development, but pytest should become the normal correctness-test runner.

## Recommended Structure

Keep service-local tests in place:

```text
adaptive_llm/tests/
orchestrator/tests/
stt/tests/
tts/tests/
media-edge/**/*_test.go
scripts/orchestration/test_*.py
pipeline_test/
```

Add a small repo-level test interface:

```text
tests/
  README.md
  run.py or test.sh
  suites.yaml
```

The top-level runner should dispatch to service-local commands. It should not
contain the test logic itself.

Example command surface:

```bash
./test unit
./test service
./test integration
./test e2e
./test all

./test adaptive_llm
./test orchestrator
./test media-edge
./test stt
./test tts
```

Internally, those commands can call the appropriate tool for each service:

```bash
python -m pytest orchestrator/tests
python -m pytest adaptive_llm/tests
python -m pytest stt/tests
python -m pytest tts/tests
go test ./...
```

Each command should run from the correct working directory and environment,
because the services do not all share the same dependency stack.

## Test Tiers

Classify every test into one of these tiers.

| Tier | Purpose | Requirements | Default? |
| --- | --- | --- | --- |
| Unit | Pure logic, no network, no model, no Docker | Language runtime only | Yes |
| Service contract | One service boundary tested with fakes or mocks | Service dependencies, no full pipeline | Yes if fast |
| Integration | Multiple services or external infrastructure | Redis, ClickHouse, live gRPC, model process, or Docker | No |
| End-to-end | Real user-like pipeline path | Running stack and representative input audio | No |
| Benchmark/performance | Throughput, latency, profiling, load behavior | Controlled machine state, often models or services | No |

## Examples

Likely unit or fast service-contract tests:

- `orchestrator/tests/test_rms.py`
- `orchestrator/tests/test_endpoint.py`
- `scripts/orchestration/test_plan.py`
- `adaptive_llm/tests/test_planner.py`
- `adaptive_llm/tests/test_facade.py` when using dry-run/fakes
- `media-edge/internal/profile/*_test.go`
- `media-edge/cmd/media-edge/bridge_test.go`

Likely heavier or opt-in tests:

- Tests requiring actual STT, TTS, or LLM model loading
- Runtime-specific LLM tests for llama.cpp, vLLM, or SGLang
- Tests requiring Redis, ClickHouse, or multiple live services
- `pipeline_test/test_stream_wav.py`
- WAV/WebRTC harness runs from `pipeline_test/stream_wav.py`
- Load and benchmark scripts

## Markers and Naming

Python tests should use pytest markers once a central pytest configuration is
introduced:

```text
unit
service
integration
e2e
model
gpu
slow
bench
```

Go tests should follow normal `go test` naming and use build tags or explicit
test names for slow, model-backed, or environment-dependent checks.

Rename ambiguous files over time when it improves clarity. For example:

| Current name | Possible clearer name |
| --- | --- |
| `orchestrator/tests/test_endpoint.py` | `test_vad_endpointing.py` |
| `tts/tests/test_client.py` | `test_tts_grpc_client.py` |
| `tts/tests/test_backends.py` | `test_tts_backend_selection.py` |

Do not rename everything at once. Prefer gradual cleanup when touching a test
for nearby work.

## What Not To Do

Do not create one giant test file for the whole repository.

Do not move all service tests into one central directory if that hides service
ownership or makes imports harder.

Do not run model-heavy, GPU-heavy, Docker, load, or end-to-end checks as part of
the default fast suite.

Do not mix benchmarks and correctness tests in the same default command.

## Phased Plan

### Phase 1: Inventory

Create a table of every existing test:

```text
service | file | tier | dependencies | default run? | notes
```

This should identify tests that are currently correctness checks, benchmarks,
manual harnesses, or model-dependent validation.

### Phase 2: Central Runner

Add a repo-level test command that can run:

- all fast Python tests
- all fast Go tests
- one service at a time
- one tier at a time

The first default suite should be conservative: fast, deterministic, and not
dependent on model downloads, GPUs, Redis, ClickHouse, or the full pipeline.

### Phase 3: Pytest Configuration

Add shared pytest configuration for markers and defaults. Keep the service test
files in place, but standardize invocation and marker usage.

### Phase 4: Separate Heavy Suites

Create explicit commands for:

- integration tests
- model-backed tests
- GPU tests
- end-to-end pipeline tests
- benchmarks and load tests

These should be opt-in and documented with prerequisites.

### Phase 5: Cleanup and Normalization

Gradually remove standalone `main()` runners from correctness tests, or leave
them only where they are genuinely useful for manual development.

Move profiling scripts and load generators out of the default test path if they
are currently named like regular correctness tests.

Normalize naming so test files describe the behavior they protect, not just the
module they touch.

## Target Outcome

The desired end state is:

- Service teams can still work locally inside their service directory.
- The repository has one obvious test command surface.
- Fast tests are deterministic and suitable for frequent local runs.
- Heavy tests are available but explicit.
- End-to-end tests validate the real pipeline without replacing service-level
  tests.
- Benchmarks and load tests are separated from correctness checks.

