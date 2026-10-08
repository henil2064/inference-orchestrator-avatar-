# Repository Test Interface

Run fast, deterministic correctness checks from the repository root:

```bash
./test
./test all
./test service
./test orchestrator
./test adaptive_llm
./test media-edge
./test stt
./test tts
./test scripts
./test pipeline-helpers
```

Opt-in checks are explicit:

```bash
./test integration
./test e2e
./test adaptive_llm-llamacpp
./test adaptive_llm-cache-bench
./test adaptive_llm-vllm-bench
./test avatar
./test media-edge-bench
./test orchestrator-replica-live
./test pipeline
./test pipeline-load-burst
./test pipeline-open-loop
./test scripts-supervisor
./test stt-tune
./test tts-client
./test tts-tune
```

The runner reads `tests/suites.yaml` and dispatches to service-local tests with
the correct working directory, `PYTHONPATH`, and service virtualenv. Test
behavior remains owned by the service directories.

Python suites use the venv declared in `tests/suites.yaml`:

| suite | interpreter |
| --- | --- |
| `adaptive_llm` | `adaptive_llm/.venv/bin/python` |
| `orchestrator` | `orchestrator/.venv/bin/python` |
| `stt` | `stt/.venv-sravaani/bin/python` |
| `tts` | `tts/.venv/bin/python` |
| `scripts` | `tests/.venv/bin/python` |
| `pipeline-*` | `pipeline_test/.venv/bin/python` |

If a venv has not been created yet, run that service's setup/launcher once, or
use `TEST_PYTHON=/path/to/python ./test <suite>` to override the manifest for a
one-off run.

Pytest itself is loaded from `tests/.venv` through `tests/pytest_bootstrap.py`.
That keeps test tooling out of the service venvs while still executing the test
code with the service interpreter and its installed modules.

Pytest tier markers are registered in `pytest.ini` and applied coarsely from
the root `conftest.py`. The repo-level runner is still the supported default
entrypoint because each service needs its own working directory and dependency
environment.

## Inventory

| service | file | tier | dependencies | default run? | notes |
| --- | --- | --- | --- | --- | --- |
| `adaptive_llm` | `adaptive_llm/tests/test_config.py` | service | Python deps | yes | facade config behavior |
| `adaptive_llm` | `adaptive_llm/tests/test_context_window.py` | unit | Python deps | yes | prompt/context trimming logic |
| `adaptive_llm` | `adaptive_llm/tests/test_conversation_cache_bench.py` | unit | Python deps | yes | summary logic for cache bench output, not a benchmark run |
| `adaptive_llm` | `adaptive_llm/tests/test_env_loading.py` | unit | Python deps | yes | dotenv parsing |
| `adaptive_llm` | `adaptive_llm/tests/test_facade.py` | service | grpc health deps, fakes | yes | adaptive facade admission/worker failure behavior |
| `adaptive_llm` | `adaptive_llm/tests/test_hardware.py` | unit | Python deps | yes | hardware parsing/normalization fixtures |
| `adaptive_llm` | `adaptive_llm/tests/test_model_store.py` | service | Python deps, tmp files/fakes | yes | model registry and planner-facing validation |
| `adaptive_llm` | `adaptive_llm/tests/test_planner.py` | unit | Python deps | yes | execution-plan selection |
| `adaptive_llm` | `adaptive_llm/tests/test_prompt_manager.py` | unit | Python deps | yes | prompt selection and validation |
| `adaptive_llm` | `adaptive_llm/tests/test_runtime_manager.py` | service | subprocess, grpc | yes | worker startup failure/timeout handling |
| `adaptive_llm` | `adaptive_llm/tests/test_vllm_engine_args.py` | unit | transformers, mocked torch | yes | vLLM env mapping and compatibility shims |
| `adaptive_llm` | `adaptive_llm/tests/test_llamacpp_interrupt.py` | model | llama-cpp-python import, fake model | no | llama.cpp cancellation semantics |
| `adaptive_llm` | `adaptive_llm/tests/test_llamacpp_prompt_format.py` | model | llama-cpp-python import | no | prompt format selection against worker helper |
| `adaptive_llm` | `adaptive_llm/tests/test_llamacpp_stream_metrics.py` | model | llama-cpp-python import, fake model | no | stream metrics/profile contract |
| `adaptive_llm` | `adaptive_llm/tests/test_sglang_interrupt.py` | service | transformers import, fake HTTP | yes | SGLang stream cancellation |
| `adaptive_llm` | `adaptive_llm/tests/test_vllm_stream_metrics.py` | service | transformers import, fake engine | yes | vLLM stream metrics/profile contract |
| `orchestrator` | `orchestrator/tests/*.py` | service | grpc, redis package, prometheus, fakes | yes | `MediaBridge`, routing, admission, metrics, logger, and session-store behavior; live-socket replica fan-out is opt-in |
| `orchestrator` | `orchestrator/tests/test_replica_fanout.py::test_replica_channel_round_robins_and_excludes_unhealthy` | integration | grpc health deps, local sockets | no | verifies real gRPC round-robin health behavior |
| `stt` | `stt/tests/test_segment.py` | service | grpc, numpy, fake model | yes | `AsrServicer` segmentation/admission/watchdog paths |
| `tts` | `tts/tests/test_backends.py` | service | numpy, grpc, fake backends | yes | backend routing, stream splitting, admission, and watchdog paths |
| `tts` | `tts/tests/test_client.py` | integration | running TTS gRPC service | no | manual live client check |
| `media-edge` | `media-edge/**/*_test.go` | service | Go toolchain | yes | Go package tests; benchmarks are skipped unless `-bench` is used |
| `media-edge` | `media-edge/cmd/media-edge/perf_bench_test.go` | bench | Go toolchain, optional Silero assets | no | hot-path benchmarks via `./test media-edge-bench` |
| `tests` | `tests/test_runner.py` | unit | pytest | yes | repo test-dispatch target expansion |
| `scripts/orchestration` | `scripts/orchestration/test_plan.py` | unit | stdlib | yes | replica sizing formula |
| `scripts/orchestration` | `scripts/orchestration/test_supervisor.py` | integration | grpc health deps, subprocesses, local sockets | no | supervisor restart and health-watchdog behavior |
| `pipeline_test` | `pipeline_test/test_stream_wav.py` | unit | pytest | yes | dependency-light `stream_wav.py` helper checks |
| `pipeline_test` | `pipeline_test/stream_wav.py` | e2e | running stack, WebRTC deps, audio input | no | end-to-end WAV/WebRTC replay |
| `pipeline_test` | `pipeline_test/load_burst.py` | e2e | running stack, WebRTC deps, audio input | no | closed-loop concurrent load generator |
| `pipeline_test` | `pipeline_test/open_loop_load.py` | e2e | running stack, WebRTC deps, audio input | no | open-loop scheduled load generator |
| `adaptive_llm` | `adaptive_llm/tools/vllm_tune.py` | bench | model path/ref, GPU/runtime deps | no | vLLM model/load benchmark |
| `adaptive_llm` | `adaptive_llm/tools/vllm_cache_tune.py` | bench | model path/ref, GPU/runtime deps | no | prefix-cache benchmark |
| `scripts` | `scripts/tune_stt.py` | bench | STT venv, benchmark audio, model deps | no | STT tuning benchmark |
| `scripts` | `scripts/tune_tts_indic_mio.py` | bench | TTS venv, model deps | no | TTS tuning benchmark |
