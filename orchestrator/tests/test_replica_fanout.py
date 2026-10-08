"""Self-check for Phase 3 STT/TTS replica fan-out: target resolution and the
multi-address round_robin + health-check channel. Run:
  python -m tests.test_replica_fanout
"""
import asyncio
import os

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

import service.server as server_module


def test_resolve_targets_falls_back_to_single_when_multi_unset():
    orig = dict(os.environ)
    try:
        os.environ.pop("STT_TARGETS", None)
        os.environ["STT_TARGET"] = "host1:1111"
        assert server_module._resolve_targets("STT_TARGETS", "STT_TARGET", "default:0") == ["host1:1111"]
    finally:
        os.environ.clear()
        os.environ.update(orig)
    print("ok test_resolve_targets_falls_back_to_single_when_multi_unset")


def test_resolve_targets_prefers_multi_and_splits_on_comma():
    orig = dict(os.environ)
    try:
        os.environ["STT_TARGETS"] = " host1:1111 , host2:2222 ,,"
        os.environ["STT_TARGET"] = "ignored:0"
        assert server_module._resolve_targets("STT_TARGETS", "STT_TARGET", "default:0") == [
            "host1:1111", "host2:2222"]
    finally:
        os.environ.clear()
        os.environ.update(orig)
    print("ok test_resolve_targets_prefers_multi_and_splits_on_comma")


async def _make_health_server(status, counter):
    class CountingInterceptor(grpc.aio.ServerInterceptor):
        async def intercept_service(self, continuation, handler_call_details):
            counter[0] += 1
            return await continuation(handler_call_details)

    servicer = health.aio.HealthServicer()
    await servicer.set("", status)
    srv = grpc.aio.server(interceptors=[CountingInterceptor()])
    health_pb2_grpc.add_HealthServicer_to_server(servicer, srv)
    port = srv.add_insecure_port("127.0.0.1:0")
    await srv.start()
    return srv, port, servicer


def test_replica_channel_round_robins_and_excludes_unhealthy():
    """The actual behavior the doc promises: with two backends, both healthy,
    traffic splits across both; once one reports NOT_SERVING (Phase 3 wedge
    self-report), round_robin stops picking it -- verified against a live
    channel, not assumed from the gRPC docs."""
    async def scenario():
        c1, c2 = [0], [0]
        srv1, p1, hs1 = await _make_health_server(health_pb2.HealthCheckResponse.SERVING, c1)
        srv2, p2, hs2 = await _make_health_server(health_pb2.HealthCheckResponse.SERVING, c2)
        try:
            channel = server_module._replica_channel([f"127.0.0.1:{p1}", f"127.0.0.1:{p2}"])
            async with channel:
                stub = health_pb2_grpc.HealthStub(channel)
                await asyncio.sleep(0.5)  # let health-check-based subchannel discovery settle
                for _ in range(20):
                    await stub.Check(health_pb2.HealthCheckRequest(service=""))
                assert c1[0] > 0 and c2[0] > 0, f"expected both backends used, got {c1} {c2}"

                await hs2.set("", health_pb2.HealthCheckResponse.NOT_SERVING)
                await asyncio.sleep(1.0)
                c1[0] = c2[0] = 0
                for _ in range(20):
                    await stub.Check(health_pb2.HealthCheckRequest(service=""))
                assert c2[0] == 0, f"round_robin still routed to a NOT_SERVING backend: {c2}"
                assert c1[0] > 0
        finally:
            await srv1.stop(grace=0)
            await srv2.stop(grace=0)

    asyncio.run(scenario())
    print("ok test_replica_channel_round_robins_and_excludes_unhealthy")


def test_downstream_health_gate_flips_serving_and_not_serving():
    async def scenario():
        statuses = []
        serving_sequence = iter([False, True, True, False])

        async def fake_downstreams_serving(*, timeout_s=1.0):
            try:
                return next(serving_sequence)
            except StopIteration:
                raise asyncio.CancelledError

        class FakeHealth:
            async def set(self, service, status):
                statuses.append(status)

        original = server_module._downstreams_serving
        server_module._downstreams_serving = fake_downstreams_serving
        try:
            try:
                await server_module._downstream_health_gate(FakeHealth(), interval_s=0, timeout_s=0)
            except asyncio.CancelledError:
                pass
        finally:
            server_module._downstreams_serving = original

        assert statuses == [
            health_pb2.HealthCheckResponse.SERVING,
            health_pb2.HealthCheckResponse.NOT_SERVING,
        ], statuses

    asyncio.run(scenario())
    print("ok test_downstream_health_gate_flips_serving_and_not_serving")
