"""Focused checks for physical-core detection (planning/hardware.py).

Not a logical_cores // 2 guess -- that's only right on a uniform 2x-SMT box.
See compatibility.py's llama.cpp cpu_threads sizing, which depends on this.
"""
from __future__ import annotations

from adaptive_llm.planning.hardware import _parse_physical_cores, collect_snapshot, load_fixture


# A trimmed but structurally real /proc/cpuinfo shape: 2 sockets x 2 cores x
# 2 threads (SMT) = 8 logical, 4 physical -- the case a logical//2 guess
# would get right by coincidence (8//2=4), used as a sanity check alongside
# the no-SMT case below where that guess would be wrong.
_TWO_SOCKET_SMT_CPUINFO = "\n\n".join(
    f"processor\t: {i}\nphysical id\t: {phys}\ncore id\t\t: {core}\n"
    for i, (phys, core) in enumerate([
        ("0", "0"), ("0", "0"),  # socket 0, core 0, 2 SMT threads
        ("0", "1"), ("0", "1"),  # socket 0, core 1
        ("1", "0"), ("1", "0"),  # socket 1, core 0
        ("1", "1"), ("1", "1"),  # socket 1, core 1
    ])
) + "\n"

# No SMT at all: 4 logical == 4 physical. logical_cores // 2 would wrongly
# report 2 -- this is the case the old formula actually got wrong.
_NO_SMT_CPUINFO = "\n\n".join(
    f"processor\t: {i}\nphysical id\t: 0\ncore id\t\t: {i}\n" for i in range(4)
) + "\n"


def test_parses_two_socket_smt_layout():
    assert _parse_physical_cores(_TWO_SOCKET_SMT_CPUINFO, logical_cores=8) == 4


def test_parses_no_smt_layout_where_the_old_half_guess_was_wrong():
    assert _parse_physical_cores(_NO_SMT_CPUINFO, logical_cores=4) == 4


def test_missing_physical_id_defaults_to_single_socket():
    # Some /proc/cpuinfo layouts omit "physical id" entirely on single-socket
    # boxes; every "core id" must still count once via the "0" default.
    text = "\n\n".join(f"processor\t: {i}\ncore id\t\t: {i}\n" for i in range(4)) + "\n"
    assert _parse_physical_cores(text, logical_cores=4) == 4


def test_empty_or_unparseable_text_falls_back_to_half_logical():
    assert _parse_physical_cores("", logical_cores=8) == 4
    assert _parse_physical_cores("garbage\nlines\n", logical_cores=1) == 1


def test_live_collect_snapshot_reports_a_sane_physical_core_count():
    """Real /proc/cpuinfo on whatever box runs this: physical_cores must be
    positive and never exceed logical_cores."""
    snap = collect_snapshot()
    assert 1 <= snap.static.physical_cores <= snap.static.logical_cores


def test_load_fixture_without_physical_cores_falls_back_to_half_logical(tmp_path):
    """A fixture predating this field must keep producing the exact planner
    output it always did -- same half-of-logical heuristic, not a surprise
    jump from re-deriving it differently."""
    import json

    fixture = {
        "static": {"logical_cores": 12, "total_ram_mib": 65536, "cuda_available": False},
        "dynamic": {"available_ram_mib": 32768},
    }
    path = tmp_path / "no_physical_cores.json"
    path.write_text(json.dumps(fixture))
    snap = load_fixture(str(path))
    assert snap.static.physical_cores == 6


def test_load_fixture_respects_an_explicit_physical_cores(tmp_path):
    import json

    fixture = {
        "static": {"logical_cores": 12, "physical_cores": 12, "total_ram_mib": 65536, "cuda_available": False},
        "dynamic": {"available_ram_mib": 32768},
    }
    path = tmp_path / "explicit_physical_cores.json"
    path.write_text(json.dumps(fixture))
    snap = load_fixture(str(path))
    assert snap.static.physical_cores == 12
