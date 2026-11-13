import pytest

from check_setup import missing_files
from synth_pilot import prepare
from synth_tools import Kit


def test_generated_fixture_does_not_require_private_source(tmp_path):
    fixture = prepare(tmp_path, "standalone", "joined", 41011)
    kit = Kit(source=tmp_path / "absent_source", data=fixture["root"], cache=tmp_path / "cache")
    result = kit.execute(fixture["name"], "ANSWER = dm.number(7)", "2025-05-16")
    assert result["ok"] and result["answer"] == {"type": "number", "answer": 7}
    with pytest.raises(FileNotFoundError, match="build_synth_source"):
        _ = kit.splits


def test_export_check_requires_controller_beyond_inference_runtime(tmp_path):
    synth, controller, runtime = (tmp_path / name for name in ("synth", "controller", "runtime"))
    for path in missing_files("sql", synth, controller, runtime):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    assert missing_files("sql", synth, controller, runtime) == []
    missing = missing_files("export", synth, controller, runtime)
    assert runtime / "datamate/snapshot.py" in missing
    assert controller / "training/offline_candidates.py" in missing
    assert controller / "harness_control/exec_guard.py" in missing
