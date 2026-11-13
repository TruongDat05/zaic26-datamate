import pickle
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'eval'), str(ROOT)]
import model_bakeoff as mb
from synth_pilot import prepare
from synth_tools import Kit


@pytest.fixture
def executions(tmp_path, monkeypatch):
    monkeypatch.setenv('SYNTH_TEST_SECRET', 'secret_value')
    tables = tmp_path / 'tables.pkl'
    tables.write_bytes(pickle.dumps({'x': pd.DataFrame({'v': [1, 2, 3]})}))
    fixture = prepare(tmp_path, 's_isolation', 'flat', 41011)
    kit = Kit(data=fixture['root'], cache=Path(fixture['root']) / 'cache')

    def raw(code):
        return mb.run_code(code, tables, tmp_path, 'isolation')

    def runtime(code):
        result = kit.execute(fixture['name'], code, '2025-05-16')
        return (result['answer']['answer'] if result['ok'] else None, result['error'])

    return raw, runtime


@pytest.mark.parametrize('path', [0, 1])
def test_generated_code_cannot_read_sentinel(executions, tmp_path, path):
    sentinel = tmp_path / 'protected.txt'
    sentinel.write_text('synthetic sentinel')
    answer, error = executions[path](f"x = pd.io.common.get_handle({str(sentinel)!r}, 'r')")
    assert answer is None
    assert 'PermissionError' in error


@pytest.mark.parametrize('path', [0, 1])
def test_generated_code_cannot_open_network_socket(executions, path):
    answer, error = executions[path]("x = pd.io.common.get_handle('http://127.0.0.1:9/', 'r')")
    assert answer is None
    assert 'Operation not permitted' in error


@pytest.mark.parametrize('path', [0, 1])
def test_analytics_and_environment(executions, path):
    code = ("answer = int(T['x'].v.sum())" if path == 0 else
            "ANSWER = dm.number(sum(len(v) for v in T.values()))")
    answer, error = executions[path](code)
    assert error is None and answer > 0
    code = ("answer = 'SYNTH_TEST_SECRET' in pd.io.common.os.environ" if path == 0 else
            "ANSWER = dm.string(str('SYNTH_TEST_SECRET' in pd.io.common.os.environ))")
    answer, error = executions[path](code)
    assert error is None
    assert answer in (False, 'False')
