import concurrent.futures
import pickle
import sqlite3
import sys
import threading
import urllib.error
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'eval'), str(ROOT)]
import llm
import model_bakeoff as mb


def test_run_code_concurrent_identical_tags(tmp_path, monkeypatch):
    barrier = threading.Barrier(2)
    original = mb.subprocess.run
    def synchronized(*args, **kwargs):
        barrier.wait(timeout=10)
        return original(*args, **kwargs)
    monkeypatch.setattr(mb.subprocess, 'run', synchronized)
    paths = []
    for i in (1, 2):
        p = tmp_path / f'{i}.pkl'
        p.write_bytes(pickle.dumps({'x': pd.DataFrame({'v': [i]})}))
        paths.append(p)
    with concurrent.futures.ThreadPoolExecutor(2) as ex:
        results = list(ex.map(lambda pair: mb.run_code(f'answer = int(T["x"].v.iloc[0]) + {pair[0]}', pair[1], tmp_path, 'same'), enumerate(paths)))
    assert results == [(1, None), (3, None)]


def test_weak_error_is_not_hard(monkeypatch):
    monkeypatch.setattr(mb, 'bo', lambda _: {'T': {}})
    monkeypatch.setattr(mb, '_known', lambda _: frozenset())
    c = {'question_goc': 'Có bao nhiêu mặt hàng trong tháng 1/2025?', 'question': 'Có bao nhiêu mặt hàng trong tháng 1/2025?', 'loai': 'number', 'raw': 2, 'loi_sinh': [], 'do_kho': 3, 'giai': {'a': {'answer': 2, 'loi': None}, 'b': {'answer': 2, 'loi': None}, 'w': {'answer': None, 'loi': 'loi_api'}}}
    result = mb.judge_t4({'bo': 'x', 'cases': [c], 'solvers': ['a', 'b'], 'weak': 'w', 'giai_xong': True})
    assert result['n_that_kho'] == 0
    assert c['giai']['w']['status'] == 'infrastructure_error'


def test_ledger_counts_failed_retries_and_enforces_guard(tmp_path, monkeypatch):
    llm.configure_run(tmp_path / 'calls.sqlite', limit=2)
    monkeypatch.setattr(llm, 'guard', lambda _: None)
    monkeypatch.setattr(llm.time, 'sleep', lambda _: None)
    monkeypatch.setattr(llm, '_post', lambda *a: (_ for _ in ()).throw(urllib.error.URLError('offline')))
    env = {'API_URL': 'https://example.invalid', 'API_KEY': 'dummy'}
    try:
        with pytest.raises(llm.BudgetExceeded):
            llm.chat('deepseek', 'synthetic', retries=3, env=env, log=False)
        with sqlite3.connect(tmp_path / 'calls.sqlite') as db:
            assert db.execute('select count(*) from requests').fetchone()[0] == 2
        with pytest.raises(ValueError):
            llm.chat('deepseek', 'synthetic', check_leak=False, env=env, log=False)
    finally:
        llm.configure_run(None)


def test_leak_propagates(monkeypatch):
    def fail(*args, **kwargs):
        raise llm.LeakError('blocked')
    monkeypatch.setattr(llm, 'chat', fail)
    with pytest.raises(llm.LeakError):
        mb._goi({'goi': []}, 'soan', 'deepseek', 'off', 'synthetic', 10)


def test_zero_and_same_numeric_answer_are_allowed():
    c = {'question_goc': 'Có bao nhiêu mặt hàng trong tháng 1/2025?', 'loai': 'number', 'raw': 0, 'loi_sinh': []}
    assert not mb.kiem_mau(c)
    a = {'_sh': {'one'}, '_typed': {'type': 'number', 'answer': 1000}}
    b = {'_sh': {'two'}, '_typed': {'type': 'number', 'answer': 1000}}
    assert not mb._trung(a, b)
    a['_sh'] = mb._shingles('Doanh thu tháng 1/2025 là bao nhiêu?')
    b['_sh'] = mb._shingles('Cho biết doanh thu của tháng 1/2025 nhé')
    assert mb._trung(a, b)
    b['_sh'] = mb._shingles('Doanh thu thuần tháng 1/2025 là bao nhiêu?')
    assert not mb._trung(a, b)
    assert {'table', 'unanswerable'} <= set(mb.LOAI)


def test_extra_cannot_replace_guarded_payload(monkeypatch):
    monkeypatch.setattr(llm, '_post', lambda *a: pytest.fail('unreviewed payload sent'))
    for key in ('messages', 'model', 'stream', 'stream_options'):
        with pytest.raises(ValueError, match='cannot override'):
            llm.chat('deepseek', 'synthetic', extra={key: 'replacement'}, log=False)


def test_nonstring_content_cannot_bypass_guard(monkeypatch):
    monkeypatch.setattr(llm, '_post', lambda *a: pytest.fail('unguarded content sent'))
    with pytest.raises(ValueError, match='string content'):
        llm.chat('deepseek', [{'role':'user','content':[{'type':'text','text':'synthetic'}]}], log=False)


def test_partial_fallback_and_main_cache_reset(tmp_path, monkeypatch):
    import gen_data
    import synth_tools
    source = tmp_path / 'shops'
    (source / 'one').mkdir(parents=True)
    (source / 'one' / 'tables.pkl').write_bytes(pickle.dumps({'x':pd.DataFrame({'v':[1]})}))
    (source / 'one' / 'rules.md').write_text('Self-authored synthetic rules')
    monkeypatch.setattr(mb, 'SHOPS', {})
    monkeypatch.setattr(llm, 'guard', lambda _: None)
    monkeypatch.setattr(mb, 'prompt_t4', lambda *a, **k: 'synthetic')
    monkeypatch.setattr(mb, '_known', lambda _: frozenset())
    mb._known.cache_clear = lambda: None
    monkeypatch.setattr(gen_data, 'VanPhong', lambda **k: None)
    monkeypatch.setattr(synth_tools, 'Kit', lambda: None)
    calls=[]
    def batch(j, *a):
        calls.append((j['provider'],j['n_req']))
        return dict(j, cases=[{'question':'synthetic'}] * j['n_req'] if len(calls)>1 else [{'question':'synthetic'}])
    monkeypatch.setattr(mb, 't4_batch', batch)
    monkeypatch.setattr(mb, 'ban_ghi_t4', lambda r:r)
    monkeypatch.setattr(mb, '_hop_le', lambda r:r['cases'])
    monkeypatch.setattr(mb, 'mau_finetune', lambda c,*a:c)
    monkeypatch.setattr(mb, 'summarize', lambda r:{})
    monkeypatch.setattr(gen_data, 'keep', lambda c,kept,kit: kept.append(c))
    try:
        args=['--shops',str(source),'--batches','1','--workers','1','--teacher','gemini:high,qwen:off','--ledger',str(tmp_path/'ledger.sqlite')]
        assert gen_data.main(args+['--out',str(tmp_path/'out')]) == 0
        assert calls == [('gemini',12),('qwen',11)]
        assert len((tmp_path/'out'/'raw_samples.jsonl').read_text().splitlines())==12
        assert not (tmp_path/'out'/'train.jsonl').exists()
        # A second main in the same interpreter must not retain the previous shop/cache.
        (source/'one').rename(source/'two')
        assert gen_data.main(args+['--dry-run']) == 0
        assert set(mb.SHOPS)=={'two'}
    finally:
        llm.configure_run(None)


def test_date_range_excludes_expiry(tmp_path):
    import gen_data
    tables = {'sales': pd.DataFrame({'order_date': pd.to_datetime(['2025-01-01', '2025-02-01']), 'expiry': pd.to_datetime(['2035-01-01', '2036-01-01'])})}
    (tmp_path / 'tables.pkl').write_bytes(pickle.dumps(tables))
    shop = gen_data.load_shop(tmp_path, 'synthetic rules')
    assert '01/02/2025' in shop['khoang']
    assert '2036' not in shop['khoang']


def test_unanswerable_and_table_are_typed():
    assert mb.typed(None, 'unanswerable')['type'] == 'unanswerable'
    assert mb.typed(3, 'unanswerable') is None
    assert mb.typed({'columns': ['x'], 'rows': [[1]]}, 'table')['type'] == 'table'


def test_missing_and_wrong_solver_classification(monkeypatch):
    monkeypatch.setattr(mb, 'bo', lambda _: {'T': {}})
    monkeypatch.setattr(mb, '_known', lambda _: frozenset())
    def case():
        return {'question_goc': 'Có bao nhiêu mặt hàng trong tháng 1/2025?', 'question': 'Có bao nhiêu mặt hàng trong tháng 1/2025?', 'loai': 'number', 'raw': 2, 'loi_sinh': [], 'do_kho': 3, 'giai': {'a': {'answer': 2, 'loi': None}, 'b': {'answer': 2, 'loi': None}}}
    for weak_answer, status, hard in [(None, 'not_measured', 0), ({'answer': 3, 'loi': None}, 'executed_wrong', 1), ({'answer': 'abc', 'loi': None}, 'invalid_output', 0)]:
        c = case()
        if weak_answer is not None:
            c['giai']['w'] = weak_answer
        r = mb.judge_t4({'bo': 'x', 'cases': [c], 'solvers': ['a', 'b'], 'weak': 'w', 'giai_xong': True})
        assert c['giai']['w']['status'] == status
        assert r['n_that_kho'] == hard


def test_ledger_cap_is_atomic_across_threads(tmp_path, monkeypatch):
    llm.configure_run(tmp_path / 'calls.sqlite', limit=3, provider_limits={'deepseek': 2})
    monkeypatch.setattr(llm, 'guard', lambda _: None)
    monkeypatch.setattr(llm, '_post', lambda *args: {'model': 'response-model', 'choices': [{'message': {'content': 'ok'}}], 'usage': {'completion_tokens': 1}})
    env = {'API_URL': 'https://example.invalid', 'API_KEY': 'dummy'}
    def call(_):
        try:
            return llm.chat('deepseek', 'synthetic', env=env, log=False, tag='stage:role')
        except llm.BudgetExceeded:
            return None
    try:
        with concurrent.futures.ThreadPoolExecutor(8) as ex:
            results = list(ex.map(call, range(8)))
        assert sum(r is not None for r in results) == 3
        with sqlite3.connect(tmp_path / 'calls.sqlite') as db:
            assert db.execute("select count(*) from requests where status='success' and response_model='response-model'").fetchone()[0] == 3
    finally:
        llm.configure_run(None)
