import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'eval')]
from build_sft_data import code_problem, training_rows
from synth_pilot import mutations, plan, prepare, records, reference
import model_bakeoff as mb


def test_numbered_programs_in_one_fence_stay_independent():
    text = '```python\n# cau 1\nanswer = 1\n\n# cau 2\nanswer = 2\n```'
    assert [c.strip() for c in mb.tach_khoi(text, 2)] == ['# cau 1\nanswer = 1', '# cau 2\nanswer = 2']


def test_oracle_small_records_and_mutations():
    d = records(41011)
    assert reference(d, 'gross') == {'type': 'number', 'answer': 294000}
    assert reference(d, 'quantity') == {'type': 'number', 'answer': 67}
    assert reference(d, 'net') == {'type': 'number', 'answer': 291000}
    assert reference(d, 'zero') == {'type': 'number', 'answer': 0}
    variants = dict(mutations(d))
    assert reference(variants['return_boundary'], 'net')['answer'] == 286000
    assert reference(variants['date_boundary'], 'gross')['answer'] == 1293000
    assert reference(variants['unit_factor'], 'quantity')['answer'] != 67
    assert reference(variants['tie'], 'top')['answer'] == ['Vở Sao', 'Bút Mây']
    assert reference(variants['join_fanout'], 'gross') == reference(d, 'gross')
    assert reference(variants['no_today'], 'exists')['answer'] is False
    assert reference(variants['unsold_item'], 'zero')['answer'] == 1
    assert reference(variants['rounding_boundary'], 'average')['answer'] == 51251


def test_code_result_dependency_not_blanket_constant_ban():
    assert code_problem('ANSWER = dm.money(294000)', 'number') == 'constant_answer'
    assert code_problem("s = T['lines']; total = s['amount'].sum(); ANSWER = dm.money(total)", 'number') is None
    assert code_problem("s = T['lines']; ANSWER = dm.money(294000)", 'number') == 'constant_answer'
    assert code_problem("ANSWER = dm.unanswerable('no salary table')", 'unanswerable') is None
    assert code_problem("s = T['lines']; x = 2025; ANSWER = dm.number(len(s[s['qty'] > 3]))", 'number') is None


def test_fixture_loader_and_runtime_are_real(tmp_path):
    from synth_tools import Kit
    for family in ('joined', 'flat', 'events'):
        f = prepare(tmp_path, 's_' + family, family, 41011)
        kit = Kit(data=f['root'], cache=Path(f['root']) / 'cache')
        code = "ANSWER = dm.number(len(T['" + next(iter(f['b']['T'])) + "']))"
        out = kit.execute(f['name'], code, '2025-05-16')
        assert out['ok']
        assert out['answer']['answer'] == len(next(iter(f['b']['T'].values())))
    assert len(plan()) == 12
    assert sum(s['period']['relative'] for s in plan()) == 4


def test_training_schema_rejects_raw_t4_and_dynamic_meta(tmp_path):
    p = tmp_path / 't4.jsonl'
    p.write_text(json.dumps({'kind': 't4_mau', 'question': 'synthetic', 'code': 'answer=1'}) + '\n')
    with pytest.raises(ValueError, match='outer contract'):
        list(training_rows(p))


def test_structured_spec_cannot_silently_change_period():
    from synth_pilot import sample_schema
    s = plan()[0]
    c = {'question': 'Doanh thu tháng 4/2025?', 'code': "answer=T['sales'].amount.sum()", 'loai': 'number', 'slot': 'gross',
         'spec': {k: s[k] for k in ('measure', 'filters', 'period', 'grouping', 'ranking', 'unit', 'tie_policy', 'answer_type')}}
    assert sample_schema(c, s, 'runtime')
    c['spec'] = dict(c['spec'], period=dict(s['period'], end='2025-05-01'))
    assert not sample_schema(c, s, 'runtime')
    c['spec'] = 'anything'
    assert not sample_schema(c, s, 'runtime')


def test_reconstructed_input_and_external_io_are_rejected(tmp_path, monkeypatch):
    assert code_problem("T={'x':pd.DataFrame({'v':[3]})}; ANSWER=dm.number(T['x'].v.sum())", 'number') == 'redefined_input_tables'
    monkeypatch.setattr(mb.subprocess, 'run', lambda *a, **k: pytest.fail('unsafe code executed'))
    for code in ("import os; answer=os.listdir('.')", "answer=pd.read_csv('private.csv')", "answer=open('private.csv').read()"):
        assert mb.run_code(code, tmp_path / 'unused.pkl', tmp_path, 'unsafe')[1].startswith('unsafe_code:')


def test_manual_audit_binds_actual_code_and_gold():
    from build_sft_data import audit_hash
    r = {k: '' for k in ('id','candidate','phase','shop','family','seed','source_hash','question','gold','plan','teacher_code','verifier_code','runtime_code')}
    original = audit_hash(r)
    assert audit_hash(dict(r, runtime_code='ANSWER=dm.number(3)')) != original
    assert audit_hash(dict(r, gold={'type':'number','answer':3})) != original


def test_wrong_tokenizer_never_certifies(tmp_path):
    from check_sft_tokens import verify_identity
    (tmp_path / 'tokenizer.json').write_text('{}')
    with pytest.raises(ValueError, match='identity mismatch'):
        verify_identity(tmp_path)


@pytest.mark.parametrize('response', [[10,20,30], {'input_ids':[10,20,30], 'attention_mask':[1,1,1]}])
def test_template_token_count_uses_ids_for_both_transformers_apis(response):
    from check_sft_tokens import template_tokens
    class Tokenizer:
        def apply_chat_template(self, *a, **k):
            return response
    assert template_tokens(Tokenizer(), [{'role':'user','content':'synthetic'}], True)==[10,20,30]


def test_forged_mutation_pass_flags_cannot_certify_bad_code(tmp_path):
    from build_sft_data import recheck_for_export
    f = prepare(tmp_path, 'forged', 'joined', 41011)
    row = {'id':'forged', 'shop':'forged', 'family':'joined', 'seed':41011, 'slot':'gross',
           'question':'Doanh thu bán tháng 4/2025?', 'question_canonical':'Doanh thu bán tháng 4/2025?',
           'gold':reference(f['truth'], 'gross'), 'reject':[],
           'runtime_code':"ANSWER=dm.money(T['lines']['amount'].sum())",
           'teacher_code':"answer=T['lines']['amount'].sum()", 'verifier_code':"answer=T['lines']['amount'].sum()",
           'mutations':[{'kind':'date_boundary','pass':True},{'kind':'join_fanout','pass':True}]}
    with pytest.raises(ValueError, match='fresh semantic mutation replay failed'):
        recheck_for_export([row], tmp_path)


def test_certification_requires_complete_families_and_fresh_final():
    from build_sft_data import require_pilot_coverage
    rows = [{'family':f, 'slot':s['slot'], 'shop':f, 'phase':'final' if f=='events' else 'dev', 'reject':[]}
            for f in ('joined','flat','events') for s in plan()]
    require_pilot_coverage(rows)
    with pytest.raises(ValueError, match='family coverage'):
        require_pilot_coverage(rows[1:])
    rejected_final = [dict(r, reject=['verifier_pass']) if r['family']=='events' and r['slot']=='gross' else r for r in rows]
    # Another accepted events row cannot hide a failed final-test slot.
    rejected_final.append(dict(rows[24], phase='expanded', shop='other'))
    with pytest.raises(ValueError, match='fresh final test'):
        require_pilot_coverage(rejected_final)
    with pytest.raises(ValueError, match='missing fresh final'):
        require_pilot_coverage([dict(r, phase='dev') for r in rows])
