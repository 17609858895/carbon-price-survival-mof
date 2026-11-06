"""Data-free mathematical smoke checks for the actual screening kernels."""
import ast
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
source = ROOT/'work/scripts/16_conformal_selection.py'
tree = ast.parse(source.read_text(encoding='utf-8'))
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {'bh', 'conformal_select', 'bayes_fdr_select'}]
scope = {'np': np}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), scope)

assert np.array_equal(scope['bh'](np.array([0.001, 0.02, 0.10, 0.80]), 0.05), [True, True, False, False])
rng = np.random.default_rng(29)
cal = rng.normal(size=200)
null = rng.random(200) < 0.65
test = np.linspace(-1.5, 4.0, 31)
selected = scope['conformal_select'](cal, null, test, 0.2)
assert selected.any() and not selected.all()
for scale in [0.5, 2.0, 10.0]:
    assert np.array_equal(selected, scope['conformal_select'](cal/scale, null, test/scale, 0.2))
perm_cal = rng.permutation(len(cal))
perm_test = rng.permutation(len(test))
permuted = scope['conformal_select'](cal[perm_cal], null[perm_cal], test[perm_test], 0.2)
assert np.array_equal(permuted, selected[perm_test])
pv = np.array([0.999, 0.98, 0.90, 0.7, 0.2])
pick = scope['bayes_fdr_select'](pv, 0.05)
assert pick.any() and np.mean(1-pv[pick]) <= 0.05

lco = np.array([60.0, 60.0, 60.0, 80.0])
cc = np.array([0.2, 1.0, 1.2, -0.1])
y = (1-cc)/lco
assert np.all(np.isfinite(y)) and np.array_equal(y > 0, cc < 1)
for price in [40.0, 100.0, 300.0]:
    physical = (cc < 1) & (lco <= price*(1-cc))
    assert np.array_equal(y >= 1/price, physical)

files = sorted(ROOT.rglob('*.py'))
for path in files:
    ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
print(f'PASS: shortlist kernels, scale/permutation invariance, reciprocal viability; {len(files)} Python sources parsed.')
