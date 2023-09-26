from biocomp import utils as ut
import numpy as np
from copy import deepcopy
import biocomp.datautils as du
import biocomp.train as train
import biocomp.parameters as pm
import biocomp.nodes as nd
import biocomp.compute as cmp
from biocomp.parameters import ParameterTree, PTree, ParamPath
import biocomp
import jax
from jax import numpy as jnp
import jax.tree_util as jtu
import pytest

b = pm.PTreeBranch()

p = ParameterTree()
assert p.data.is_empty()

assert len(p.data) == 0
assert len(list(p.data.iter_leaves())) == 0

p['a'] = 1
p.data.get_at('a')
assert PTree.is_leaf_at(p.data, 'a')
assert len(p.data) == 1
assert p['a'] == 1

p['b/arr'] = np.array([0,1,2])
p

assert np.all(p['b/arr'] == np.array([0,1,2]))
assert not PTree.is_leaf_at(p.data, 'b')
assert PTree.is_leaf_at(p.data, 'b/arr')

assert p['a'] == 1
assert len(p.data) == 2
del p.data['a']
assert len(p.data) == 1
p['a'] = 1
assert len(p.data) == 2

# trying to create a branch on leaf node should fail:
with pytest.raises(KeyError):
    p['a/newbranch'] = 1

r = p.at('b/arr2', np.array([3,4]))
assert np.all(r == np.array([3,4]))
assert np.all(p['b/arr2'] == np.array([3,4]))
assert len(p.data) == 3
assert len(p['b']) == 2


ref = pm.ArrayRef(p.data)
ref.push_back('b/arr', 1)
ref.push_back('b/arr2', 0)
assert np.all(ref.view() == np.array([1, 3]))
p['ref'] = ref
assert np.all(p['ref'] == ref.view())
v = p.at('sub/ref', ref)
assert np.all(v == ref.view())
v2 = p.at('sub/ref', 3, overwrite=True)
assert v2 == 3 == p['sub/ref']
v = p.at('sub/ref', ref, overwrite=True)
v = p.at('sub/ref', 3, overwrite=False)
assert np.all(v == ref.view())
assert np.all(p['sub/ref'] == ref.view())


assert type(p.data.get_at('ref', get_leaf_value=False)) == pm.PTree
assert (id(p.data.get_at('ref', get_leaf_value=False)) != id(ref))
assert type(p.data.get_at('ref', get_leaf_value=False).value) == pm.ArrayRef
assert id(p.data.get_at('ref', get_leaf_value=False).value) == id(ref)
assert len(p.data) == 5

pcopy = deepcopy(p)
assert pcopy == p

def test_parampath():
    pa = ParamPath('a')
    assert pa.path == ['a']
    pab = ParamPath('a/b')
    assert pab.path == ['a', 'b']
    pab2 = ParamPath(['a', 'b'])
    assert pab2.path == ['a', 'b']
    assert pab == pab2
    assert pab != pa
    assert pa < pab
    assert pab > pa
    paaa = ParamPath('a/a/a')
    assert paaa > pa
    assert paaa < pab
    pb = ParamPath('b')
    pba = ParamPath('b/a')
    assert pb > pa
    assert pba > pab
    paroot = ParamPath('/a/')
    assert str(pa) == 'a'
    assert str(paroot) == 'a'
    assert str(pab) == 'a/b'
    assert str(pba) == 'b/a'
    assert str(paaa) == 'a/a/a'
    empty = ParamPath('')
    assert str(empty) == ''
    assert empty < pa
    padr = ParamPath('a/d/r')
    assert (pab < padr)
    arp = pm.ArrayRefPath(ParamPath('a/d/r'), [], [])
    assert pab < arp
    assert arp > pab

test_parampath()

pleaves, pstruct = jtu.tree_flatten(p)
assert len(pleaves) == 3
[id(x) for x in pleaves]

reconstructed = jtu.tree_unflatten(pstruct, pleaves)
assert reconstructed == p
assert (id(reconstructed.data.get_at('ref', get_leaf_value=False)) != id(ref))

assert PTree.is_leaf_at(p.data, 'a')
assert PTree.is_leaf_at(reconstructed.data, 'a')
assert not PTree.is_leaf_at(reconstructed.data, 'b')

assert p.tags.is_empty()
p.tag('a', 'tag1')
assert not p.tags.is_empty()
assert p.tags['a'] == np.array([True])
p.tag('b/arr', 'tag1')
assert p.tags['b/arr'] == np.array([True])

p.tag('a', 'tag2')
assert np.all(p.tags['a'] == np.array([True, True]))
assert np.all(p.tags['b/arr'] == np.array([True, False]))

t1, not1 = p.filter_by_tag('tag1')

assert t1.tagnames == ['tag1', 'tag2']

assert len(t1.data) == 2
assert len(not1.data) == 3

merged = ParameterTree.merge(not1, t1)
assert merged.data == p.data
assert merged.tags == p.tags
assert merged.tagnames == p.tagnames
assert merged == p


pl, ps = jtu.tree_flatten(p)
ml, ms = jtu.tree_flatten(merged)
assert pl == ml
assert ms == ps



added = jtu.tree_map(lambda x0, x1: x0 + x1, merged, p)

PTree.check(added.data)
assert added['a'] == 2
assert np.all(added['b/arr'] == np.array([0,2,4]))
assert np.all(added['b/arr2'] == np.array([6,8]))
assert np.all(added['ref'] == np.array([2,6]))
assert added.tagnames == p.tagnames
assert added.tags == p.tags


assert id(not1.data.get_at('ref', get_leaf_value=False).value) != id(merged.data.get_at('ref', get_leaf_value=False).value)
assert id(not1.data.get_at('ref', get_leaf_value=False).value.tree) != id(merged.data.get_at('ref', get_leaf_value=False).value.tree)
assert not1.tagnames == t1.tagnames == p.tagnames
assert merged.tagnames == t1.tagnames

t1.tag('b', 'tag2')

merged2 = ParameterTree.merge(not1, t1)
assert merged2 != merged
assert merged2 != p

t1l, t1s = jtu.tree_flatten(t1)
not1l, not1s = jtu.tree_flatten(not1)


rec_t1 = jtu.tree_unflatten(t1s, t1l)
rec_not1 = jtu.tree_unflatten(not1s, not1l)

rec_merged = ParameterTree.merge(rec_not1, rec_t1)
assert rec_merged == merged2


p.at('b/n/arr3', np.array([5,6]), ['tag1'])
assert np.all(p['b/n/arr3'] == np.array([5,6]))
assert np.all(p.tags['b/n/arr3'] == np.array([True, False]))

pcopy = deepcopy(p)
assert pcopy == p

p2 = ParameterTree()
a = p2.at('b/arr', np.array([0,1,2]), ['tag1'])
assert np.all(p2['b/arr'] == np.array([0,1,2]))
assert np.all(a == np.array([0,1,2]))



def conv_to_float(x):
    if isinstance(x, (np.ndarray, jnp.ndarray)):
        return x.astype(np.float32)
    return float(x)
t1_f = jtu.tree_map(conv_to_float, t1)
not1_f = jtu.tree_map(conv_to_float, not1)

assert t1_f['b/arr'].dtype == np.float32
assert t1['b/arr'].dtype == np.int64

def f(dyn, stat):
    m = pm.ParameterTree.merge(dyn, stat)
    return jnp.mean(m['sub/ref'] * 2)
v,g = jax.jit(jax.value_and_grad(f))(t1_f, not1_f)
assert v == 4.0
assert np.all(g['b/arr'] == np.array([0.0, 1.0, 0.0]))

pf = jtu.tree_map(conv_to_float, p)

@jax.jit
def full_f(par):
    t, nt = par.filter_by_tag('tag1')
    return f(t, nt)


v,g = jax.jit(jax.value_and_grad(full_f))(pf)
assert v == 4.0
assert np.all(g['b/arr'] == np.array([0.0, 1.0, 0.0]))

jtu.tree_map(lambda x: type(x), p.data)

p.data.check()
t1.data.check()
not1.data.check()
merged.data.check()
merged2.data.check()
rec_merged.data.check()
p2.data.check()


# serialization
# with pickle
import pickle
s = pickle.dumps(p)
p3 = pickle.loads(s)
assert p3 == p
p3.data.check()
# let's use pickle for now...


##

# tree_at

pt = pm.ParameterTree()
pt.at('a', 1)
pt.at('b/c', np.array([1,2,3], dtype=np.float32))

@jax.jit
def add1(x):
    c = x['b/c'] + 1
    x = x.tree_set_at('b/c', c)
    return x

@jax.jit
def addn(x, n):
    x = jax.lax.fori_loop(0, n, lambda i, x: add1(x), x)
    return x



pt = addn(pt, 3)

assert np.all(pt['b/c'] == np.array([4,5,6]))

##
print()
print()
print('test_params.py: all tests passed')

# # Array likes
# for t in (np.int32, np.int64, np.float32, np.float64, bool, int, float):
    # for shape in [(2,), (2,3), (2,3,4), (), (1,)]:
        # a = jnp.arange(np.prod(shape)).reshape(shape).astype(t)
        # sa = pm.serialize(a)
        # da = pm.deserialize(sa)
        # assert np.all(da == a)
        # assert da.dtype == a.dtype
        # assert da.shape == a.shape
        # assert type(da) == np.ndarray




