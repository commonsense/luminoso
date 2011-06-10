from luminoso2.assoc_space import AssocSpace
import numpy as np
from divisi2 import DenseMatrix, DenseVector
from nose.tools import eq_

def test_make_empty():
    concepts, axes = 100, 5
    m = AssocSpace.make_empty(concepts, axes)
    eq_(m.n_rows, concepts)
    eq_(m.n_cols, concepts)
    eq_(m.k, axes)
    assert np.all(m.u == 0)
    assert np.all(m.sigma == 0)
    assert np.all(m.v == 0)

def test_expanded_to():
    k = 5
    u = DenseMatrix(np.zeros((2,k)))
    sigma = DenseVector(np.zeros(k))
    u[0, 0] = 1.0
    u[1, 1] = 1.0
    sigma[0] = 3
    sigma[1] = 1
    assoc = AssocSpace(u, sigma)
    eq_(assoc.u[0, 0], 1.0)
    eq_(assoc.u[0, 1], 0.0)
    eq_(assoc.v[0, 0], 1.0)

    expanded = assoc.expanded_to(4, 4)
    eq_(expanded.u.shape[0], 4)
    eq_(expanded.u.shape[1], k)
    eq_(expanded.k, k)
    eq_(expanded.v.shape[0], 4)
    eq_(expanded.v.shape[1], k)
    eq_(expanded.u[0, 0], 1.0)
    eq_(expanded.u[0, 1], 0.0)
    eq_(expanded.v[0, 0], 1.0)
    assert np.all(assoc.sigma == expanded.sigma)
