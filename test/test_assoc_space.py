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

