from luminoso2.assoc_space import AssocSpace
import numpy as np
from divisi2 import DenseMatrix, DenseVector
from nose.tools import eq_

def test_dimensionality():
    k = 5
    u = DenseMatrix(np.zeros((100, k)))
    sigma = DenseVector(np.zeros(k))
    m = AssocSpace(u, sigma)
    eq_(m.k, k)
