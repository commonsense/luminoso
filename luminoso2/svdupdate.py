# This code is taken from
# https://github.com/piskvorky/gensim/blob/develop/src/gensim/models/lsimodel.py#L499
# License: LGPL

import numpy
import logging
logger = logging.getLogger('luminoso2.svdupdate')

def svdUpdate(U, S, V, a, b):
    """
    Update SVD of an (m x n) matrix `X = U * diag(S) * V^T` so that
    `[X + a * b^T] = U' * S' * V'^T`
    and return `U'`, `diag(S')`, `V'`.
    
    The original matrix X is not needed at all, so this function implements one-pass
    streaming rank-1 updates to an existing decomposition.
    
    `a` and `b` are (m, 1) and (n, 1) matrices.
    
    You can set V to None if you're not interested in the right singular
    vectors. In that case, the returned V' will also be None (saves memory).
    
    This is the rank-1 update as described in
    **Brand, 2006: Fast low-rank modifications of the thin singular value decomposition**,
    but without separating the basis from rotations.
    """
    # convert input to matrices (no copies of data made if already numpy.ndarray or numpy.matrix)
    S = numpy.asarray(S).ravel()
    U = numpy.asmatrix(U)
    if V is not None:
        V = numpy.asmatrix(V)
    a = numpy.asmatrix(a).reshape(a.size, 1)
    b = numpy.asmatrix(b).reshape(b.size, 1)

    rank = len(S)

    # eq (6)
    m = U.T * a
    p = a - U * m
    Ra = numpy.sqrt(p.T * p)
    if float(Ra) < 1e-10:
        logger.debug("input already contained in a subspace of U; skipping update")
        return U, S, V
    P = (1.0 / float(Ra)) * p

    if V is not None:
        # eq (7)
        n = V.T * b
        q = b - V * n
        Rb = numpy.sqrt(q.T * q)
        if float(Rb) < 1e-10:
            logger.debug("input already contained in a subspace of V; skipping update")
            return U, S, V
        Q = (1.0 / float(Rb)) * q
    else:
        n = numpy.matrix(numpy.zeros((rank, 1)))
        Rb = numpy.matrix([[1.0]])

    if float(Ra) > 1.0 or float(Rb) > 1.0:
        logger.debug("insufficient target rank (Ra=%.3f, Rb=%.3f); this update will result in major loss of information"
                      % (float(Ra), float(Rb)))

    # eq (8)
    S = numpy.r_[S, 0.0]
    K = numpy.matrix(numpy.diag(S)) + numpy.r_[m, Ra] * numpy.r_[n, Rb].T

    # eq (5)
    u, s, vt = numpy.linalg.svd(K, full_matrices = False)
    tUp = numpy.matrix(u[:, :rank])
    tVp = numpy.matrix(vt.T[:, :rank])
    tSp = s[: rank]
    Up = numpy.c_[U, P] * tUp
    if V is not None:
        Vp = numpy.c_[V, Q] * tVp
    else:
        Vp = None
    Sp = tSp

    return Up, Sp, Vp
