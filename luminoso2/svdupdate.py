import numpy as np
import numpy.linalg as LA
import logging
logger = logging.getLogger('luminoso2.svdupdate')

def general_svd_update(U, S, V, A, B, rank, eps=1e-10):
    '''
    Given the SVD X=USV^T, finds an updated decomposition X+AB^T = Up
    Sp Vp^T. The result is trimmed to the specified rank.

    S, a diagonal matrix, is represented by its diagonal elements.

    Returns: Up, Sp, Vp.

    The approach follows Brand 2006 (MERL TR2006-059).

    Note that all the matrices involved should be plain NumPy arrays, not subclasses.
    '''

    # Let P be an orthogonal basis of the column-space of
    # (I-UU^T)A--the component of A that is orthogonal to U, and set
    # Ra = P^T(I-UU^T)A
    M = np.dot(U.T, A) # roughly: component of A in direction of U
    Pp = A - np.dot(U, M) # component of A orthogonal to U
    P, Ra = LA.qr(Pp) # Finds Pp=P*Ra, where P is orthonormal and Ra upper triangular

    # # Truncate Ra to only the non-negligible values.
    # mag = np.sum(Ra, axis=1)
    # cutoff = np.min(np.nonzero(mag < S[0]*eps)[0])
    # assert np.all(mag[cutoff:] < S[0]*eps)
    # Ra = P[:,:cutoff]

    # Similarly, let QRb=(I-VV^T)B.
    N = np.dot(V.T, B)
    Qq = B - np.dot(V, N)
    Q, Rb = LA.qr(Qq)

    # Here is the main equation (4)
    K = np.dot(np.r_[M, Ra], np.r_[N, Rb].T)
    for i in xrange(len(S)):
        K[i,i] += S[i]

    # Now diagonalize K.
    Uu, Ss, Vv = LA.svd(K, full_matrices=False)

    # Construct the new SVD.
    Up = np.dot(np.c_[U, P], Uu)
    Sp = Ss
    Vp = np.dot(np.c_[V, Q], Vv)

    return trim_svd(Up, Sp, Vp, rank)

def trim_svd(U, s, V, rank):
    return U[:,:rank], s[:rank], V[:,:rank]
