import divisi2
import numpy as np
from luminoso2.svdupdate import general_svd_update

class AssocSpace(object):
    '''
    An AssocSpace projects terms into a vector space shaped by their
    associations. It can also update that vector space to account for
    new term association evidence.
    '''
    def __init__(self, u, sigma, v=None):
        '''
        Creates an AssocSpace from the `U` (and possibly `V`) matrices
        and the vector of singular values `sigma`.
        '''
        if v is None: v = u # symmetric hack
        self.u = u
        self.sigma = sigma
        self.v = v
        self.labels = u.row_labels
        self.assoc = self.spectrally_associate(self.u).normalize_rows(offset=0.001)

    def __getstate__(self):
        return dict(
            version=0,
            u=self.u,
            sigma=self.sigma,
            v=self.v)

    def __setstate__(self, state):
        assert state['version'] == 0
        self.__init__(state['u'], state['sigma'], state['v'])

    @property
    def k(self):
        return len(self.sigma)

    @classmethod
    def from_matrix(cls, mat, k):
        '''
        Creates an AssocSpace with dimensionality `k` from a matrix of
        new associations `mat`.
        '''
        u, s, v = mat.svd(k=k)
        return cls(u, s, v)

    def on_drop(self, index, key):
        """
        Handle when a key falls out of the PrioritySet.
        """
        self.u[index, :] = 0
        # TODO: re-orthonormalize.

    def spectrally_associate(self, vec):
        """
        Multiply vec by e^(sqrt(S)/2).
        """
        return vec.multiply(np.exp(np.sqrt(self.sigma)/2))

    def vector_from_terms(self, terms, input_only=False):
        """
        Get a category vector representing the given set of weighted terms,
        expressed as (term, weight) tuples.
        """
        total_weight = 0.000001
        vec = divisi2.DenseVector(
            np.zeros((len(self.labels),)),
            labels=self.labels
        )
        for _, weight in terms:
            total_weight += abs(weight)

        for term, weight in terms:
            try:
                index = self.labels.index(term)
            except KeyError:
                continue
            # The old function had this line but never used it.
            #assoc_weight = np.linalg.norm(self.u.row_named(term))
            vec[index] = weight
        if input_only:
            return vec
        return divisi2.dot(vec, self.assoc).normalize()

    def assoc_between_two_terms(self, term1, term2):
        return divisi2.dot(self.assoc.row_named(term1),
                           self.assoc.row_named(term2))

    def terms_similar_to_vector(self, vec):
        """
        Take in a category vector, and returns a weighted vector of
        associated terms. You can run the `top_items()` method of this vector
        to get the most associated terms.
        """
        return divisi2.dot(self.assoc, vec)

    def merged_with(self, other):
        '''
        Construct a new AssocSpace formed by merging this model with another.

        Does not modify the current model.
        '''
        # Strip to bare NumPy
        U = np.asarray(self.u)
        V = np.asarray(self.v)
        A = np.multiply(np.asarray(other.u), np.asarray(other.sigma))
        B = np.asarray(other.v)

        # Apply SVD update.
        new_u, new_sigma, new_v = general_svd_update(U, self.sigma, V, A, B, U.shape[1])

        return self.__class__(divisi2.DenseMatrix(new_u, self.labels, None),
                              new_sigma,
                              divisi2.DenseMatrix(new_v, self.labels, None))
