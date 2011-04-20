"""
Classify terms and documents (in general, vectors) as being associated with a canonical document or not.

Since terms or documents can be associated with several or no canonical documents, a one-class SVM is used.
"""
import numpy as np
from scikits.learn import svm

def get_classifier(vectors, **kw):
    kw.setdefault('kernel', 'linear')
    classifier = svm.OneClassSVM(**kw)
    classifier.fit(vectors)
    return classifier

def vectors_for_terms(model, terms):
    return [model.vector_from_terms([(term, 1.0)]) for term in terms]

#def is_vec_in_classifier(classifier, vec):#

def distances_to_classifier(classifier, vectors):
    return classifier.decision_function(vectors)

def closest_terms(classifier, model, terms, n):
    vecs = vectors_for_terms(model, terms)
    distances = distances_to_classifier(classifier, vecs)
    closest_idxes = np.argsort(np.abs(distances.ravel()))
    return [(terms[i], distances[i]) for i in closest_idxes[:n]]
