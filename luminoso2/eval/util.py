import numpy as np
from scipy.stats import ttest_1samp

## Rough statistical significance tests
def diagonal_vs_non(mat):
    diags, nondiags = [], []
    for row in range(mat.shape[0]):
        for col in range(mat.shape[1]):
            (diags if row==col else nondiags).append(mat[row,col])
    return diags, nondiags

def diagonal_significance(mat):
    diags, nondiags = diagonal_vs_non(mat)
    print 'diags', np.mean(diags), 'nondiags', np.mean(nondiags)
    return ttest_1samp(diags, np.mean(nondiags))

def confusion_matrix(model, studies, canonical_names):
    '''
    Computes the "confusion matrix" of classifying each document in each study according to which canonical it is closest to.
    '''
    # Look up all the canonical vectors.
    canonical_mat = model.get_doc_matrix('Canonical')
    if '/' not in canonical_names[0]:
        canonical_basename = canonical_mat.row_labels[0].rsplit('/', 1)[0]
        canonical_names = ['%s/%s.txt' % (canonical_basename, cname) for cname in canonical_names]
    canonical_vecs = np.array([canonical_mat.row_named(cname).hat() for cname in canonical_names])

    confusion = []
    for study in studies:
        study_mat = model.get_doc_matrix(study)
        matches = np.zeros(len(canonical_names))
        # Classify each document in the study by its proximity to all the canonicals.
        for idx, vec in enumerate(study_mat):
            best_match = np.argmax(np.dot(canonical_vecs, vec.hat()))
            matches[best_match] += 1
        confusion.append(matches)
    return np.array(confusion)

def precision_vs_recall(model, studies, canonical_names, max_match=5):
    # Look up all the canonical vectors.
    canonical_mat = model.get_doc_matrix('Canonical')
    if '/' not in canonical_names[0]:
        canonical_basename = canonical_mat.row_labels[0].rsplit('/', 1)[0]
        canonical_names = ['%s/%s.txt' % (canonical_basename, cname) for cname in canonical_names]
    canonical_vecs = np.array([canonical_mat.row_named(cname).hat() for cname in canonical_names])
    
    matches = np.zeros(max_match)
    retrieved = np.zeros(max_match)
    relevant = np.zeros(max_match)
    for nmatch in xrange(1, max_match+1):
        for study_idx, study in enumerate(studies):
            study_mat = model.get_doc_matrix(study)
            
            # Classify each document in the study by its proximity to all the canonicals.
            for idx, vec in enumerate(study_mat):
                sort_matches = np.argsort(np.dot(canonical_vecs, vec.hat()))
                best_matches = sort_matches[-nmatch:]
                if study_idx in set(best_matches):
                    matches[nmatch-1] += 1
                retrieved[nmatch-1] += nmatch
                relevant[nmatch-1] += 1
    print matches
    print retrieved
    print relevant
    precision = matches / retrieved
    recall = matches / relevant
    return precision, recall
