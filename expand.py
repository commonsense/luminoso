from csc import divisi2
from luminoso.study import StudyDirectory
import sys

def expand_study(study_name):
    study = StudyDirectory(study_name).get_study()
    theblend, concepts = study.get_assoc_blend()
    U, S, V = theblend.normalize_all().svd(k=50)
    doc_rows = divisi2.aligned_matrix_multiply(study.get_documents_matrix(), U)
    projections = U.extend(doc_rows)
    spectral = divisi2.reconstruct_activation(projections, S, post_normalize=True)
    divisi2.save(spectral, study_name+'/Results/expanded.rmat')

if __name__ == '__main__':
    if len(sys.argv) > 1:
        expand_study(sys.argv[1])
    else:
        print "Please give the name of a study directory on the command line."
