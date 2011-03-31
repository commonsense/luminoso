from csc import divisi2
import random

def randomClustersMatrix(terms, k):
    n = len(terms)
    clusters = []
    term_names = []
    cluster_names = []
    for i in range(n):
        # get random possition for possible cluster
        r = random.randint(0, k-1)
        # add the name to term_names to construct the matrix
        term_names.extend([terms[i]])
        # add cluster name to the list
        cluster_names.extend([str(i)])
        # create cluster for term and place 1 in assigned cluster
        cluster = [0]*k
        cluster[r] = 1
        # add cluster to cluster list 'clusters'
        clusters.extend([cluster])
    #create and return sparce matrix
    return divisi2.SparseMatrix(clusters, term_names, cluster_names)
        

def createSpectralMatrix():
    # proj is a ReconstructedMatrix of the form terms:terms.
    proj = divisi2.load('C:\Users\LLPadmin\Desktop\luminoso\ThaiFoodStudy\Results\spectral.rmat')

    # create sparse matrix for clusters of the form terms:clusters (row, col).
    clusterMatrix = randomClustersMatrix(proj.col_labels, 10)


    return divisi2.multiply(clusterMatrix, proj)

x = createSpectralMatrix()
