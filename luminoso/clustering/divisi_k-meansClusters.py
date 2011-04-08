from csc import divisi2
import random

def create_file(counter, stored_values):
        '''
        Takes the path to the folder to write to and writes a file containing the tag's contents.
        '''
        f = open('C:\Users\LLPadmin\Desktop\luminoso\luminoso\clustering\\' + str(counter) + '.txt', 'w')
        for value in stored_values:
            f.write(str(value) + ' ')
        f.close()

def randomClustersMatrix(terms, k):
    n = len(terms)
    clusters = []
    term_names = []
    cluster_names = []
    termsDict = dict()
    # Create (n = # of terms) clusters.
    for i in range(n):
        # get random possition for possible cluster
        r = random.randint(0, k-1)
        # add the name to term_names to construct the matrix and add term to termsDict
        term_names.extend([terms[i]])
        termsDict[terms[i]] = [None, None]
        # create cluster for term and place 1 in assigned cluster
        cluster = [0]*k
        cluster[r] = 1
        # add cluster to cluster list 'clusters'
        clusters.extend([cluster])
    # Name the k clusters
    for i in range(k):
        # add cluster name to the list
        cluster_names.extend(['cluster'+str(i)])
    #create and return sparce matrix
    #return divisi2.SparseMatrix(clusters, term_names, cluster_names)
    return divisi2.DenseMatrix(clusters, term_names, cluster_names), cluster_names, term_names, termsDict

def normalize(matrix, dict, cluster_names, term_names):
    # List to store a boolean that says if the term was placed on the same cluster twice in a row.
    repeated_cluster = [None]*len(term_names)
    for i in range(len(matrix)):
        # Find max, normalize by replacing max_cluster with a 1 and others with 0.
        maximum = matrix[i].argmax()
        new = [0]*10
        new[maximum] = 1
        # Enter new cluster term is in in possition 0, and store the previous one on pos 1 of a dictionary.
        dict[term_names[i]][1] = dict[term_names[i]][0]
        dict[term_names[i]][0] = maximum
        # Populate repeated_cluster
        repeated_cluster[i] = dict[term_names[i]][1]==dict[term_names[i]][0]
        # Create a new DenseVector to replace the old one.
        matrix[i] = divisi2.dense.DenseVector(new, cluster_names)
    # Copy results to a .txt file.
    #create_file(1, dict.values())
    return repeated_cluster.count(True)/len(repeated_cluster) > .95
        
def createSpectralMatrix():
    # proj is a ReconstructedMatrix of the form terms:terms.
    proj = divisi2.load('C:\Users\LLPadmin\Desktop\luminoso\ThaiFoodStudy\Results\spectral.rmat')

    # create sparse matrix for clusters of the form terms:clusters (row, col).
    clusterMatrix, cluster_names, term_names, termsDict = randomClustersMatrix(proj.col_labels, 10)
    count = 0

    while True:
        count += 1
        clusterMatrix = divisi2.aligned_matrix_multiply(proj.left, divisi2.aligned_matrix_multiply(proj.right,clusterMatrix))
        repeat = normalize(clusterMatrix, termsDict, cluster_names, term_names)
        if repeat:
            print count
            break
    
    return clusterMatrix

x = createSpectralMatrix()
print x

#divisi2.DenseMatrix([[12,1],[2,23]], ['12','1223'], ['1','2'])
