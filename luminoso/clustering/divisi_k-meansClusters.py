from csc import divisi2
import random
import os

def create_file(name, stored_values):
        '''
        Takes the path to the folder to write to and writes a file containing the tag's contents.
        '''
        f = open(os.path.abspath('.') + os.sep + name + '.txt', 'w')
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
        #r = random.randint(0, k-1)
        r = i%k
        # add the name to term_names to construct the matrix and add term to termsDict
        term_names.extend([terms[i]])
        termsDict[terms[i]] = [None, None]
        # create cluster for term and place 1 in assigned cluster
        cluster = [0.0]*k
        cluster[r] = 1.0
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
    k = len(cluster_names)
    for i in range(len(matrix)):
        # Find max, normalize by replacing max_cluster with a 1 and others with 0.
        maximum = matrix[i].argmax()
        new = [0.0]*k
        new[maximum] = 1.0
        # Enter new cluster term is in in possition 0, and store the previous one on pos 1 of a dictionary.
        dict[term_names[i]][1] = dict[term_names[i]][0]
        dict[term_names[i]][0] = maximum
        # Populate repeated_cluster
        repeated_cluster[i] = dict[term_names[i]][1]==dict[term_names[i]][0]
        # Create a new DenseVector to replace the old one.
        matrix[i] = divisi2.dense.DenseVector(new, cluster_names)
    # If the clusters have not change in 2 iterations in a row, stop.
    if repeated_cluster.count(True)/len(repeated_cluster) > .95:
        return True
    # Create Normalize Vector for clusters i.e., sum(values in a row)/total elements)
    sums = []
    terms = float(len(term_names))
    for j in cluster_names:
        sums.append(sum(matrix.col_named(j))/terms)
    v = divisi2.DenseVector(sums, cluster_names)
    # Normalize by multiplying each row with vector v.
    for i in range(len(matrix)):
        matrix[i] = matrix[i]*v
    # Copy results to a .txt file.
    create_file('cluster_assignment', dict.values())
        
def createSpectralMatrix(k):
    # proj is a ReconstructedMatrix of the form terms:terms.
    proj = divisi2.load(os.path.abspath('..\\..\\ThaiFoodStudy')+'\\Results\\spectral.rmat')

    # create sparse matrix for clusters of the form terms:clusters (row, col).
    clusterMatrix, cluster_names, term_names, termsDict = randomClustersMatrix(proj.col_labels, k)
    count = 0

    while True:
        count += 1
        clusterMatrix = divisi2.aligned_matrix_multiply(proj.left, divisi2.aligned_matrix_multiply(proj.right,clusterMatrix))
        repeat = normalize(clusterMatrix, termsDict, cluster_names, term_names)
        if repeat:
            print "Aftert "+str(count)+" iterations, we got acceptable clusters."
            break
    
    return clusterMatrix

x = createSpectralMatrix(20)
print x
