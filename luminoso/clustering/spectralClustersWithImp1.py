from csc import divisi2
from os import sep
from luminoso import *
from clusterImp1 import KMeansClustering
import time

def create(filePath, resultPath):
    f = open('spectralMatrix.txt', 'w')
    proj = divisi2.load(filePath)
    labels = proj.row_labels

    for i in labels:
        f.write(i+' ')
        count = 0
        p = proj.row_named(i)
        while count < len(p):
            if count == len(p) - 1:
                f.write(str(p[count]) + '\n')
            else:
                f.write(str(p[count])+' ')
            count+=1
    f.close()

##create('C:\Users\LLPadmin\Desktop\luminoso\ThaiFoodStudy\Results\spectral.rmat','C:\Users\LLPadmin\Desktop\luminoso\luminoso\k-means\\')

def createTuples(filePath):
    print "Creating Tuples..."
    proj = divisi2.load(filePath)
    labels = proj.row_labels
    d = dict()
    for i in range(len(labels)):
        d[tuple(proj[i])] = labels[i]
    return d

def checkEquality(dict):
    for i in range(len(dict)):
        temp = dict[dict.keys()[i]]
        for j in range(i+1, len(dict)):
            if temp == dict[dict.keys()[j]]:
                return True
    return False

def getClusters(dict, k):
    print "Creating clusters..."
    c = KMeansClustering(dict.keys())
    clusters = c.getclusters(k)

    for cluster in clusters:
        for word in range(len(cluster)):
            cluster[word] = dict[cluster[word]]

    return clusters

def clusterFiles(clusters):
    print "Creating Files..."
    for i in range(len(clusters)):
        f = open('cluster_'+str(i)+'.txt', 'w')
        for j in clusters[i]:
            f.write(j+'\n')
        f.close()

start = time.time()
tup = createTuples('C:\Users\LLPadmin\Desktop\luminoso\ThaiFoodStudy\Results\spectral.rmat')
#checkEquality(tup)
clusters = getClusters(tup, 30)
mid = time.time()
print "It took "+str(mid-start)+" seconds to get clusters."
clusterFiles(clusters)
end = time.time()
print "It took "+str(end-mid)+" seconds to file clusters."
print "It took "+str(end-start)+" seconds time."

## STATS:

## K = 20 (15 minutes 47 seconds)
##Creating Tuples...
##Creating clusters...
##It took 947.165999889 seconds to get clusters.
##Creating Files...
##It took 0.0150001049042 seconds to file clusters.
##It took 947.180999994 seconds time.
##
## K = 30 (34 minutes 11 seconds)
##Creating Tuples...
##Creating clusters...
##It took 2051.28199983 seconds to get clusters.
##Creating Files...
##It took 0.0989999771118 seconds to file clusters.
##It took 2051.3809998 seconds time.
##
