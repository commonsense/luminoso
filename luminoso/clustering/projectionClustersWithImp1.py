from csc import divisi2
from os import sep
from luminoso import *
from clusterImp1 import KMeansClustering
import time

def create(filePath, resultPath):
    f = open('matrix.txt', 'w')
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

#create('C:\Users\LLPadmin\Desktop\luminoso\ThaiFoodStudy\Results\projections.dmat','C:\Users\LLPadmin\Desktop\luminoso\luminoso\k-means\\')

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
tup = createTuples('C:\Users\LLPadmin\Desktop\luminoso\ThaiFoodStudy\Results\projections.dmat')
#checkEquality(tup)
clusters = getClusters(tup, 20)
mid = time.time()
print "It took "+str(mid-start)+" secs to get clusters."
clusterFiles(clusters)
end = time.time()
print "It took "+str(end-mid)+" secs to file clusters."
print "It took "+str(end-start)+" total seconds."

## STATS:
##Creating Tuples...
##Creating clusters...
##It took 100.353999853 secs to get clusters.
##Creating Files...
##It took 0.0629999637604 secs to file clusters.
##It took 100.416999817 total seconds.
##
