from luminoso.whereami import *
from csc.conceptnet.analogyspace import conceptnet_2d_from_db
import cPickle as pickle

picklefile = open(package_dir+'/study_skel/Matrices/conceptnet.pickle', 'w')
cnet = conceptnet_2d_from_db('en')
pickle.dump(cnet, picklefile)
picklefile.close()

