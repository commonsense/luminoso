import sys
import luminoso2

model_path, = sys.argv[1:]

model = luminoso2.load(model_path)
canonical_basename = model.docs_in_study('Canonical')[0].rsplit('/', 1)[0]

newsgroups = [
    ('sci.med', 'med', 'sci_med'),
    ('sci.space', 'space', 'sci_space'),
    ('sci.crypt', 'crypt', 'sci_crypt'),
    ('sci.electronics', 'electronics', 'sci_electronics'),
    ('comp.graphics', 'graphics', 'comp_graphics'),
    ('comp.os.ms-windows.misc', 'windows', 'os_ms_windows'),
    ('comp.sys.ibm.pc.hardware', 'pc', 'pc_hardware'),
    ('comp.sys.mac.hardware', 'machardware', 'mac_hardware'),
    ('comp.windows.x', 'xwindows', 'windows_x'),
    ('misc.forsale', 'forsale', 'misc_forsale'),
    ('rec.autos', 'autos', 'rec_autos'),
    ('rec.motorcycles', 'motorcycles', 'rec_motorcycle'),
    ('rec.sport.baseball', 'baseball', 'sport_baseball'),
    ('rec.sport.hockey', 'hockey', 'sport_hockey'),
    ('talk.politics.guns', 'guns', 'poltics_guns'),
    ('talk.politics.mideast', 'middleeast', 'politics_mideast'),
    ('talk.politics.misc', 'politics_misc', 'politics_misc'),
    ('alt.atheism', 'atheism', 'alt_atheism'),
    ('soc.religion.christian', 'christian', 'religion_christian'),
    ('talk.religion.misc', 'religion_misc', 'religion_misc')
]
group_names, study_names, canonical_names = zip(*newsgroups)
canonical_fullnames = [u'%s/%s.txt' % (canonical_basename, canonical) for canonical in canonical_names]
N = len(newsgroups)

stats = [model.canonical_stats(study) for study in study_names]
terms = [model.database.get_document(canonical).text.strip() for canonical in canonical_fullnames]

centralities = [[stats[newsgroup_idx].centrality[canonical_fullnames[canonical_idx]]
                 for canonical_idx in range(N)]
                for newsgroup_idx in range(N)]

def plot_centralities(centralities):
    from matplotlib import pyplot as plt
    import numpy as np
    plt.yticks(np.arange(N)+0.5, group_names)
    plt.xticks(np.arange(N)+0.5, terms, rotation='vertical')
    plt.pcolor(np.array(centralities), cmap='hot')
    plt.title('Topic alignment with common sense')
    plt.subplots_adjust(bottom=0.22, left=0.3)
    plt.clim(-60, 10)
    cbar = plt.colorbar(ticks=[-60, -50, -40, -30, -20, -10, 0, 10])
    cbar.set_label('Centrality')
    
"""
from pylab import *
from scipy.stats import ttest_rel
import pickle

figure(1)
ary = pickle.load(open('centrality.pickle'))
row_order = row_order[::-1]
sense_diffs = []
arranged = np.zeros((20, 20))
for row in xrange(20):
    for col in xrange(20):
        arranged[row,col] = ary[row_order[row][1], col_order[col][1]]
        if row != col: sense_diffs.append(arranged[row,row]-arranged[row,col])

figure(2)
ary = pickle.load(open('nosense.centrality.pickle'))

arranged = np.zeros((20, 20))
no_sense_diffs = []
for row in xrange(20):
    for col in xrange(20):
        arranged[row,col] = ary[row_order[row][1], col_order[col][1]]
        if row != col: no_sense_diffs.append(arranged[row,row]-arranged[row,col])
yticks(arange(20)+0.5, [ro[0] for ro in row_order])
xticks(arange(20)+0.5, [co[0] for co in col_order], rotation='vertical')
pcolor(arranged, cmap='hot')
title('Topic alignment without common sense')
subplots_adjust(bottom=0.22, left=0.3)
clim(-60, 10)
cbar = colorbar(ticks=[-60, -50, -40, -30, -20, -10, 0, 10])
cbar.set_label('Centrality')

print np.mean(sense_diffs), np.mean(no_sense_diffs)
print ttest_rel(no_sense_diffs, sense_diffs)
show()
"""
