import luminoso2
import numpy as np

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
N = len(newsgroups)

def load_model(model_path, regen_doc_matrices):
    model = luminoso2.load(model_path)
    if regen_doc_matrices:
        for study in study_names + ['Canonical']:
            model.update_doc_matrix(study)

    canonical_basename = model.docs_in_study('Canonical')[0].rsplit('/', 1)[0]
    canonical_fullnames = [u'%s/%s.txt' % (canonical_basename, canonical) for canonical in canonical_names]

    stats = [model.canonical_stats(study) for study in study_names]
    terms = [model.database.get_document(canonical).text.strip() for canonical in canonical_fullnames]

    centralities = [[stats[newsgroup_idx].centrality[canonical_fullnames[canonical_idx]]
                     for canonical_idx in range(N)]
                    for newsgroup_idx in range(N)]
    return model, canonical_basename, stats, terms, centralities

def plot_alignment(centralities, terms):
    from matplotlib import pyplot as plt
    plt.yticks(np.arange(N)+0.5, group_names)
    plt.xticks(np.arange(N)+0.5, terms, rotation='vertical')
    plt.pcolor(centralities, cmap='hot')
    plt.subplots_adjust(bottom=0.22, left=0.3)
    #plt.clim(-60, 10)
    cbar = plt.colorbar()#ticks=[-60, -50, -40, -30, -20, -10, 0, 10])
    cbar.set_label('Centrality')

def do_model(path, regen, title, filename):
    model, canonical_basename, stats, terms, centralities = load_model(path, regen)
    centralities = np.array(centralities)
    plot_alignment(centralities, terms)
    from matplotlib import pyplot as plt
    plt.title(title)
    plt.savefig(filename)
    non_diag_elements = np.ravel(centralities - np.diag(np.diag(centralities)))
    return centralities, non_diag_elements
    
def cs1_assoc0(regen):
    return do_model('20news_cs1_assoc0', regen, 'Topic alignment with common sense', 'cs1_assoc0.png')

def cs0_assoc0(regen):
    return do_model('20news_cs0_assoc0', regen, 'Topic alignment without common sense', 'cs0_assoc0.png')

#def compare_commonsense_with_not():
    
    
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
title()
subplots_adjust(bottom=0.22, left=0.3)
clim(-60, 10)
cbar = colorbar(ticks=[-60, -50, -40, -30, -20, -10, 0, 10])
cbar.set_label('Centrality')

print np.mean(sense_diffs), np.mean(no_sense_diffs)
print ttest_rel(no_sense_diffs, sense_diffs)
show()
"""
