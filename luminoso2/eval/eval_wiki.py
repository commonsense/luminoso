# -*- coding: utf-8 -*-
import luminoso2
import numpy as np
from scipy.stats import ttest_1samp

def get_names(model):
    # cut in half for training purposes
    names = [value for tag, value in model.database.all_tags()
             if tag == u'study' and value != 'Canonical' and value != 'FA'
             and value != u'総記']
    return names[::2]

def load_model(model_path, regen_doc_matrices):
    model = luminoso2.load(model_path)
    names = get_names(model)
    if regen_doc_matrices:
        for study in list(names) + ['Canonical']:
            model.update_doc_matrix(study)
    return model

def analyze_model(model):
    names = get_names(model)
    canonical_basename = model.docs_in_study('Canonical')[0].rsplit('#', 1)[0]
    canonical_fullnames = [u'%s#%s' % (canonical_basename, canonical) for canonical in names]

    stats = [model.canonical_stats(study) for study in names]
    terms = [model.database.get_document(canonical).text.strip() for canonical in canonical_fullnames]

    correlation_means = [[stats[group_idx].correlation[canonical_fullnames[canonical_idx]]['mean']
                          for canonical_idx in range(len(canonical_fullnames))]
                         for group_idx in range(len(names))]
    return canonical_fullnames, names, stats, terms, correlation_means

def plot_alignment(correlation_means, terms, names):
    from matplotlib import pyplot as plt
    plt.yticks(np.arange(len(names))+0.5, names[::-1])
    plt.xticks(np.arange(len(terms))+0.5, terms, rotation='vertical')
    plt.pcolor(np.flipud(correlation_means / np.sqrt(np.mean(correlation_means*correlation_means, axis=0))[np.newaxis,:]), cmap='hot')
    plt.subplots_adjust(bottom=0.22, left=0.3)
    #plt.clim(-60, 10)
    cbar = plt.colorbar()   #ticks=[-60, -50, -40, -30, -20, -10, 0, 10])
    cbar.set_label('Correlation mean')

def do_model(path, regen, title, filename):
    model = load_model(path, regen)
    canonical_fullnames, names, stats, terms, correlation_means = analyze_model(model)
    correlation_means = np.array(correlation_means)
    plot_alignment(correlation_means, terms, names)
    from matplotlib import pyplot as plt
    plt.title(title)
    plt.savefig(filename)
    return correlation_means

def wiki_cs1(regen=True):
    return do_model('wiki_en_cs1_assoc0', regen, 'Topic alignment with common sense', 'wiki_en_cs1_assoc0.png')

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

def is_model_good(model, filename=None):
    names = get_names(model)
    for study in list(names) + ['Canonical']:
        model.update_doc_matrix(study)
    canonical_fullnames, names, stats, terms, correlation_means = analyze_model(model)
    correlation_means = np.array(correlation_means)
    if filename is not None:
        from matplotlib import pyplot as plt
        plot_alignment(correlation_means, terms, names)
        plt.savefig(filename)

    return diagonal_significance(correlation_means)

def multiple_models(model, name='enwiki'):
    import divisi2
    background = divisi2.network.conceptnet_assoc('en')
    assoc_powers = (0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001)
    blend_factors = (0.75, 0.5, 0.25)
    out = open(name+'_model.log', 'a')
    for a in assoc_powers:
        for b in blend_factors:
            print "running", (a,b)
            model.recompute(background, partial=True, newconfig={
                'blend_factor': b,
                'assoc_power': a
            })
            filename="%s_%s_%s.png" % (name, a, b)
            sig, pvalue = is_model_good(model, filename)
            print "%s,%s,%s,%s" % (a, b, sig, pvalue)
            print >> out, "%s,%s,%s,%s" % (a, b, sig, pvalue)
    out.close()

