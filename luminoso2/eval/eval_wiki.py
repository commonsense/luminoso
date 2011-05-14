# -*- coding: utf-8 -*-
import numpy as np
import luminoso2
from luminoso2.eval.util import diagonal_significance, confusion_matrix

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
    plt.pcolor(np.flipud(correlation_means / np.sum(correlation_means, axis=1)[:,np.newaxis]), cmap='hot')
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

def is_model_good(model, filename=None):
    names = get_names(model)
    #for study in list(names) + ['Canonical']:
    #    model.update_doc_matrix(study)
    canonical_fullnames, names, stats, terms, correlation_means = analyze_model(model)
    correlation_means = np.array(correlation_means)
    if filename is not None:
        from matplotlib import pyplot as plt
        plt.clf()
        plot_alignment(correlation_means, terms, names)
        plt.savefig(filename)

    return diagonal_significance(correlation_means)

def is_model_confused(model, filename=None):
    names = get_names(model)
    for study in list(names) + ['Canonical']:
        model.update_doc_matrix(study)
    canonical_fullnames, names, stats, terms, correlation_means = analyze_model(model)
    confusion = confusion_matrix(model, names, canonical_fullnames)
    if filename is not None:
        from matplotlib import pyplot as plt
        plt.clf()
        plot_alignment(confusion, terms, names)
        plt.savefig(filename)
    #confusion /= np.sum(confusion, axis=1)
    precision = np.sum(np.diag(confusion)) / np.sum(confusion)
    return confusion, precision

def multiple_models(model, name='enwiki_post_normalize'):
    import divisi2
    background = divisi2.network.conceptnet_assoc('en')
    assoc_powers = (1.0,)
    blend_factors = (
        0.70, 0.30, 0.50, 0.99, 0.01, 0.95, 0.90, 0.85, 0.80, 0.75, 0.65,
        0.60, 0.55, 0.45, 0.40, 0.35, 0.25, 0.20, 0.15, 0.10, 0.05
    )
    out = open(name+'_confusion.log', 'a')
    for a in assoc_powers:
        for b in blend_factors:
            print "running", (a,b)
            model.recompute(background, partial=True, newconfig={
                'blend_factor': b,
                'assoc_power': a
            })
            filename="%s_%s_%s.png" % (name, a, b)
            conf, precision = is_model_confused(model, filename)
            separability, pvalue = is_model_good(model, 'sep_'+filename)
            print "%s,%s,%s,%s" % (a, b, precision, separability)
            print >> out, "%s,%s,%s,%s" % (a, b, precision, separability)
            out.flush()
    out.close()

