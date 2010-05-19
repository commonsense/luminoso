"""
This class provides the model to SVDView's view, calculating a blend of all
the components of a study that it finds in the filesystem.
"""
from PyQt4 import QtCore
import os, codecs, time
import cPickle as pickle
import numpy as np
import traceback
import logging
import zipfile
import chardet
logger = logging.getLogger('luminoso')

from standalone_nlp.lang_en import nltools as en_nl
from csc.util.persist import get_picklecached_thing
from csc.divisi.labeled_view import make_sparse_labeled_tensor, LabeledView
from csc.divisi.ordered_set import OrderedSet
from csc.divisi.tensor import DenseTensor, data
from csc.divisi.blend import Blend

from luminoso.whereami import package_dir
from luminoso.report import render_info_page, default_info_page

import shutil

try:
    import json
except ImportError:
    import simplejson as json

class Document(object):
    '''
    A Document is an entity in a Study.
    '''
    def __init__(self, name, text):
        self.name = name
        self.text = text

    @classmethod
    def from_file(cls, filename, name):
        # Open in text mode.
        rawtext = open(filename, 'r')
        encoding = chardet.detect(rawtext.read())['encoding']
        rawtext.close()
        text = codecs.open(filename, encoding=encoding, errors='replace').read()
        return cls(name, text)

    def extract_concepts_with_negation(self):
        return extract_concepts_with_negation(self.text)

class CanonicalDocument(Document):
    pass


NEGATION = ['no', 'not', 'never', 'stop', 'lack', "n't"]
PUNCTUATION = ['.', ',', '!', '?', '...', '-']
def extract_concepts_with_negation(text):
    words = en_nl.normalize(en_nl.tokenize(text)).split()
    
    # FIXME: this may join together words from different contexts...
    positive_words = []
    negative_words = []
    positive = True
    for word in words:
        if word in NEGATION:
            positive = False
        else:
            if positive:
                positive_words.append(word)
            else:
                negative_words.append(word)
            if word in PUNCTUATION:
                positive = True
    positive_concepts = [(c, 1) for c in en_nl.extract_concepts(' '.join(positive_words))]
    negative_concepts = [(c, -1) for c in en_nl.extract_concepts(' '.join(negative_words))]
    return positive_concepts + negative_concepts

def load_json_from_file(file):
    with open(file) as f:
        return json.load(f)

def write_json_to_file(data, file):
    with open(file, 'w') as f:
        json.dump(data, f)

class Study(QtCore.QObject):
    '''
    A Study is a collection of documents and other matrices that can be analyzed.
    '''
    def __init__(self, name, documents, other_matrices):
        QtCore.QObject.__init__(self)
        self.name = name
        self.documents = documents
        self.other_matrices = other_matrices
        self.num_axes = 20
        
    def _step(self, msg):
        logger.info(msg)
        self.emit(QtCore.SIGNAL('step(QString)'), msg)

    @property
    def num_documents(self):
        return len(self.documents)
    
    def get_documents_matrix(self):
        if not self.documents: return None
        documents_matrix = make_sparse_labeled_tensor(ndim=2)
        for doc in self.documents:
            for concept, value in doc.extract_concepts_with_negation():
                if not en_nl.is_blacklisted(concept):
                    documents_matrix[concept, doc.name] += value
        return documents_matrix.normalized(mode=[0,1]).bake()

    def get_blend(self):
        doc_matrix = self.get_documents_matrix()
        other_matrices = self.other_matrices

        if doc_matrix is not None:
            blend = Blend([doc_matrix] + other_matrices)
            study_concepts = set(doc_matrix.label_list(0)) | set(doc_matrix.label_list(1))
        else:
            if len(other_matrices) == 1:
                blend = other_matrices[0]
            else:
                blend = Blend(other_matrices)
            study_concepts = set(blend.label_list(0))

        return blend, study_concepts

    def get_projections(self, svd, study_concepts):
        # Put concepts and features together.
        # FIXME: maybe make this a real blend?
        concatenated = svd.get_weighted_u().concatenate(svd.v)
        # Extract just the study concepts. (FIXME: really this complicated?)
        new_concepts = OrderedSet(study_concepts)
        new_data = np.zeros((len(new_concepts), svd.u.shape[1]))
        new_matrix = LabeledView(DenseTensor(new_data), [new_concepts, None])
        for index in xrange(len(new_concepts)):
            new_data[index, :] = data(concatenated[new_concepts[index], :])
        
        # Find the principal components of that matrix.
        # maybe FIXME: don't shove them into the matrix?
        newsvd = new_matrix.svd(k=2)
        axis_labels = OrderedSet(['DefaultXAxis', 'DefaultYAxis'])
        extra_axis_data = data(newsvd.v.T)[0:2, :] / 1000
        extra_axis_matrix = LabeledView(DenseTensor(extra_axis_data), [axis_labels, None])
        return new_matrix.concatenate(extra_axis_matrix)

    def compute_stats(self, projections, svd, study_concepts):
        """
        FIXME: On large datasets, calculating every pairwise similarity might
        be too expensive. Cut down the size of the working matrix somehow?
        """
        projdata = data(projections)
        svals = data(svd.svals)
        
        # build an array of documents vs. axes
        docdata = [data(self.projections[doc.name,:])
                   for doc in self.documents]
        if not docdata: return {}
        docdata = np.array(docdata)
        simdata = np.dot(projdata * svals, docdata.T)

        mean = np.mean(simdata)
        stdev = np.std(simdata)
        n = simdata.shape[0] * simdata.shape[1]
        stderr = stdev/np.sqrt(n)

        congruence = {}
        for index, concept in enumerate(study_concepts):
            if not en_nl.is_blacklisted(concept):
                vec = simdata[index]
                cmean = np.mean(vec)
                cstdev = np.std(vec)
                cstderr = cstdev / np.sqrt(len(vec))
                z = (cmean - mean) / cstderr
                congruence[concept] = z
        consistency = mean/stderr
        return {
            'mean': mean,
            'stdev': stdev,
            'n': n,
            'consistency': consistency,
            'congruence': congruence,
            'timestamp': list(time.localtime())
        }

    def analyze(self):
        # TODO: make it possible to blend multiple directories

        self._step('Blending...')
        blend, study_concepts = self.get_blend()
        svd = blend.svd(k=self.num_axes)
         
        self._step('Finding interesting axes...')
        projections = self.get_projections(svd, study_concepts)

        self._step('Calculating stats...')
        stats = self.compute_stats(projections, svd, study_concepts)
        
        return StudyResults(blend, svd, projections, stats)


class StudyResults(QtCore.QObject):
    def __init__(self, blend, svd, projections, stats):
        QtCore.QObject.__init__(self)
        self.blend = blend
        self.svd = svd
        self.projections = projections
        self.stats = stats


    def write_coords_as_csv(self, filename):
        import csv
        x_axis = self.projections['DefaultXAxis',:].hat()
        y_axis = self.projections['DefaultYAxis',:].hat()
        output = open(filename, 'w')
        writer = csv.writer(output)
        writer.writerow(['Concept', 'X projection', 'Y projection', 'Coordinates'])
        for concept in self.study_concepts:
            xproj = self.projections[concept,:] * x_axis
            yproj = self.projections[concept,:] * y_axis
            coords = self.projections[concept,:].values()
            row = [concept.encode('utf-8'), xproj, yproj] + coords
            writer.writerow(row)
        output.close()

    def write_report(self, filename):
        if self.stats is None: return
        self.info = render_info_page(self)
        with open(filename, 'w') as out:
            out.write(self.info)

    def get_consistency(self):
        return self.stats['consistency']

    def get_congruence(self, concept):
        return self.stats['congruence'][concept]

    def get_info(self):
        if self.info is not None: return self.info
        else: return default_info_page(self)

    def save(self, dir):
        def tgt(name): return os.path.join(dir, name)
        def save_pickle(name, obj):
            with open(tgt(name), 'wb') as out:
                pickle.dump(obj, out, -1)

        self._step('Saving blend...')
        save_pickle("blend.pickle", self.blend)

        self._step('Saving SVD...')
        save_pickle("svd.pickle", self.svd)
        
        self._step('Saving projections...')
        save_pickle('projections.pickle', self.projections)

        self._step('Writing reports...')
        # Save stats
        write_json_to_file(self.stats, tgt("stats.json"))
        self.write_coords_as_csv(tgt("coords.csv"))
        self.write_report(tgt("report.html"))

    @classmethod
    def load(cls, dir):
        def tgt(name): return os.path.join(dir, name)
        def load_pickle(name):
            with open(tgt(name), 'rb') as f:
                return pickle.load(f)

        # Either this will all fail or all succeed.
        #self._step("Loading blend")
        blend = load_pickle("blend.pickle")

        #self._step("Loading SVD")
        svd = load_pickle("svd.pickle")

        #self._step("Loading projections")
        projections = load_pickle("projections.pickle")
        # HACK: infer what the study concepts were
        study_concepts = set(projections.label_list(0))
        study_concepts.remove('DefaultXAxis')
        study_concepts.remove('DefaultYAxis')

        # Load stats
        stats = load_json_from_file(tgt("stats.json"))

        return cls(blend, svd, projections, stats)

class StudyDirectory(QtCore.QObject):
    '''
    A StudyDirectory manages the directory representing a study. It has three responsibilites:
     - loading the documents, both study and canonical
     - storing settings, such as the number of axes to compute
     - caching analysis results for speed
    '''
    def __init__(self, dir):
        self.dir = dir.rstrip(os.path.sep)
        QtCore.QObject.__init__(self)

        self.load_settings()

    @staticmethod
    def make_new(destdir):
        # make a new study... the hard way.
        def dest_path(x): return os.path.join(destdir, x)
        os.mkdir(destdir)
        for dir in ['Canonical', 'Documents', 'Matrices', 'Results']:
            os.mkdir(dest_path(dir))
        shutil.copy(os.path.join(package_dir, 'study_skel', 'Matrices', 'conceptnet.pickle'), os.path.join(destdir, 'Matrices', 'conceptnet.pickle'))
        write_json_to_file({}, dest_path('settings.json'))

        return StudyDirectory(destdir)

    def load_settings(self):
        try:
            self.settings = load_json_from_file(self.get_settings_file())
        except (IOError, ValueError):
            self.settings = {}
            traceback.print_exc()

    def save_settings(self):
        write_json_to_file(self.settings, self.get_settings_file())
    
    def load_cache(self):
        self._load_blend()
        self._load_projections()
        self._load_svd()
        self._load_stats()

    def _step(self, msg):
        logger.info(msg)
        self.emit(QtCore.SIGNAL('step(QString)'), msg)
    
    def study_path(self, path):
        return self.dir + os.path.sep + path

    def get_settings_file(self):
        return self.study_path("settings.json")

    def get_canonical_dir(self):
        return self.study_path("Canonical")

    def get_documents_dir(self):
        return self.study_path("Documents")

    def get_matrices_dir(self):
        return self.study_path("Matrices")
        
    def get_results_dir(self):
        return self.study_path("Results")
    
    def listdir(self, dir, text_only, full_names):
        files = os.listdir(self.study_path(dir))
        if text_only: files = [x for x in files if x.endswith('.txt')]
        if full_names:
            return [self.study_path(os.path.join(dir, x)) for x in files]
        else:
            return files

    def get_matrices_files(self):
        return self.listdir('Matrices', text_only=False, full_names=True)

    def get_documents(self):
        study_documents = [Document.from_file(filename, name=os.path.basename(filename))
                           for filename in self.listdir('Documents', text_only=True, full_names=True)]
        canonical_documents = [CanonicalDocument.from_file(filename, name=os.path.basename(filename))
                               for filename in self.listdir('Canonical', text_only=True, full_names=True)]
        return study_documents + canonical_documents
    #self.canonical_docs = [doc.name for doc in self.canonical_documents]
        

    def get_matrices(self):
        return [get_picklecached_thing(filename).normalized(mode=[0,1]).bake()
                for filename in self.get_matrices_files()]
    

    def get_study(self):
        return Study(name=self.dir.split(os.path.sep)[-1],
                     documents=self.get_documents(), other_matrices=self.get_matrices())

    def analyze(self):
        study = self.get_study()
        results = study.analyze()
        results.save(self.study_path('Results'))

    def set_num_axes(self, axes):
        self.settings['axes'] = axes
        self.study.num_axes = axes
        self.save_settings()

def test():
    study = StudyDirectory('../ThaiFoodStudy')
    study.analyze()

if __name__ == '__main__':
    DO_PROFILE=False
    if DO_PROFILE:
        import cProfile as profile
        import pstats
        profile.run('test()', 'study.profile')
        p = pstats.Stats('study.profile')
        p.sort_stats('time').print_stats(50)
    else:
        test()


