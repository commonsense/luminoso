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

def write_json_to_file(data, file):
    f = open(file, 'w')
    json.dump(data, f)
    f.close()

class LuminosoStudy(QtCore.QObject):
    def __init__(self, dir):
        QtCore.QObject.__init__(self)
        self.dir = dir.rstrip(os.path.sep)
        self.load_settings()

        self.blend = None
        self.svd = None
        self.projections = None
        self.study_concepts = None
        self.info = None
        self.stats = None
        self.update_documents()
        #self.load_pickle_cache()

    @staticmethod
    def make_new(destdir):
        # make a new study... the hard way.
        def dest_path(x): return os.path.join(destdir, x)
        os.mkdir(destdir)
        for dir in ['Canonical', 'Documents', 'Matrices', 'Results']:
            os.mkdir(dest_path(dir))
        shutil.copy(os.path.join(package_dir, 'study_skel', 'Matrices', 'conceptnet.pickle'), os.path.join(destdir, 'Matrices', 'conceptnet.pickle'))
        write_json_to_file({}, dest_path('settings.json'))

        return LuminosoStudy(destdir)

    def load_settings(self):
        try:
            settings_file = open(self.get_settings_file())
            self.settings = json.load(settings_file)
            settings_file.close()
        except (IOError, ValueError):
            self.settings = {}
            traceback.print_exc()

    def save_settings(self):
        write_json_to_file(self.settings, self.get_settings_file())
    
    def load_pickle_cache(self):
        self._load_blend()
        self._load_projections()
        self._load_svd()
        self._load_stats()

    def _step(self, msg):
        logger.info(msg)
        self.emit(QtCore.SIGNAL('step(QString)'), msg)
    
    def _load_blend(self):
        self._step("Loading blend")
        try:
            blend_p = open(self.study_path("Results/blend.pickle"))
            self.blend = pickle.load(blend_p)
            blend_p.close()
        except IOError:
            logger.info("No blend file in study")

    def _load_projections(self):
        self._step("Loading projections")
        try:
            projections_p = open(self.study_path("Results/projections.pickle"))
            self.projections = pickle.load(projections_p)
            self.study_concepts = set(self.projections.label_list(0))
            self.study_concepts.remove('DefaultXAxis')
            self.study_concepts.remove('DefaultYAxis')
            projections_p.close()
        except IOError:
            logger.info("No projections file in study")

    def _load_svd(self):
        self._step("Loading SVD")
        try:
            svd_p = open(self.study_path("Results/svd.pickle"))
            self.svd = pickle.load(svd_p)
            svd_p.close()
        except IOError:
            logger.info("No svd file in study")

    def _load_stats(self):
        self._step("Loading stats")
        try:
            stats_p = open(self.study_path("Results/stats.json"))
            self.stats = json.load(stats_p)
            stats_p.close()
            self.make_info_page()
        except (IOError, ValueError):
            logger.info("No stats file in study")

    def study_path(self, path):
        return self.dir + os.path.sep + path

    def get_name(self):
        return self.dir.split(os.path.sep)[-1]

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

    def update_documents(self):
        self.study_documents = [Document.from_file(filename, name=os.path.basename(filename))
                           for filename in self.listdir('Documents', text_only=True, full_names=True)]
        self.canonical_documents = [CanonicalDocument.from_file(filename, name=os.path.basename(filename))
                               for filename in self.listdir('Canonical', text_only=True, full_names=True)]
        self.documents = self.study_documents + self.canonical_documents
        self.canonical_docs = [doc.name for doc in self.canonical_documents]
        

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

    def load_matrices(self):
        matrices = []
        for filename in self.get_matrices_files():
            matrices.append(get_picklecached_thing(filename).normalized(mode=[0,1]).bake())
        return matrices
    
    def get_blend(self):
        other_matrices = self.load_matrices()
        doc_matrix = self.get_documents_matrix()

        if doc_matrix is not None:
            blend = Blend([doc_matrix] + other_matrices)
            self.study_concepts = set(doc_matrix.label_list(0)) | set(doc_matrix.label_list(1))
        else:
            if len(other_matrices) == 1:
                blend = other_matrices[0]
            else:
                blend = Blend(other_matrices)
            self.study_concepts = set(blend.label_list(0))
        
        out = open(self.study_path("Results/blend.pickle"), 'wb')
        pickle.dump(blend, out)
        out.close()

        self.blend = blend
        if doc_matrix:
            self.settings['documents'] = list(doc_matrix.label_list(1))
        else:
            self.settings['documents'] = []
        self.save_settings()
        return blend
    
    def study_filter(self, concept):
        return concept in self.study_concepts
    
    def write_csv(self):
        import csv
        docs = self.settings.get('documents', [])
        x_axis = self.projections['DefaultXAxis',:].hat()
        y_axis = self.projections['DefaultYAxis',:].hat()
        output = open(self.study_path("Results/coords.csv"), 'w')
        writer = csv.writer(output)
        writer.writerow(['Concept', 'X projection', 'Y projection', 'Coordinates'])
        for concept in self.study_concepts:
            xproj = self.projections[concept,:] * x_axis
            yproj = self.projections[concept,:] * y_axis
            coords = self.projections[concept,:].values()
            row = [concept.encode('utf-8'), xproj, yproj] + coords
            writer.writerow(row)
        output.close()

    def calculate_stats(self):
        """
        FIXME: On large datasets, calculating every pairwise similarity might
        be too expensive. Cut down the size of the working matrix somehow?
        """
        projdata = data(self.projections)
        svals = data(self.svd.svals)
        
        # build an array of documents vs. axes
        docdata = []
        document_list = self.settings.get('documents')
        if document_list:
            for doc in self.settings['documents']:
                docdata.append(data(self.projections[doc,:]))
            docdata = np.array(docdata)
            simdata = np.dot(projdata * svals, docdata.T)
            
            mean = np.mean(simdata)
            stdev = np.std(simdata)
            n = simdata.shape[0] * simdata.shape[1]
            stderr = stdev/np.sqrt(n)

            congruence = {}
            for index, concept in enumerate(self.study_concepts):
                if not en_nl.is_blacklisted(concept):
                    vec = simdata[index]
                    cmean = np.mean(vec)
                    cstdev = np.std(vec)
                    cstderr = cstdev / np.sqrt(len(vec))
                    z = (cmean - mean) / cstderr
                    congruence[concept] = z
            consistency = mean/stderr
            self.stats = {
                'mean': mean,
                'stdev': stdev,
                'n': n,
                'consistency': consistency,
                'congruence': congruence,
                'timestamp': list(time.localtime())
            }
            write_json_to_file(self.stats, self.study_path("Results/stats.json"))
            self.report_stats()
            return self.stats
        else:
            return {}
    
    def make_info_page(self):
        if self.stats is not None:
            self.info = render_info_page(self)

    def report_stats(self):
        self.make_info_page()
        if self.info:
            out = open(self.study_path("Results/report.html"), 'w')
            out.write(self.info)
            out.close()

    def get_consistency(self):
        return self.stats['consistency']

    def get_congruence(self, concept):
        return self.stats['congruence'][concept]

    def set_num_axes(self, axes):
        self.settings['axes'] = axes
        self.save_settings()

    def get_info(self):
        if self.info is not None: return self.info
        else: return default_info_page(self)

    def analyze(self):
        # TODO: recursive svd, congruence, consistency

        # TODO: make it possible to read in multiple directories and
        # blend them
        k = self.settings.get('axes', 20)

        self._step('Blending...')
        blend = self.get_blend()
        svd = blend.svd(k=k)
        self.svd = svd
         
        self._step('Concatenating...')
        concatenated = svd.get_weighted_u().concatenate(svd.v)
        new_concepts = OrderedSet(self.study_concepts)
        new_data = np.zeros((len(new_concepts), svd.u.shape[1]))
        new_matrix = LabeledView(DenseTensor(new_data), [new_concepts, None])
        for index in xrange(len(new_concepts)):
            new_data[index, :] = data(concatenated[new_concepts[index], :])
        
        self._step('Finding interesting axes...')
        newsvd = new_matrix.svd(k=k)
        axis_labels = OrderedSet(['DefaultXAxis', 'DefaultYAxis'])
        extra_axis_data = data(newsvd.v.T)[0:2, :] / 1000
        extra_axis_matrix = LabeledView(DenseTensor(extra_axis_data), [axis_labels, None])
        self.projections = new_matrix.concatenate(extra_axis_matrix)

        self._step('Saving SVD...')
        out = open(self.study_path("Results/svd.pickle"), 'wb')
        pickle.dump(self.svd, out)
        out.close()
        
        self._step('Saving projections...')
        out = open(self.study_path("Results/projections.pickle"), 'wb')
        pickle.dump(self.projections, out)
        out.close()

        self._step('Calculating stats...')
        self.calculate_stats()
        self.report_stats()
        self._step('Writing CSV...')
        self.write_csv()
        
    
        if self.stats is not None:
            return (blend, self.projections, self.svd, self.stats)
        else:
            return (blend, self.projections, self.svd, None)

def test():
    study = LuminosoStudy('../ThaiFoodStudy')
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


