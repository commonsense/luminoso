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
import hashlib
import chardet
logger = logging.getLogger('luminoso')

from standalone_nlp.lang_en import nltools as en_nl
from csc import divisi2
from csc.divisi2.blending import blend
from csc.divisi2.ordered_set import OrderedSet

from luminoso.whereami import package_dir
from luminoso.report import render_info_page, default_info_page

import shutil

try:
    import json
except ImportError:
    import simplejson as json

class OutdatedAnalysisError(Exception):
    pass

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

    def get_sentences(self):
        words = en_nl.tokenize(self.text).split()
        sentences = []
        current_sentence = []
        for word in words:
            if word in PUNCTUATION:
                if len(current_sentence) > 2:
                    sentences.append(current_sentence)
                    current_sentence = []
            else:
                current_sentence.append(word)
        sentences.append(' '.join(current_sentence))
        return sentences

class CanonicalDocument(Document):
    pass


NEGATION = ['no', 'not', 'never', 'stop', 'lack', "n't"]
PUNCTUATION = ['.', ',', '!', '?', '...', '-', ':', ';']
def extract_concepts_with_negation(text):
    words = en_nl.tokenize(text).split()
    return extract_concepts_from_words(words)

def extract_concepts_from_words(words):
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
        self._documents_matrix = None
        self.other_matrices = other_matrices
        self.num_axes = 20
        
    def _step(self, msg):
        logger.info(msg)
        self.emit(QtCore.SIGNAL('step(QString)'), msg)

    def get_contents_hash(self):
        def sha1(txt): return hashlib.sha1(txt).hexdigest()
        docs = dict((doc.name, (isinstance(doc, CanonicalDocument), sha1(doc.text))) for doc in self.documents)
        matrices = dict((name, hash(mat)) for name, mat in self.other_matrices.items())
        # TODO: make sure matrices have a meaningful `hash`.
        return dict(docs=docs, matrices=matrices)

    @property
    def num_documents(self):
        return len(self.documents)
    
    def get_documents_matrix(self):
        """
        Get a matrix of documents vs. concepts.

        This is temporarily cached (besides what StudyDir does) because it
        will be needed multiple times in analyzing a study.
        """
        self._step('Building document matrix...')
        if self.num_documents == 0: return None
        if self._documents_matrix is not None:
            return self._documents_matrix
        entries = []
        for doc in self.documents:
            for concept, value in doc.extract_concepts_with_negation():
                if not en_nl.is_blacklisted(concept):
                    entries.append((value, doc.name, concept))
        self._documents_matrix = divisi2.make_sparse(entries)
        return self._documents_matrix
    
    def get_documents_assoc(self):
        self._step('Finding associated concepts...')
        if self.num_documents == 0: return None
        entries = []
        for doc in self.documents:
            if isinstance(doc, CanonicalDocument):
                # canonical docs must not affect the analysis
                continue

            for sentence in doc.get_sentences():
                # avoid insane space usage by limiting to 100 words
                concepts = extract_concepts_from_words(sentence[:100])
                for concept1, value1 in concepts:
                    for concept2, value2 in concepts:
                        entries.append( (value1*value2, concept1, concept2) )
                        entries.append( (value1*value2, concept2, concept1) )
        return divisi2.SparseMatrix.square_from_named_entries(entries)

    def get_assoc_blend(self):
        self._step('Blending...')
        other_matrices = []
        doc_matrix = self.get_documents_assoc()
        for name, matrix in self.other_matrices.items():
            # use association matrices only
            # (unless we figure out how to do both kinds of blending)
            if name.endswith('.assoc.smat'):
                if matrix.shape[0] != matrix.shape[1]:
                    raise ValueError("The matrix %s is not square" % name)
                other_matrices.append(matrix)

        if doc_matrix is None:
            theblend = blend(other_matrices)
            study_concepts = set(theblend.row_labels)
        else:
            theblend = blend([doc_matrix] + other_matrices)
            study_concepts = set(doc_matrix.row_labels)
        return theblend, study_concepts

    def get_eigenstuff(self):
        self._step('Finding eigenvectors...')
        document_matrix = self.get_documents_matrix()
        theblend, study_concepts = self.get_assoc_blend()
        U, Sigma, V = theblend.normalize_all().svd(k=self.num_axes)
        indices = [U.row_index(concept) for concept in study_concepts]
        reduced_U = U[indices]
        doc_rows = divisi2.aligned_matrix_multiply(document_matrix, reduced_U)
        projections = reduced_U.extend(doc_rows)
        return document_matrix, projections, Sigma

    def compute_stats(self, docs, projections, spectral):
        """
        DOUBLE FIXME: this isn't divisi2 ready

        FIXME: On large datasets, calculating every pairwise similarity might
        be too expensive. Cut down the size of the working matrix somehow?
        """
        return None

        standard_docs_projections = divisi2.aligned_matrix_multiply(docs,
        projections)

        # left off here

        for doc in self.get_documents():
            if isinstance(doc, CanonicalDocument): continue
            for doc2 in self.get_documents():
                pass
        
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
        self._documents_matrix = None
        docs, projections, Sigma = self.get_eigenstuff()
        spectral = divisi2.reconstruct_activation(projections, Sigma)
        self._step('Calculating stats...')
        stats = self.compute_stats(docs, projections, spectral)
        
        return StudyResults(self, docs, projections, spectral, stats)


class StudyResults(QtCore.QObject):
    def __init__(self, study, docs, projections, spectral, stats):
        QtCore.QObject.__init__(self)
        self.study = study
        self.docs = docs
        self.spectral = spectral
        self.projections = projections
        self.stats = stats

    def write_coords_as_csv(self, filename):
        # FIXME: not divisi2 ready
        raise NotImplementedError

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

        #self._step('Saving blend...')
        save_pickle("documents.smat", self.docs)

        #self._step('Saving eigenvectors...')
        save_pickle("spectral.rmat", self.spectral)
        
        #self._step('Saving projections...')
        save_pickle('projections.dmat', self.projections)

        #self._step('Writing reports...')
        # Save stats
        write_json_to_file(self.stats, tgt("stats.json"))
        self.write_report(tgt("report.html"))

        # Save input contents hash to know if the study has changed.
        save_pickle('input_hash.pickle', self.study.get_contents_hash())

    @classmethod
    def load(cls, dir, for_study):
        def tgt(name): return os.path.join(dir, name)
        def load_pickle(name):
            with open(tgt(name), 'rb') as f:
                return pickle.load(f)

        # Either this will all fail or all succeed.
        input_hash = load_pickle('input_hash.pickle')
        cur_hash = for_study.get_contents_hash()
        if input_hash != cur_hash:
            raise OutdatedAnalysisError()
        
        docs = load_pickle("documents.smat")
        spectral = load_pickle("spectral.rmat")
        projections = load_pickle("projections.dmat")

        # Load stats
        stats = load_json_from_file(tgt("stats.json"))

        return cls(for_study, docs, projections, spectral, stats)

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
        return dict((os.path.basename(filename), divisi2.load(filename))
                    for filename in self.get_matrices_files())
    

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


