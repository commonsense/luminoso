"""
This class provides the model to SVDView's view, calculating a blend of all
the components of a study that it finds in the filesystem.
"""
import sys, os
if __name__ == '__main__':
    # prepare for other imports
    sys.path.extend([os.path.join(os.path.dirname(sys.argv[0]), "lib"),
                     os.path.dirname(sys.argv[0])])

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
from csc.divisi2.blending import blend, blend_svd
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
        sentences.append(current_sentence)
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
    def __init__(self, name, documents, canonical, other_matrices, num_axes=20):
        QtCore.QObject.__init__(self)
        self.name = name
        self.study_documents = documents
        self.canonical_documents = canonical
        # self.documents is now a property
        self._documents_matrix = None
        self.other_matrices = other_matrices
        self.num_axes = num_axes
        
    def _step(self, msg):
        logger.info(msg)
        self.emit(QtCore.SIGNAL('step(QString)'), msg)

    def get_contents_hash(self):
        def sha1(txt):
            if isinstance(txt, unicode): txt = txt.encode('utf-8')
            return hashlib.sha1(txt).hexdigest()

        docs = dict((doc.name, (isinstance(doc, CanonicalDocument),
                                sha1(doc.text)))
                    for doc in self.documents)
        matrices = tuple(sorted(self.other_matrices.keys()))

        # TODO: make sure matrices have a meaningful `hash`.
        #matrices = dict((name, hash(mat)) for name, mat in self.other_matrices.items())
        return dict(docs=docs, matrices=matrices)

    @property
    def num_documents(self):
        return len(self.documents)

    @property
    def documents(self):
        return self.study_documents + self.canonical_documents
    
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
        for doc in self.study_documents:
            concepts = doc.extract_concepts_with_negation()[:100]
            for concept1, value1 in concepts:
                for concept2, value2 in concepts:
                    entries.append( (value1*value2, concept1, concept2) )
                    entries.append( (value1*value2, concept2, concept1) )
            for sentence in doc.get_sentences():
                # avoid insane space usage by limiting to 100 words
                concepts = extract_concepts_from_words(sentence[:100])
                for concept1, value1 in concepts:
                    for concept2, value2 in concepts:
                        entries.append( (value1*value2, concept1, concept2) )
                        entries.append( (value1*value2, concept2, concept1) )
        return divisi2.SparseMatrix.square_from_named_entries(entries)

    def get_assoc_blend_svd(self):
        other_matrices = []
        doc_matrix = self.get_documents_assoc()
        self._step('Blending...')
        for name, matrix in self.other_matrices.items():
            # use association matrices only
            # (unless we figure out how to do both kinds of blending)
            if name.endswith('.assoc.smat'):
                if matrix.shape[0] != matrix.shape[1]:
                    raise ValueError("The matrix %s is not square" % name)
                other_matrices.append(matrix)

        if doc_matrix is None:
            blend_entries = other_matrices
            study_concepts = set(theblend.row_labels)
        else:
            blend_entries = [doc_matrix] + other_matrices
            study_concepts = set(doc_matrix.row_labels)
        blendsvd = blend_svd([mat.normalize_all() for mat in blend_entries],
                             k=self.num_axes)
        return blendsvd, study_concepts

    def get_eigenstuff(self):
        self._step('Finding eigenvectors...')
        document_matrix = self.get_documents_matrix()
        blendsvd, study_concepts = self.get_assoc_blend_svd()
        U, Sigma, V = blendsvd
        indices = [U.row_index(concept) for concept in study_concepts]
        reduced_U = U[indices]
        doc_rows = divisi2.aligned_matrix_multiply(document_matrix.normalize_rows(), reduced_U)
        projections = reduced_U.extend(doc_rows)
        return document_matrix, projections, Sigma

    def compute_stats(self, docs, spectral):
        """
        Calculate statistics.

        Consistency: how tightly-clustered the documents are in the spectral
        decomposition space.

        Centrality: a Z-score for how "central" each concept and document
        is. Same general idea as "congruence" from Luminoso 1.0.
        """

        if len(self.study_documents) == 0:
            # consistency and centrality are undefined
            consistency = None
            centrality = None
            core = None
        else:
            concept_sums = docs.col_op(np.sum)
            doc_indices = [spectral.left.row_index(doc.name) for doc in
                           self.study_documents]
            
            # Compute the association of all study documents with each other
            reference_assoc = np.asarray(spectral[doc_indices, doc_indices].to_dense())
            reference_mean = np.mean(reference_assoc)
            reference_stdev = np.std(reference_assoc)
            reference_stderr = reference_stdev / len(doc_indices)
            consistency = reference_mean / reference_stderr

            ztest_stderr = reference_stdev / np.sqrt(len(doc_indices))

            all_assoc = np.asarray(spectral[:, doc_indices].to_dense())
            all_means = np.mean(all_assoc, axis=1)
            all_stdev = np.std(all_assoc, axis=1)
            all_stderr = all_stdev / np.sqrt(len(doc_indices))
            centrality = divisi2.DenseVector((all_means - reference_mean) /
            ztest_stderr, spectral.row_labels)
            core = centrality.top_items(50)
            core = [c[0] for c in core
                    if c[0] in concept_sums.labels
                    and concept_sums.entry_named(c[0]) >= 2][:10]

            c_centrality = {}
            for doc in self.canonical_documents:
                c_centrality[doc.name] = centrality.entry_named(doc.name)
        
        return {
            'num_documents': self.num_documents,
            'num_concepts': spectral.shape[0] - self.num_documents,
            'consistency': consistency,
            'centrality': c_centrality,
            'core': core,
            'timestamp': list(time.localtime())
        }
    
    def analyze(self):
        # TODO: make it possible to blend multiple directories
        self._documents_matrix = None
        docs, projections, Sigma = self.get_eigenstuff()
        spectral = divisi2.reconstruct_activation(projections, Sigma, post_normalize=False)
        self._step('Calculating stats...')
        stats = self.compute_stats(docs, spectral)
        
        results = StudyResults(self, docs, spectral.left, spectral, stats)
        return results


class StudyResults(QtCore.QObject):
    def __init__(self, study, docs, projections, spectral, stats):
        QtCore.QObject.__init__(self)
        self.study = study
        self.docs = docs
        self.spectral = spectral
        self.projections = projections
        self.stats = stats
        self.canonical_filenames = [doc.name for doc in study.canonical_documents]
        self.info = None

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
        return self.info

    def get_consistency(self):
        return self.stats['consistency']

    def get_congruence(self, concept):
        return self.stats['congruence'][concept]

    def get_info(self):
        if self.info is not None: return self.info
        else: return default_info_page(self.study)

    def save(self, dir):
        def tgt(name): return os.path.join(dir, name)
        def save_pickle(name, obj):
            with open(tgt(name), 'wb') as out:
                pickle.dump(obj, out, -1)

        self.study._step('Saving document matrix...')
        save_pickle("documents.smat", self.docs)

        self.study._step('Saving eigenvectors...')
        save_pickle("spectral.rmat", self.spectral)
        
        self.study._step('Saving projections...')
        save_pickle('projections.dmat', self.projections)

        self.study._step('Writing reports...')
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
        try:
            input_hash = load_pickle('input_hash.pickle')
        except IOError:
            raise OutdatedAnalysisError()
        cur_hash = for_study.get_contents_hash()
        if input_hash != cur_hash:
            raise OutdatedAnalysisError()
        
        for_study._step('Loading document matrix...')
        docs = load_pickle("documents.smat")
        for_study._step('Loading eigenvectors...')
        spectral = load_pickle("spectral.rmat")
        for_study._step('Loading projections...')
        projections = load_pickle("projections.dmat")

        # Load stats
        for_study._step('Loading stats...')
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
    
    def _ensure_dir_exists(self, targetdir):
        path = os.path.join(self.dir, targetdir)
        if not os.path.exists(path):
            os.mkdir(path)

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
        self._ensure_dir_exists("Canonical")
        return self.study_path("Canonical")

    def get_documents_dir(self):
        self._ensure_dir_exists("Documents")
        return self.study_path("Documents")

    def get_matrices_dir(self):
        self._ensure_dir_exists("Matrices")
        return self.study_path("Matrices")
        
    def get_results_dir(self):
        self._ensure_dir_exists("Results")
        return self.study_path("Results")
    
    def listdir(self, dir, text_only, full_names):
        files = os.listdir(self.study_path(dir))
        if text_only: files = [x for x in files if x.endswith('.txt')]
        if full_names:
            return [self.study_path(os.path.join(dir, x)) for x in files]
        else:
            return files

    def get_matrices_files(self):
        self._ensure_dir_exists("Matrices")
        return self.listdir('Matrices', text_only=False, full_names=True)

    def get_documents(self):
        study_documents = [Document.from_file(filename, name=os.path.basename(filename))
                           for filename in self.listdir('Documents', text_only=True, full_names=True)]
        return study_documents

    def get_canonical_documents(self):
        canonical_documents = [CanonicalDocument.from_file(filename, name=os.path.basename(filename))
                               for filename in self.listdir('Canonical', text_only=True, full_names=True)]
        return canonical_documents

    def get_matrices(self):
        return dict((os.path.basename(filename), divisi2.load(filename))
                    for filename in self.get_matrices_files())
    

    def get_study(self):
        return Study(name=self.dir.split(os.path.sep)[-1],
                     documents=self.get_documents(),
                     canonical=self.get_canonical_documents(),
                     other_matrices=self.get_matrices(),
                     num_axes = self.settings.get('axes',20))

    def analyze(self):
        study = self.get_study()
        results = study.analyze()
        self._ensure_dir_exists('Results')
        results.save(self.study_path('Results'))
        return results

    def set_num_axes(self, axes):
        self.settings['axes'] = axes
        self.study.num_axes = axes
        self.save_settings()
    
    def get_existing_analysis(self):
        # FIXME: this loads the study twice, I think
        try:
            return StudyResults.load(self.study_path('Results'), self.get_study())
        except OutdatedAnalysisError:
            return None

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


