#!/usr/bin/env python
"""
This module defines the LuminosoModel object in luminoso2.

Overall design:
- Luminoso as a whole defines some canonicals that can be easily included
- A LuminosoModel contains many LuminosoStudies, plus canonicals
- A LuminosoStudy contains many Documents (many of which also go into the model)
- Spaces and studies are configured using the `config` module, giving
  configurations that are both easily human-readable and computer-readable
"""

from __future__ import with_statement   # for Python 2.5 support
import divisi2
import numpy as np
from divisi2.fileIO import load_pickle, save_pickle
from divisi2.reconstructed import ReconstructedMatrix
from divisi2.ordered_set import PrioritySet
from luminoso2.term_database import TermDatabase, _BIG
from luminoso2.text_readers import get_reader, DOCUMENT, TAG
from luminoso2.document_handlers import handle_url
from collections import defaultdict
import os
import sys
from config import Config
import logging
import codecs
try:
    import json
except ImportError:
    import simplejson as json
LOG = logging.getLogger(__name__)

class StudyExists(Exception):
    pass

class LuminosoModel(object):
    """
    A LuminosoModel is a semantic space. You supply it with as many documents
    as possible from the domain of documents you intend to analyze, or
    possibly other forms of domain-specific knowledge.

    The LuminosoModel represents the semantic similarities between things
    as a Divisi2 reconstructed association matrix. This matrix can be
    updated incrementally to take new data into account, which is how
    Luminoso learns new domain-specific knowledge.
    """
    CONFIG_FILENAME = 'luminoso.cfg'
    ASSOC_FILENAME = 'associations.rmat'
    DB_FILENAME = 'terms.sqlite'

    def __init__(self, model_dir):
        """
        A LuminosoModel is constructed from `dir`, a path to a directory.
        This directory will contain saved versions of various matrices, as
        well as a SQLite database of terms and documents.
        """
        if not isinstance(model_dir, unicode):
            # Ensure that paths are unicode.
            model_dir = model_dir.decode(sys.getfilesystemencoding())
        if not os.access(model_dir, os.R_OK):
            raise IOError("Cannot read the study directory %s. "
                          "Use LuminosoModel.make() to make a new one."
                          % model_dir)
        self.dir = model_dir
        self._load_config()
        self._load_assoc()
        self.database = TermDatabase(
          self.filename_in_dir(LuminosoModel.DB_FILENAME)
        )
        self.connections_cache = {}
        self.idf_cache = {}
    
    def filename_in_dir(self, filename):
        """
        Given a filename relative to this LuminosoModel's directory, get its
        complete path.
        """
        return self.dir + os.sep + filename
    
    def file_exists_in_dir(self, filename):
        """
        Determine whether a file exists in this LuminosoModel's directory.
        """
        return os.access(self.filename_in_dir(filename), os.F_OK)

    def _load_config(self):
        "Load the configuration file."
        if self.file_exists_in_dir(LuminosoModel.CONFIG_FILENAME):
            self.config = Config(
              open(self.filename_in_dir(LuminosoModel.CONFIG_FILENAME))
            )
        else:
            raise IOError("This model is missing a config file.")

    def save_config(self):
        "Save the current configuration to the configuration file."
        save_config_file(
          self.config,
          self.filename_in_dir(LuminosoModel.CONFIG_FILENAME)
        )

    def save_canonical_stats(self, study='all'):
        """
        Given a study named 'foo', this saves its statistics to
        'foo.stats.json'.
        """
        stats = self.canonical_stats(study)
        out = codecs.open(self.filename_in_dir(study+'.stats.json'), 'w',
                          encoding='utf-8')
        json.dump(stats, out, indent=2, ensure_ascii=False)
        out.close()

    def _load_assoc(self):
        "Load the association matrix and priority queue from a file."
        if self.file_exists_in_dir(LuminosoModel.ASSOC_FILENAME):
            self.assoc = load_pickle(
                self.filename_in_dir(LuminosoModel.ASSOC_FILENAME)
            )
            assert isinstance(self.assoc, ReconstructedMatrix)
        else:
            raise IOError("This LuminosoModel does not have an "
                          "'associations.rmat' file. Use LuminosoModel.make() "
                          "to make a valid LuminosoModel.")
        self.assoc.make_symmetric()
        assert isinstance(self.assoc.row_labels, PrioritySet)

        self.priority = self.assoc.row_labels
        self.priority.listen_for_drops(self.on_drop)

    def save_assoc(self):
        "Save the association matrix to a file."
        save_pickle(self.assoc, 
                    self.filename_in_dir(LuminosoModel.ASSOC_FILENAME))
    
    def on_drop(self, index, key):
        """
        Handle when a key falls out of the PrioritySet.
        """
        self.assoc.left[index, :] = 0

    def add_document(self, doc, reader_name=None):
        """
        Take in a document, pass it through the reader, and store its terms
        in the term database.

        The document should be expressed as a dictionary, containing at least
        these keys:
        
        - name: the unique identifier for the document
        - text: the plain text of the document, possibly including text-encoded
          tags
        - url: a unique identifier for the document, preferably one that
          actually locates it relative to the study

        Optionally, it may contain:
        
        - tags: (key, value) tuples representing tags
        """
        LOG.info("Reading document: %r" % doc['url'])
        if reader_name is None:
            reader_name = self.config['reader']
        reader = get_reader(reader_name)
        text = doc['text']
        tags = doc.get('tags', [])
        doc_terms = []
        connections = list(reader.extract_connections(text))
        self.connections_cache[doc['url']] = connections
        for weight, term1, term2 in connections:
            if term1 == DOCUMENT:
                if isinstance(term2, tuple) and term2[0] == TAG:
                    tags.append(term2[1:])
                else:
                    doc_terms.append((term2, weight))
                    relevance = self.database.term_relevance(term2)
                    self.index_term(term2, relevance)

        doc['reader'] = reader_name
        doc['terms'] = doc_terms
        doc['tags'] = tags
        self.database.add_document(doc)
        self.idf_cache = {}   # invalidate the cache of term IDFs
        return doc['url']
    
    def get_document_connections(self, docid):
        """
        Given a previously added document, get the list of connections
        produced from it.
        """
        if docid in self.connections_cache:
            connections = self.connections_cache[docid]
        else:
            doc = self.database.get_document(docid)
            reader = get_reader(doc.reader)
            connections = list(reader.extract_connections(doc.text))
        return connections
    
    def get_document_terms(self, docid):
        """
        Given a previously added document, get the list of weighted terms
        that appear in it, as (term, weight) tuples.
        """
        return [(term2, weight) for (weight, term1, term2)
                in self.get_document_connections(docid)
                if term1 == DOCUMENT]
    
    def get_document_tags(self, docid):
        """
        Get the list of tags on a document from the database.
        """
        return self.database.get_document_tags(docid)
        
    def document_assoc_updates(self, docid):
        """
        Given a previously added document, yield triples to use to update the 
        association matrix.
        """
        LOG.info("Collecting connections from: %r" % docid)
        connections = self.get_document_connections(docid)
        for weight, term1, term2 in connections:
            if weight > 0 and term1 != DOCUMENT:
                norm_factor = ((self.database.count_term(term1) + 1)
                               * (self.database.count_term(term2) + 1)) ** .5
                yield weight/norm_factor, term1, term2
    
    def index_term(self, term, priority=None):
        """
        Ensure that a term is in both the database and the PrioritySet.
        If `priority` is specified, this will update its priority value.

        Returns the index of the term in the set.
        """
        index = self.priority.add(term)
        if priority:
            self.priority.update(term, priority)
        return index

    def get_term_idf(self, term):
        if term in self.idf_cache:
            return self.idf_cache[term]
        else:
            idf = self.database.term_idf(term)
            self.idf_cache[term] = idf
            return idf

    def learn_assoc(self, weight, term1, term2):
        """
        Learn the strength of the association between term1 and term2,
        both of which should exist in self.priority for efficiency's sake.
        For the purpose of testing, however, we can still add the terms.
        """
        try:
            row = self.priority.index(term1)
        except KeyError:
            row = self.priority.add(term1)
        try:
            col = self.priority.index(term2)
        except KeyError:
            col = self.priority.add(term2)

        mse = self.assoc.hebbian_increment(row, col, weight)
        return mse

    def learn_from(self, url, study=None, iterations=1):
        """
        Given a URL or file path that points to a collection of documents,
        learn from all of those documents. They may also be added to a
        study at the same time.

        Default to 1 iteration, because on a reasonable amount of data that
        will be all you need.
        """
        self.add_from_url(url, study, learn_iterations=iterations)
        self.save_assoc()
        self.save_config()
    # compatibility with previous docs
    learn_from_url = learn_from

    def add_from(self, url, study=None, learn_iterations=0, batch_size=1000):
        """
        Given a URL or file path that points to a collection of documents,
        add all the documents to the database. If `learn_iterations` is 0,
        the concept model will not change. When greater than 0, this 
        implements `learn_from_url`.

        This is the main loop that one should use to train a model with a
        batch of documents.
        """
        stream = handle_url(url, sample_frac=self.config.get('sample_frac', 1.0))
        batch = []
        while True:
            if len(batch) == batch_size:
                self.add_batch(lambda: iter(batch), study, learn_iterations)
                batch[:] = []
            try:
                batch.append(next(stream))
            except StopIteration:
                self.add_batch(lambda: iter(batch), study, learn_iterations)
                batch[:] = []
                break
    add_from_url = add_from

    def add_batch(self, stream_func, study=None, learn_iterations=0):
        """
        Add a batch of documents from some source, a `stream_func` that
        when called returns an iterator over the documents.
        """
        fulltext_cache = {}
        self.connections_cache = {}

        # First pass: add documents to the term database, and meanwhile
        # collect full texts and tags.
        for doc in stream_func():
            docid = self.add_document(doc)
            reader = get_reader(doc['reader'])
            for term, fulltext in reader.extract_term_texts(doc['text']):
                fulltext_cache[term] = fulltext
            if study is not None:
                self.database.set_tag_on_document(docid, 'study', study)
        
        LOG.info("Committing documents to the database")
        self.database.commit()

        LOG.info("Collecting relevant terms")
        self.database.update_relevance()

        if learn_iterations:
            # Second pass (optional): find how much we should update the
            # ReconstructedMatrix entries based on the word associations
            # we discover.
            learn_accumulator = defaultdict(float)
            for doc in stream_func():
                for weight, term1, term2\
                 in self.document_assoc_updates(doc['url']):
                    if term1 in self.priority and term2 in self.priority:
                        learn_accumulator[(term1, term2)] += weight

            # Now actually apply those total updates. Multiple times, if asked.
            total = len(learn_accumulator)
            for iter in xrange(learn_iterations):
                LOG.info("Updating association matrix: pass %d" % (iter+1))
                i = 0
                avg_err = 1.0
                for term1, term2 in learn_accumulator:
                    i += 1
                    if (i % 100) == 0:
                        LOG.info("Learned %d/%d; err=%4.4f"
                                 % (i, total, avg_err))
                    weight = learn_accumulator[(term1, term2)]
                    err = self.learn_assoc(weight, term1, term2)
                    avg_err = (.999 * avg_err) + (.001 * err)
        
        # Finally, update the full texts of the terms we saw.
        LOG.info("Updating full texts")
        for term, fulltext in fulltext_cache.items():
            self.database.set_term_text(term, fulltext)
        self.database.commit()
        
        # If this was a study, make a document matrix for it.
        if study is not None:
            LOG.info("Making document matrix for %r" % study)
            self.update_doc_matrix(study)
        LOG.info("Updating tag matrix")
        self.update_tag_matrix()
    
    def docs_in_study(self, study_name='all'):
        """
        Get a list of all documents in the given study.
        """
        return list(self.database.documents_with_tag_value(u'study',
                                                           study_name))

    def update_doc_matrix(self, study_name='all'):
        """
        Collect the documents in a particular study, and make a dense matrix
        from them representing their positions in this semantic space.

        FIXME: this filename may conflict with other things like 'tags'.
        """
        docs = self.docs_in_study(study_name)
        npmat = np.zeros((len(docs), self.config['num_axes']))
        dmat = divisi2.DenseMatrix(npmat, row_labels=docs)
        for docid in docs:
            row = dmat.row_index(docid)
            dmat[row] = self.vector_from_document(docid)
        divisi2.save(dmat, self.filename_in_dir(study_name+'.dmat'))

    def get_doc_matrix(self, study_name='all'):
        """
        Get the matrix of all documents in a particular study.
        """
        return divisi2.load(self.filename_in_dir(study_name+'.dmat'))

    def add_default_study(self, study_name='all'):
        """
        Ensure that every known document is in a study with the given name
        (default 'all'). Many methods for working with documents require a
        study name. This will help with experimenting with those methods
        on documents that weren't added as part of a study.
        """
        for doc in self.database.all_documents():
            self.database.set_tag_on_document(doc, 'study', study_name)

    def update_tag_matrix(self):
        """
        Collect the tags in a particular study, and make a dense matrix
        from them representing their average positions in this semantic space.
        """
        all_tags = self.database.all_tags()
        npmat = np.zeros((len(all_tags), self.config['num_axes']))
        dmat = divisi2.DenseMatrix(npmat, row_labels=all_tags)
        for key, value in all_tags:
            row = dmat.row_index((key, value))
            ndocs = 0
            for docid in self.database.documents_with_tag_value(key, value):
                dmat[row] += self.vector_from_document(docid)
                ndocs += 1
            if ndocs > 0:
                dmat[row] /= ndocs
        divisi2.save(dmat, self.filename_in_dir('tags.dmat'))
        self._tag_matrix = dmat
        return dmat
    
    def get_tag_matrix(self):
        """
        Get the matrix of all tags in a particular study.
        """
        if hasattr(self, '_tag_matrix'):
            return self._tag_matrix
        else:
            return divisi2.load(self.filename_in_dir('tags.dmat'))

    def export_svdview(self, study_name='all', num=10000):
        from divisi2.export_svdview import write_packed
        def denormalize(concept_text):
            doc = self.database.get_document(concept_text)
            if doc:
                return doc.name
            else:
                return concept_text
                #return self.database.get_term_text(concept_text)
        
        top_terms = [term.term for term in self.database.top_terms(num)]
        num = len(top_terms)
        term_mat = divisi2.DenseMatrix(
          np.zeros((num, self.config['num_axes'])),
          row_labels=top_terms
        )
        for i in xrange(num):
            term = top_terms[i]
            term_mat[i, :] = self.assoc.left[self.priority.index(term)]

        mat = term_mat.concatenate(self.get_doc_matrix(study_name))
        write_packed(mat, self.filename_in_dir(study_name), denormalize)

    def vector_from_terms(self, terms):
        """
        Get a category vector representing the given set of weighted terms,
        expressed as (term, weight) tuples. This will apply TF-IDF weighting.
        """
        total_weight = 0.0
        for _, weight in terms:
            total_weight += abs(weight)

        vec = divisi2.DenseVector(
            np.zeros((len(self.priority),)),
            labels=self.priority
        )
        for term, weight in terms:
            if term in self.priority:
                index = self.priority.index(term)
                tfidf_weight = weight * self.get_term_idf(term) * self.database.normalized_relevance(term)
                vec[index] = tfidf_weight / total_weight
        category = divisi2.dot(vec, self.assoc.left)
        return category

    def vector_from_text(self, text, reader_name=None):
        """
        Get a category vector in this model representing the given text,
        with TF-IDF applied.
        """
        if reader_name is None:
            reader_name = self.config['reader']
        reader = get_reader(reader_name)
        terms = []
        for weight, term1, term2 in reader.extract_connections(text):
            if term1 == DOCUMENT:
                terms.append((term2, weight))
        return self.vector_from_terms(terms)
    
    def vector_from_input(self):
        """
        Get a category vector representing the given line of input from
        standard in (which is a good way to enter Unicode that iPython can't
        deal with).
        """
        text = raw_input('> ')
        return self.vector_from_text(text.decode('utf-8'))

    def vector_from_document(self, doc_id):
        """
        Get a category vector for the given known document, with TF-IDF
        applied.
        """
        terms = self.get_document_terms(doc_id)
        return self.vector_from_terms(terms)
    
    def terms_similar_to_vector(self, vec):
        """
        Take in a category vector, and returns a weighted vector of
        associated terms. You can run the `top_items()` method of this vector
        to get the most associated terms.
        """
        return divisi2.dot(self.assoc.left, vec)
    
    def domain_terms_similar_to_vector(self, vec):
        """
        Take in a category vector, and returns a weighted vector of
        associated terms, but leave out ones that only appear in common
        sense background knowledge.

        You can run the `top_items()` method of this vector
        to get the most associated terms.
        """
        # FIXME: this way of finding domain concepts is such a hack.
        mask = np.zeros((len(self.priority),), 'b')
        for i, item in enumerate(self.priority.items):
            if (self.priority.priority.has_key(i) and
                self.priority.priority[i] < 1e6):
                mask[i] = True
        return divisi2.multiply(divisi2.dot(self.assoc.left.normalize_rows(offset=1.0), vec), mask)
    
    def docs_similar_to_vector(self, vec, study='all'):
        """
        Take in a category vector, and returns a weighted vector of
        associated documents in the study. You can run the `top_items()`
        method of this vector to get the most associated documents.
        """
        return divisi2.dot(self.get_doc_matrix(study).normalize_rows(offset=1.0), vec)

    def tags_similar_to_vector(self, vec):
        """
        Take in a category vector, and returns a weighted vector of
        associated tags in the study. You can run the `top_items()`
        method of this vector to get the most associated tags.
        """
        return divisi2.dot(self.get_tag_matrix().normalize_rows(offset=1.0), vec)
    
    def show_sim(self, similarities, n=10):
        """
        Display similar terms or documents in a human-readable form
        at the command line.
        """
        for name, value in similarities.top_items(n):
            doc = self.database.get_document(name)
            if doc:
                printable_name = doc.name
            else:
                printable_name = self.database.get_term_text(name)
            print "%40s  %+4.4f" % (printable_name[:40].encode('utf-8'), value)

    def canonical_stats(self, study='all', canonicals='Canonical'):
        """
        Get the correlation/centrality stats from a study, as compared to
        the documents in another study designated 'canonical'. That study
        is probably not a real study, it's just a set of documents like it's
        always been, but it's represented the same way.

        The default canonical study is in fact the one named 'Canonical'.

        TODO: average the documents as the study is being learned, allowing
        streaming and very large studies.
        """
        stats = {'correlation': {}, 'centrality': {}}

        # calculate the rms concept-concept similarity, as a scale factor
        mean_concept = np.mean(self.assoc.left_view, axis=0)
        concept_concept = np.dot(self.assoc.left_view, mean_concept)
        baseline = np.sqrt(np.mean(concept_concept ** 2))

        # np.asarray it so that we can apply numpy functions to it safely.
        study_matrix = np.asarray(self.get_doc_matrix(study))
        canonical_matrix = self.get_doc_matrix(canonicals)

        # First, find the (presumed) normal distribution for how much the
        # documents in this study are like each other. That distribution
        # (and particularly its mean) is called "consistency".
        mean_document = np.mean(np.asarray(study_matrix), axis=0)
        mean_doc_projections = np.dot(self.assoc.left_view, mean_document)

        stats['consistency'] = _mean_var_stats(mean_doc_projections, baseline)

        # Next, find a similar distribution for how much the documents in
        # this study are like each canonical document.

        for c_row in xrange(canonical_matrix.shape[0]):
            canonical_id = canonical_matrix.row_label(c_row)
            canonical_vec = np.asarray(canonical_matrix[c_row, :])
            canonical_projections = np.dot(self.assoc.left_view,
                                           canonical_vec)
            correlation_stats = _mean_var_stats(canonical_projections,
                                                baseline)
            stats['correlation'][canonical_id] = correlation_stats

            centrality = ((correlation_stats['mean'] -
                           stats['consistency']['mean'])
                          / correlation_stats['stderr'])
            stats['centrality'][canonical_id] = centrality
        return stats

    def __repr__(self):
        return "<LuminosoModel: %r>" % self.dir

    @staticmethod
    def make(model_dir, orig_dmat, config):
        """
        Make a new LuminosoModel in the (nonexistent) directory `dir`,
        with initial half-association matrix `orig_dmat`. (A half-association
        matrix is a matrix that gives an association matrix when it is
        multiplied by its transpose.)
        """
        # Adjust the size of the matrix to match the config, if necessary.
        if os.access(model_dir, os.F_OK):
            raise StudyExists("The model directory %r already exists." % model_dir)
        rows = config['num_concepts']
        cols = config['num_axes']
        if orig_dmat.shape != (rows, cols):
            dmat = divisi2.DenseMatrix((rows, cols))
            rows_to_copy = orig_dmat.shape[0]
            if rows < rows_to_copy:
                raise ValueError("num_concepts is too small to fit the "
                                 "existing concepts.")
            cols_to_copy = min(cols, orig_dmat.shape[1])
            dmat[:rows_to_copy, :cols_to_copy] = \
              orig_dmat[:rows_to_copy, :cols_to_copy]
            dmat.row_labels = orig_dmat.row_labels
        else:
            dmat = orig_dmat
        
        # Make sure that the matrix has a PrioritySet for its row labels.
        _prioritize_labels(dmat, rows)    
        rmat = divisi2.reconstruct_symmetric(dmat)

        # Make the model directory and populate its initial files.
        os.mkdir(model_dir)
        rmat_file = model_dir + os.sep + LuminosoModel.ASSOC_FILENAME
        config_file = model_dir + os.sep + LuminosoModel.CONFIG_FILENAME
        save_pickle(rmat, rmat_file)
        save_config_file(config, config_file)

        # Now load the model from that directory and return it.
        model = LuminosoModel(model_dir)
        return model

    @staticmethod
    def make_empty(model_dir, config=None):
        """
        Make a LuminosoModel that starts from an empty matrix.
        """
        if config is None:
            config = _default_config()
        mat = divisi2.DenseMatrix((config['num_concepts'], config['num_axes']))
        model = LuminosoModel.make(model_dir, mat, config)
        return model

    @staticmethod
    def make_english(model_dir, config=None):
        """
        Make a LuminosoModel whose initial matrix contains common sense
        in English.
        """
        return LuminosoModel.make_common_sense(model_dir, 'en', config)

    @staticmethod
    def make_japanese(model_dir, config=None):
        """
        Make a LuminosoModel whose initial matrix contains common sense
        in Japanese.
        """
        return LuminosoModel.make_common_sense(model_dir, 'ja', config)

    @staticmethod
    def make_common_sense(model_dir, lang='en', config=None):
        """
        Make a LuminosoModel whose initial matrix contains common sense
        for some language.
        """
        if config is None:
            config = _default_config()
            config['reader'] = 'simplenlp.'+lang
        if os.access(model_dir, os.F_OK):
            raise StudyExists("The model directory %r already exists." % model_dir)
        LOG.info("Making common sense matrix")
        assoc = divisi2.network.conceptnet_assoc(lang)
        (mat_U, diag_S, _) = assoc.normalize_all().svd(k=100)
        rmat = divisi2.reconstruct_activation(
            mat_U, diag_S, post_normalize=True
        )
        model = LuminosoModel.make(model_dir, rmat.left, config)
        model.config['iteration'] = 1000
        return model

def _default_config():
    "The default configuration for new studies."
    config = Config()
    config['num_concepts'] = 100000
    config['num_axes'] = 100
    config['reader'] = 'simplenlp.en'
    config['iteration'] = 0
    config['sample_frac'] = 1.0
    return config

def _prioritize_labels(mat, num_concepts):
    """
    Ensure that a dense matrix has a PrioritySet for its row labels.
    """
    if not isinstance(mat.row_labels, PrioritySet):
        # turn an ordinary matrix into one that has a
        # PrioritySet for row indices
        priority = PrioritySet(num_concepts)
        if mat.row_labels is not None:
            items = mat.row_labels
            item_tuples = zip(items, [_BIG] * len(items))
            priority.load_items(item_tuples)
        mat.row_labels = priority
    return mat

def _mean_var_stats(samples, baseline=1):
    mean = np.mean(samples) / baseline
    stdev = np.std(samples) / baseline
    stderr = stdev / np.sqrt(len(samples)-0.999999)
    return {'mean': mean, 'stdev': stdev, 'stderr': stderr, 'n': len(samples)}

def convert_config(config_dict):
    """
    Convert a dictionary to a Config object.
    """
    config = Config()
    for key, value in config_dict.items():
        config[key] = value
    return config

def save_config_file(config, filename):
    """
    Save a specified config object to a file. (This can be done before the
    LuminosoModel instance exists, so it can be loaded with a valid
    config.)
    """
    if not isinstance(config, Config):
        config = convert_config(config)
    out = open(filename, 'w')
    config.save(out)
    out.close()

def run_in_directory(dir, lang='en'):
    """
    When run as a script, this module will take in a path to a Luminoso1-style
    study directory, and will save or update the Luminoso2 model in it.

    The directory should contain:
        `Canonical/` or `canonical.json`: contains canonical documents
        `Documents/`: contains documents that will be learned from
    
    Afterward, it will contain:
        `Model/`: the luminoso2 model

    If the Model/ directory exists already, it will update the existing model.
    """
    model_dir = dir+'/Model'
    docs_dir = dir+'/Documents'
    canonical_url = dir+'/Canonical'

    # test for a new canonical.json
    if os.access(dir+'/canonical.json', os.R_OK):
        canonical_url = dir+'/canonical.json'

    try:
        model = LuminosoModel.make_common_sense(model_dir, lang)
    except StudyExists:
        model = LuminosoModel(model_dir)
    
    model.add_from_url(canonical_url, 'Canonical')
    model.learn_from_url(docs_dir, 'all')
    LOG.info("Saving stats")
    model.save_canonical_stats()
    return model

def main():
    if len(sys.argv) <= 1:
        print "Usage: model.py <study dir> <language>"
    else:
        dir = sys.argv[1].rstrip('/')
        if len(sys.argv) <= 2:
            lang = 'en'
        else:
            lang = sys.argv[2]
        if lang == 'jp': # easy to confuse ja with JP
            lang = 'ja'
        run_in_directory(dir, lang)

if __name__ == '__main__':
    main()

