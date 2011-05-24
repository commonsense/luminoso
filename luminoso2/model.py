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
from divisi2.ordered_set import PrioritySet
from luminoso2.svdupdate import general_svd_update
from luminoso2.term_database import TermDatabase, _BIG
from luminoso2.text_readers import get_reader, DOCUMENT, TAG
from luminoso2.document_handlers import handle_url
import os
import sys
from config import Config
import logging
import codecs
import hashlib, zlib
import random
try:
    import simplejson as json
except ImportError:
    import json
LOG = logging.getLogger(__name__)

class StudyExists(Exception):
    pass

CONNECTION_COMPRESSION_LEVEL = 2

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
    DATA_FILENAME = 'data.pkl'
    DB_FILENAME = 'terms.sqlite'
    CONNECTIONS_DIR = 'connections_cache'
    _update_k = 100

    def __init__(self, model_dir, model_data=None):
        """
        A LuminosoModel is constructed from `dir`, a path to a directory.
        This directory will contain saved versions of various matrices, as
        well as a SQLite database of terms and documents.

        model_data, if passed, is a tuple (U, Sigma) of model data. Otherwise it's loaded from a pickle.
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
        if model_data:
            self._set_data(model_data)
        else:
            self._load_data()
        self.database = TermDatabase(
          self.filename_in_dir(LuminosoModel.DB_FILENAME)
        )
        self.idf_cache = {}
        self._connections_path = self.filename_in_dir(self.CONNECTIONS_DIR)
        if not os.path.isdir(self._connections_path):
            os.mkdir(self._connections_path)
    
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
            _set_defaults(self.config)
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
        json.dump(stats.as_dict(), out, indent=2, ensure_ascii=False)
        out.close()

    def _load_data(self):
        "Load the model data from a file."
        data = load_pickle(
            self.filename_in_dir(LuminosoModel.DATA_FILENAME)
            )
        self._set_data(data)

    def _set_data(self, model_data):
        self.u, self.sigma = model_data
        self.assoc = self.spectrally_associate(self.u).normalize_rows(offset=0.001)
        self.save_data()
        self.priority = self.u.row_labels
        assert isinstance(self.priority, PrioritySet)
        self.priority.listen_for_drops(self.on_drop)

    def save_data(self):
        "Save the model data to a file."
        save_pickle((self.u, self.sigma),
                    self.filename_in_dir(LuminosoModel.DATA_FILENAME))
    
    def on_drop(self, index, key):
        """
        Handle when a key falls out of the PrioritySet.
        """
        self.u[index, :] = 0

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
        text = doc['text']
        tags = doc.get('tags', [])
        doc_terms = []
        connections = self.get_connections_for_text(text, reader_name)
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

    def _get_cache_for_text(self, text, reader_name=None):
        # Connections are cached within the model directory as
        # zlib-compressed json-encoded files named by the sha1 of
        # their utf8-encoded text contents.

        if reader_name is None:
            reader_name = self.config['reader']

        # Check if it's cached on disk yet.
        hasher = hashlib.sha1()
        hasher.update(reader_name+':')
        hasher.update(text.encode('utf8'))
        digest = hasher.hexdigest()
        filename = os.path.join(self._connections_path, digest)

        # Note that the json module encodes results as utf8 by default.
        if os.path.exists(filename):
            return json.loads(zlib.decompress(open(filename,'rb').read()))
        else:
            reader = get_reader(reader_name)
            connections = list(reader.extract_connections(text))
            term_texts = list(reader.extract_term_texts(text))
            cache = connections, term_texts
            # Atomically overwrite the file (why isn't this in the standard library??)
            # Choose a random tempfile name because another process may be trying to do the same thing.
            tmp = '%s_tmp_%d' % (filename, random.randrange(1000))
            with open(tmp, 'wb') as f:
                f.write(zlib.compress(json.dumps(cache), CONNECTION_COMPRESSION_LEVEL))
            os.rename(tmp, filename)

            return cache
    
    def get_connections_for_text(self, text, reader_name=None):
        '''
        Get connections for the given text, using the given reader.
        '''
        connections, term_texts = self._get_cache_for_text(text, reader_name)
        return connections

    def get_term_texts_for_text(self, text, reader_name=None):
        '''
        Get connections for the given text, using the given reader.
        '''
        connections, term_texts = self._get_cache_for_text(text, reader_name)
        return term_texts

    def get_document_text(self, docid):
        doc = self.database.get_document(docid)
        return doc.text

    def get_document_connections(self, docid):
        """
        Given a previously added document, get the list of connections
        produced from it.
        """
        connections = self.get_connections_for_text(self.get_document_text(docid))

        if self.config.ignore_negation:
            connections = [(abs(weight), term1, term2)
                           for (weight, term1, term2) in connections]
        return connections
    
    def get_document_terms(self, docid):
        """
        Given a previously added document, get the list of weighted terms
        that appear in it, as (term, weight) tuples.
        """
        doc = self.database.get_document(docid)
        terms = json.loads(doc.terms)
        if self.config.ignore_negation:
            terms = [(term, abs(weight)) for (term, weight) in terms]
        return terms
    
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
        for item in self.text_assoc_updates(self.get_document_text(docid)):
            yield item

    def text_assoc_updates(self, text):
        connections = self.get_connections_for_text(text)
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
        if self.config.ignore_bigrams and ' ' in term:
            return None
        index = self.priority.add(term, priority)
        return index

    def get_term_idf(self, term):
        if term in self.idf_cache:
            return self.idf_cache[term]
        else:
            idf = self.database.term_idf(term)
            self.idf_cache[term] = idf
            return idf

    def learn_from(self, url, study=None, batch_size=1000):
        """
        Given a URL or file path that points to a collection of documents,
        learn from all of those documents. They may also be added to a
        study at the same time.
        """
        self.add_from_url(url, study, learn=True, batch_size=batch_size)
        self.save_data()
        self.save_config()
    # compatibility with previous docs
    learn_from_url = learn_from

    def add_from(self, url, study=None, learn=False, batch_size=1000):
        """
        Given a URL or file path that points to a collection of
        documents, add all the documents to the database. If `learn`
        is False, the concept model will not change; otherwise, this
        implements `learn_from`.

        This is the main loop that one should use to train a model with a
        batch of documents.
        """
        stream = handle_url(url, sample_frac=self.config.get('sample_frac', 1.0))
        batch = []
        while True:
            if len(batch) == batch_size:
                self.add_batch(batch, study, learn=learn)
                batch[:] = []
            try:
                batch.append(next(stream))
            except StopIteration:
                self.add_batch(batch, study, learn=learn)
                batch[:] = []
                break
        
        # If this was a study, make a document matrix for it.
        if study is not None:
            LOG.info("Making document matrix for %r" % study)
            self.update_doc_matrix(study)
        #LOG.info("Updating tag matrix")
        #self.update_tag_matrix()

    add_from_url = add_from

    def _update_matrix(self, batch_texts):
        nterms, k = self.u.shape
        values, rows, cols = [], [], []
        for text in batch_texts:
            for weight, term1, term2 in self.text_assoc_updates(text):
                try:
                    row = self.priority.index(term1)
                    col = self.priority.index(term2)
                except KeyError:
                    # Terms fell out of the priorityset.
                    continue
                values.append(weight)
                rows.append(row)
                cols.append(col)
                values.append(weight)
                rows.append(col)
                cols.append(row)
        return divisi2.SparseMatrix.from_lists(values, rows, cols,
                                               nrows=nterms, ncols=nterms)

    def add_batch(self, batch, study=None, learn=False):
        """
        Add a batch of documents. If `study` is specified, add them to
        that study. If `learn` is True, update the association model
        to account for those new documents.
        """
        fulltext_cache = {}

        # First pass: add documents to the term database, and meanwhile
        # collect full texts and tags.
        for doc in batch:
            docid = self.add_document(doc)
            for term, fulltext in self.get_term_texts_for_text(doc['text']):
                fulltext_cache[term] = fulltext
            if study is not None:
                self.database.set_tag_on_document(docid, 'study', study)

        LOG.info("Committing documents to the database")
        self.database.commit()

        LOG.info("Collecting relevant terms")
        self.database.update_relevance()

        if learn:
            # Second pass (optional): update the projection based on
            # the word associations we discover.
            update_matrix = self._update_matrix([doc['text'] for doc in batch])

            # Compute the SVD of this set of updates.
            LOG.info('Analyzing new associations...')
            update_u, update_s, update_v = update_matrix.svd(k=self._update_k)

            # Strip to bare NumPy
            U = np.asarray(self.u)
            A = np.multiply(np.asarray(update_u), np.asarray(update_s))
            B = np.asarray(update_v)

            # Apply SVD update.
            LOG.info('Updating projection with new associations...')
            new_u, new_sigma, _ = general_svd_update(U, self.sigma, U, A, B, U.shape[1])

            # Update the stored model.
            self.u = divisi2.DenseMatrix(new_u, self.priority, None)
            self.sigma = new_sigma
            self.assoc = self.spectrally_associate(self.u).normalize_rows(offset=0.001)

        # Finally, update the full texts of the terms we saw.
        LOG.info("Updating full texts")
        for term, fulltext in fulltext_cache.items():
            self.database.set_term_text(term, fulltext)
        self.database.commit()

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
        Collect the tags in a model, and make a dense matrix
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
            term_mat[i, :] = self.u[self.priority.index(term)]

        mat = term_mat.concatenate(self.get_doc_matrix(study_name))
        write_packed(mat, self.filename_in_dir(study_name), denormalize)

    def spectrally_associate(self, vec):
        """
        Multiply vec by e^(sqrt(S)/2).
        """
        return vec.multiply(np.exp(np.sqrt(self.sigma)/2))

    def vector_from_tag(self, tag, value, input_only=False):
        vecs = [self.vector_from_document(doc, input_only=input_only)
                for doc in self.database.documents_with_tag_value(tag, value)]
        return sum(vecs)

    def vector_from_terms(self, terms, input_only=False):
        """
        Get a category vector representing the given set of weighted terms,
        expressed as (term, weight) tuples. This will apply TF-IDF weighting.
        """
        total_weight = 0.000001
        vec = divisi2.DenseVector(
            np.zeros((len(self.priority),)),
            labels=self.priority
        )
        for _, weight in terms:
            total_weight += abs(weight)

        for term, weight in terms:
            if isinstance(term, basestring) and term in self.priority:
                index = self.priority.index(term)
                tfidf_weight = weight * self.get_term_idf(term)
                assoc_weight = np.linalg.norm(self.u.row_named(term))
                vec[index] = tfidf_weight / total_weight
        if input_only:
            return vec
        return divisi2.dot(vec, self.assoc).normalize()

    def vector_from_text(self, text, reader_name=None, input_only=False):
        """
        Get a category vector in this model representing the given text,
        with TF-IDF applied.
        """
        terms = []
        for weight, term1, term2 in self.get_connections_for_text(text, reader_name):
            if term1 == DOCUMENT:
                terms.append((term2, abs(weight)))
        return self.vector_from_terms(terms, input_only=input_only)
    
    def vector_from_input(self):
        """
        Get a category vector representing the given line of input from
        standard in (which is a good way to enter Unicode that iPython can't
        deal with).
        """
        text = raw_input('> ')
        return self.vector_from_text(text.decode('utf-8'))

    def vector_from_document(self, doc_id, input_only=False):
        """
        Get a category vector for the given known document, with TF-IDF
        applied.
        """
        terms = self.get_document_terms(doc_id)
        return self.vector_from_terms(terms, input_only=input_only)
    
    def terms_similar_to_vector(self, vec):
        """
        Take in a category vector, and returns a weighted vector of
        associated terms. You can run the `top_items()` method of this vector
        to get the most associated terms.
        """
        return divisi2.dot(self.assoc, vec)

    def term_association(self, term1, term2):
        return divisi2.dot(self.assoc.row_named(term1),
                           self.assoc.row_named(term2))
    
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
        return divisi2.multiply(self.terms_similar_to_vector(vec), mask)
    
    def docs_similar_to_vector(self, vec, study='all'):
        """
        Take in a category vector, and returns a weighted vector of
        associated documents in the study. You can run the `top_items()`
        method of this vector to get the most associated documents.
        """
        return divisi2.dot(self.get_doc_matrix(study), vec)

    def tags_similar_to_vector(self, vec):
        """
        Take in a category vector, and returns a weighted vector of
        associated tags in the study. You can run the `top_items()`
        method of this vector to get the most associated tags.
        """
        return divisi2.dot(self.get_tag_matrix(), vec)
    
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
        study_matrix = self.get_doc_matrix(study)
        canonical_matrix = self.get_doc_matrix(canonicals)
        return CanonicalStats(self.assoc, study_matrix, canonical_matrix)

    def __repr__(self):
        return "<LuminosoModel: %r>" % self.dir

    @staticmethod
    def make(model_dir, model_data, config):
        """
        Make a new LuminosoModel in the (nonexistent) directory `dir`,
        with initial data `model_data`.
        """
        # Adjust the size of the matrix to match the config, if necessary.
        if os.access(model_dir, os.F_OK):
            raise StudyExists("The model directory %r already exists." % model_dir)
        rows = config['num_concepts']
        cols = config['num_axes']
        U, Sigma = model_data
        if U.shape != (rows, cols):
            dmat = divisi2.DenseMatrix((rows, cols))
            rows_to_copy = U.shape[0]
            if rows < rows_to_copy:
                raise ValueError("num_concepts is too small to fit the "
                                 "existing concepts.")
            cols_to_copy = min(cols, U.shape[1])
            dmat[:rows_to_copy, :cols_to_copy] = \
              U[:rows_to_copy, :cols_to_copy]
            dmat.row_labels = U.row_labels
            Sigma = Sigma[:cols_to_copy]
        else:
            dmat = U
        
        # Make sure that the matrix has a PrioritySet for its row labels.
        _prioritize_labels(dmat, rows)    

        # Make the model directory and populate its initial files.
        os.mkdir(model_dir)
        config_file = model_dir + os.sep + LuminosoModel.CONFIG_FILENAME
        save_config_file(config, config_file)

        # Now load the model from that directory and return it.
        model = LuminosoModel(model_dir, model_data=(dmat, Sigma))
        return model

    @staticmethod
    def make_empty(model_dir, config=None):
        """
        Make a LuminosoModel that starts from an empty matrix.
        """
        if config is None:
            config = Config()
            _set_defaults(config)
        mat = divisi2.DenseMatrix((config['num_concepts'], config['num_axes']))
        diag_S = np.zeros(config['num_axes'])
        model = LuminosoModel.make(model_dir, (mat, diag_S), config)
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
            config = Config()
            _set_defaults(config)
            config['reader'] = 'simplenlp.'+lang
        if os.access(model_dir, os.F_OK):
            raise StudyExists("The model directory %r already exists." % model_dir)
        LOG.info("Making common sense matrix")
        assoc = divisi2.network.conceptnet_assoc(lang)
        (mat_U, diag_S, _) = assoc.normalize_all().svd(k=100)
        
        model = LuminosoModel.make(model_dir, (mat_U, diag_S), config)
        model.config['iteration'] = 1000
        return model

    def recompute(self, background_data, newconfig=None, partial=False):
        """
        Re-learn a model with new parameters or new background knowledge,
        given its existing term database.

        TODO: un-duplicate code between this and .make
        """
        config = self.config
        if newconfig is not None:
            for key, value in newconfig.items():
                config[key] = value

        rows = config['num_concepts']
        cols = config['num_axes']
        blend_factor = config.get('blend_factor', 0.5)
        assoc_power = config.get('assoc_power', 1.0)

        if background_data is None:
            newU = divisi2.DenseMatrix((rows, cols))
            newSigma = np.zeros((cols,))
            newU.row_labels = PrioritySet(rows)
        else:
            U, Sigma, _ = background_data.svd(k=cols)

            newU = divisi2.DenseMatrix((rows, cols))
            rows_to_copy = U.shape[0]
            if rows < rows_to_copy:
                raise ValueError("num_concepts is too small to fit the "
                                 "existing concepts.")
            cols_to_copy = min(cols, U.shape[1])
            newU[:rows_to_copy, :cols_to_copy] = \
              U[:rows_to_copy, :cols_to_copy]
            newU.row_labels = U.row_labels
            newSigma = Sigma[:cols_to_copy]
            
        # Make sure that the matrix has a PrioritySet for its row labels.
        _prioritize_labels(newU, rows)

        from luminoso2.term_database import Term
        from sqlalchemy.sql.expression import desc
        query = self.database.sql_session.query(Term)\
                   .filter(Term.term != '*')
        if self.config.ignore_bigrams:
            query = query.filter(~Term.term.like('% %'))
        query = query.order_by(desc(Term.relevance))

        for termobj in query:
            newU.row_labels.add(termobj.term, termobj.relevance)
            if len(newU.row_labels.items) == rows:
                break
        
        self.priority = newU.row_labels

        if partial:
            # set partial=True when you already have the unblended.svd file
            # and you just want to blend it with something different
            combined_update = divisi2.load(self.filename_in_dir('unblended.svd'))
        else:
            LOG.info('computing update svds')
            updates = compute_update_svds(self)
            LOG.info('reducing update svds')
            combined_update = reduce_update_svds(updates)
            LOG.info('saving reduced matrix')
            divisi2.save(combined_update, self.filename_in_dir('unblended.svd'))
        
        LOG.info('computing blend')
        # rescale to perform rough blending
        combinedU, combinedS, combinedV = combined_update
        combinedS = combinedS / combinedS[0] * (1.0-blend_factor) * assoc_power
        newSigma = newSigma / newSigma[0] * blend_factor * assoc_power
        
        if background_data is None:
            blendedU, blendedS = combinedU, combinedS
        else:
            final_updates = [(combinedU, combinedS, combinedU), (newU, newSigma, newU)]
            LOG.info('reducing blend')
            blendedU, blendedS, blendedV = reduce_update_svds(final_updates)
        
        self.u = divisi2.DenseMatrix(
            blendedU[:, :cols],
            row_labels=self.priority
        )
        self.sigma = blendedS[:cols]
        self.assoc = self.spectrally_associate(self.u).normalize_rows(offset=0.001)

        LOG.info('saving final results')
        # set my U and S accordingly
        self.save_config()
        self.save_data()

class CanonicalStats(object):
    def __init__(self, assoc, study_matrix, canonical_matrix):
        canonical_matrix = canonical_matrix.normalize_rows()
        self.canonical_similarity = canonical_matrix.dot(canonical_matrix.T)
        self.correlation = {}
        self.centrality = {}

        # np.asarray it so that we can apply numpy functions to it safely.
        assoc = np.asarray(assoc)
        self.doc_labels = study_matrix.row_labels
        study_matrix = np.asarray(study_matrix)

        # calculate the rms concept-concept similarity, as a scale factor
        mean_concept = np.mean(assoc, axis=0)
        concept_concept = np.dot(assoc, mean_concept)
        #self.baseline = np.sqrt(np.mean(concept_concept ** 2))
        self.baseline = 1.0 # FIXME: should be rms(cosine similarity between concepts)

        # First, find the (presumed) normal distribution for how much the
        # documents in this study are like each other. That distribution
        # (and particularly its mean) is called "consistency".
        mean_document = np.mean(study_matrix, axis=0)
        mean_doc_projections = np.dot(study_matrix, mean_document)

        self.consistency = _mean_var_stats(mean_doc_projections, self.baseline)
        #assert -1 <= self.consistency['mean'] <= 1

        # Next, find a similar distribution for how much the documents in
        # this study are like each canonical document.
        self.canonical_projections = {}
        for c_row, canonical_id in enumerate(canonical_matrix.row_labels):
            canonical_vec = np.asarray(canonical_matrix[c_row, :])
            canonical_projections = np.dot(study_matrix,
                                           canonical_vec)
            self.canonical_projections[canonical_id] = canonical_projections
            correlation_stats = _mean_var_stats(canonical_projections,
                                                self.baseline)
            self.correlation[canonical_id] = correlation_stats

            centrality = ((correlation_stats['mean'] -
                           self.consistency['mean'])
                          / correlation_stats['stderr'])
            self.centrality[canonical_id] = centrality

    def as_dict(self):
        return dict(consistency=self.consistency,
                    correlation=self.correlation,
                    centrality=self.centrality)

    def __unicode__(self):
        return unicode(self.as_dict)


_default_config = dict(
    num_concepts = 100000,
    num_axes = 100,
    reader = 'simplenlp.en',
    iteration = 0,
    sample_frac = 1.0,
    blend_factor = 1.0,
    ignore_negation = False,
    ignore_bigrams = True,
)

def _set_defaults(config):
    "The default configuration for new studies."
    for k, v in _default_config.iteritems():
        if k not in config:
            config[k] = v

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

SENTINEL = 'STOP'
def _matrix_worker(input, output, model, k):
    from multiprocessing import current_process
    for batch in iter(input.get, SENTINEL):
        proc = current_process()
        LOG.info('worker %s (pid=%d) computing update matrix', proc.name, proc.pid)
        update_mat = model._update_matrix(batch)
        LOG.info('worker %s (pid=%d) computing update SVD', proc.name, proc.pid)
        svd = update_mat.svd(k=k)
        output.put(svd)

k = 200 # FIXME
num_workers = 1

def compute_update_svds(model, batch_size=1000):
    from multiprocessing import Process, Queue
    from csc_utils import in_groups_of

    task_queue = Queue()
    done_queue = Queue()

    LOG.info('Getting docs list...')
    docs = model.database.list_documents()
    LOG.info('Computing update matrices and SVDs...')
    LOG.info('- starting workers')
    for i in range(num_workers):
        Process(target=_matrix_worker, args=(task_queue, done_queue, model, k)).start()
    mat_count = 0
    LOG.info('- getting document texts')
    for batch in in_groups_of(batch_size, docs):
        text_batch = [model.get_document_text(docid) for docid in batch]
        task_queue.put(text_batch)
        mat_count += 1
    # Tell workers they can stop when done.
    for i in range(num_workers): task_queue.put(SENTINEL)
    LOG.info('Put %d batches of document texts, now awaiting results', mat_count)
    update_svds = []
    for i in range(mat_count):
        update_svds.append(done_queue.get())
        LOG.info('Got %d of %d SVDs', len(update_svds), mat_count)
    LOG.info('Done!')
    return update_svds

def reduce_update_svds(update_svds):
    from csc_utils import in_groups_of
    # Reduce, in a tree.
    todo = update_svds[:]
    while len(todo) > 1:
        pairs = list(in_groups_of(2, todo))
        LOG.info('Blending %d pairs', len(pairs))
        todo = []
        for pair in pairs:
            if len(pair) == 1:
                # Last one.
                todo.append(pair[0])
                continue

            # Blend the pair of SVDs.
            (u_a, s_a, v_a), (u_b, s_b, v_b) = pair
            
            # Strip to bare NumPy
            U = np.asarray(u_a)
            S = s_a
            A = np.multiply(np.asarray(u_b), np.asarray(s_b))
            B = np.asarray(v_b)

            # Compute updated SVD.
            new_u, new_sigma, _ = general_svd_update(U, S, U, A, B, k)

            todo.append((new_u, new_sigma, new_u))

    return todo[0]
    
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

