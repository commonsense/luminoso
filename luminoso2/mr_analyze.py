'''
Extract connections from a set of documents.
'''
import random
from mrjob.job import MRJob
from luminoso2.text_readers import get_reader, DOCUMENT, TAG

class MRAnalyzer(MRJob):
    DEFAULT_INPUT_PROTOCOL = 'json_value'

    def steps(self):
        '''
        1. Extract connections from documents -> (term, [(doc, weight), ...])
        2. Compute count, distinct_docs, 
        3. PROFIT!
        '''

    def configure_options(self):
        super(MRDocumentReader, self).configure_options()
        self.add_passthrough_option(
            '--reader', dest='reader', default='simplenlp.en',
            help="Name of the reader to use.")
        self.add_passthrough_option(
            '--num-batches', dest='num_batches', default=10,
            help="Number of batches to split up the input into.")

    def random_batch(self):
        return random.randrange(10)
    
    def mapper(self, _, doc):
        '''Extract a list of connections from the given document.

        Takes: an ignored key, and the document dictionary (see document_handlers).
        Yields: the document URL as key; dict(reader='', connections=[], tags=[]).'''
        reader = self.options.reader
        tags = doc.get('tags', [])
        connections, tags = [], []
        for weight, term1, term2 in get_reader(reader).extract_connections(doc['text']):
            if term1 == DOCUMENT and isinstance(term2, tuple) and term2[0] == TAG:
                tags.append(term2[1:])
            else:
                connections.append((weight, term1, term2))

        yield doc['url'], dict(reader=reader, connections=connections, tags=tags)


if __name__ == '__main__':
    MRDocumentReader.run()

