'''
Extract connections from a set of documents.
'''
import random
from mrjob.job import MRJob
from luminoso2.text_readers import get_reader, DOCUMENT, TAG

class ConnectionExtractor(MRJob):
    DEFAULT_INPUT_PROTOCOL = 'json_value'

    def steps(self):
        return [self.mr(mapper=self.extract_connections)]

    def configure_options(self):
        super(ConnectionExtractor, self).configure_options()
        self.add_passthrough_option(
            '--reader', dest='reader', default='simplenlp.en',
            help="Name of the reader to use.")

    def extract_connections(self, _, doc):
        reader = self.options.reader
        url = doc['url']
        tags = doc.get('tags', [])
        for item in get_reader(reader).extract_connections(doc['text']):
            yield url, item
        # weight, term1, term2
        #     if term1 == DOCUMENT:
        #         yield term2, (url,)
        #     yield term1, (url, term2, weight)
        #     yield term2, (url, term1, weight)

    # def count_term_use(self, term, values):
    #     total_weight = 0.0
    #     docs = set()
        
    #     other_terms = defaultdict(float)
    #     for url, other_term, weight in values:
    #         docs.add(url)
    #         total_weight += abs(weight)
    #         other_terms[other_term] += weight
    #     yield term, (total_weight, len(docs))

    # def compute_term_stats(self, key, values):
    #     key_type, item = key
    #     if key_type == KEYTYPE_UNIGRAM:
    #         yield term_count
    #     elif key_type == KEYTYPE_BIGRAM:
    #         return bigram_likelihood_ratio(
    #             self.count_term(term),
    #             self.count_term(words[0]),
    #             self.count_term(words[1]),
    #             self.count_term(ANY)
    #         )


if __name__ == '__main__':
    ConnectionExtractor.run()

