'''
Prepares a set of documents for a map-reduce job.
'''

import os
from mrjob.protocol import JSONValueProtocol
from luminoso2.document_handlers import handle_url
from csc_utils import in_groups_of

def encode_document(doc):
    return JSONValueProtocol.write(None, doc) + '\n'

def encode_documents(url, dest_dir, batch_size=100, start_at=0):
    for idx, batch in enumerate(in_groups_of(batch_size, handle_url(url)), start_at):
        with open(os.path.join(dest_dir, 'docs_%d' % idx), 'wb') as outfile:
            for doc in batch:
                outfile.write(encode_document(doc))

if __name__ == '__main__':
    import sys
    url, dest_dir = sys.argv[1:]
    encode_documents(url, dest_dir)
