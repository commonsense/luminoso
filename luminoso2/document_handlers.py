"""
A document handler is the step before a TextReader: it takes in some source of
raw data, and yields dictionaries representing individual documents. A
TextReader can then scan these for terms and tags.

Document handlers are not currently pluggable; they are top-level functions.
The job of `handle_url` is to dispatch to the appropriate function.
"""
import chardet
import os
import sys
import random
try:
    import json
except ImportError:
    import simplejson as json
import logging
LOG = logging.getLogger(__name__)

def ensure_unicode(text):
    """
    Take in something. Make it Unicode.
    """
    if isinstance(text, unicode):
        return text
    elif isinstance(text, str):
        return text.decode('utf-8', 'replace')
    else:
        return unicode(text)

def get_file_text(filename):
    """
    Get the contents of a text file as unicode; guess its encoding.
    """
    f = open(filename, 'rb')
    rawtext = f.read()
    f.close()
    encoding = chardet.detect(rawtext)['encoding']
    # When the encoding is None, assume it's utf-8 because that will probably work.
    if encoding is None: encoding = 'utf-8'
    try:
        return rawtext.decode(encoding, 'replace')
    except LookupError:
        raise UnicodeDecodeError(
          'File %r uses an unimplemented encoding %r' % (filename, encoding))

def handle_text_file(filename, name=None):
    """
    Handle a file that we believe to contain plain text, in some reasonable
    encoding.

    May raise a UnicodeDecodeError if the file uses a codec that Python does
    not yet implement, such as EUC-TW.
    """
    text = get_file_text(filename)
    for result in handle_text(text, filename, name):
        yield result

def handle_text(text, url, name=None):
    """
    Given plain text content, return it as a document dictionary.
    """
    text = ensure_unicode(text)
    if name is None:
        # get it from the beginning of the text
        name = text.split('\n')[0][:40]
    else:
        name = ensure_unicode(name)
    yield {u'name': name, u'url': url, u'text': ensure_unicode(text)}

def _check_document(document):
    """
    Upon retrieving a document dictionary, see if it fits the format Luminoso
    expects.
    """
    return (isinstance(document, dict) and 
            u'name' in document and u'text' in document)

def handle_json_file(filename, name=None):
    """
    Dig into a JSON file, and find documents containing "name" and "text"
    entries.
    """
    stream = open(filename)
    obj = json.load(stream)
    stream.close()
    for result in handle_json_obj(obj, filename, name):
        yield result

def handle_json_obj(obj, url, name=None):
    """
    Handle a JSON object, which is either a document itself or may contain
    a number of subdocuments.

    The `url` parameter should contain a filename or URL to ensure that
    document names are not completely ambiguous.
    """
    # TODO: split these cases into separate functions that are reusable
    # (pylint has a valid complaint!)
    if isinstance(obj, basestring):
        doc = {
            u'url': url,
            u'name': name or os.path.basename(url),
            u'text': obj
        }
        yield doc
    elif isinstance(obj, list):
        for document in obj:
            # discard the name; it won't be unique
            for result in handle_json_obj(document, url):
                yield result
    elif isinstance(obj, dict):
        baseurl = os.path.dirname(url)
        if u'text' in obj:
            # this is a single document on its own
            obj[u'url'] = url + u'#' + obj.get(u'name', name)
            obj.setdefault(u'name', name)
            yield obj
        elif u'url' in obj:
            for result in handle_url(baseurl + os.path.sep + obj[u'url'],
                                     obj.get(u'name', name)):
                yield result
        else:
            # assume it's a dictionary mapping name -> document
            for newname, document in obj.iteritems():
                fullurl = url + '#' + newname
                for result in handle_json_obj(document, fullurl, newname):
                    yield result
    else:
        LOG.warn("could not find a valid document in %r#%r" % (url, name))

def handle_directory(dirname, sample_frac=1.0):
    """
    Handle a directory and get a stream of documents out of it.

    See handle_url for documentation of sample_frac.
    """
    if not isinstance(dirname, unicode):
        # Ensure that paths are unicode.
        dirname = dirname.decode(sys.getfilesystemencoding())
    for filename in os.listdir(dirname):
        if (not filename.startswith(u'.') and
            (sample_frac >= 1.0 or random.random() < sample_frac)):
            for result in handle_url(dirname+os.sep+filename):
                yield result

def handle_url(url, name=None, sample_frac=1.0):
    """
    Handle a file specified by its URL (by default, a local file).

    If `sample_frac` is given, it gives the fraction of documents to
    consider. Whether to keep a particular document is chosen by a
    flip (Beta distribution) with this parameter, so the actual number
    of documents used will follow Binomial(num_docs, sample_frac).

    TODO: handle schemas that aren't local files.
    """
    if os.access(url, os.F_OK):
        if os.path.isdir(url): # it's a directory
            for result in handle_directory(url, sample_frac=sample_frac):
                yield result
        elif url.endswith(u'.json') or url.endswith(u'.js'):
            for result in handle_json_file(url, name):
                yield result
        else:
            # assume text file
            for result in handle_text_file(url, name):
                yield result
    else:
        if name:
            LOG.warn("could not open %r#%r" % (url, name))
        else:
            LOG.warn("could not open %r" % url)
