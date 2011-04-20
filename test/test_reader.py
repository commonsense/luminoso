# coding:utf-8
from luminoso2.text_readers import get_reader, DOCUMENT
from nose.tools import assert_equal, raises

def test_simple():
    reader = get_reader('simplenlp.en')
    tokenized = list(reader.extract_tokens('one two three four'))
    assert_equal(tokenized, ['two', 'three', 'four'])

def test_tagged():
    reader = get_reader('simplenlp.en')
    tokenized = list(reader.extract_tokens('one two three #four'))
    assert_equal(tokenized, ['two', 'three', '#four'])

@raises(KeyError)
def test_no_reader():
    get_reader('simplenlp.foo')

def test_japanese():
    reader = get_reader('simplenlp.ja')
    connections = list(reader.extract_connections(
        u'私は日本語をあんまり出来ません。困りましたね。'
    ))
    pos_doc_terms = [term2 for weight, term1, term2 in connections
                           if term1 == DOCUMENT and weight > 0]
    neg_doc_terms = [term2 for weight, term1, term2 in connections
                           if term1 == DOCUMENT and weight < 0]
    
    assert u'出来る' in neg_doc_terms
    assert u'日本語' in neg_doc_terms
    assert u'日本語 出来る' in neg_doc_terms
    assert u'困る' in pos_doc_terms