# coding:utf-8
import luminoso2
from luminoso2.model import LuminosoModel
import tempfile, shutil
from nose.tools import eq_
from nose.plugins.attrib import attr
import logging
logging.basicConfig()
from luminoso2.whereami import get_project_filename
TEMPDIR = None

def setup_module():
    global TEMPDIR
    TEMPDIR = tempfile.mkdtemp()

def teardown_module():
    shutil.rmtree(TEMPDIR)

@attr('ja')
def test_japanese():
    japanese = luminoso2.make_japanese(TEMPDIR+'/japanese')
    assoc1 = japanese.term_association(u'猫', u'犬')
    assoc2 = japanese.term_association(u'猫', u'好き')
    assert assoc1 > 0.3
    neko = {
        'url': u'猫.txt',
        'name': u'猫',
        'text': u'猫が好きです',
    }
    inu = {
        'url': u'犬.txt',
        'name': u'犬',
        'text': u'犬が好きです',        
    }
    japanese.add_batch([neko, inu], study=u'テスト', learn=True)
    assoc3 = japanese.term_association(u'猫', u'好き')
    assert assoc3 > assoc2

def test_small():
    model = LuminosoModel.make_empty(
        TEMPDIR + '/small',
        {
            'num_concepts': 3,
            'num_axes': 2,
            'iteration': 0,
            'reader': 'simplenlp.en'
        }
    )
    eq_(model.config['num_concepts'], 3)
    eq_(model.index_term('a', 2), 0)
    eq_(model.index_term('b', 1), 1)
    eq_(model.index_term('c', 3), 2)
    eq_(model.index_term('d', 4), 1)
    eq_(model.index_term('e', 0), 0)
    eq_(model.index_term('e', 0), 0)
    eq_(model.priority.items, ['e', 'd', 'c'])

def test_read_from_url():
    model = LuminosoModel.make_empty(
        TEMPDIR + '/testdocs',
        {
            'num_concepts': 5,
            'num_axes': 2,
            'iteration': 0,
            'reader': 'simplenlp.en'
        }
    )
    model.learn_from_url(get_project_filename('test/TestDocuments'),
                         study=u'test')
    tags = model.database.get_document_tags(
      get_project_filename('test/TestDocuments/happytest.txt')
    )
    eq_(sorted(tags), [(u'study', u'test')])
    terms = model.get_document_terms(
      get_project_filename('test/TestDocuments/happytest.txt')
    )
    eq_(sorted(map(tuple, terms)), [(u'happy', 1.0), (u'sad', -0.5)])

def test_english():
    english = luminoso2.make_english(TEMPDIR+'/english')
    assoc_before = english.term_association('cat', 'dog')
    assert assoc_before > 0.5
    doc1 = dict(url='doc1.txt', name='doc1',
                text='cats and dogs')
    doc2 = dict(url='doc2.txt', name='doc2',
                text='cats. dogs.')
    english.add_batch([doc1, doc2], learn=True)
    assoc_after = english.term_association('cat', 'dog')
    # after seeing the new documents, association should increase
    assert assoc_after > assoc_before

if __name__ == '__main__':
    import cProfile
    import simplenlp
    en = simplenlp.get_nl('en')
    en.lemma_split('test')
    en.is_stopword('test')
    setup_module()
    model = LuminosoModel.make_empty(
        TEMPDIR + '/testdocs',
        {
            'num_concepts': 5,
            'num_axes': 2,
            'iteration': 0,
            'reader': 'simplenlp.en'
        }
    )
    cProfile.run('for i in xrange(10): model.learn_from_url("TestDocuments")', sort=2)
    #model = LuminosoModel('../models/PLDBStudy_test3')
    #cProfile.run("model.learn_from_url('../models/PLDBStudy/Documents')", sort=2)

    teardown_module()
