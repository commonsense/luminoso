import luminoso2
from luminoso2.model import LuminosoModel
import tempfile, shutil
from nose.tools import assert_equal

TEMPDIR = None

def setup_module():
    global TEMPDIR
    TEMPDIR = tempfile.mkdtemp()

def teardown_module():
    shutil.rmtree(TEMPDIR)

def test_documents():
    themodel = LuminosoModel.make_empty(TEMPDIR+'/test_documents')
    #themodel.add_document(dict(url='#test1', name='test one',
    #                           text='one two three'))
    themodel.add_document(dict(url='#test1', name='test one',
                               text='two three'))
    assert_equal(themodel.database.count_term('*'), 3)
    assert_equal(themodel.database.count_documents(), 1)
    themodel.add_document(dict(url='#test2', name='test two',
                               text='two three four'))
    assert_equal(themodel.database.count_documents(), 2)
    assert themodel.database.get_document('#test1')
    assert themodel.database.get_document('#test2')
    assert_equal(themodel.database.count_term('two'), 2)
    assert_equal(themodel.database.count_term('two three'), 2)
    assert_equal(themodel.database.count_term('three four'), 1)
    assert_equal(themodel.database.count_term('*'), 8)

def test_tags():
    themodel = LuminosoModel.make_empty(TEMPDIR+'/test_tags')
    themodel.add_document(dict(url='#test1', name='test one',
                               text='#foo +bar -baz #quux=4'))
    themodel.add_document(dict(url='#test2', name='test two',
                               text='foo',
                               tags=[('foo', None), ('bar', True),
                                     ('baz', False), ('quux', 4)]))
    assert_equal(sorted(themodel.database.get_document_tags('#test1')),
                 sorted(themodel.database.get_document_tags('#test2')))
    
