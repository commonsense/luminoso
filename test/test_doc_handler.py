from luminoso2.document_handlers import *
from luminoso2.whereami import get_project_filename
import nose

def test_basics():
    assert ensure_unicode('foo') == u'foo'
    assert ensure_unicode(chr(128)) == u'\ufffd'
    assert isinstance(ensure_unicode('foo'), unicode)
    assert ensure_unicode(None) == u'None'

def test_directory():
    test_dir = get_project_filename('test/TestDocuments')
    good = [{u'name': u'i am happy and not sad',
             u'text': u'i am happy and not sad\n',
             u'url': test_dir + u'/happytest.txt'},
            {u'name': u'string.json',
             u'text': u"\u00a1I'm a string!",
             u'url': test_dir + u'/string.json'
            },
            {u'name': u'Entry 1',
             u'text': u'one two three',
             u'url': test_dir + u'/list.json#Entry 1'},
            {u'name': u'Entry 2',
             u'text': u'two three four',
             u'url': test_dir + u'/list.json#Entry 2'},
            {u'name': u'Entry 3',
             u'text': u'three four five\n',
             u'url': test_dir + u'/../MoreTestDocuments/relative.txt'},
            {u'name': u'Entry 4',
             u'text': u"I'm somewhere else",
             u'url': test_dir + u'/../MoreTestDocuments/relative2.json'},
            {u'name': u'JSON test',
             u'text': u'one two three',
             u'url': test_dir + u'/simple.json#JSON test'},
            {u'name': u'unicode test',
             u'text': u'\u4e00\u4e8c\u4e09\u56db\u4e94',
             u'url': test_dir + u'/unicode.json#unicode test'},
            {u'name': u'\u4e94\u56db\u4e09\u4e8c\u4e00',
             u'text': u'\u4e94\u56db\u4e09\u4e8c\u4e00\n',
             u'url': test_dir + u'/unicode.txt'},
            {u'name': u'small entry',
             u'text': u'small',
             u'url': test_dir + u'/list.json#small entry'},
            {u'name': u'another entry',
             u'text': u'another',
             u'url': test_dir + u'/list.json#another entry'}]
    got = handle_url(get_project_filename('test/TestDocuments'))
    nose.tools.assert_equal(sorted(list(good)), sorted(list(got)))