from luminoso.study import extract_concepts_with_negation, Study, Document, get_picklecached_thing
from nose.tools import *

def create_doc():
    global doc
    doc = Document('obvious', demo_text)

demo_text = "The customer really doesn't like bad service." 
demo_pos = sorted(['customer', 'customer really', 'really'])
demo_neg = sorted(['like', 'bad', 'service', 'like bad', 'bad service'])
@with_setup(create_doc)
def test_document_extract_concepts():
    concepts = doc.extract_concepts_with_negation()

    pos = [concept for concept, polarity in concepts if polarity == 1]
    neg = [concept for concept, polarity in concepts if polarity == -1]
    eq_(len(pos)+len(neg), len(concepts))

    eq_(sorted(pos), demo_pos)
    eq_(sorted(neg), demo_neg)

@with_setup(create_doc)
def test_study_no_cnet():
    study = Study('test', [doc], [])
    res = study.analyze()
    
# FIXME: temporary
CONCEPTNET_PATH = '/Users/kcarnold/code/luminoso/ThaiFoodStudy/Matrices/conceptnet.pickle'
def test_study():
    doc = Document('obvious', demo_text)
    cnet = get_picklecached_thing(CONCEPTNET_PATH).normalized(mode=[0,1]).bake()
    study = Study('test', [doc], [cnet])
    study.analyze()
    
