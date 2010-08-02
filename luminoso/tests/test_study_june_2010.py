from luminoso.study import *
from csc.divisi2.sparse import SparseMatrix
import unittest

'''
This is a unit test for study.py
'''

class TestStudy(unittest.TestCase):

    def setUp(self):        
        self.empty_doc = Document('empty document', '')
        self.doc = Document('test document', "#posTag #-negTag The boy doesn't like tests. I don't like tests either.")

        #NOTE: The path to the directory used to create the StudyDirectory() is to be changed as needed.
        #This test was created with the ThaiFoodStudy as it was on June 16, 2010. 
        self.study = StudyDirectory('C:/Users/Rafael/Desktop/luminoso/ThaiFoodStudy')


    '''
    Test function that checks functionality of the extract_concepts_from_words() method.
    The tag used as a default for extract_concepts_from_words() is '#5star'. Any word
    containing this tag should not be modified.
    '''
    def test_tagged_words(self):
        #First extract words from the sentences the document contains, then check the results.
        words = []
        for sentence in self.doc.get_sentences():
            for word in sentence:
                words.append(word)

        #Extract concepts.
        concepts = extract_concepts_from_words(words)

        #Check correctness of tagged words. Tagged words should not be modified.
        self.assertTrue( ('#posTag', 1) in concepts)
        self.assertTrue( ('#-negTag', -1) in concepts)
        
    '''
    Test function that checks functionality of the Document class by separating concepts by polarity
    and then making sure that the sum of positive, negative and tagged concepts add up to the total
    number of concepts extracted from the document.
    '''
    def test_doc(self):
        #Extract concepts from the document.
        concepts = self.doc.extract_concepts_with_negation()
        empty_concepts = self.empty_doc.extract_concepts_with_negation()

        #Separate pos, tag and neg concepts.
        pos = [concept for concept, polarity in concepts if polarity == 1]
        neg = [concept for concept, polarity in concepts if polarity == -1]
        
        #Check correctness. 
        self.assertTrue( (len(pos) + len(neg)) == len(concepts))
        self.assertTrue( 0 == len(empty_concepts))

    '''
    Test the different functions in StudyDirectory.
    '''
    def test_study_directory(self):
        #canonical and docuements in ThaiFoodStudy as of june 16, 2010. These may change.
        canonical_in_ThaiFoodStudy = ['canonical_chinese.txt', 'canonical_thai.txt', 'good_review.txt']
        documents_in_ThaiFoodStudy = ['alicez.txt','annmarief.txt','catl.txt','christiang.txt','danc.txt',
                                      'daveh.txt','DianeS.txt','ejc.txt','frankl.txt','georget.txt','geraldinek.txt','id.txt',
                                      'jessicar.txt','johannaw.txt','js.txt','kellyz.txt','kevint.txt','laurenk.txt',
                                      'manifreds.txt','melissaw.txt','richardr.txt','ronu.txt','ruthp.txt','sandrac.txt',
                                      'shannond.txt','sos.txt','thomasc.txt','timg.txt','tonys.txt','tracyb.txt','yanz.txt']

        #There is only 1 matrix instance in the ThaiFoodStudy.
        self.assertTrue( len(self.study.get_matrices()) == 1)

        #Check that the canonicals and documents extracted from the study match the ones copied in
        #the list created above.
        for can in self.study.get_canonical_documents():
            self.assertTrue( can.name in canonical_in_ThaiFoodStudy)

        for doc in self.study.get_documents():
            self.assertTrue( doc.name in documents_in_ThaiFoodStudy)

        existing = self.study.get_existing_analysis()
        analyzed = self.study.analyze()

        #Assert that the study results we created above, namely existing and analyzed, contain
        #documents in the desired format and are of StudyResult.
        self.assertTrue( existing.docs.__class__ is SparseMatrix)
        self.assertTrue( analyzed.docs.__class__ is SparseMatrix)

        self.assertTrue( existing.__class__ is StudyResults)
        self.assertTrue( analyzed.__class__ is StudyResults)
        

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestStudy)
    unittest.TextTestRunner(verbosity=2).run(suite)
