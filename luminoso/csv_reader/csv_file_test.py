import unittest
from csv_reader import *

'''
This is a test suite for the CSVFile classe.
'''
class TestCSVFile(unittest.TestCase):

    def setUp(self):
        self.path = os.path.abspath('.ssh/../')
        self.csv = CSVFile(self.path+os.sep+'FakeFile.csv')
        #Create study dir
        self.csv.create_study_dir()

    def test_path(self):
        self.assertTrue(self.csv.path == (self.path+os.sep+'FakeFile.csv'))
        self.assertTrue(self.csv.study_path == (self.path+os.sep+'FakeFile_Study'))
        self.assertTrue(self.csv.documents_path() == (self.path+os.sep+'FakeFile_Study'+os.sep+'Documents'))
        self.assertTrue(self.csv.canonical_path() == (self.path+os.sep+'FakeFile_Study'+os.sep+'Canonical'))
        self.assertTrue(self.csv.matrices_path() == (self.path+os.sep+'FakeFile_Study'+os.sep+'Matrices'))

    def test_create_study_dir(self):
        #Checks if the study directory was created.
        self.assertTrue(os.path.isdir(self.csv.study_path))
        self.assertTrue(os.path.isdir(self.csv.canonical_path()))
        self.assertTrue(os.path.isdir(self.csv.documents_path()))
        self.assertTrue(os.path.isdir(self.csv.matrices_path()))
        self.assertTrue(os.path.isdir(self.csv.results_path()))


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCSVFile)
    unittest.TextTestRunner(verbosity=2).run(suite)
