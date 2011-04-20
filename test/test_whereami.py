from luminoso2.whereami import get_project_filename
import os

def test_filenames():
    assert os.access(get_project_filename('test/test_whereami.py'), os.F_OK)
    assert os.access(get_project_filename('setup.py'), os.F_OK)
