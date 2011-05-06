"""
This is the top-level module of Luminoso version 2.
"""
from luminoso2.model import LuminosoModel, run_in_directory
import warnings
import logging, sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# suppress warnings internal to PyLint
warnings.simplefilter("ignore")

# Import commonly used methods.
make_common_sense = LuminosoModel.make_common_sense
make_english = LuminosoModel.make_english
make_japanese = LuminosoModel.make_japanese
make_empty = LuminosoModel.make_empty

def load(model_dir):
    """
    Load a LuminosoModel.
    """
    return LuminosoModel(model_dir)

