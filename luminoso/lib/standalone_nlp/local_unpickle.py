from pickle import Unpickler
from StringIO import StringIO

module_subst = {
    'csc.nl': 'standalone_nlp.nl',
    #'csc.nl.en': 'standalone_nlp.nl',
    'csc.nl.euro': 'standalone_nlp.euro',
    'csc.nl.mblem.trie': 'standalone_nlp.trie'
}

class LocalUnpickler(Unpickler):
    def find_class(self, module, name):
        return Unpickler.find_class(self, module_subst.get(module, module), name)

def loads(str):
    file = StringIO(str)
    return LocalUnpickler(file).load()
