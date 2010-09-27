from PySide import QtGui, QtCore
import codecs, os

class StudyTreeView(QtGui.QTreeView):
    def __init__(self):
        QtGui.QTreeView.__init__(self)
        self.setMinimumSize(QtCore.QSize(200, 400))
        self.setAcceptDrops(True)

        
    
    def dragEnterEvent(self, event):
        event.acceptProposedAction()
    def dragMoveEvent(self, event):
        event.acceptProposedAction()
    def dropEvent(self, event):
        print event.mimeData()

    def get_filename_at(self, index):
        filename = self.model().filePath(index)
        return os.path.basename(unicode(filename))

    def get_file_contents_at(self, index):
        filename = unicode(self.model().filePath(index))
        if filename.endswith(u'.pickle'): return ''
        try:
            text = codecs.open(filename, encoding='utf-8', errors='replace').read()
            return text
        except IOError:
            return ''

    def find_filesystem_index(self, filename, rootIndex=None):
        """
        Finds the tree index of a directory entry with a certain name.
        """
        if rootIndex is None: rootIndex = self.rootIndex()
        for i in range(self.model().rowCount(rootIndex)):
            index = self.model().index(i,0,rootIndex)
            if self.get_filename_at(index) == filename: return index
        return None

    def find_document_index(self, text):
        """
        Finds the tree index of a document (canonical or otherwise) with a
        certain name.
        """
        documents_folder_index = self.find_filesystem_index('Documents')
        canon_folder_index = self.find_filesystem_index('Canonical')
        doc_index = self.find_filesystem_index(text, documents_folder_index)
        if doc_index is not None: return doc_index
        else: return self.find_filesystem_index(text, canon_folder_index)

