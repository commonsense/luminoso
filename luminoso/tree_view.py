from PyQt4 import QtGui, QtCore
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


    def find_filename_index(self, text):
        documents_folder_index = self.model().index(1, 0, self.rootIndex())
        canon_folder_index = self.model().index(0,0,self.rootIndex())
        for i in range(self.model().rowCount(documents_folder_index)):
            index = self.model().index(i,0,documents_folder_index)
            if self.get_filename_at(index) == text:
                return index
        for i in range(self.model().rowCount(canon_folder_index)):
            index = self.model().index(i,0,canon_folder_index)
            if self.get_filename_at(index) == text:
                return index

        return None

