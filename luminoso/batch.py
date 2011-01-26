from PySide import QtCore, QtGui
from PySide.QtCore import Qt

class progress_reporter(QtCore.QObject):
    def __init__(self, parent, name, num_steps, modal=True):
        QtCore.QObject.__init__(self)
        self.progress = QtGui.QProgressDialog(name, "", 0, num_steps-1, parent)
        if modal:
            self.progress.setWindowModality(Qt.WindowModal)

        self.cur_step = 0
        self.progress.setValue(0)
        self.progress.setAutoClose(True)
        self.progress.setAutoReset(True)

    def __enter__(self):
        self.progress.forceShow()
        return self

    def set_text(self, text):
        self.progress.setLabelText(text)

    def tick(self, msg=None):
        self.cur_step += 1
        if msg is not None:
            self.set_text(msg)
        self.progress.setValue(self.cur_step)
        QtGui.QApplication.processEvents()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.progress.setValue(self.progress.maximum())
        QtGui.QApplication.processEvents()
        return False # propagate any exceptions
