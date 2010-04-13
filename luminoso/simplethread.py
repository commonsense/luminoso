from PyQt4 import QtCore

class ThreadRunner(QtCore.QThread):
    """
    A straightforward way to run a function in a new thread:

        self.thread = ThreadRunner(func)
        self.thread.start()
    """
    def __init__(self, func, args=None, parent=None):
        QtCore.QThread.__init__(self, parent)
        if args is None: args = []
        self.func = func
        self.args = args
        self.result = None

    def run(self):
        self.result = self.func(*self.args)

