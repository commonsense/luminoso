#!/usr/bin/env python
from luminoso.whereami import get_icon
from PySide import QtCore, QtGui
from PySide.QtCore import Qt
import sys, os
import traceback

# Back up stdout before spyder redirects it to its console.
sys._stdout = sys.stdout
sys._stderr = sys.stderr

def initialize():
    app = QtGui.QApplication(sys.argv)

    #----Monkey patching sys.exit
    def fake_sys_exit(arg=[]):
        pass
    sys.exit = fake_sys_exit
    return app

def qt_exception_hook(ex_type, ex_value, ex_traceback):
    QtCore.qWarning('Sorry, an internal error occurred. Could you please send the authors the text below and a brief note about what you were doing? Thanks!\n\n'+
                    '\n'.join(traceback.format_exception(ex_type, ex_value, ex_traceback)))

def main(app=None):
    if app is None: app = initialize()
    try:
        # Set up splash screen
        pixmap = QtGui.QPixmap(get_icon("luminoso_splash.png"))
        splash = QtGui.QSplashScreen(pixmap)
        splash.show()
        splash.raise_()
        app.processEvents()
        # Set up error handling
        QtGui.QErrorMessage.qtHandler()
        sys.excepthook = qt_exception_hook
        
        from luminoso.window import MainWindow
        window = MainWindow()
        window.setup()
        window.show()
        window.activateWindow()
        window.raise_()
        app.setOrganizationName("Common Sense Computing Initiative")
        app.setApplicationName("Luminoso")
        splash.finish(window)
        app.exec_()
    except:
        traceback.print_exc(file=sys._stderr)

if __name__ == '__main__':
    main()
