from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import Qt

from luminoso import svdview
from luminoso.tree_view import StudyTreeView


class LuminosoUI(QtGui.QWidget):
    def __init__(self, parent):
        QtGui.QWidget.__init__(self, parent)
        
        # Lay out the interface
        self.main_layout = QtGui.QHBoxLayout(self)
        self.lr_splitter = QtGui.QSplitter()
        self.main_layout.addWidget(self.lr_splitter)
        
        self.left_panel = QtGui.QFrame(self.lr_splitter)

        self.tree_view = StudyTreeView()
        
        self.study_options = QtGui.QGroupBox()
        self.study_options.setTitle("Study options")
        self.options_layout = QtGui.QVBoxLayout(self.study_options)
        self.axes_box = QtGui.QFrame(self.study_options)
        self.cutoff_box = QtGui.QFrame(self.study_options)
        self.options_layout.addWidget(self.axes_box)
        self.options_layout.addWidget(self.cutoff_box)

        self.axes_layout = QtGui.QHBoxLayout(self.axes_box)
        self.axes_label = QtGui.QLabel()
        self.axes_layout.addWidget(self.axes_label)
        self.axes_label.setText("Dimensions:")
        self.axes_spinbox = QtGui.QSpinBox()
        self.axes_layout.addWidget(self.axes_spinbox)
        self.axes_spinbox.setMinimum(2)
        self.axes_spinbox.setMaximum(1000)
        self.axes_spinbox.setValue(20)
        
        self.cutoff_layout = QtGui.QHBoxLayout(self.cutoff_box)
        self.cutoff_label = QtGui.QLabel()
        self.cutoff_layout.addWidget(self.cutoff_label)
        self.cutoff_label.setText("Concept threshold:")
        self.cutoff_spinbox = QtGui.QSpinBox()
        self.cutoff_layout.addWidget(self.cutoff_spinbox)
        self.cutoff_spinbox.setMinimum(1)
        self.cutoff_spinbox.setMaximum(100)
        self.cutoff_spinbox.setValue(2)
        #self.analyze_button = QtGui.QPushButton("&Analyze >>")
        
        self.left_layout = QtGui.QVBoxLayout(self.left_panel)
        self.left_layout.addWidget(self.tree_view)
        self.left_layout.addWidget(self.study_options)
        #self.left_layout.addWidget(self.analyze_button)
        
        self.ud_splitter = QtGui.QSplitter(Qt.Vertical, self.lr_splitter)
        
        self.svdview_panel = svdview.SVDViewPanel()
        self.ud_splitter.addWidget(self.svdview_panel)

        self.svdview_panel.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)

        self.lr_splitter.setStretchFactor(0, 1)
        self.lr_splitter.setStretchFactor(1, 3)
        
        self.tab_stack = QtGui.QTabWidget(self.ud_splitter)
        self.tab_stack2 = QtGui.QTabWidget(self.ud_splitter)
        self.info_page = QtGui.QTextEdit()
        self.empty_tab = QtGui.QTextEdit()
        self.info_page.setReadOnly(True)
        self.tab_stack.addTab(self.svdview_panel, "Main")
        self.tab_stack2.addTab(self.info_page, "Info")
        self.tab_stack.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Maximum)

        #Add search text box to toolbar
        self.search_panel = QtGui.QFrame()
        #self.search_panel.setTitle("Search")
        self.search_layout = QtGui.QHBoxLayout(self.search_panel)
        self.search_box = QtGui.QLineEdit()
        self.search_layout.addWidget(self.search_box)
        self.search_button = QtGui.QPushButton("&Search")
        self.search_layout.addWidget(self.search_button)

    def show_info(self, html):
        "Display a message in the info pane."
        self.info_page.setHtml(html)


    def show_document_info(self, doc_name, info):
        "Displays information about the document with name doc_name and contents info in the info pane"
        title = "<h2>"+doc_name+"</h2>"
        info = "<p>" + info + "</p>"
        self.show_info(title+info)
    
    def get_num_axes(self):
        return self.axes_spinbox.value()

    def set_num_axes(self, value):
        self.axes_spinbox.setValue(value)

    def get_concept_cutoff(self):
        return self.cutoff_spinbox.value()

    def set_concept_cutoff(self, value):
        self.cutoff_spinbox.setValue(value)

