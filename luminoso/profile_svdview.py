import cProfile as profile
import svdview
import pstats

import sys
from PyQt4.QtGui import QApplication

app = QApplication(sys.argv)
profile.run('svdview.main(app)', 'svdview.profile')

p = pstats.Stats('svdview.profile')
p.sort_stats('time').print_stats(50)
