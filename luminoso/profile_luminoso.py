import cProfile as profile
import run_luminoso
import pstats
import sys
real_stdout = sys.stdout
from PySide.QtGui import QApplication

app = QApplication(sys.argv)
profile.run('run_luminoso.main(app)', 'luminoso.profile')

sys.stdout = real_stdout
p = pstats.Stats('luminoso.profile')
p.sort_stats('cumulative').print_stats(50)
