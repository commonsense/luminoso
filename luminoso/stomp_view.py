from svdview import SVDViewer
from stompy.simple import Client
from csc.divisi.recycling_set import RecyclingSet
from csc.util.vector import unpack64
import simplejson as json
import numpy as np

import sys
from PySide.QtCore import SIGNAL, QThread
from PySide.QtGui import QApplication

if __name__ == '__main__':
    app = QApplication(sys.argv)

class StompThread(QThread):
    def __init__(self, host, port, channel, nconcepts, k=19):
        QThread.__init__(self)
        self.stomp = Client(host, port)
        self.channel = channel

        self.array = np.zeros((nconcepts, k))
        self.k = k
        self.nconcepts = nconcepts
        self.labels = RecyclingSet(nconcepts)
        self.labels.listen_for_drops(self.on_drop)

        self.exiting = False
    
    def on_message(self, body):
        message = json.loads(body)
        print message
        if 'text' not in message: return
        if message['text'].startswith('('): return
        vec = unpack64(message['coordinates'])
        self.handle_vector(vec[1:], message['text'])
        for concept, value in message['concepts'].items():
            vec = unpack64(value)
            self.handle_vector(vec[1:], concept)

    def on_drop(self, index, label):
        self.array[index,:] = 0
        self.emit(SIGNAL("newData()"))
    
    def handle_vector(self, vec, text):
        if len(text) < 20:
            idx = self.labels.add(text)
            norm = max(0.0000001, np.linalg.norm(vec))
            self.array[idx,:] = vec/norm
            self.emit(SIGNAL("newData()"))

    def run(self):
        self.stomp.connect(username=None, password=None)
        self.stomp.subscribe(destination=self.channel, ack='auto')

        while not self.exiting:
            msg = self.stomp.get()
            self.on_message(msg.body)

def main(app):
    thread = StompThread('localhost', 61613, '/topic/SocNOC/redfishbluefish', 10000, 9)
    #thread = StompThread('localhost', 61613, '/topic/SocNOC/lmds', 10000, 9)
    signal = SIGNAL("newData()")
    view = SVDViewer.make(thread.array, thread.labels)
    view.connect(thread, signal, view.refreshData)
    view.setGeometry(300, 300, 800, 600)
    view.setWindowTitle("TwitterMap")
    view.show()

    thread.start()

    app.exec_()

if __name__ == '__main__':
    main(app)

