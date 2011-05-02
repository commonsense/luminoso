import tornado.ioloop
import tornado.web
import tornadio
import luminoso2
import sys, os
import json
_model = None

def get_model():
    global _model
    if _model is None:
        _model = luminoso2.load(model_path)
    return _model

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")

class DocsHandler(tornado.web.RequestHandler):
    def initialize(self):
        get_model()

    def get(self):
        url = self.get_argument('url', None)
        if url is None:
            self.list_docs()
        else:
            self.get_doc(url)

    def list_docs(self):
        model = get_model()
        self.write(json.dumps(dict(docs=model.database.list_documents())))

    def get_doc(self, url):
        model = get_model()
        doc = model.database.get_document(url)
        tags = model.database.get_document_tags(url)
        self.write(json.dumps(dict(
            name=doc.name,
            reader=doc.reader,
            text=doc.text,
            tags=tags)))

class ViewerHandler(tornado.web.RequestHandler):
    def get(self, study, kind):
        self.set_header('Content-Type', 'application/octet-stream')
        model = get_model()
        assert kind in ['names', 'coords']
        filename = "%s.%s" % (model.filename_in_dir(study), kind)
        if not os.path.exists(filename):
            model.export_svdview(study)
        self.write(open(filename ,'rb').read())

class Channel(tornadio.SocketConnection):
    active_channels = set()

    def on_open(self, request, *a, **kw):
        self.active_channels.add(self)
        self.send('Connected.')

    def on_message(self, message):
        print "Message"

    def on_close(self):
        self.active_channels.remove(self)

    @classmethod
    def send_to_all(self, message):
        for channel in self.active_channels:
            channel.send(message)

settings = dict(
    template_path = 'templates',
    static_path = os.path.join(os.path.dirname(__file__), "static"),
    enabled_protocols = ['websocket', 'xhr-multipart', 'xhr-polling'],
    socket_io_port = 8888,
)

application = tornado.web.Application([
  (r'/', MainHandler),
  (r'/docs', DocsHandler),
  (r'/viewer/(\w+)/(\w+)', ViewerHandler),
  tornadio.get_router(Channel).route()
], **settings)

def tick():
    import datetime
    timestamp = datetime.datetime.now().isoformat()
    print 'test', timestamp
    Channel.send_to_all(timestamp)

periodic = tornado.ioloop.PeriodicCallback(tick, 1000)
periodic.start()

if __name__ == '__main__':
    model_path = sys.argv[1]
    print 'Listening on http://localhost:8888/'
    application.listen(8888)
    tornado.ioloop.IOLoop.instance().start()
