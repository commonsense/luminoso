import tornado.ioloop
import tornado.web
import sys, luminoso2
import json
model = None

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")

class DocsHandler(tornado.web.RequestHandler):
    def initialize(self):
        global model
        if model is None:
            model = luminoso2.load(model_path)

    def get(self):
        url = self.get_argument('url', None)
        if url is None:
            self.list_docs()
        else:
            self.get_doc(url)

    def list_docs(self):
        self.write(json.dumps(dict(docs=model.database.list_documents())))

    def get_doc(self, url):
        doc = model.database.get_document(url)
        tags = model.database.get_document_tags(url)
        self.write(json.dumps(dict(
            name=doc.name,
            reader=doc.reader,
            text=doc.text,
            tags=tags)))

application = tornado.web.Application([
  (r'/', MainHandler),
  (r'/docs', DocsHandler)
], template_path='templates')

if __name__ == '__main__':
    model_path = sys.argv[1]
    print 'Listening on http://localhost:8888/'
    application.listen(8888)
    tornado.ioloop.IOLoop.instance().start()
