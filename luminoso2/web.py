import sys, luminoso2
model = luminoso2.load(sys.argv[1])

from flask import Flask, render_template, request
app = Flask(__name__)

import json

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/docs')
def docs():
    return json.dumps(dict(docs=model.database.list_documents()))

@app.route('/doc/')
def doc_by_url():
    url = request.args['url']
    doc = model.database.get_document(url)
    tags = model.database.get_document_tags(url)
    return json.dumps(dict(
        name=doc.name,
        reader=doc.reader,
        text=doc.text,
        tags=tags))

if __name__ == '__main__':
    app.run(debug=True)
