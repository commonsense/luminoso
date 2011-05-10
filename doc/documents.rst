.. _documents:

Using documents
===============

Luminoso learns about your data by reading in documents. The complete process
contains three steps:

1. A **document handler** finds documents and extracts their titles, their
   plain-text contents, and possibly the structured *tags* that are associated
   with those documents.
2. A **text reader** takes the plain text from each document and applies basic
   NLP operations to it -- including combining different inflections of the
   same word, discovering multiple-word phrases, and discovering negations.
3. Optionally, the **association matrix** will be updated to learn from the
   contexts that terms appeared in.

Document handlers
-----------------
A *document handler* takes in some initial *source*, and outputs a stream of
dictionaries representing documents. Sources may contain other sources, in
which case they will be scanned recursively. When you use a method such as
:meth:`LuminosoModel.learn_from`, you provide a URL or path to a source.

One very common document source is a directory on the local disk. This will
scan all `.txt` and `.json` files in the directory, as well as recursively
scanning all directories within it.

A .txt file always represents a single document with exactly that text.

A .json file may be a single document, in which case it should have the
following keys:

- `text`: the text to read
- `name`: a human-readable name for the document
- `tags`: an optional list of [key, value] pairs that provide structured
  information about the document

A .json file may also be a dictionary whose keys are document names, and whose
values are document texts or URLs of other document sources.

This can be a quick and easy way to define canonical documents. For example, in
the example `PLDBStudy`, the beginning of `canonical.json` looks like this::

    {"Architecture and Design": "architecture, build, design",
     "Art and Technology": "art, draw, color, sculpt, create",
     "Artificial Intelligence": "artificial intelligence",
     "Banking": "account, bank, money",
     ...
    }

.. note::

   Python's JSON parser insists that strings should appear in double quotes,
   never in single quotes. If you use single quotes, your .json file will
   be unreadable.

Encodings
`````````
When reading plain text files, Luminoso attempts to automatically detect the
text encoding, using UTF-8 if it is unsure. If it detects an
encoding incorrectly, please report it to us as a bug.

The document handler API
````````````````````````
.. automodule:: luminoso2.document_handlers
    :members:

.. _readers:

Readers
-------
A *reader* is responsible for taking the output of a handler and processing it
as natural language. This requires knowledge of which natural language the text
will be in.

When creating a model, you choose which reader to use -- for example, the
:func:`luminoso.make_common_sense` function takes in a language code.

The two readers we currently support are:

- `en`: Reads English text using the `simplenlp` module.
- `ja`: Reads Japanese text, using the `simplenlp` module together with an
  installed version of `mecab`.

The text reader API
```````````````````
.. automodule:: luminoso2.text_readers
    :members:

Associations
------------
Whenever you use the :meth:`LuminosoModel.learn_from` method, Luminoso will
periodically iterate over the documents it has scanned and use them to update
the model's *association matrix*. In short, it will remember which terms were
used together in your domain and make them more strongly associated with each
other.

Changing the association matrix will change the :ref:`statistics <statistics>`
for your studies. It is also the most time-consuming part of the learning
process. If you want to add documents to the model *without* taking their word
associations into account, use :meth:`LuminosoModel.add_from` instead of
:meth:`LuminosoModel.learn_from`.
