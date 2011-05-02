.. _model:

Creating and using a model
==========================

What is a Luminoso model?
-------------------------

A *model* represents what Luminoso knows about the meanings of words, which it
gradually learns from the documents that you input to it.

One straightforward way of creating a model, especially if you have experience
with Luminoso 1, is described at :ref:`luminoso1`. This page describes another
way that does not depend on where your documents are stored.

In Python, a model is represented as an instance of :class:`LuminosoModel`.
Every model is backed by a directory where it keeps its persistent information.
(If you're creating a model to experiment with and throw away, try creating it
in `/tmp`).

The files in the model directory are a complete representation of the model's
state. They are sufficient to quickly load a previously-created model at the
Python prompt and resume working with it.

Creating a model
----------------

To create a model, choose a directory name that can be created but that does
not already exist, and run::

    >>> import luminoso
    >>> model = luminoso.make_common_sense('path/to/dir', 'en')

That's how you make a model that understands English text, anyway. You could
also make one for Japanese text by changing `'en'` to `'ja'`.

Loading an existing model
-------------------------
To resume working with a model that already exists, run::
    
    >>> import luminoso
    >>> model = luminoso.load('path/to/dir', 'en')

Teaching a model
----------------
`learn_from` is the main method you will use to teach Luminoso a set of
documents. It takes two parameters: a URL or path, and a study name.

The first argument, the "URL", can be just a path on your local disk. It 
can even be relative to the current directory. It should point to a
source of documents (see :ref:`documents`).

If you provide a study name, it will automatically add all the documents it
finds to that study.

::

    >>> model.learn_from('path/to/docs/directory', 'study_name')

Studies within a model
----------------------
Because Luminoso 2 has added the concept of a persistent "model", it is now
possible to run multiple studies on the same model. Each document you add may
be associated with a study, so that a study refers to a group of documents
within the model. By default, documents will be added to the study named "all".

Some operations take in a study name so that you can perform them over a subset
of documents. For example, when you calculate statistics (see :ref:`statistics`) using :meth:`LuminosoModel.canonical_stats`, you supply the study name that the statistics should be calculated for.

The model API
-------------
.. module:: luminoso2.model

.. autoclass:: LuminosoModel
   :members: learn_from_url, add_from_url, add_document,
             docs_in_study, vector_from_terms, vector_from_text,
             vector_from_input, vector_from_document, terms_similar_to_vector,
             domain_terms_similar_to_vector, docs_similar_to_vector,
             tags_similar_to_vector, learn_assoc, canonical_stats

