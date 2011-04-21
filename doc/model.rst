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
TODO

::

    >>> model.learn_from_url('path/to/docs/directory', 'study_name')

The model API
-------------
.. module:: luminoso2.model

.. autoclass:: LuminosoModel
   :members: learn_from_url, add_from_url, add_document,
             docs_in_study, vector_from_terms, vector_from_text,
             vector_from_input, vector_from_document, terms_similar_to_vector,
             domain_terms_similar_to_vector, docs_similar_to_vector,
             tags_similar_to_vector, learn_assoc, canonical_stats

Studies within a model
----------------------
TODO
