.. _luminoso1:

Working with Luminoso 1 studies
===============================

In Lumioso 1, a study was always represented by a directory, whose
subdirectories contained documents. In Luminoso 2, this is still a valid way of
providing input, among other forms of input described at :ref:`model`.

There are two equivalent ways to analyze this kind of study.

At the command line, run Luminoso2's `model.py` with two arguments -- the
study directory and the language to use:

    $ model.py ./PLDBStudy en

Or, run the `luminoso2.run_in_directory` function from the Python prompt,
which takes the same arguments:

    >>> import luminoso2
    >>> luminoso2.run_in_directory('./PLDBStudy', 'en')

The language can be either `en` for English (which is the default if it is
unspecified), or `ja` for Japanese. See :ref:`readers` for more information.

The directory you provide should contain:

    `Canonical/`
        A directory of canonical documents. Alternatively, you could supply a
        JSON file that is a source of canonical documents called
        `canonical.json` -- see :ref:`documents`. This may be empty if you
        do not intend to use canonical documents.

    `Documents/`
        A directory containing the documents to process, as .txt or .json
        files.

This process will produce (or update if it already exists):

    `Model/`
        The Luminoso2 model (see :ref:`model`) that results from analyzing
        those documents. It can be loaded into Python with:
            
            >>> model = luminoso2.load('Model')

    `Model/all.stats.json`
        A JSON file containing statistics that compare the documents in the
        study to each canonical document (see :ref:`statistics`).

