.. _statistics:

Canonical documents and statistics
==================================

Canonical documents are example text files that you supply. You can use them as
reference points in a Luminoso model, representing known meanings to which you
compare your unknown documents.

A canonical document can be an actual example document, representing for
example an idealized review. It can also contain simply a set of words that
define what you are looking for, such as "car train bus transport travel". Or
it can simply contain a single word defining a topic, such as "music" --
Luminoso's common sense background knowledge will pull in related words for
you.

One common use of canonical documents is to contrast two sets of words, using
the keyword "not". For example, here is a canonical document that distinguishes
positive expressions from negative ones, which we call `positive-emotion.txt`::

    good like love enjoy happy nice easy,
    NOT bad dislike hate sad angry hard difficult

Another use of canonical documents is to create one for each topic you are
interested in, so that you can use those topics to organize the space.

Luminoso will calculate statistics called "centrality" and "correlation" that
compare sets of documents (called "studies") to each canonical document. 

Correlation and centrality
--------------------------
TODO
