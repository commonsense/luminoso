Lumongous: A Big Data Extension of Luminoso
===========================================

Steps:

- Find an incremental version of spectral association, preferably with
  MapReduce. (The incremental part has been done, the MapReduce part probably
  hasn't.)
- Store results in a database.
- Sum documents over those results.

Spectral association
--------------------
Term co-occurrence is a matrix C. Its primary eigenvector is the result of
iterating v * C^n for an arbitrary v. To find later eigenvectors, subtract out
previous eigenvectors at each step.

This could be run continually, with some cleverness and some processing power.

MapReduce for SA:

    Repeat until convergence:
        filter out the lowest-scored things in the input vector
        for each word in input vector:
            for each doc matching word:
                output vector += doc * input[word]
        output vector <- input vector

Result: a mapping of words -> assoc vectors
