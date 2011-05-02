# -*- coding: utf-8 -*-
"""
This module defines a variety of classes that can be used to read plain text
and return word associations. Which reader to use depends on the complexity
of the task and what language the text is in.
"""
from simplenlp import get_nl
import logging
LOG = logging.getLogger(__name__)

# TODO:
# - Handle situations that aren't just about discovering terms. (Negations,
#   different strengths of terms, etc.) Change the API to make this possible.

DOCUMENT = u'*Document*'
TAG = u'*Tag*'

def is_tag(token):
    """
    Is this a token that represents a tag?
    """
    return (token.startswith(u'#') or token.startswith(u'+') or
            token.startswith(u'-')) and len(token) > 2

class TextReader(object):
    """
    An abstract class showing the interface that TextReaders must implement.
    """

    def extract_connections(self, text):
        """
        Outputs a list of tuples, containing two terms and the strength with
        which they are connected. These connections are symmetrical, so there
        is no need to output them once in each direction.
        
        It should also output a special connection at least once per term,
        with the first term being the special symbol called DOCUMENT.
        """
        raise NotImplementedError
    
    def extract_term_texts(self, text):
        """
        Yields a list of (term, phrase) tuples, mapping the normalized form
        of a term to the full phrase it came from.
        """
        raise NotImplementedError

class SimpleNLPReader(TextReader):
    """
    An abstract class for readers that use simplenlp.

    The model of connections between words is one that decreases geometrically
    until a cutoff value. The default values allow for associations between
    words that are up to 10 tokens apart.

    It also includes negative contexts. When one term in a pair appears in a
    negative context, the sign of the connection is flipped. The sign of the
    negated term's connection to the document is also flipped.
    """
    HARD_PUNCT = [u'//']

    # include English and Japanese punctuation mixed together
    PUNCT = [u'.', u'?', u'!', u':', u'…', u'—', u'、', u'。',
             u'「', u'」', u'『', u'』', u'【', u'】', u'〖', u'〗',
             u'？', u'！']

    # punctuation that has to appear as the entire token to count
    # (so we don't accidentally treat "'s" as punctuation, for example)
    PUNCT_TOKENS = [u"''", u"``", u"'", u"`", u":", u";", u'-', u'--', u'---']

    # These have to be language-specific
    NEGATIONS = []

    EXTRA_STOPWORDS = []

    # override this for right-head languages like Japanese
    HEAD_DIRECTION = 'left'

    nl = None

    def __init__(self, distance_weight, negation_weight, cutoff):
        """
        Set parameters that all SimpleNLPReaders use.
        """
        self.negation_weight = negation_weight
        self.distance_weight = distance_weight
        self.cutoff = cutoff
        self.SPECIAL_TOKENS = set(self.HARD_PUNCT + self.PUNCT +
                                  self.PUNCT_TOKENS + self.NEGATIONS)

    def extract_tokens(self, text):
        """
        Return a list of the (lemmatized, non-stopword) tokens that appear in
        the text.
        """
        tokens = self.tokenize(text)
        tokens_out = []
        for token in tokens:
            if is_tag(token):
                tokens_out.append(token)
            elif self.nl.is_stopword(token):
                pass
            else:
                tokens_out.extend(self.tokenize(self.nl.normalize(token)))
        return tokens_out
    
    def tokenize(self, text):
        """
        Tokenize text as appropriate for the language.
        """
        raise NotImplementedError

    def untokenize(self, text):
        """
        Untokenize text as appropriate for the language.
        """
        raise NotImplementedError

    def is_normal_word(self, token):
        """
        Classify a token very generally as "normal" or "abnormal". When
        outputting phrases, we want them to begin and end with normal
        tokens.
        """
        return (token and (token not in self.SPECIAL_TOKENS)
                      and (token[0] not in self.PUNCT)
                      and not self.nl.is_stopword(token))
    
    def _attenuate(self, memory):
        """
        Decrease the weights of the terms that are currently in memory, as
        they get farther from our place in the text.
        Remove them if they fall below the given cutoff value.
        """
        for i in reversed(xrange(len(memory))):
            memory[i][1] *= self.distance_weight
            if memory[i][1] < self.cutoff:
                del memory[i]

    def extract_connections(self, text):
        """
        Determine which terms appear near each other in the given document,
        and output a generator of (weight, term1, term2) connections that
        can be used to train a LuminosoSpace.

        The special term named `DOCUMENT` is used to mark the strength with
        which each term appears in the document itself.
        """
        tokens = self.extract_tokens(text)
        if self.HEAD_DIRECTION == 'right':
            tokens.reverse()
        weight = 1.0
        memory = []
        prev_token = None

        for token in tokens:
            if token: # protect in case we get an empty token somehow
                self._attenuate(memory)
                active_terms = []
                if token in self.HARD_PUNCT:
                    memory = []
                    weight = 1.0
                    prev_token = None
                elif token in self.PUNCT_TOKENS or token[0] in self.PUNCT:
                    weight = 1.0
                    prev_token = None
                elif token in self.NEGATIONS:
                    weight *= self.negation_weight
                    prev_token = None
                elif token in self.EXTRA_STOPWORDS:
                    continue
                elif is_tag(token):
                    tag = parse_tag(token)
                    yield (0, DOCUMENT, tag)
                    prev_token = None
                else:
                    # this is an ordinary token, not a negation or punctuation
                    active_terms = [token]
                    if prev_token is not None:
                        if self.HEAD_DIRECTION == 'right':
                            bigram = token + ' ' + prev_token
                        else:
                            bigram = prev_token + u' ' + token
                        active_terms.append(bigram)
                    for term in active_terms:
                        yield (weight, DOCUMENT, term)
                        memory.append([term, weight])
                    prev_token = token
                
                for term in active_terms:
                    for prev, prev_weight in memory:
                        # Currently, this will associate each term with
                        # itself, in addition to previous terms.
                        # Is this the right way to do that?
                        yield (weight*prev_weight, prev, term)
    
    def extract_term_texts(self, text):
        """
        Yield tuples of (term, phrase), showing the original text that
        various terms came from.
        """
        tokens = self.tokenize(text)
        for window in xrange(1, 5):
            for left in xrange(len(tokens) - window + 1):
                right = left + window
                if (self.is_normal_word(tokens[left]) and
                    self.is_normal_word(tokens[right-1])):
                    phrase = self.untokenize(tokens[left:right])
                    term = self.nl.normalize(phrase)
                    yield (term, phrase)


class SimpleNLPEnglishReader(SimpleNLPReader):
    """
    A SimpleNLPReader for handling English text.
    """
    # words that flip the context from positive to negative
    NEGATIONS = [u'no', u'not', u'never', u'stop', u'lack',
                 u"n't", u'without']
    
    EXTRA_STOPWORDS = [
        'also', 'no', 'not', 'without', 'ever', 'because', 'then', 
        'than', 'do', 'just', 'how', 'out', 'much', 'both', 'other'
    ]

    def __init__(self, distance_weight=0.8, negation_weight=-0.5, cutoff=0.1):
        """
        Create a reader that reads English text using `simplenlp`. You can
        optionally adjust the weights for how much various terms affect
        each other.
        """
        SimpleNLPReader.__init__(self, distance_weight, negation_weight,
                                 cutoff)
        self.nl = get_nl('en')

    def tokenize(self, text):
        """
        Simply tokenize the text, without dealing with stopwords.
        """
        return self.nl.tokenize(text).split()

    def untokenize(self, tokens):
        """
        Given a list of tokens, re-assemble them into a phrase.
        """
        spaced = ' '.join(tokens)
        return self.nl.untokenize(spaced)

class SimpleNLPJapaneseReader(SimpleNLPReader):
    """
    A SimpleNLPReader for handling Japanese text.
    """
    HEAD_DIRECTION = 'right'

    # Words that flip the context from positive to negative, to their *left*.
    NEGATIONS = [u'ない']

    # After simplenlp deals with stopwords and luminoso has detected negations,
    # we include "nai" as a stopword as well.
    EXTRA_STOPWORDS = [u'ない']
    
    # include both English and Japanese punctuation
    PUNCT = [u'.', u'?', u'!', u':', u'…', u'—', u'、', u'。',
             u'「', u'」', u'『', u'』', u'【', u'】', u'〖', u'〗']
    
    # Count は as a punctuation token. It's actually the topic marker, but
    # the scope of a negation will rarely cross it.
    PUNCT_TOKENS = [u"''", u"``", u"'", u"`", u":", u";", u'-', u'--', u'---',
                    u"は"]

    def __init__(self, distance_weight=0.8, negation_weight=-0.5, cutoff=0.1):
        """
        Create a reader that reads English text using `simplenlp`. You can
        optionally adjust the weights for how much various terms affect
        each other.
        """
        SimpleNLPReader.__init__(self, distance_weight, negation_weight,
                                 cutoff)
        self.nl = get_nl('ja')

    def tokenize(self, text):
        """
        Simply tokenize the text, without dealing with stopwords.
        """
        return self.nl.tokenize_list(text)
    
    def extract_term_texts(self, text):
        """
        Get a list of (term, phrase) mappings from the text.
        """
        return self.nl.extract_phrases(text)

    def extract_tokens(self, text):
        """
        Return a list of the (lemmatized, non-stopword) tokens that appear in
        the text.
        """
        analysis = self.nl.analyze(text)
        return [self.nl.get_record_root(record) for record in analysis
                if not self.nl.is_stopword_record(record)]

    def untokenize(self, tokens):
        """
        Given a list of tokens, re-assemble them into a phrase.
        """
        return ''.join(tokens)    

def parse_tag(token):
    """
    If this token is a specially-marked tag, parse it into a consistent format.

    Returns a tuple of `(TAG, key, value)`, where `TAG` is a special flag
    value, `key` is the name of the tag, and `value` is the value it is
    assigned (or None).
    """
    # a way to deal with PLDB-style tags
    token = token.replace(':', '=')

    if token.startswith(u'#'):
        if u'=' in token:
            key, value = token[1:].split(u'=', 1)
            # handle numeric values
            try:
                value = float(value)
            except ValueError:
                pass
        else:
            key = token[1:]
            value = None
    elif token.startswith(u'+'):
        key = token[1:]
        value = True
    elif token.startswith(u'-'):
        key = token[1:]
        value = False
    else:
        # this shouldn't happen, but if it does, it's not worth throwing
        # an error.
        LOG.warn('Got a weird tag: %s' % token)
        key = token
        value = None
    return (TAG, key, value)

READERS = {
    'simplenlp.en': SimpleNLPEnglishReader,
    'simplenlp.ja': SimpleNLPJapaneseReader,
}

LANGUAGE_DEFAULTS = {
    'en': 'simplenlp.en',
    'ja': 'simplenlp.ja'
}

def get_reader(name):
    """
    Gets a TextReader instance by specifying its name. Often, the reader
    you want will be 'simplenlp.en'.
    """
    try:
        return READERS[name]()
    except KeyError:
        raise KeyError("There is no text reader named %r." % name)

## Other things to include eventually:
# class JapaneseTextReader(TextReader):
#   (uses our CaboCha interface)
# 
# class ParsingEnglishTextReader(TextReader):
#   (takes constituents into account)
