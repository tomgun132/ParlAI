import collections
import six
import sys
import unicodedata

import tensorflow as tf
import sentencepiece as sp

SENTPIECE_MODEL = 'ja_wiki_sentpiece/wiki-ja.model'
SENTPIECE_VOCAB = 'ja_wiki_sentpiece/wiki-ja.vocab'

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with tf.io.gfile.GFile(vocab_file, "r") as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            token, _ = token.split("\t")
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab

def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")

def convert_by_vocab(vocab, items, unk_info):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for item in items:
        if item in vocab:
            output.append(vocab[item])
        else:
            output.append(unk_info)
    return output

class FullTokenizer(object):
    """Runs end-to-end tokenziation."""

    def __init__(self, path, do_lower_case=True):
        model_file = path + SENTPIECE_MODEL
        vocab_file = path + SENTPIECE_VOCAB
        self.tokenizer = SentencePieceTokenizer(model_file, do_lower_case=do_lower_case)
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def parse(self, text):
        split_tokens = self.tokenizer.tokenize(text)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        """Id of <unk> is assumed as 0 accroding to sentencepiece"""
        return convert_by_vocab(self.vocab, tokens, unk_info=0)

    def convert_ids_to_tokens(self, ids):
        """Token of unknown word is assumed as <unk> according to sentencepiece"""
        return convert_by_vocab(self.inv_vocab, ids, unk_info="<unk>")

class SentencePieceTokenizer(object):
    """Runs SentencePiece tokenization (from raw text to tokens list)"""

    def __init__(self, model_file=None, do_lower_case=True):
        """Constructs a SentencePieceTokenizer."""
        self.tokenizer = sp.SentencePieceProcessor()
        if self.tokenizer.Load(model_file):
            print("Loaded a trained SentencePiece model.")
        else:
            print("You have to give a path of trained SentencePiece model.")
            sys.exit(1)
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = convert_to_unicode(text)
        if self.do_lower_case:
            text = text.lower()
        output_tokens = self.tokenizer.EncodeAsPieces(text)
        return ' '.join(output_tokens)