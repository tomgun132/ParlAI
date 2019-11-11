import os
import codecs
import json
import numpy as np

from nltk import ngrams
from collections import defaultdict, deque
from parlai.core.dict import DictionaryAgent
from parlai.utils.misc import AttrDict

def escape(s):
    r"""
    Replace potential special characters with escaped version.

    For example, \n => \\n and \t => \\t

    :param s:
        string to escape
    """
    return s.replace('\n', '\\n').replace('\t', '\\t').replace('\r', '\\r')

def unescape(s):
    r"""
    Revert escaped characters back to their special version.

    For example, \\n => \n and \\t => \t

    :param s:
        string to unescape
    """
    return s.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')

def make_dir(path):
    """Make the directory and any nonexistent parent directories (`mkdir -p`)."""
    # the current working directory is a fine path
    if path != '':
        os.makedirs(path, exist_ok=True)

class Batch(AttrDict):
    def __init__(
        self,
        text_vec=None,
        text_lengths=None,
        bi_text_vec=None,
        bi_text_lengths=None,
        label_vec=None,
        label_lengths=None,
        bi_label_vec=None,
        bi_label_lengths=None,
        labels=None,
        valid_indices=None,
        candidates=None,
        candidate_vecs=None,
        bi_candidate_vecs=None,
        observations=None,
        **kwargs,
    ):
        super().__init__(
            text_vec=text_vec,
            text_lengths=text_lengths,
            bi_text_vec=bi_text_vec,
            bi_text_lengths=bi_text_lengths,
            label_vec=label_vec,
            label_lengths=label_lengths,
            bi_label_vec=bi_label_vec,
            bi_label_lengths=bi_label_lengths,
            labels=labels,
            valid_indices=valid_indices,
            candidates=candidates,
            candidate_vecs=candidate_vecs,
            bi_candidate_vecs=bi_candidate_vecs,
            observations=observations,
            **kwargs,
        )

class UniBiDictionaryAgent(DictionaryAgent):

    def __init__(self, opt, shared=None):
        self.bi2ind = {}
        self.ind2bi = {}
        self.bi_freq = defaultdict(int)
        try:
            from nltk import ngrams
        except ImportError:
            raise ImportError('Please install NLTK (pip install nltk)')

        super().__init__(opt, shared)

        if self.null_token:
            self.add_bi(self.null_token)
            self.bi_freq[self.null_token] = 1000000003

        if self.start_token:
            # set special start of sentence word token
            self.add_bi(self.start_token)
            self.bi_freq[self.start_token] = 1000000002

        if self.end_token:
            # set special end of sentence word token
            self.add_bi(self.end_token)
            self.bi_freq[self.end_token] = 1000000001

        if self.unk_token:
            # set special unknown word token
            self.add_bi(self.unk_token)
            self.bi_freq[self.unk_token] = 1000000000

    def add_bi(self, bi):
        if bi not in self.bi2ind:
            index = len(self.bi2ind)
            self.bi2ind[bi] = index
            self.ind2bi[index] = bi

    def __getitem__(self, key):
        if type(key) == str:
            # return index from token, or unk_token's index, or None
            return self.tok2ind.get(key, self.bi2ind.get(key, self.tok2ind.get(self.unk_token, None)))
        else:
            raise TypeError("Key type must be string")

    def __contains__(self, key):
        """
        Return if the dictionary contains the key.

        If key is an int, returns whether the key is in the indices.
        If key is a str, return if the token is in the dict of tokens.
        """
        if type(key) == int:
            return key in self.ind2tok or key in self.ind2bi
        elif type(key) == str:
            return key in self.tok2ind or key in self.bi2ind

    def __len__(self):
        return len(self.tok2ind) + len(self.bi2ind)

    @staticmethod
    def space_tokenize(text):
        """Tokenize exactly on spaces. Useful when text is pre-tokenized."""
        return text.strip().split(' ')

    def getitem(self, key):
        if type(key) == int:
            # return token from index, or unk_token
            return self.ind2tok.get(key, self.unk_token)
        elif type(key) == str:
            # return index from token, or unk_token's index, or None
            return self.tok2ind.get(key, self.tok2ind.get(self.unk_token, None))

    def getbiitem(self, key):
        if type(key) == int:
            # return token from index, or unk_token
            return self.ind2bi.get(key, self.unk_token)
        elif type(key) == str:
            # return index from token, or unk_token's index, or None
            return self.bi2ind.get(key, self.bi2ind.get(self.unk_token, None))

    def add_bi_to_dict(self, tokens):
        """Build dictionary from the list of provided tokens."""
        self.built = False
        for token in tokens:
            self.add_bi(token)
            self.bi_freq[token] += 1

    def remove_tail(self, min_freq):
        """Remove elements below the frequency cutoff from the dictionary."""
        to_remove = []
        for token, freq in self.freq.items():
            if freq < min_freq:
                # queue up removals since can't mutate dict during iteration
                to_remove.append(token)
        for token, freq in self.bi_freq.items():
            if freq < min_freq:
                to_remove.append(token)

        for token in to_remove:
            if token in self.tok2ind:
                del self.freq[token]
                idx = self.tok2ind.pop(token)
                del self.ind2tok[idx]
            elif token in self.bi2ind:
                del self.bi_freq[token]
                idx = self.bi2ind.pop(token)
                del self.ind2bi[idx]

    def resize_to_max(self, maxtokens):
        """Trims the dictionary to the maximum number of tokens."""
        if maxtokens >= 0 and len(self.tok2ind) > maxtokens:
            for k in range(maxtokens, len(self.ind2tok)):
                v = self.ind2tok[k]
                del self.ind2tok[k]
                del self.tok2ind[v]
                del self.freq[v]
        if maxtokens >= 0 and len(self.bi2ind) > maxtokens:
            for k in range(maxtokens, len(self.ind2bi)):
                v = self.ind2bi[k]
                del self.ind2bi[k]
                del self.bi2ind[v]
                del self.bi_freq[v]

    def load(self, filename):
        """
        Load pre-existing dictionary in 'token[<TAB>count]' format.

        Initialize counts from other dictionary, or 0 if they aren't included.
        """
        print('Dictionary: loading dictionary from {}'.format(filename))

        lower_special = self.null_token == self.null_token.lower()
        SPECIAL_TOKENS = {'__UNK__', '__NULL__', '__END__', '__START__'}
        with codecs.open(filename, 'r', encoding='utf-8', errors='ignore') as read:
            for line in read:
                split = line.strip().split('\t')
                token = unescape(split[0])
                if lower_special and token in SPECIAL_TOKENS:
                    token = token.lower()
                cnt = int(split[1]) if len(split) > 1 else 0

                if len(token.split(' ')) == 2:
                    self.bi_freq[token] = cnt
                    self.add_bi(token)
                elif token.startswith('__'):
                    self.bi_freq[token] = cnt
                    self.add_bi(token)
                    self.freq[token] = cnt
                    self.add_token(token)
                else:
                    self.freq[token] = cnt
                    self.add_token(token)
        print('[ num words =  %d ]' % len(self))

    def save(self, filename=None, append=False, sort=True):
        """
        Save dictionary to file.

        Format is 'token<TAB>count' for every token in the dictionary, sorted
        by count with the most frequent words first.

        If ``append`` (default ``False``) is set to ``True``, appends instead of
        overwriting.

        If ``sort`` (default ``True``), then first sort the dictionary before saving.
        """
        filename = self.opt['dict_file'] if filename is None else filename

        if self.tokenizer == 'bpe':
            needs_removal = self.bpehelper.finalize(
                self.freq, num_symbols=self.maxtokens, minfreq=self.minfreq
            )
            if needs_removal:
                self._remove_non_bpe()
            elif filename != self.opt['dict_file']:
                # need to copy over the old codecs file
                self.bpehelper.copy_codecs_file(filename + '.codecs')
            if sort:
                self.sort(trim=False)
        elif sort:
            self.sort(trim=True)

        print('Dictionary: saving dictionary to {}'.format(filename))

        make_dir(os.path.dirname(filename))
        mode = 'a' if append else 'w'
        with open(filename, mode, encoding='utf-8') as write:
            for i in self.ind2tok.keys():
                tok = self.ind2tok[i]
                cnt = self.freq[tok]
                write.write('{tok}\t{cnt}\n'.format(tok=escape(tok), cnt=cnt))
            for i in self.ind2bi.keys():
                tok = self.ind2bi[i] # ?Should combining bigram with "_" be done here or in act?
                cnt = self.bi_freq[tok]
                if not tok.startswith('__'):
                    write.write('{tok}\t{cnt}\n'.format(tok=escape(tok), cnt=cnt))


        # save opt file
        with open(filename + '.opt', 'w', encoding='utf-8') as handle:
            json.dump(self.opt, handle)

    def sort(self, trim=True):
        """
        Sort the dictionary.

        Inline operation. Rearranges the dictionary so that the elements with
        the lowest index have the highest counts. This reindexes the dictionary
        according to the sorted frequencies, breaking ties alphabetically by
        token.

        :param bool trim:
            If True, truncate the dictionary based on minfreq and maxtokens.
        """
        # sort first by count, then alphabetically
        if trim:
            self.remove_tail(self.minfreq)
        sorted_pairs = sorted(self.freq.items(), key=lambda x: (-x[1], x[0]))
        new_tok2ind = {}
        new_ind2tok = {}
        for i, (tok, _) in enumerate(sorted_pairs):
            new_tok2ind[tok] = i
            new_ind2tok[i] = tok

        sorted_pairs = sorted(self.bi_freq.items(), key=lambda x: (-x[1], x[0]))
        new_bi2ind = {}
        new_ind2bi = {}
        for i, (tok, _) in enumerate(sorted_pairs):
            new_bi2ind[tok] = i
            new_ind2bi[i] = tok

        self.tok2ind = new_tok2ind
        self.ind2tok = new_ind2tok
        self.bi2ind = new_bi2ind
        self.ind2bi = new_ind2bi

        if trim:
            self.resize_to_max(self.maxtokens)
        assert len(self.freq) == len(self.ind2tok) == len(self.tok2ind)
        assert len(self.bi_freq) == len(self.ind2bi) == len(self.bi2ind)
        return sorted_pairs

    def txt2vec(self, text, vec_type=list):
        """
        Convert a string to a vector (list of ints).

        First runs a sentence tokenizer, then a word tokenizer.

        :param type vec_type:
            The type of the returned vector if the input is a string. Suggested
            ``list``, ``tuple``, ``set``, or ``np.ndarray``.
        """
        text = text.strip().replace('、', '')
        if vec_type == list or vec_type == tuple or vec_type == set:
            tokens = self.tokenize(str(text))
            bigrams = list(ngrams(tokens, 2))
            uni_res = vec_type((self.getitem(token) for token in tokens))
            bi_res = vec_type((self.getbiitem(' '.join(bi)) for bi in bigrams))
        elif vec_type == np.ndarray:
            tokens = self.tokenize(str(text))
            uni_res = np.fromiter((self.getitem(token) for token in tokens), np.int)
            bi_res = np.fromiter((self.getbiitem(' '.join(bi)) for bi in ngrams(tokens, 2)), np.int)
        else:
            raise RuntimeError('Type {} not supported by dict'.format(vec_type))
        return uni_res, bi_res

    def vec2txt(self, vector, delimiter=' '):
        """
        Convert a vector of IDs to a string.

        Converts a vector (iterable of ints) into a string, with each token
        separated by the delimiter (default ``' '``).
        """
        print('Not supported')
        pass

    def act(self):
        """
        Add words in the last observation to the dictionary.

        This checks any fields in the message present in the --dict-textfields
        argument (e.g. "text,labels").
        """
        for textfield in self.textfields:
            source = self.observation.get(textfield)
            if source is None:
                continue
            # fields may be singleton strings or lists of strings.
            # wrap the singleton strings in a list to iterate over them
            if type(source) is str:
                source = [source]
            for text in source:
                if text:
                    bitokens = self.tokenize(text.replace('、', ''))
                    bigrams = [' '.join(bi) for bi in ngrams(bitokens, 2)]
                    self.add_to_dict(self.tokenize(text))
                    self.add_bi_to_dict(bigrams)
        return {'id': 'Dictionary'}
