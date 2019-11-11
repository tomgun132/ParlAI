import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import math

from nltk import ngrams
from parlai.agents.transformer.polyencoder import PolyencoderAgent
from parlai.agents.transformer.transformer import add_common_cmdline_args
from parlai.agents.bert_ranker.bi_encoder_ranker import BiEncoderRankerAgent, BiEncoderModule ,to_bert_input
from parlai.core.torch_ranker_agent import TorchRankerAgent
from parlai.core.torch_agent import Output

from parlai.utils.misc import (
    AttrDict,
    argsort,
    fp16_optimizer_wrapper,
    padded_tensor,
    warn_once,
    padded_3d,
    round_sigfigs,
)

from .modules import PolyAIEncoder, PolyEncoderBert
from .utils import UniBiDictionaryAgent, Batch

from tqdm import tqdm
from itertools import islice
from collections import namedtuple

class PolyAIJPRanker(TorchRankerAgent):

    @staticmethod
    def dictionary_class():
        """
        Return the dictionary class that this agent expects to use.

        Can be overriden if a more complex dictionary is required.
        """
        return UniBiDictionaryAgent

    @classmethod
    def add_cmdline_args(cls, argparser):
        agent = argparser.add_argument_group('PolyAIJPRanker')
        add_common_cmdline_args(agent)
        agent.add_argument(
            '--h-layer-num',
            type=int,
            default=3,
            help='Number of Feed forward network layer',
            hidden=True
        )
        agent.add_argument(
            '--h-dim',
            type=int,
            default=1024,
            help='Dimension size of Feed forward network layer',
            hidden=True
        )
        agent.add_argument(
            '--h-act-func',
            type=str,
            default='swish',
            help='Type of feed forward network activation function',
            hidden=True
        )
        agent.add_argument(
            '--linear-dim',
            type=int,
            default=512,
            help='Dimension size of linear layer',
            hidden=True
        )
        agent.add_argument(
            '--scoring-func',
            type=str,
            default='scaled',
            help='Type of scoring function whether it is scaled cosine function or dot product',
            hidden=True
        )

        super(PolyAIJPRanker, cls).add_cmdline_args(argparser)
        return agent

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.id = 'jp_ranker'

    # TODO Need to be overridded to add P1_TOKEN to bigram dict
    def build_dictionary(self):
        """
        Return the constructed dictionary, which will be set to self.dict.

        If you need to add additional tokens to the dictionary, this is likely
        the right place to do it.
        """
        d = self.dictionary_class()(self.opt)
        if self.opt.get('person_tokens'):
            d[self.P1_TOKEN] = 999_999_999
            d[self.P2_TOKEN] = 999_999_998
        return d

    def build_model(self, states=None):
        return PolyAIEncoder(self.opt, self.dict)

    def _vectorize_text(
        self, text, add_start=False, add_end=False, truncate=None, truncate_left=True
    ):
        """
        Return vector from text.

        :param text:
            String to vectorize.

        :param add_start:
            Add the start token to the front of the tensor.

        :param add_end:
            Add the end token to the end of the tensor.

        :param truncate:
            Truncate to this many tokens >= 0, or None.

        :param truncate_left:
            Truncate from the left side (keep the rightmost tokens). You
            probably want this True for inputs, False for targets.
        """
        # ? Need to be checked to make sure bigram tensor is correct
        vec, bi_vec = self.dict.txt2vec(text)
        vec = self._add_start_end_tokens(vec, add_start, add_end)
        vec = self._check_truncate(vec, truncate, truncate_left)
        tensor = torch.LongTensor(vec)
        bi_vec = self._add_start_end_tokens(bi_vec, add_start, add_end)
        bi_vec = self._check_truncate(bi_vec, truncate, truncate_left)
        bi_tensor = torch.LongTensor(bi_vec)
        return tensor, bi_tensor

    def _get_history_vec(self, hist):
        # ? Need to be checked to make sure bigram tensor is correct
        history_vecs = hist.get_history_vec_list()
        if len(history_vecs) == 0:
            return None

        # if self.vec_type == 'deque':
        #     history = deque(maxlen=hist.max_len)
        #     bi_history = deque(maxlen=hist.max_len)
        #     for vec, bi_vec in zip(history_vecs[:-1]):
        #         history.extend(vec)
        #         history.extend(hist.delimiter_tok)
        #         bi_history.extend(bi_vec)
        #         bi_history.extend(hist.delimiter_tok)
        #     history.extend(history_vecs[-1][0])
        #     bi_history.extend(history_vecs[-1][1])
        # else:
            # vec type is a list
        history = []
        bi_history = []
        if type(hist.delimiter_tok) == tuple:
            delimiter_tok = hist.delimiter_tok[0]
        for vecs in history_vecs[:-1]:
            vec, bi_vec = vecs
            history += vec
            history += delimiter_tok
            bi_history += bi_vec
            bi_history += delimiter_tok
        history += history_vecs[-1][0]
        bi_history += history_vecs[-1][1]


        return history, bi_history

    def _set_text_vec(self, obs, history, truncate):

        if 'text' not in obs:
            return obs

        if 'text_vec' not in obs:
            # text vec is not precomputed, so we set it using the history
            history_string = history.get_history_str()
            # when text not exist, we get text_vec from history string
            # history could be none if it is an image task and 'text'
            # filed is be empty. We don't want this
            if history_string is None:
                return obs
            obs['full_text'] = history_string
            # ? Need to be checked to make sure bigram tensor is correct
            if history_string:
                vec, bi_vec = self._get_history_vec(history)
                obs['text_vec'] = vec
                obs['bi_text_vec'] = bi_vec

        # check truncation
        if obs.get('text_vec') is not None:
            truncated_vec = self._check_truncate(obs['text_vec'], truncate, True)
            obs.force_set('text_vec', torch.LongTensor(truncated_vec))
        if obs.get('bi_text_vec') is not None:
            truncated_vec = self._check_truncate(obs['bi_text_vec'], truncate, True)
            obs.force_set('bi_text_vec', torch.LongTensor(truncated_vec))
        return obs

    def _set_label_vec(self, obs, add_start, add_end, truncate):
        # convert 'labels' or 'eval_labels' into vectors
        if 'labels' in obs:
            label_type = 'labels'
        elif 'eval_labels' in obs:
            label_type = 'eval_labels'
        else:
            label_type = None

        if label_type is None:
            return

        elif label_type + '_vec' in obs:
            # check truncation of pre-computed vector
            truncated_vec = self._check_truncate(obs[label_type + '_vec'], truncate)
            obs.force_set(label_type + '_vec', torch.LongTensor(truncated_vec))
            if 'bi_' + label_type + '_vec' in obs:
                truncated_vec = self._check_truncate(obs['bi_' + label_type + '_vec'], truncate)
                obs.force_set('bi_' + label_type + '_vec', torch.LongTensor(truncated_vec))
        else:
            # pick one label if there are multiple
            lbls = obs[label_type]
            label = lbls[0] if len(lbls) == 1 else self.random.choice(lbls)
            vec_label, bi_vec_label = self._vectorize_text(label, add_start, add_end, truncate, False)
            obs[label_type + '_vec'] = vec_label
            obs[label_type + '_choice'] = label
            obs['bi_' + label_type + '_vec'] = bi_vec_label

        return obs

    def _set_label_cands_vec(self, obs, add_start, add_end, truncate):
        if 'label_candidates_vecs' in obs:
            if truncate is not None:
                # check truncation of pre-computed vectors
                vecs = obs['label_candidates_vecs']
                for i, c in enumerate(vecs):
                    vecs[i] = self._check_truncate(c, truncate)
                if 'bi_label_candidates_vecs' in obs:
                    vecs = obs['bi_label_candidates_vecs']
                    for i, c in enumerate(vecs):
                        vecs[i] = self._check_truncate(c, truncate)
        elif self.rank_candidates and obs.get('label_candidates'):
            obs.force_set('label_candidates', list(obs['label_candidates']))
            label_vecs, bi_label_vecs = [], []
            for c in obs['label_candidates']:
                vec, bi_vec = self._vectorize_text(c, add_start, add_end, truncate, False)
                label_vecs.append(vec)
                bi_label_vecs.append(bi_vec)
            obs['label_candidates_vecs'] = label_vecs
            obs['bi_label_candidates_vecs'] = bi_label_vecs
        return obs

    def batchify(self, obs_batch, sort=False):
        """Override batchify options for seq2seq."""
        # kwargs['sort'] = True  # need sorted for pack_padded
        # batch = super().batchify(*args, **kwargs)
        if len(obs_batch) == 0:
            return Batch()

        is_valid = (
            lambda obs: 'text_vec' in obs and 'bi_text_vec' in obs
        )
        valid_obs = [(i, ex) for i, ex in enumerate(obs_batch) if is_valid(ex)]

        if len(valid_obs) == 0:
            return Batch()

        valid_inds, exs = zip(*valid_obs)

        # ? Need to be checked to make sure bigram tensor is correct
        # TEXT
        xs, bi_xs, x_lens, bi_xs_lens = None, None, None, None
        if any(ex.get('text_vec') is not None for ex in exs) and any(ex.get('bi_text_vec') is not None for ex in exs):
            _xs = [ex.get('text_vec', self.EMPTY) for ex in exs]
            _bi_xs = [ex.get('bi_text_vec', self.EMPTY) for ex in exs]
            xs, x_lens = padded_tensor(
                _xs, self.NULL_IDX, self.use_cuda, fp16friendly=self.opt.get('fp16')
            )
            bi_xs, bi_x_lens = padded_tensor(
                _bi_xs, self.NULL_IDX, self.use_cuda, fp16friendly=self.opt.get('fp16')
            )

            if sort:
                print('currently does not support sorted function because there are two inputs')
                # sort = False  # now we won't sort on labels
                # xs, x_lens, valid_inds, exs = argsort(
                #     x_lens, xs, x_lens, valid_inds, exs, descending=True
                # )
                # bi_xs, bi_x_lens, valid_inds, exs = argsort(
                #     x_lens, xs, x_lens, valid_inds, exs, descending=True
                # )

        # LABELS
        labels_avail = any('labels_vec' in ex for ex in exs) and any('bi_labels_vec' in ex for ex in exs)
        some_labels_avail = labels_avail or (any('eval_labels_vec' in ex for ex in exs) and any('eval_labels_vec' in ex for ex in exs))

        ys, bi_ys, y_lens, bi_y_lens, labels = None, None, None, None, None
        if some_labels_avail:
            field = 'labels' if labels_avail else 'eval_labels'

            label_vecs = [ex.get(field + '_vec', self.EMPTY) for ex in exs]
            bi_label_vecs = [ex.get('bi_' + field + '_vec', self.EMPTY) for ex in exs]
            labels = [ex.get(field + '_choice') for ex in exs]
            y_lens = [y.shape[0] for y in label_vecs]
            bi_y_lens = [y.shape[0] for y in bi_label_vecs]

            ys, y_lens = padded_tensor(
                label_vecs,
                self.NULL_IDX,
                self.use_cuda,
                fp16friendly=self.opt.get('fp16'),
            )
            bi_ys, bi_y_lens = padded_tensor(
                bi_label_vecs,
                self.NULL_IDX,
                self.use_cuda,
                fp16friendly=self.opt.get('fp16'),
            )
            if sort and xs is None:
                print('currently does not support sorted function because there are two inputs')
                # ys, valid_inds, label_vecs, labels, y_lens = argsort(
                #     y_lens, ys, valid_inds, label_vecs, labels, y_lens, descending=True
                # )

        # LABEL_CANDIDATES
        cands, cand_vecs, bi_cand_vecs = None, None, None
        if any('label_candidates_vecs' in ex for ex in exs):
            cands = [ex.get('label_candidates', None) for ex in exs]
            cand_vecs = [ex.get('label_candidates_vecs', None) for ex in exs]
            bi_cand_vecs = [ex.get('bi_label_candidates_vecs', None) for ex in exs]

        # IMAGE
        imgs = None
        if any('image' in ex for ex in exs):
            imgs = [ex.get('image', None) for ex in exs]

        return Batch(
            text_vec=xs,
            text_lengths=x_lens,
            bi_text_vec=bi_xs,
            bi_text_lengths=bi_x_lens,
            label_vec=ys,
            label_lengths=y_lens,
            bi_label_vec=bi_ys,
            bi_label_lengths=bi_y_lens,
            labels=labels,
            valid_indices=valid_inds,
            candidates=cands,
            candidate_vecs=cand_vecs,
            bi_candidate_vecs=bi_cand_vecs,
            image=imgs,
            observations=exs,
        )

    def score_candidates(self, batch, cand_vecs, cand_encs=None):
        if cand_encs:
            scores = self.model(batch.text_vec, batch.bi_text_vec, None, None, cand_encs)
        else:
            scores = self.model(batch.text_vec, batch.bi_text_vec, cand_vecs['uni'], cand_vecs['bi'])
        return scores

    def train_step(self, batch):
        out = super().train_step(batch)
        if self.opt['scoring_func'] == 'scaled':
            w = self.model.C.data
            w.clamp_(0, math.sqrt(self.opt['linear_dim']))
        return out

    def eval_step(self, batch):
        """Evaluate a single batch of examples."""
        if batch.text_vec is None and batch.image is None:
            return
        batchsize = (
            batch.text_vec.size(0)
            if batch.text_vec is not None
            else batch.image.size(0)
        )
        self.model.eval()

        cands, cand_vecs, label_inds = self._build_candidates(
            batch, source=self.eval_candidates, mode='eval'
        )

        cand_encs = None
        if self.encode_candidate_vecs and self.eval_candidates in ['fixed', 'vocab']:
            # if we cached candidate encodings for a fixed list of candidates,
            # pass those into the score_candidates function
            if self.eval_candidates == 'fixed':
                cand_encs = self.fixed_candidate_encs
            elif self.eval_candidates == 'vocab':
                cand_encs = self.vocab_candidate_encs

        scores = self.score_candidates(batch, cand_vecs, cand_encs=cand_encs)
        if self.rank_top_k > 0:
            _, ranks = scores.topk(
                min(self.rank_top_k, scores.size(1)), 1, largest=True
            )
        else:
            _, ranks = scores.sort(1, descending=True)

        # Update metrics
        if label_inds is not None:
            loss = self.criterion(scores, label_inds)
            self.metrics['loss'] += loss.item()
            self.metrics['examples'] += batchsize
            for b in range(batchsize):
                rank = (ranks[b] == label_inds[b]).nonzero()
                rank = rank.item() if len(rank) == 1 else scores.size(1)
                self.metrics['rank'] += 1 + rank
                self.metrics['mrr'] += 1.0 / (1 + rank)

        ranks = ranks.cpu()
        max_preds = self.opt['cap_num_predictions']
        cand_preds = []
        for i, ordering in enumerate(ranks):
            if cand_vecs['uni'].dim() == 2:
                cand_list = cands
            elif cand_vecs['uni'].dim() == 3:
                cand_list = cands[i]
            # using a generator instead of a list comprehension allows
            # to cap the number of elements.
            cand_preds_generator = (
                cand_list[rank] for rank in ordering if rank < len(cand_list)
            )
            cand_preds.append(list(islice(cand_preds_generator, max_preds)))

        if (
            self.opt.get('repeat_blocking_heuristic', True)
            and self.eval_candidates == 'fixed'
        ):
            cand_preds = self.block_repeats(cand_preds)

        preds = [cand_preds[i][0] for i in range(batchsize)]
        return Output(preds, cand_preds)

    # ? Need to be checked to make sure bigram tensors is correct
    def _build_candidates(self, batch, source, mode):
        label_vecs = batch.label_vec  # [bsz] list of lists of LongTensors
        bi_label_vecs = batch.bi_label_vec  # [bsz] list of lists of LongTensors
        label_inds = None
        batchsize = (
            batch.text_vec.size(0)
            if batch.text_vec is not None
            else batch.image.size(0)
        )

        if label_vecs is not None:
            assert label_vecs.dim() == 2
        if bi_label_vecs is not None:
            assert bi_label_vecs.dim() == 2

        if source == 'batch':
            warn_once(
                '[ Executing {} mode with batch labels as set of candidates. ]'
                ''.format(mode)
            )
            if batchsize == 1:
                warn_once(
                    "[ Warning: using candidate source 'batch' and observed a "
                    "batch of size 1. This may be due to uneven batch sizes at "
                    "the end of an epoch. ]"
                )
            if label_vecs is None or bi_label_vecs is None:
                raise ValueError(
                    "If using candidate source 'batch', then batch.label_vec or batch.bi_label_vec"
                    "cannot be None."
                )

            cands = batch.labels
            cand_vecs ={
                'uni': label_vecs,
                'bi' : bi_label_vecs,
            }
            label_inds = label_vecs.new_tensor(range(batchsize))

        elif source == 'batch-all-cands':
            warn_once(
                '[ Executing {} mode with all candidates provided in the batch ]'
                ''.format(mode)
            )
            if batch.candidate_vecs is None or batch.bi_candidate_vecs is None:
                raise ValueError(
                    "If using candidate source 'batch-all-cands', then batch."
                    "candidate_vecs cannot be None. If your task does not have "
                    "inline candidates, consider using one of "
                    "--{m}={{'batch','fixed','vocab'}}."
                    "".format(m='candidates' if mode == 'train' else 'eval-candidates')
                )
            # initialize the list of cands with the labels
            cands = []
            all_cands_vecs = []
            all_bi_cands_vecs = []
            # dictionary used for deduplication
            cands_to_id = {}
            for i, cands_for_sample in enumerate(batch.candidates):
                for j, cand in enumerate(cands_for_sample):
                    if cand not in cands_to_id:
                        cands.append(cand)
                        cands_to_id[cand] = len(cands_to_id)
                        all_cands_vecs.append(batch.candidate_vecs[i][j])
                        all_bi_cands_vecs.append(batch.bi_candidate_vecs[i][j])
            _cand_vecs, _ = padded_tensor(
                all_cands_vecs,
                self.NULL_IDX,
                use_cuda=self.use_cuda,
                fp16friendly=self.fp16,
            )
            _bi_cand_vecs, _ = padded_tensor(
                all_bi_cands_vecs,
                self.NULL_IDX,
                use_cuda=self.use_cuda,
                fp16friendly=self.fp16,
            )
            cand_vecs = {'uni': _cand_vecs, 'bi': _bi_cand_vecs}
            label_inds = label_vecs.new_tensor(
                [cands_to_id[label] for label in batch.labels]
            )

        elif source == 'inline':
            raise NotImplementedError("inline candidates is not supported")

        elif source == 'fixed':
            if self.fixed_candidates is None:
                raise ValueError(
                    "If using candidate source 'fixed', then you must provide the path "
                    "to a file of candidates with the flag --fixed-candidates-path or "
                    "the name of a task with --fixed-candidates-task."
                )
            warn_once(
                "[ Executing {} mode with a common set of fixed candidates "
                "(n = {}). ]".format(mode, len(self.fixed_candidates))
            )

            cands = self.fixed_candidates
            cand_vecs = self.fixed_candidate_vecs

            if label_vecs is not None:
                raise ValueError(
                    "currently cannot use fixed candidates together with label"
                )
                label_inds = label_vecs.new_empty((batchsize))
                bad_batch = False
                for batch_idx, label_vec in enumerate(label_vecs):
                    max_c_len = cand_vecs.size(1)
                    label_vec_pad = label_vec.new_zeros(max_c_len).fill_(self.NULL_IDX)
                    if max_c_len < len(label_vec):
                        label_vec = label_vec[0:max_c_len]
                    label_vec_pad[0 : label_vec.size(0)] = label_vec
                    label_inds[batch_idx] = self._find_match(cand_vecs, label_vec_pad)
                    if label_inds[batch_idx] == -1:
                        bad_batch = True
                if bad_batch:
                    if self.ignore_bad_candidates and not self.is_training:
                        label_inds = None
                    else:
                        raise RuntimeError(
                            'At least one of your examples has a set of label candidates '
                            'that does not contain the label. To ignore this error '
                            'set `--ignore-bad-candidates True`.'
                        )

        elif source == 'vocab':
            raise NotImplementedError
            warn_once(
                '[ Executing {} mode with tokens from vocabulary as candidates. ]'
                ''.format(mode)
            )
            cands = self.vocab_candidates
            cand_vecs = self.vocab_candidate_vecs
            # NOTE: label_inds is None here, as we will not find the label in
            # the set of vocab candidates
        else:
            raise Exception("Unrecognized source: %s" % source)

        return (cands, cand_vecs ,label_inds)

    def set_fixed_candidates(self, shared):
        """
        Load a set of fixed candidates and their vectors (or vectorize them here).

        self.fixed_candidates will contain a [num_cands] list of strings
        self.fixed_candidate_vecs will contain a [num_cands, seq_len] LongTensor

        See the note on the --fixed-candidate-vecs flag for an explanation of the
        'reuse', 'replace', or path options.

        Note: TorchRankerAgent by default converts candidates to vectors by vectorizing
        in the common sense (i.e., replacing each token with its index in the
        dictionary). If a child model wants to additionally perform encoding, it can
        overwrite the vectorize_fixed_candidates() method to produce encoded vectors
        instead of just vectorized ones.
        """
        if shared:
            self.fixed_candidates = shared['fixed_candidates']
            self.fixed_candidate_vecs = shared['fixed_candidate_vecs']
            self.fixed_candidate_encs = shared['fixed_candidate_encs']
        else:
            opt = self.opt
            cand_path = self.fixed_candidates_path
            if 'fixed' in (self.candidates, self.eval_candidates):
                if not cand_path:
                    # Attempt to get a standard candidate set for the given task
                    path = self.get_task_candidates_path()
                    if path:
                        print("[setting fixed_candidates path to: " + path + " ]")
                        self.fixed_candidates_path = path
                        cand_path = self.fixed_candidates_path
                # Load candidates
                print("[ Loading fixed candidate set from {} ]".format(cand_path))
                with open(cand_path, 'r', encoding='utf-8') as f:
                    cands = [line.strip() for line in f.readlines()]
                # Load or create candidate vectors
                if os.path.isfile(self.opt['fixed_candidate_vecs']):
                    vecs_path = opt['fixed_candidate_vecs']
                    vecs = self.load_candidates(vecs_path)
                else:
                    setting = self.opt['fixed_candidate_vecs']
                    model_dir, model_file = os.path.split(self.opt['model_file'])
                    model_name = os.path.splitext(model_file)[0]
                    cands_name = os.path.splitext(os.path.basename(cand_path))[0]
                    vecs_path = os.path.join(
                        model_dir, '.'.join([model_name, cands_name, 'vecs'])
                    )
                    if setting == 'reuse' and os.path.isfile(vecs_path):
                        vecs = self.load_candidates(vecs_path)
                    else:  # setting == 'replace' OR generating for the first time
                        vecs = self._make_candidate_vecs(cands)
                        self._save_candidates(vecs, vecs_path)

                self.fixed_candidates = cands
                self.fixed_candidate_vecs = vecs
                if self.use_cuda:
                    for k, v in self.fixed_candidate_vecs.items():
                        self.fixed_candidate_vecs[k] = v.cuda()

                if self.encode_candidate_vecs:
                    # candidate encodings are fixed so set them up now
                    enc_path = os.path.join(
                        model_dir, '.'.join([model_name, cands_name, 'encs'])
                    )
                    if setting == 'reuse' and os.path.isfile(enc_path):
                        encs = self.load_candidates(enc_path, cand_type='encodings')
                    else:
                        encs = self._make_candidate_encs(self.fixed_candidate_vecs)
                        self._save_candidates(
                            encs, path=enc_path, cand_type='encodings'
                        )
                    self.fixed_candidate_encs = encs
                    if self.use_cuda:
                        self.fixed_candidate_encs = self.fixed_candidate_encs.cuda()
                    if self.fp16:
                        self.fixed_candidate_encs = self.fixed_candidate_encs.half()
                    else:
                        self.fixed_candidate_encs = self.fixed_candidate_encs.float()
                else:
                    self.fixed_candidate_encs = None

            else:
                self.fixed_candidates = None
                self.fixed_candidate_vecs = None
                self.fixed_candidate_encs = None

    def _make_candidate_vecs(self, cands):
        """Prebuild cached vectors for fixed candidates."""
        cand_batches = [cands[i : i + 512] for i in range(0, len(cands), 512)]
        print(
            "[ Vectorizing fixed candidate set ({} batch(es) of up to 512) ]"
            "".format(len(cand_batches))
        )
        cand_vecs = []
        bi_cand_vecs = []
        for batch in tqdm(cand_batches):
            batch_vecs, batch_bi_vecs = self.vectorize_fixed_candidates(batch)
            cand_vecs.extend(batch_vecs)
            bi_cand_vecs.extend(batch_bi_vecs)
        return {'uni': padded_3d([cand_vecs], dtype=cand_vecs[0].dtype).squeeze(0),
            'bi': padded_3d([bi_cand_vecs], dtype=bi_cand_vecs[0].dtype).squeeze(0)}

    def encode_candidates(self, padded_cands, bi_padded_cands):
        padded_cands = padded_cands.unsqueeze(1)
        bi_padded_cands = bi_padded_cands.unsqueeze(1)
        _, cand_rep = self.model.encode(None, None, padded_cands, bi_padded_cands)
        return cand_rep

    def _make_candidate_encs(self, vecs):
        """
        Encode candidates from candidate vectors.

        Requires encode_candidates() to be implemented.
        """

        cand_encs = []

        vec_batches = [vecs['uni'][i : i + 256] for i in range(0, len(vecs['uni']), 256)]
        bi_vec_batches = [vecs['bi'][i : i + 256] for i in range(0, len(vecs['bi']), 256)]
        print(
            "[ Encoding fixed candidates set from ({} batch(es) of up to 256) ]"
            "".format(len(vec_batches))
        )
        # Put model into eval mode when encoding candidates
        self.model.eval()
        with torch.no_grad():
            for vec_batch, bi_vec_batch in tqdm(zip(vec_batches, bi_vec_batches)):
                cand_encs.append(self.encode_candidates(vec_batch, bi_vec_batch))
        return torch.cat(cand_encs, 0)

    def vectorize_fixed_candidates(self, cands_batch, add_start=False, add_end=False):
        vecs = []
        bi_vecs = []
        for cand in cands_batch:
            vec, bi_vec = self._vectorize_text(
                    cand,
                    truncate=self.label_truncate,
                    truncate_left=False,
                    add_start=add_start,
                    add_end=add_end,
                    )
            vecs.append(vec)
            bi_vecs.append(bi_vec)
        return vecs, bi_vecs

class JPRanker(PolyencoderAgent):
    @classmethod
    def add_cmdline_args(cls, argparser):
        """Add command-line arguments specifically for this agent."""
        agent = argparser.add_argument_group('jpseq2seq Arguments')
        agent.add_argument(
            '--train-folder',
            type=str,
            default='jp_dialogue',
            help='training data location'
        )
        super(JPRanker, cls).add_cmdline_args(argparser)
        return agent

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.id = 'jp_ranker'

    # Overrided to get japanese fasttext vector
    def _get_embtype(self, emb_type):
        try:
            import torchtext.vocab as vocab
        except ImportError as ex:
            print('Please install torch text with `pip install torchtext`')
            raise ex
        if emb_type.startswith('fasttext'):
            from ..fasttext_cc_jp.build import download
            init = 'fasttext_cc_jp'
            embs = download(self.opt.get('datapath'))
        else:
            raise RuntimeError(
                'embedding type {} not implemented. '
                'For Japanese vectors, only fasttext has been implemented'
                ''.format(emb_type)
            )

        return embs, init

class BertJPRanker(BiEncoderRankerAgent):
    @classmethod
    def add_cmdline_args(cls, argparser):
        """Add command-line arguments specifically for this agent."""
        agent = argparser.add_argument_group('jpseq2seq Arguments')
        agent.add_argument(
            '--train-folder',
            type=str,
            default='jp_dialogue',
            help='training data location'
        )
        agent.add_argument(
            '--attention-type',
            type=str,
            default='basic',
            choices=['basic', 'sqrt'],
            help='Type of the top aggregation layer of the poly-'
            'encoder (where the candidate representation is'
            'the key)',
        )
        agent.add_argument(
            '--n-codes',
            type=int,
            default=64,
            help='number of vectors used to represent the context'
            'these are the number of vectors that are considered.',
        )
        agent.add_argument(
            '--context-model',
            type=str,
            default='basic',
            choices=['basic', 'poly'],
            help='type of context encoder that will be used:'
            'basic means it is a normal bi-encoder bert model,'
            'poly is by using polyencoder-like for computing'
            'context representation and scoring',
        )
        super(BertJPRanker, cls).add_cmdline_args(argparser)
        return agent

    def __init__(self, opt, shared=None):
        self.model_type = opt['context_model']
        super().__init__(opt, shared)
        self.id = 'bert_jp_ranker'

    def build_model(self):
        if self.model_type == 'basic':
            return BiEncoderModule(self.opt)
        elif self.model_type == 'poly':
            return PolyEncoderBert(self.opt)

    def batchify(self, *args, **kwargs):
        """Override batchify"""
        kwargs['sort'] = True  # need sorted for pack_padded
        batch = super().batchify(*args, **kwargs)

        obs_batch = args[0]
        sort = kwargs['sort']
        is_valid = (
            lambda obs: 'text_vec' in obs or 'image' in obs
        )

        if len(obs_batch) == 0:
            return Batch()

        valid_obs = [(i, ex) for i, ex in enumerate(obs_batch) if is_valid(ex)]

        if len(valid_obs) == 0:
            return Batch()

        valid_inds, exs = zip(*valid_obs)

        if any('topic' in ex for ex in exs):
            topics = [ex.get('topic', None) for ex in exs]
            new_batch = namedtuple('Batch', tuple(batch.keys()) + tuple(['topics']))
            batch = new_batch(topics=topics, **dict(batch))

        return batch

    def eval_step(self, batch):
        """Evaluate a single batch of examples."""
        if batch.text_vec is None and batch.image is None:
            return
        batchsize = (
            batch.text_vec.size(0)
            if batch.text_vec is not None
            else batch.image.size(0)
        )
        self.model.eval()

        cands, cand_vecs, label_inds = self._build_candidates(
            batch, source=self.eval_candidates, mode='eval'
        )

        cand_encs = None
        if self.encode_candidate_vecs and self.eval_candidates in ['fixed', 'vocab']:
            # if we cached candidate encodings for a fixed list of candidates,
            # pass those into the score_candidates function
            if self.eval_candidates == 'fixed':
                cand_encs = self.fixed_candidate_encs
            elif self.eval_candidates == 'vocab':
                cand_encs = self.vocab_candidate_encs

        """
        If we group responses based on the topic
        """
        if type(cand_vecs) == dict:
            # raise NotImplementedError("Currently, interactive with topic cannot be called from act() function")
            if batch.topics is None:
                raise AttributeError("Topic must exist in the batch if the fixed candidates come from json file")
            topic = batch.topics[0] # This is assuming interactive mode where batch_size == 1
            if topic is not None:
                cands = cands[topic]
                cand_vecs = cand_vecs[topic]
                cand_encs = cand_encs[topic]
            else:
                raise AttributeError("Topic must exist in the batch if the fixed candidates come from json file")
        scores = self.score_candidates(batch, cand_vecs, cand_encs=cand_encs)

        if self.rank_top_k > 0:
            sorted_scores, ranks = scores.topk(
                min(self.rank_top_k, scores.size(1)), 1, largest=True
            )
        else:
            sorted_scores, ranks = scores.sort(1, descending=True)

        sorted_scores = F.softmax(sorted_scores, dim=-1)
        # Update metrics
        if label_inds is not None:
            loss = self.criterion(scores, label_inds)
            self.metrics['loss'] += loss.item()
            self.metrics['examples'] += batchsize
            for b in range(batchsize):
                rank = (ranks[b] == label_inds[b]).nonzero()
                rank = rank.item() if len(rank) == 1 else scores.size(1)
                self.metrics['rank'] += 1 + rank
                self.metrics['mrr'] += 1.0 / (1 + rank)

        ranks = ranks.cpu()
        max_preds = self.opt['cap_num_predictions']
        cand_preds = []
        for i, ordering in enumerate(ranks):
            if cand_vecs.dim() == 2:
                cand_list = cands
            elif cand_vecs.dim() == 3:
                cand_list = cands[i]
            # using a generator instead of a list comprehension allows
            # to cap the number of elements.
            cand_preds_generator = (
                cand_list[rank] for rank in ordering if rank < len(cand_list)
            )
            cand_preds.append(list(islice(cand_preds_generator, max_preds)))

        if (
            self.opt.get('repeat_blocking_heuristic', True)
            and self.eval_candidates == 'fixed'
        ):
            cand_preds = self.block_repeats(cand_preds)

        preds = [cand_preds[i][0] for i in range(batchsize)]
        return Output(preds, cand_preds, scores=sorted_scores)

    def match_batch(self, batch_reply, valid_inds, output=None):
        if output is None:
            return batch_reply
        if output.text is not None:
            for i, response in zip(valid_inds, output.text):
                batch_reply[i]['text'] = response
        if output.text_candidates is not None:
            for i, cands in zip(valid_inds, output.text_candidates):
                batch_reply[i]['text_candidates'] = cands
        if 'scores' in output and output.scores is not None:
            for i, score in zip(valid_inds, output.scores):
                batch_reply[i]['scores'] = score
        return batch_reply

    def score_candidates(self, batch, cand_vecs, cand_encs=None):
        # Encode contexts first
        token_idx_ctxt, segment_idx_ctxt, mask_ctxt = to_bert_input(
            batch.text_vec, self.NULL_IDX
        )
        ctxt_out, _ = self.model(
            token_idx_ctxt, segment_idx_ctxt, mask_ctxt, None, None, None
        )

        if self.model_type == 'poly':
            embedding_ctxt, mask_ctxt = ctxt_out
        else:
            embedding_ctxt = ctxt_out
        bsz = token_idx_ctxt.size(0)
        if cand_encs is not None:
            if self.model_type == 'poly':
                if bsz == 1:
                    cand_rep = cand_encs.unsqueeze(0)
                else:
                    cand_rep = cand_encs.unsqueeze(0).expand(bsz, cand_encs.size(0), -1)
                return self.model.score(embedding_ctxt, mask_ctxt, cand_rep)
            else:
                return embedding_ctxt.mm(cand_encs.t())

        if len(cand_vecs.size()) == 2 and cand_vecs.dtype == torch.long:
            # train time. We compare with all elements of the batch
            token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
                cand_vecs, self.NULL_IDX
            )
            _, embedding_cands = self.model(
                None, None, None, token_idx_cands, segment_idx_cands, mask_cands
            )

            if self.model_type == 'poly':
                num_cands = embedding_cands.size(0)  # will be bsz if using batch cands
                cand_rep = embedding_cands.unsqueeze(1).expand(num_cands, bsz, -1).transpose(0, 1).contiguous()
                return self.model.score(embedding_ctxt, mask_ctxt, cand_rep)
            else:
                return embedding_ctxt.mm(embedding_cands.t())

        # predict time with multiple candidates
        if len(cand_vecs.size()) == 3:
            csize = cand_vecs.size()  # batchsize x ncands x sentlength
            cands_idx_reshaped = cand_vecs.view(csize[0] * csize[1], csize[2])
            token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
                cands_idx_reshaped, self.NULL_IDX
            )
            _, embedding_cands = self.model(
                None, None, None, token_idx_cands, segment_idx_cands, mask_cands
            )
            embedding_cands = embedding_cands.view(
                csize[0], csize[1], -1
            )  # batchsize x ncands x embed_size

            if self.model_type == 'poly':
                return self.model.score(embedding_ctxt, mask_ctxt, embedding_cands)
            else:
                embedding_cands = embedding_cands.transpose(
                    1, 2
                )  # batchsize x embed_size x ncands
                embedding_ctxt = embedding_ctxt.unsqueeze(1)  # batchsize x 1 x embed_size
                scores = torch.bmm(
                    embedding_ctxt, embedding_cands
                )  # batchsize x 1 x ncands
                scores = scores.squeeze(1)  # batchsize x ncands
                return scores

        # otherwise: cand_vecs should be 2D float vector ncands x embed_size
        return embedding_ctxt.mm(cand_vecs.t())

    def _build_candidates(self, batch, source, mode):
        label_vecs = batch.label_vec
        if type(self.fixed_candidate_vecs) == dict and label_vecs is not None:
            raise TypeError("Fixed candidate with topic can only be done in eval step without label")

        return super()._build_candidates(batch, source, mode)

    def set_fixed_candidates(self, shared):
        """
        Load a set of fixed candidates and their vectors (or vectorize them here).

        self.fixed_candidates will contain a [num_cands] list of strings
        self.fixed_candidate_vecs will contain a [num_cands, seq_len] LongTensor

        See the note on the --fixed-candidate-vecs flag for an explanation of the
        'reuse', 'replace', or path options.

        Note: TorchRankerAgent by default converts candidates to vectors by vectorizing
        in the common sense (i.e., replacing each token with its index in the
        dictionary). If a child model wants to additionally perform encoding, it can
        overwrite the vectorize_fixed_candidates() method to produce encoded vectors
        instead of just vectorized ones.
        """
        if shared:
            self.fixed_candidates = shared['fixed_candidates']
            self.fixed_candidate_vecs = shared['fixed_candidate_vecs']
            self.fixed_candidate_encs = shared['fixed_candidate_encs']
        else:
            opt = self.opt
            cand_path = self.fixed_candidates_path
            if 'fixed' in (self.candidates, self.eval_candidates):
                if not cand_path:
                    # Attempt to get a standard candidate set for the given task
                    path = self.get_task_candidates_path()
                    if path:
                        print("[setting fixed_candidates path to: " + path + " ]")
                        self.fixed_candidates_path = path
                        cand_path = self.fixed_candidates_path
                # Load candidates
                print("[ Loading fixed candidate set from {} ]".format(cand_path))
                is_topical = False
                with open(cand_path, 'r', encoding='utf-8') as f:
                    if 'json' in cand_path:
                        cands = json.load(f)
                        is_topical = True
                    else:
                        cands = [line.strip() for line in f.readlines()]
                # Load or create candidate vectors
                if os.path.isfile(self.opt['fixed_candidate_vecs']):
                    vecs_path = opt['fixed_candidate_vecs']
                    vecs = self.load_candidates(vecs_path)
                else:
                    setting = self.opt['fixed_candidate_vecs']
                    model_dir, model_file = os.path.split(self.opt['model_file'])
                    model_name = os.path.splitext(model_file)[0]
                    cands_name = os.path.splitext(os.path.basename(cand_path))[0]
                    vecs_path = os.path.join(
                        model_dir, '.'.join([model_name, cands_name, 'vecs'])
                    )
                    if setting == 'reuse' and os.path.isfile(vecs_path):
                        vecs = self.load_candidates(vecs_path)
                    else:  # setting == 'replace' OR generating for the first time
                        vecs = self._make_candidate_vecs(cands)
                        self._save_candidates(vecs, vecs_path)

                self.fixed_candidates = cands
                self.fixed_candidate_vecs = vecs
                if self.use_cuda:
                    if is_topical:
                        for topic, vec in self.fixed_candidate_vecs.items():
                            self.fixed_candidate_vecs[topic] = vec.cuda()
                    else:
                        self.fixed_candidate_vecs = self.fixed_candidate_vecs.cuda()

                if self.encode_candidate_vecs:
                    # candidate encodings are fixed so set them up now
                    enc_path = os.path.join(
                        model_dir, '.'.join([model_name, cands_name, 'encs'])
                    )
                    if setting == 'reuse' and os.path.isfile(enc_path):
                        encs = self.load_candidates(enc_path, cand_type='encodings')
                    else:
                        encs = self._make_candidate_encs(self.fixed_candidate_vecs)
                        self._save_candidates(
                            encs, path=enc_path, cand_type='encodings'
                        )
                    self.fixed_candidate_encs = encs
                    if self.use_cuda:
                        if is_topical:
                            for topic, enc in self.fixed_candidate_encs.items():
                                self.fixed_candidate_encs[topic] = enc.cuda()
                        else:
                            self.fixed_candidate_encs = self.fixed_candidate_encs.cuda()
                    if self.fp16:
                        if is_topical:
                            for topic, enc in self.fixed_candidate_encs.items():
                                self.fixed_candidate_encs[topic] = enc.half()
                        else:
                            self.fixed_candidate_encs = self.fixed_candidate_encs.half()
                    else:
                        if is_topical:
                            for topic, enc in self.fixed_candidate_encs.items():
                                self.fixed_candidate_encs[topic] = enc.float()
                        else:
                            self.fixed_candidate_encs = self.fixed_candidate_encs.float()
                else:
                    self.fixed_candidate_encs = None

            else:
                self.fixed_candidates = None
                self.fixed_candidate_vecs = None
                self.fixed_candidate_encs = None

    def _make_candidate_vecs(self, cands):
        """Prebuild cached vectors for fixed candidates."""
        if isinstance(cands, dict):
            vec_dict = dict()
            for topic, cand_list in cands.items():
                vecs = self.vectorize_fixed_candidates(cand_list)
                vec_dict[topic] = padded_3d([vecs], dtype=vecs[0].dtype).squeeze(0)
            return vec_dict
        else:
            cand_batches = [cands[i : i + 512] for i in range(0, len(cands), 512)]
            print(
                "[ Vectorizing fixed candidate set ({} batch(es) of up to 512) ]"
                "".format(len(cand_batches))
            )
            cand_vecs = []
            for batch in tqdm(cand_batches):
                cand_vecs.extend(self.vectorize_fixed_candidates(batch))
            return padded_3d([cand_vecs], dtype=cand_vecs[0].dtype).squeeze(0)

    def _make_candidate_encs(self, vecs):
        """
        Encode candidates from candidate vectors.

        Requires encode_candidates() to be implemented.
        """
        if isinstance(vecs, dict):
            encs_dict = {}
            for topic, cand_vec in vecs.items():
                cand_encs = []
                vec_batches = [cand_vec[i : i + 256] for i in range(0, len(cand_vec), 256)]
                print(
                    "[ Encoding fixed candidates set from ({} batch(es) of up to 256) ]"
                    "".format(len(vec_batches))
                )
                # Put model into eval mode when encoding candidates
                self.model.eval()
                with torch.no_grad():
                    for vec_batch in tqdm(vec_batches):
                        cand_encs.append(self.encode_candidates(vec_batch))

                encs_dict[topic] = torch.cat(cand_encs, 0)

            return encs_dict
        else:
            cand_encs = []
            vec_batches = [vecs[i : i + 256] for i in range(0, len(vecs), 256)]
            print(
                "[ Encoding fixed candidates set from ({} batch(es) of up to 256) ]"
                "".format(len(vec_batches))
            )
            # Put model into eval mode when encoding candidates
            self.model.eval()
            with torch.no_grad():
                for vec_batch in tqdm(vec_batches):
                    cand_encs.append(self.encode_candidates(vec_batch))
            return torch.cat(cand_encs, 0)
