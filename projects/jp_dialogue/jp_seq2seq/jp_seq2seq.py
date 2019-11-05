from parlai.agents.seq2seq.seq2seq import Seq2seqAgent
from parlai.core.torch_ranker_agent import TorchRankerAgent
from parlai.core.dict import DictionaryAgent
from parlai.core.utils import neginf
from parlai.core.torch_agent import Output, Batch

from .util import ConvAI2History, show_beam_cands, reorder_extrep2gram_qn
from .controls import (
    CONTROL2DEFAULTNUMBUCKETS,
    CONTROL2DEFAULTEMBSIZE,
    WDFEATURE2UPDATEFN,
    get_ctrl_vec,
    get_wd_features,
    initialize_control_information,
)

from collections import defaultdict, namedtuple, Counter
from operator import attrgetter

import torch
import torch.nn as nn
import torch.nn.functional as F

import codecs
import os
import json
import numpy as np
import math

def _transpose_hidden_state(hidden_state):
    """
    Transpose the hidden state so that batch is the first dimension.

    RNN modules produce (num_layers x batchsize x dim) hidden state, but
    DataParallel expects batch size to be first. This helper is used to
    ensure that we're always outputting batch-first, in case DataParallel
    tries to stitch things back together.
    """
    if isinstance(hidden_state, tuple):
        return tuple(map(_transpose_hidden_state, hidden_state))
    elif torch.is_tensor(hidden_state):
        return hidden_state.transpose(0, 1)
    else:
        raise ValueError("Don't know how to transpose {}".format(hidden_state))

class JPSeq2seq(Seq2seqAgent):

    @classmethod
    def add_cmdline_args(cls, argparser):
        """Add command-line arguments specifically for this agent."""
        agent = argparser.add_argument_group('jpseq2seq Arguments')
        agent.add_argument(
            '--train-folder',
            type=str,
            help='training data location'
        )
        super(JPSeq2seq, cls).add_cmdline_args(argparser)
        return agent

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.id = 'jp_seq2seq'

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

class ControllableJPSeq2seq(Seq2seqAgent):

    @classmethod
    def add_cmdline_args(cls, argparser):
        """Add command-line arguments specifically for this agent."""
        agent = argparser.add_argument_group('ctrljpseq2seq Arguments')
        agent.add_argument(
            '--train-folder',
            type=str,
            help='training data location'
        )
        agent.add_argument(
            '--weighted-decoding',
            type=str,
            default='',
            help='List of WD features and their corresponding weights '
            'For example, intrep_word:-1,extrep_2gram:-1,nidf:3',
        )
        agent.add_argument(
            '--beam-reorder',
            default='none',
            choices=['none', 'best_extrep2gram_qn'],
            help='Choices: none, best_extrep2gram_qn.'
            'Apply the specified function for reordering the '
            'n-best beam search candidates. '
            'If best_extrep2gram_qn, then pick candidate which '
            'contains question mark and has lowest extrep_2gram',
        )
        agent.add_argument(
            '--beam-block-ngram',
            type=int,
            default=0,
            help='Block all repeating ngrams up to history length n-1',
        )
        super(ControllableJPSeq2seq, cls).add_cmdline_args(argparser)
        return agent

    def __init__(self, opt, shared=None):
        if not shared:
            initialize_control_information(opt)
        super().__init__(opt, shared)
        self.id = 'ctrljp_seq2seq'
        self.multigpu = (
            opt.get('multigpu') and self.use_cuda and (opt.get('batchsize') > 1)
        )
        self.beam_block_ngram = opt.get('beam_block_ngram', 0)

        if self.opt.get('weighted_decoding', '') != "":
            if self.beam_size == 1:
                raise ValueError(
                    "WD control is not currently implemented for greedy "
                    "search. Either increase --beam-size to be greater "
                    "than 1, or do not enter --weighted-decoding (-wd)."
                )

            # Get a list of (feature, weight) i.e. (string, float) pairs
            wd_feats_wts = [
                (s.split(':')[0], float(s.split(':')[1]))
                for s in self.opt['weighted_decoding'].split(',')
            ]
            self.wd_features = [f for (f, w) in wd_feats_wts]  # list of strings
            for wd_feat in self.wd_features:
                if wd_feat not in WDFEATURE2UPDATEFN:
                    raise ValueError(
                        "'%s' is not an existing WD feature. Available WD "
                        "features: %s" % (wd_feat, ', '.join(WDFEATURE2UPDATEFN.keys()))
                    )
            self.wd_wts = [w for (f, w) in wd_feats_wts]  # list of floats
        else:
            self.wd_features, self.wd_wts = [], []

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

    def batchify(self, *args, **kwargs):
        """Override batchify options for seq2seq.""" 
        # kwargs['sort'] = True  # need sorted for pack_padded
        batch = super().batchify(*args, **kwargs)
        obs_batch = args[0]
        is_valid = (
            lambda obs: 'text_vec' in obs or 'image' in obs
        )  # from TorchAgent.batchify

        if len(obs_batch) == 0:
            return Batch()

        valid_obs = [(i, ex) for i, ex in enumerate(obs_batch) if is_valid(ex)]

        if len(valid_obs) == 0:
            return Batch()

        valid_inds, exs = zip(*valid_obs)

        # Add history to the batch
        history = [ConvAI2History(ex['full_text'], dictionary=self.dict) for ex in exs]

        ControlBatch = namedtuple(
            'Batch', tuple(batch.keys()) + ('ctrl_vec', 'history')
        )
        batch = ControlBatch(ctrl_vec=None, history=history, **dict(batch))

        return batch


    @staticmethod
    def beam_search(
        model,
        batch,
        beam_size,
        dictionary,
        start=1,
        end=2,
        pad=0,
        min_length=3,
        min_n_best=5,
        max_ts=40,
        block_ngram=0,
        wd_features=None,
        wd_wts=None,
    ):
        """
        Beam search given the model and Batch.

        This function uses model with the following reqs:

        - model.encoder takes input returns tuple (enc_out, enc_hidden, attn_mask)
        - model.decoder takes decoder params and returns decoder outputs after attn
        - model.output takes decoder outputs and returns distr over dictionary

        :param model: nn.Module, here defined in modules.py
        :param batch: Batch structure with input and labels
        :param beam_size: Size of each beam during the search
        :param start: start of sequence token
        :param end: end of sequence token
        :param pad: padding token
        :param min_length: minimum length of the decoded sequence
        :param min_n_best: minimum number of completed hypothesis generated
            from each beam
        :param max_ts: the maximum length of the decoded sequence
        :param wd_features: list of strings, the WD features to use
        :param wd_weights: list of floats, the WD weights to use

        :return:
            - beam_preds_scores : list of tuples (prediction, score) for each
              sample in Batch
            - n_best_preds_scores : list of n_best list of tuples (prediction,
              score) for each sample from Batch
            - beams : list of Beam instances defined in Beam class, can be used
              for any following postprocessing, e.g. dot logging.
        """
        if wd_features is None:
            wd_features = []
        if wd_wts is None:
            wd_wts = []
        encoder_states = model.encoder(batch.text_vec)
        current_device = encoder_states[0][0].device
        vocab_size = len(dictionary)

        batch_size = len(batch.text_lengths)
        beams = [
            Beam(
                beam_size,
                min_length=min_length,
                padding_token=pad,
                bos_token=start,
                eos_token=end,
                min_n_best=min_n_best,
                cuda=current_device,
                block_ngram=block_ngram,
            )
            for i in range(batch_size)
        ]
        decoder_input = (
            torch.Tensor([start])
            .detach()
            .expand(batch_size, 1)
            .long()
            .to(current_device)
        )
        # repeat encoder_outputs, hiddens, attn_mask
        decoder_input = decoder_input.repeat(1, beam_size).view(
            beam_size * batch_size, -1
        )

        inds = torch.arange(batch_size).to(current_device).unsqueeze(1).repeat(1, beam_size).view(-1)
        encoder_states = model.reorder_encoder_states(encoder_states, inds)
        # ctrl_input is shape (bsz, num_controls)
        # we want it to be (bsz*beam_size, num_controls)
        # ctrl_input = batch.ctrl_vec
        # if batch.ctrl_vec is not None:
        #     ctrl_input = batch.ctrl_vec.repeat(beam_size, 1)

        # enc_out = (
        #     enc_out.unsqueeze(1)
        #     .repeat(1, beam_size, 1, 1)
        #     .view(batch_size * beam_size, -1, enc_out.size(-1))
        # )
        # attn_mask = (
        #     encoder_states[2]
        #     .repeat(1, beam_size)
        #     .view(attn_mask.size(0) * beam_size, -1)
        # )
        # repeated_hiddens = []
        # if isinstance(enc_hidden, tuple):  # LSTM
        #     for i in range(len(enc_hidden)):
        #         repeated_hiddens.append(
        #             enc_hidden[i].unsqueeze(2).repeat(1, 1, beam_size, 1)
        #         )
        #     num_layers = enc_hidden[0].size(0)
        #     hidden_size = enc_hidden[0].size(-1)
        #     enc_hidden = tuple(
        #         [
        #             repeated_hiddens[i].view(
        #                 num_layers, batch_size * beam_size, hidden_size
        #             )
        #             for i in range(len(repeated_hiddens))
        #         ]
        #     )
        # else:  # GRU
        #     num_layers = enc_hidden.size(0)
        #     hidden_size = enc_hidden.size(-1)
        #     enc_hidden = (
        #         enc_hidden.unsqueeze(2)
        #         .repeat(1, 1, beam_size, 1)
        #         .view(num_layers, batch_size * beam_size, hidden_size)
        #     )
        hidden = None
        for ts in range(max_ts):
            if all((b.done() for b in beams)):
                break
            output, hidden = model.decoder(
                decoder_input, encoder_states, hidden
            )
            
            score = model.output(output)
            # score contains softmax scores for batch_size * beam_size samples
            score = score.view(batch_size, beam_size, -1)
            score = F.log_softmax(score, dim=-1)
            for i, b in enumerate(beams):
                if not b.done():
                    scores_in = score[i]

                    # If using WD, update scores_in to reflect the WD features
                    if len(wd_features) > 0:

                        # Obtain wd_feat_vecs, the sum of the weighted features
                        # across the whole vocabulary
                        wd_feat_vecs = torch.zeros((beam_size, vocab_size))
                        for hyp_idx in range(beam_size):  # For each hypothesis

                            # Get the partial hypothesis (None if first timestep)
                            partial_hyp = b.partial_hyps[hyp_idx] if ts > 0 else None

                            # Get the WD feature vector (a tensor) for this hypothesis
                            wd_feat_vec = get_wd_features(
                                dictionary,
                                partial_hyp,
                                batch.history[i],
                                wd_features,
                                wd_wts,
                            )  # shape (vocab_size)

                            wd_feat_vecs[hyp_idx, :] = wd_feat_vec
                        wd_feat_vecs = wd_feat_vecs.to(current_device)

                        # Add the WD features to the log probability scores
                        scores_in = scores_in + wd_feat_vecs

                    # Update the beam as usual
                    b.advance(scores_in)

            decoder_input = torch.cat(
                [b.get_output_from_current_step() for b in beams]
            ).unsqueeze(-1)
            permute_hidden_idx = torch.cat(
                [
                    beam_size * i + b.get_backtrack_from_current_step()
                    for i, b in enumerate(beams)
                ]
            )
            # permute decoder hiddens with respect to chosen hypothesis now
            if isinstance(hidden, tuple):  # LSTM
                for i in range(len(hidden)):
                    hidden[i].data.copy_(
                        hidden[i].data.index_select(dim=0, index=permute_hidden_idx)
                    )
            else:  # GRU
                hidden.data.copy_(
                    hidden.data.index_select(dim=0, index=permute_hidden_idx)
                )
        for b in beams:
            b.check_finished()

        beam_preds_scores = [list(b.get_top_hyp()) for b in beams]
        for pair in beam_preds_scores:
            pair[0] = Beam.get_pretty_hypothesis(pair[0])

        n_best_beams = [b.get_rescored_finished(n_best=min_n_best) for b in beams]
        n_best_beam_preds_scores = []
        for i, beamhyp in enumerate(n_best_beams):
            this_beam = []
            for hyp in beamhyp:
                pred = beams[i].get_pretty_hypothesis(
                    beams[i].get_hyp_from_finished(hyp)
                )
                score = hyp.score
                this_beam.append((pred, score))
            n_best_beam_preds_scores.append(this_beam)

        return beam_preds_scores, n_best_beam_preds_scores, beams

    def _pick_cands(self, cand_preds, cand_inds, cands):
        cand_replies = [None] * len(cands)
        for idx, order in enumerate(cand_preds):
            batch_idx = cand_inds[idx]
            cand_replies[batch_idx] = [cands[batch_idx][i] for i in order]
        return cand_replies

    def truncate_output(self, out):
        """Truncate the output."""
        new_out_0 = out[0][:-1]
        new_out_1 = None if out[1] is None else out[1][:-1]
        new_out_2 = [vec[:-1] for vec in out[2]]
        return tuple([new_out_0, new_out_1, new_out_2])

    def eval_step(self, batch):
        if self.opt['weighted_decoding'] and self.beam_size > 1:
            needs_truncation = self.multigpu and batch.text_vec.size(0) % 2 != 0
            orig_batch = batch  # save for evaluation
            out = ControllableJPSeq2seq.beam_search(
                self.model,
                batch,
                self.beam_size,
                self.dict,
                start=self.START_IDX,
                end=self.END_IDX,
                pad=self.NULL_IDX,
                min_length=self.beam_min_length,
                min_n_best=self.beam_min_n_best,
                block_ngram=self.beam_block_ngram,
                wd_features=self.wd_features,
                wd_wts=self.wd_wts,
            )
            if needs_truncation:
                out = self.truncate_output(out)
            beam_preds_scores, n_best_preds_scores, beams = out

            # Optionally print out the n-best beam search candidates
            # if self.opt['verbose']:
            #     for cands, hist in zip(n_best_preds_scores, batch.history):
            #         show_beam_cands(cands, hist, self.dict)

            # If we have a special reordering function, apply it to choose the best
            # one of the candidates.
            if self.opt['beam_reorder'] == 'best_extrep2gram_qn':
                beam_preds_scores = [
                    reorder_extrep2gram_qn(cands, hist, self.dict, self.opt['verbose'])
                    for cands, hist in zip(n_best_preds_scores, batch.history)
                ]

            preds, scores = (
                [p[0] for p in beam_preds_scores],
                [p[1] for p in beam_preds_scores],
            )
            # if self.beam_dot_log is True:
            #     for i, b in enumerate(beams):
            #         dot_graph = b.get_beam_dot(dictionary=self.dict, n_best=3)
            #         image_name = (
            #             self._v2t(batch.text_vec[i, -20:])
            #             .replace(' ', '-')
            #             .replace('__null__', '')
            #         )
            #         dot_graph.write_png(
            #             os.path.join(self.beam_dot_dir, "{}.png".format(image_name))
            #         )

            if batch.label_vec is not None:
                # calculate loss on targets with teacher forcing
                seq_len = None if not self.multigpu else batch.text_vec.size(1)
                out = self.model(
                    batch.text_vec, batch.ctrl_vec, batch.label_vec, seq_len=seq_len
                )
                if needs_truncation:
                    out = self.truncate_output(out)
                f_scores = out[0]  # forced scores
                _, f_preds = f_scores.max(2)  # forced preds
                score_view = f_scores.view(-1, f_scores.size(-1))
                loss = self.criterion(score_view, orig_batch.label_vec.view(-1))
                # save loss to metrics
                notnull = orig_batch.label_vec.ne(self.NULL_IDX)
                target_tokens = notnull.long().sum().item()
                correct = ((orig_batch.label_vec == f_preds) * notnull).sum().item()
                self.metrics['correct_tokens'] += correct
                self.metrics['loss'] += loss.item()
                self.metrics['num_tokens'] += target_tokens

            cand_choices = None
            # if cand_scores is not None:
            #     cand_preds = cand_scores.sort(1, descending=True)[1]
            #     # now select the text of the cands based on their scores
            #     cand_choices = self._pick_cands(
            #         cand_preds, cand_params[1], orig_batch.candidates
            #     )

            text = [self._v2t(p) for p in preds]

            return Output(text, cand_choices)
        else:
            return super().eval_step(batch)
    # Optional override in case if loading vector by using torchtext does not work
    # This method assumes that emb vectors is loaded using built-in fasttext load model
    # def _copy_embeddings(self, weight, emb_type, log=True):
    #     if not is_primary_worker():
    #         # we're in distributed mode, copying embeddings in the workers
    #         # slows things down considerably
    #         return
    #     embs, name = self._get_embtype(emb_type)
    #     cnt = 0
    #     for w, i in self.dict.tok2ind.items():
    #         for w in embs.get_words(on_unicode_error='replace'):
    #             vec = self._project_vec(embs.vectors[embs.stoi[w]], weight.size(1))
    #             weight.data[i] = vec
    #             cnt += 1

class Beam(object):
    """Generic beam class. It keeps information about beam_size hypothesis."""

    def __init__(
        self,
        beam_size,
        min_length=3,
        padding_token=0,
        bos_token=1,
        eos_token=2,
        min_n_best=3,
        cuda='cpu',
        block_ngram=0,
    ):
        """
        Instantiate Beam object.
        :param beam_size:
            number of hypothesis in the beam
        :param min_length:
            minimum length of the predicted sequence
        :param padding_token:
            Set to 0 as usual in ParlAI
        :param bos_token:
            Set to 1 as usual in ParlAI
        :param eos_token:
            Set to 2 as usual in ParlAI
        :param min_n_best:
            Beam will not be done unless this amount of finished hypothesis
            (with EOS) is done
        :param cuda:
            What device to use for computations
        """
        self.beam_size = beam_size
        self.min_length = min_length
        self.eos = eos_token
        self.bos = bos_token
        self.pad = padding_token
        self.device = cuda
        # recent score for each hypo in the beam
        self.scores = torch.Tensor(self.beam_size).float().zero_().to(self.device)
        # self.scores values per each time step
        self.all_scores = [torch.Tensor([0.0] * beam_size).to(self.device)]
        # backtracking id to hypothesis at previous time step
        self.bookkeep = []
        # output tokens at each time step
        self.outputs = [
            torch.Tensor(self.beam_size).long().fill_(self.bos).to(self.device)
        ]
        # keeps tuples (score, time_step, hyp_id)
        self.finished = []
        self.HypothesisTail = namedtuple(
            'HypothesisTail', ['timestep', 'hypid', 'score', 'tokenid']
        )
        self.eos_top = False
        self.eos_top_ts = None
        self.n_best_counter = 0
        self.min_n_best = min_n_best
        self.block_ngram = block_ngram
        self.partial_hyps = [[self.bos] for i in range(beam_size)]

    @staticmethod
    def find_ngrams(input_list, n):
        """Get list of ngrams with context length n-1."""
        return list(zip(*[input_list[i:] for i in range(n)]))

    def get_output_from_current_step(self):
        """Get the outputput at the current step."""
        return self.outputs[-1]

    def get_backtrack_from_current_step(self):
        """Get the backtrack at the current step."""
        return self.bookkeep[-1]

    def advance(self, softmax_probs):
        """Advance the beam one step."""
        voc_size = softmax_probs.size(-1)
        current_length = len(self.all_scores) - 1
        if current_length < self.min_length:
            # penalize all eos probs to make it decode longer
            for hyp_id in range(softmax_probs.size(0)):
                softmax_probs[hyp_id][self.eos] = neginf(softmax_probs.dtype)
        if len(self.bookkeep) == 0:
            # the first step we take only the first hypo into account since all
            # hypos are the same initially
            beam_scores = softmax_probs[0]
        else:
            # we need to sum up hypo scores and curr softmax scores before topk
            # [beam_size, voc_size]
            beam_scores = softmax_probs + self.scores.unsqueeze(1).expand_as(
                softmax_probs
            )
            for i in range(self.outputs[-1].size(0)):
                if self.block_ngram > 0:
                    current_hypo = self.partial_hyps[i][1:]
                    current_ngrams = []
                    for ng in range(self.block_ngram):
                        ngrams = Beam.find_ngrams(current_hypo, ng)
                        if len(ngrams) > 0:
                            current_ngrams.extend(ngrams)
                    counted_ngrams = Counter(current_ngrams)
                    if any(v > 1 for k, v in counted_ngrams.items()):
                        # block this hypothesis hard
                        beam_scores[i] = neginf(softmax_probs.dtype)

                #  if previous output hypo token had eos
                # we penalize those word probs to never be chosen
                if self.outputs[-1][i] == self.eos:
                    # beam_scores[i] is voc_size array for i-th hypo
                    beam_scores[i] = neginf(softmax_probs.dtype)

        flatten_beam_scores = beam_scores.view(-1)  # [beam_size * voc_size]
        with torch.no_grad():
            best_scores, best_idxs = torch.topk(
                flatten_beam_scores, self.beam_size, dim=-1
            )

        self.scores = best_scores
        self.all_scores.append(self.scores)
        # get the backtracking hypothesis id as a multiple of full voc_sizes
        hyp_ids = best_idxs / voc_size
        # get the actual word id from residual of the same division
        tok_ids = best_idxs % voc_size

        self.outputs.append(tok_ids)
        self.bookkeep.append(hyp_ids)
        self.partial_hyps = [
            self.partial_hyps[hyp_ids[i]] + [tok_ids[i].item()]
            for i in range(self.beam_size)
        ]

        #  check new hypos for eos label, if we have some, add to finished
        for hypid in range(self.beam_size):
            if self.outputs[-1][hypid] == self.eos:
                #  this is finished hypo, adding to finished
                eostail = self.HypothesisTail(
                    timestep=len(self.outputs) - 1,
                    hypid=hypid,
                    score=self.scores[hypid],
                    tokenid=self.eos,
                )
                self.finished.append(eostail)
                self.n_best_counter += 1

        if self.outputs[-1][0] == self.eos:
            self.eos_top = True
            if self.eos_top_ts is None:
                self.eos_top_ts = len(self.outputs) - 1

    def done(self):
        """Return whether beam search is complete."""
        return self.eos_top and self.n_best_counter >= self.min_n_best

    def get_top_hyp(self):
        """
        Get single best hypothesis.
        :return: hypothesis sequence and the final score
        """
        top_hypothesis_tail = self.get_rescored_finished(n_best=1)[0]
        return (
            self.get_hyp_from_finished(top_hypothesis_tail),
            top_hypothesis_tail.score,
        )

    def get_hyp_from_finished(self, hypothesis_tail):
        """
        Extract hypothesis ending with EOS at timestep with hyp_id.
        :param timestep:
            timestep with range up to len(self.outputs)-1
        :param hyp_id:
            id with range up to beam_size-1
        :return:
            hypothesis sequence
        """
        assert self.outputs[hypothesis_tail.timestep][hypothesis_tail.hypid] == self.eos
        assert hypothesis_tail.tokenid == self.eos
        hyp_idx = []
        endback = hypothesis_tail.hypid
        for i in range(hypothesis_tail.timestep, -1, -1):
            hyp_idx.append(
                self.HypothesisTail(
                    timestep=i,
                    hypid=endback,
                    score=self.all_scores[i][endback],
                    tokenid=self.outputs[i][endback],
                )
            )
            endback = self.bookkeep[i - 1][endback]

        return hyp_idx

    @staticmethod
    def get_pretty_hypothesis(list_of_hypotails):
        """Return prettier version of the hypotheses."""
        hypothesis = []
        for i in list_of_hypotails:
            hypothesis.append(i.tokenid)

        hypothesis = torch.stack(list(reversed(hypothesis)))

        return hypothesis

    def get_rescored_finished(self, n_best=None):
        """
        Return finished hypotheses in rescored order.
        :param n_best:
            how many n best hypothesis to return
        :return:
            list with hypothesis
        """
        rescored_finished = []
        for finished_item in self.finished:
            current_length = finished_item.timestep + 1
            # these weights are from Google NMT paper
            length_penalty = math.pow((1 + current_length) / 6, 0.65)
            rescored_finished.append(
                self.HypothesisTail(
                    timestep=finished_item.timestep,
                    hypid=finished_item.hypid,
                    score=finished_item.score / length_penalty,
                    tokenid=finished_item.tokenid,
                )
            )

        srted = sorted(rescored_finished, key=attrgetter('score'), reverse=True)

        if n_best is not None:
            srted = srted[:n_best]

        return srted

    def check_finished(self):
        """
        Check if self.finished is empty and add hyptail in that case.
        This will be suboptimal hypothesis since the model did not get any EOS
        """
        if len(self.finished) == 0:
            # we change output because we want outputs to have eos
            # to pass assert in L102, it is ok since empty self.finished
            # means junk prediction anyway
            self.outputs[-1][0] = self.eos
            hyptail = self.HypothesisTail(
                timestep=len(self.outputs) - 1,
                hypid=0,
                score=self.all_scores[-1][0],
                tokenid=self.outputs[-1][0],
            )

            self.finished.append(hyptail)

    def get_beam_dot(self, dictionary=None, n_best=None):
        """
        Create pydot graph representation of the beam.
        :param outputs:
            self.outputs from the beam
        :param dictionary:
            tok 2 word dict to save words in the tree nodes
        :returns:
            pydot graph
        """
        try:
            import pydot
        except ImportError:
            print("Please install pydot package to dump beam visualization")

        graph = pydot.Dot(graph_type='digraph')
        outputs = [i.tolist() for i in self.outputs]
        bookkeep = [i.tolist() for i in self.bookkeep]
        all_scores = [i.tolist() for i in self.all_scores]
        if n_best is None:
            n_best = int(self.beam_size / 2)

        # get top nbest hyp
        top_hyp_idx_n_best = []
        n_best_colors = ['aquamarine', 'chocolate1', 'deepskyblue', 'green2', 'tan']
        sorted_finished = self.get_rescored_finished(n_best=n_best)
        for hyptail in sorted_finished:
            # do not include EOS since it has rescored score not from original
            # self.all_scores, we color EOS with black
            top_hyp_idx_n_best.append(self.get_hyp_from_finished(hyptail))

        # create nodes
        for tstep, lis in enumerate(outputs):
            for hypid, token in enumerate(lis):
                if tstep == 0:
                    hypid = 0  # collapse all __NULL__ nodes
                node_tail = self.HypothesisTail(
                    timestep=tstep,
                    hypid=hypid,
                    score=all_scores[tstep][hypid],
                    tokenid=token,
                )
                color = 'white'
                rank = None
                for i, hypseq in enumerate(top_hyp_idx_n_best):
                    if node_tail in hypseq:
                        if n_best <= 5:  # color nodes only if <=5
                            color = n_best_colors[i]
                        rank = i
                        break
                label = (
                    "<{}".format(
                        dictionary.vec2txt([token]) if dictionary is not None else token
                    )
                    + " : "
                    + "{:.{prec}f}>".format(all_scores[tstep][hypid], prec=3)
                )

                graph.add_node(
                    pydot.Node(
                        node_tail.__repr__(),
                        label=label,
                        fillcolor=color,
                        style='filled',
                        xlabel='{}'.format(rank) if rank is not None else '',
                    )
                )

        # create edges
        for revtstep, lis in reversed(list(enumerate(bookkeep))):
            for i, prev_id in enumerate(lis):
                from_node = graph.get_node(
                    '"{}"'.format(
                        self.HypothesisTail(
                            timestep=revtstep,
                            hypid=prev_id,
                            score=all_scores[revtstep][prev_id],
                            tokenid=outputs[revtstep][prev_id],
                        ).__repr__()
                    )
                )[0]
                to_node = graph.get_node(
                    '"{}"'.format(
                        self.HypothesisTail(
                            timestep=revtstep + 1,
                            hypid=i,
                            score=all_scores[revtstep + 1][i],
                            tokenid=outputs[revtstep + 1][i],
                        ).__repr__()
                    )
                )[0]
                newedge = pydot.Edge(from_node.get_name(), to_node.get_name())
                graph.add_edge(newedge)

        return graph
