import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from parlai.agents.transformer.modules import TransformerEncoder, BasicAttention
from parlai.core.utils import neginf
from parlai.agents.bert_ranker.helpers import (
    get_bert_optimizer,
    BertWrapper,
    BertModel,
    BertConfig,
    BertLayer,
    add_common_args,
    surround,
    MODEL_PATH,
)

def gelu(tensor):
    """
    Compute gelu function.

    c.f. https://arxiv.org/abs/1606.08415
    """
    return 0.5 * tensor * (1.0 + torch.erf(tensor / math.sqrt(2.0)))

def swish(tensor):
    """
    Swish activation is simply f(x)=x⋅sigmoid(x)
    Paper: https://arxiv.org/abs/1710.05941
    """

    return tensor * torch.sigmoid(tensor)

class PolyAIEncoder(nn.Module):

    def __init__(self, opt, dictionary):
        # h_layer_num=3, h_dim=1024,h_act_func='swish', linear_dim=512,
        # scoring_func='scaled'):
        super(PolyAIEncoder, self).__init__()
        null_idx = dictionary[dictionary.null_token]
        self.scoring_type = opt['scoring_func']
        # Unigram Encoder
        self.x_unigram_encoder = self.build_encoder(opt, dictionary.tok2ind, null_idx)
        self.y_unigram_encoder = self.build_encoder(opt, dictionary.tok2ind, null_idx)

        # Bigram Encoder
        self.x_bigram_encoder = self.build_encoder(opt, dictionary.bi2ind, null_idx)
        self.y_bigram_encoder = self.build_encoder(opt, dictionary.bi2ind, null_idx)

        # H Fully connected hidden layers
        self.x_h_layers = nn.ModuleList()
        self.y_h_layers = nn.ModuleList()
        self.h_layer_num = opt['h_layer_num']
        for i in range(self.h_layer_num):
            if i == 0:
                in_dim = opt['embedding_size']
            else:
                in_dim = opt['h_dim']
            self.x_h_layers.append(PolyAIFFN(
                in_dim=in_dim,
                out_dim=opt['h_dim'],
                dropout=opt['relu_dropout'],
                activation=opt['h_act_func']
            ))
            self.y_h_layers.append(PolyAIFFN(
                in_dim=in_dim,
                out_dim=opt['h_dim'],
                dropout=opt['relu_dropout'],
                activation=opt['h_act_func']
            ))

        # Last linear layer
        self.x_linear = nn.Linear(opt['h_dim'], opt['linear_dim'])
        self.y_linear = nn.Linear(opt['h_dim'], opt['linear_dim'])
        self.dropout = nn.Dropout(p=opt['dropout'])
        if self.scoring_type == 'scaled':
            if opt['no_cuda']:
                self.C = torch.randn(1, requires_grad=True)
            else:
                self.C = torch.randn(1, requires_grad=True).cuda()

    def build_encoder(self, opt, dictionary, null_idx):
        embeddings = nn.Embedding(
            len(dictionary), opt['embedding_size'], padding_idx=null_idx
        )
        nn.init.normal_(embeddings.weight, 0, opt['embedding_size'] ** -0.5)
        return TransformerEncoder(
            n_heads=opt['n_heads'],
            n_layers=opt['n_layers'],
            embedding_size=opt['embedding_size'],
            ffn_size=opt['ffn_size'],
            vocabulary_size=len(dictionary),
            embedding=embeddings,
            dropout=opt['dropout'],
            attention_dropout=opt['attention_dropout'],
            relu_dropout=opt['relu_dropout'],
            padding_idx=null_idx,
            learn_positional_embeddings=opt['learn_positional_embeddings'],
            embeddings_scale=opt['embeddings_scale'],
            activation=opt['activation'],
            variant=opt['variant'],
            output_scaling=opt['output_scaling'],
        )

    def encode(self, x_uni, x_bi, y_uni, y_bi):
        x_emb = None
        y_emb = None
        if y_uni is not None and y_bi is not None:
            assert len(y_uni.shape) == 3
            bsz, cand_num, seq_len = y_uni.shape
            y_uni_emb = self.y_unigram_encoder(y_uni.view(bsz*cand_num, -1))
            y_bi_emb = self.y_bigram_encoder(y_bi.view(bsz*cand_num, -1))
            y_emb = (y_uni_emb + y_bi_emb) / math.sqrt(seq_len)
            for i in range(self.h_layer_num):
                y_emb = self.y_h_layers[i](y_emb)
            y_emb = self.dropout(self.y_linear(y_emb))
            y_emb = y_emb.view(bsz, cand_num, -1)

        if x_uni is not None and x_bi is not None:
            assert len(x_uni.shape) == 2
            x_uni_emb = self.x_unigram_encoder(x_uni)
            x_bi_emb = self.x_bigram_encoder(x_bi)
            x_emb = (x_uni_emb + x_bi_emb) / math.sqrt(x_uni.shape[-1])
            for i in range(self.h_layer_num):
                x_emb = self.x_h_layers[i](x_emb)
            x_emb = self.dropout(self.x_linear(x_emb))

        return x_emb, y_emb

    def score(self, x_emb, y_emb):
        """
        Input (probably) will have size:
            x_emb: batch_size x hidden_dim
            y_emb: batch_size x num_cand x hidden_dim

        Output:
            scores: batch_size x num_cand
        """
        if self.scoring_type == 'dot':
            scores = torch.bmm(x_emb.unsqueeze(1), y_emb.transpose(1,2)).squeeze(1)
        elif self.scoring_type == 'scaled':
            x_emb_extended = x_emb.unsqueeze(1).expand(y_emb.shape)
            scores = self.C * F.cosine_similarity(x_emb_extended, y_emb, dim=-1)
        else:
            raise NotImplementedError
        return scores

    def forward(self, x_uni, x_bi, y_uni, y_bi, y_enc = None):
        bsz = x_uni.size(0)
        if y_enc is not None:
            if bsz == 1:
                y_emb = y_enc
            else:
                y_emb = y_enc.expand(bsz, y_enc.size(1), -1)
            x_emb = self.encode(x_uni, x_bi, None, None)
        elif len(y_uni.shape) == 3:
            x_emb, y_emb = self.encode(x_uni, x_bi, y_uni, y_bi)
        elif len(y_uni.shape) == 2: # bsz x seq len (if batch cands) or num_cands x seq len (if fixed cands)
            x_emb, y_emb = self.encode(x_uni, x_bi, y_uni.unsqueeze(1), y_bi.unsqueeze(1))
            num_cands = y_emb.size(0) # will be bsz if using batch cands
            y_emb = y_emb.expand(num_cands, bsz, -1).transpose(0,1).contiguous()
        output = self.score(x_emb, y_emb)
        return output

class PolyAIFFN(nn.Module):

    def __init__(self, in_dim, out_dim, dropout=0.0, activation='swish'):
        super(PolyAIFFN, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        if activation == 'relu':
            self.act_func = F.relu
        elif activation == 'gelu':
            self.act_func = gelu
        elif activation == 'swish':
            self.act_func = swish

        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        out = self.act_func(self.linear(x))
        return self.dropout(out)

class PolyEncoderBert(nn.Module):

    def __init__(self, opt):
        super(PolyEncoderBert, self).__init__()
        self.context_encoder = PolyBertWrapper(
            BertModel.from_pretrained(opt['pretrained_path']),
            opt['out_dim'],
            layer_pulled=-1,
            n_codes=opt['n_codes'],
        )
        self.cand_encoder = BertWrapper(
            BertModel.from_pretrained(opt['pretrained_path']),
            opt['out_dim'],
            add_transformer_layer=opt['add_transformer_layer'],
            layer_pulled=opt['pull_from_layer'],
            aggregation=opt['bert_aggregation'],
        )
        self.attention_type = opt['attention_type']
        self.attention = BasicAttention(dim=2, attn=self.attention_type, get_weights=False)

    def score(self, ctxt_rep, ctxt_rep_mask, cand_embed):
        ctxt_final_rep = self.attention(cand_embed, ctxt_rep, mask_ys=ctxt_rep_mask)
        return torch.sum(ctxt_final_rep * cand_embed, 2)

    def forward(
        self,
        token_idx_ctxt,
        segment_idx_ctxt,
        mask_ctxt,
        token_idx_cands,
        segment_idx_cands,
        mask_cands,
    ):
        embedding_ctxt = None
        ctxt_mask = None
        if token_idx_ctxt is not None:
            embedding_ctxt, ctxt_mask = self.context_encoder(
                token_idx_ctxt, segment_idx_ctxt, mask_ctxt
            )
        embedding_cands = None
        if token_idx_cands is not None:
            embedding_cands = self.cand_encoder(
                token_idx_cands, segment_idx_cands, mask_cands
            )
        return (embedding_ctxt, ctxt_mask), embedding_cands


class PolyBertWrapper(nn.Module):
    """Adds a optional transformer layer and a linear layer on top of BERT."""
    """Adds one more aggregation to make it works for polyencoder model."""

    def __init__(
        self,
        bert_model,
        output_dim,
        add_transformer_layer=False,
        layer_pulled=-1,
        n_codes=64
    ):
        super(PolyBertWrapper, self).__init__()
        self.layer_pulled = layer_pulled
        self.add_transformer_layer = add_transformer_layer
        # deduce bert output dim from the size of embeddings
        bert_output_dim = bert_model.embeddings.word_embeddings.weight.size(1)

        if add_transformer_layer:
            config_for_one_layer = BertConfig(
                0,
                hidden_size=bert_output_dim,
                num_attention_heads=int(bert_output_dim / 64),
                intermediate_size=3072,
                hidden_act='gelu',
            )
            self.additional_transformer_layer = BertLayer(config_for_one_layer)
        self.additional_linear_layer = torch.nn.Linear(bert_output_dim, output_dim)
        self.bert_model = bert_model
        self.n_codes = n_codes

    def forward(self, token_ids, segment_ids, attention_mask):
        """Forward pass."""
        outputs = self.bert_model(
            token_ids, segment_ids, attention_mask
        )
        _, output_pooler = outputs[:2]
        try:
            output_bert = outputs[2]
        except IndexError:
            raise IndexError("output_hidden_states in bert config must be true")
        # output_bert is a list of 12 (for bert base) layers.
        layer_of_interest = output_bert[self.layer_pulled]
        dtype = next(self.parameters()).dtype
        if self.add_transformer_layer:
            # Follow up by yet another transformer layer
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = (~extended_attention_mask).to(dtype) * neginf(
                dtype
            )
            embedding_layer = self.additional_transformer_layer(
                layer_of_interest, extended_attention_mask
            )
        else:
            embedding_layer = layer_of_interest

        # Expand the output if it is not long enough
        if embedding_layer.size(1) < self.n_codes:
            difference = self.n_codes - embedding_layer.size(1)
            extra_rep = embedding_layer.new_zeros(token_ids.size(0), difference, embedding_layer.size(2))
            embeddings = torch.cat([embedding_layer, extra_rep], dim=1)
            extra_mask = attention_mask.new_zeros(token_ids.size(0), difference)
            ctxt_mask = torch.cat([attention_mask, extra_mask], dim=1)
        else:
            embeddings = embedding_layer[:, 0:self.n_codes, :]
            ctxt_mask = attention_mask[:, 0:self.n_codes]

        # We need this in case of dimensionality reduction
        result = self.additional_linear_layer(embeddings)

        # Sort of hack to make it work with distributed: this way the pooler layer
        # is used for grad computation, even though it does not change anything...
        # in practice, it just adds a very (768*768) x (768*batchsize) matmul
        result += 0 * torch.sum(output_pooler)
        return result, ctxt_mask
