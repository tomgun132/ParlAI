import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from parlai.agents.transformer.modules import TransformerEncoder

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

    return tensor * F.sigmoid(tensor)

class PolyAIEncoder(nn.Module):

    def __init__(self, opt, dictionary):
        # h_layer_num=3, h_dim=1024,h_act_func='swish', linear_dim=512,
        # scoring_func='scaled'):
        super(PolyAIEncoder, self).__init__()
        null_idx = dictionary[dictionary.null_token]
        self.scoring_type = opt['scoring_func']
        # Unigram Encoder
        self.x_unigram_encoder = self.build_encoder(opt, dictionary.uni, null_idx)
        self.y_unigram_encoder = self.build_encoder(opt, dictionary.uni, null_idx)

        # Bigram Encoder
        self.x_bigram_encoder = self.build_encoder(opt, dictionary.bi)
        self.y_bigram_encoder = self.build_encoder(opt, dictionary.bi)

        # H Fully connected hidden layers
        self.x_h_layers = nn.ModuleList()
        self.y_h_layers = nn.ModuleList()
        self.h_layer_num = opt['h_layer_num']
        for _ in range(self.h_layer_num):
            self.x_h_layers.append(PolyAIFFN(
                in_dim=opt['embedding_size'],
                out_dim=opt['h_dim'],
                dropout=opt['relu_dropout'],
                activation=opt['h_act_func']
            ))
            self.y_h_layers.append(PolyAIFFN(
                in_dim=opt['embedding_size'],
                out_dim=opt['h_dim'],
                dropout=opt['relu_dropout'],
                activation=opt['h_act_func']
            ))

        # Last linear layer
        self.x_linear = nn.Linear(opt['h_dim'], opt['linear_dim'])
        self.y_linear = nn.Linear(opt['h_dim'], opt['linear_dim'])
        self.dropout = nn.Dropout(p=opt['dropout'])
        if self.scoring_type == 'scaled':
            self.C = torch.randn(1, requires_grad=True)

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
            y_emb = self.y_linear(y_emb)

        if x_uni is not None and x_bi is not None:
            assert len(x_uni.shape) == 2
            x_uni_emb = self.x_unigram_encoder(x_uni)
            x_bi_emb = self.x_bigram_encoder(x_bi)
            x_emb = (x_uni_emb + x_bi_emb) / math.sqrt(x_uni.shape[-1])
            for i in range(self.h_layer_num):
                x_emb = self.x_h_layers[i](x_emb)
            x_emb = self.x_linear(x_emb)

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
            x_emb_extended = x_emb.unsqueeze(1).extend(y_emb.shape)
            scores = self.C * F.cosine_similarity(x_emb_extended, y_emb, dim=-1)
        else:
            raise NotImplementedError
        return scores

    def forward(self, x_uni, x_bi, y_uni, y_bi):
        x_emb, y_emb = self.encode(x_uni, x_bi, y_uni, y_bi)
        x_emb = self.dropout(x_emb)
        y_emb = self.dropout(y_emb)
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
