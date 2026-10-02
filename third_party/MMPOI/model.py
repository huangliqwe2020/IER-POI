import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from param_parser import parameter_parser
import torch.nn.init as init
args = parameter_parser()

class NodeAttnMap(nn.Module):
    def __init__(self, in_features, nhid, use_mask=False):
        super(NodeAttnMap, self).__init__()
        self.use_mask = use_mask
        self.out_features = nhid
        self.W = nn.Parameter(torch.empty(size=(in_features, nhid)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * nhid, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, X, A):
        Wh = torch.mm(X, self.W)

        e = self._prepare_attentional_mechanism_input(Wh)

        if self.use_mask:
            e = torch.where(A > 0, e, torch.zeros_like(e))

        A = A + 1
        e = e * A

        return e

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, ninput, nhid, noutput, dropout):
        super(GCN, self).__init__()

        self.gcn = nn.ModuleList()
        self.dropout = dropout
        self.leaky_relu = nn.LeakyReLU(0.2)

        channels = [ninput] + nhid + [noutput]
        for i in range(len(channels) - 1):
            gcn_layer = GraphConvolution(channels[i], channels[i + 1])
            self.gcn.append(gcn_layer)

    def forward(self, x, adj):
        for i in range(len(self.gcn) - 1):
            x = self.leaky_relu(self.gcn[i](x, adj))

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcn[-1](x, adj)

        return x


class UserEmbeddings(nn.Module):
    def __init__(self, num_users, embedding_dim):
        super(UserEmbeddings, self).__init__()

        self.user_embedding = nn.Embedding(
            num_embeddings=num_users,
            embedding_dim=embedding_dim,
        )

    def forward(self, user_idx):
        embed = self.user_embedding(user_idx)
        return embed


class CategoryEmbeddings(nn.Module):
    def __init__(self, num_cats, embedding_dim, one_hot_rlt):
        super(CategoryEmbeddings, self).__init__()

        if args.cat_type == 'Random':
            self.cat_embedding = nn.Embedding(
                num_embeddings=num_cats,
                embedding_dim=embedding_dim,
            )
        elif args.cat_type == 'SentenceTransformer':
            if args.cat_freeze:
                one_hot_rlt = torch.from_numpy(one_hot_rlt).float()
                self.cat_embedding = nn.Embedding.from_pretrained(one_hot_rlt, freeze=True)
            else:
                one_hot_rlt = torch.from_numpy(one_hot_rlt).float()
                self.cat_embedding = nn.Embedding.from_pretrained(one_hot_rlt, freeze=False)



    def forward(self, cat_idx):
        embed = self.cat_embedding(cat_idx)
        return embed


class FuseEmbeddings(nn.Module):
    def __init__(self, user_embed_dim, poi_embed_dim):
        super(FuseEmbeddings, self).__init__()
        embed_dim = user_embed_dim + poi_embed_dim
        self.fuse_embed = nn.Linear(embed_dim, embed_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, user_embed, poi_embed):
        if args.fusion_method == "fusion":
            x = self.fuse_embed(torch.cat((user_embed, poi_embed), -1))
            x = self.leaky_relu(x)
            return x
        elif args.fusion_method == "concat":
            x = torch.cat((user_embed, poi_embed), -1)
            return x

class FuseMultimodalEmbeddings(nn.Module):
    def __init__(self, multimodal_embed_dim, multimodal_num):
        super(FuseMultimodalEmbeddings, self).__init__()
        embed_dim = multimodal_embed_dim * multimodal_num
        self.fuse_embed = nn.Linear(embed_dim, embed_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, *embeddings):
        if args.fusion_method == 'fusion':
            x = self.fuse_embed(torch.cat(embeddings, -1))
            x = self.leaky_relu(x)
            return x
        elif args.fusion_method == 'concat':
            x = torch.cat(embeddings, -1)
            return x


class RobustReviewEvidenceAggregator(nn.Module):
    """User-conditioned selection and redundancy-aware review aggregation.

    The output is a zero-initialized residual over MMPOI's original pooled
    review embedding.  This makes a baseline checkpoint a numerically exact
    starting point while allowing individual reviews to improve it.
    """

    def __init__(self, user_dim, poi_dim, evidence_dim, output_dim,
                 topk=5, temperature=0.2, redundancy_weight=0.2,
                 evidence_dropout=0.1, duplicate_threshold=0.85):
        super().__init__()
        self.topk = topk
        self.temperature = temperature
        self.redundancy_weight = redundancy_weight
        self.evidence_dropout = evidence_dropout
        self.duplicate_threshold = duplicate_threshold
        self.query = nn.Linear(user_dim + poi_dim, evidence_dim)
        self.evidence_projection = nn.Linear(evidence_dim, evidence_dim)
        self.reliability = nn.Linear(evidence_dim, 1)
        self.output_projection = nn.Linear(evidence_dim, output_dim)
        self.residual_gate = nn.Parameter(torch.tensor(-2.0))
        # Exact warm start: before learning, evidence output equals baseline.
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def _aggregate(self, query, evidence, valid_mask, apply_dropout=False):
        evidence = F.normalize(self.evidence_projection(evidence), dim=-1)
        query = F.normalize(query, dim=-1)
        mask = valid_mask.bool().clone()
        if apply_dropout and self.evidence_dropout > 0 and mask.sum() > 1:
            dropped = torch.rand_like(mask.float()) < self.evidence_dropout
            candidate = mask & ~dropped
            if candidate.any():
                mask = candidate

        pairwise = evidence @ evidence.transpose(0, 1)
        pair_mask = mask[:, None] & mask[None, :]
        eye = torch.eye(evidence.size(0), dtype=torch.bool, device=evidence.device)
        neighbour_mask = pair_mask & ~eye
        duplicate_mass = torch.relu(pairwise - self.duplicate_threshold)
        density = (duplicate_mass * neighbour_mask.float()).sum(-1)
        density = density / neighbour_mask.float().sum(-1).clamp_min(1.0)

        relevance = evidence @ query
        reliability = torch.log(torch.sigmoid(self.reliability(evidence).squeeze(-1)).clamp_min(1e-6))
        scores = relevance + reliability - self.redundancy_weight * density
        scores = scores.masked_fill(~mask, -torch.inf)
        k = min(self.topk, int(mask.sum().item()))
        if k == 0:
            return None, scores
        top_scores, top_idx = torch.topk(scores, k=k)
        weights = torch.softmax(top_scores / self.temperature, dim=0)
        pooled = (weights.unsqueeze(-1) * evidence[top_idx]).sum(0)
        return pooled, scores

    def forward(self, user_embedding, poi_embedding, evidence, valid_mask,
                baseline_review, compute_regularizers=False):
        query = self.query(torch.cat((user_embedding, poi_embedding), dim=-1))
        pooled, scores = self._aggregate(
            query, evidence, valid_mask, apply_dropout=self.training
        )
        zero = baseline_review.new_zeros(())
        if pooled is None:
            return baseline_review, zero, zero

        red_loss = zero
        sens_loss = zero
        if compute_regularizers and valid_mask.sum() > 1:
            best_idx = int(torch.argmax(scores).item())
            duplicate_evidence = torch.cat((evidence, evidence[best_idx:best_idx + 1]), dim=0)
            duplicate_mask = torch.cat((valid_mask, valid_mask.new_ones(1)), dim=0)
            duplicate_pooled, _ = self._aggregate(
                query, duplicate_evidence, duplicate_mask, apply_dropout=False
            )
            red_loss = F.mse_loss(pooled, duplicate_pooled)

            removed_mask = valid_mask.clone()
            removed_mask[best_idx] = False
            removed_pooled, _ = self._aggregate(
                query, evidence, removed_mask, apply_dropout=False
            )
            if removed_pooled is not None:
                change = 1.0 - F.cosine_similarity(
                    pooled.unsqueeze(0), removed_pooled.unsqueeze(0)
                ).squeeze(0)
                sens_loss = torch.relu(pooled.new_tensor(0.05) - change)

        residual = self.output_projection(pooled)
        output = baseline_review + torch.sigmoid(self.residual_gate) * residual
        return output, red_loss, sens_loss

class OnlyTimeOrCatEmbeddings(nn.Module):
    def __init__(self, user_embed_dim):
        super(OnlyTimeOrCatEmbeddings, self).__init__()
        embed_dim = user_embed_dim
        self.fuse_embed = nn.Linear(embed_dim, embed_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, user_embed):
        x = self.fuse_embed(user_embed)
        x = self.leaky_relu(x)
        return x

def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    return torch.cat([v1, v2], 1)


class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.empty(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.empty(in_features, 1))
        self.w = nn.parameter.Parameter(torch.empty(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.empty(in_features, out_features - 1))
        init.normal_(self.w0)
        init.normal_(self.b0)
        init.normal_(self.w)
        init.normal_(self.b)
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class CosineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.empty(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.empty(in_features, 1))
        self.w = nn.parameter.Parameter(torch.empty(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.empty(in_features, out_features - 1))
        init.normal_(self.w0)
        init.normal_(self.b0)
        init.normal_(self.w)
        init.normal_(self.b)
        self.f = torch.cos

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class Time2Vec(nn.Module):
    def __init__(self, activation, out_dim):
        super(Time2Vec, self).__init__()
        if activation == "sin":
            self.l1 = SineActivation(1, out_dim)
        elif activation == "cos":
            self.l1 = CosineActivation(1, out_dim)

    def forward(self, x):
        x = self.l1(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Batch-first layout: (1, sequence, embedding).
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, num_poi, num_cat, num_geo, embed_size, nhead, nhid, nlayers, use_time, use_cat, geo_graph_enabled, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.use_time = use_time
        self.use_cat = use_cat
        self.geo_graph_enabled = geo_graph_enabled
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        # pad_sequence(..., batch_first=True) produces (batch, sequence,
        # embedding), so the encoder must use the same layout.
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid, dropout,
                                                  batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # self.encoder = nn.Embedding(num_poi, embed_size)
        self.embed_size = embed_size
        self.decoder_poi = nn.Linear(embed_size, num_poi)
        if self.use_time:
            self.decoder_time = nn.Linear(embed_size, 1)
        if self.geo_graph_enabled:
            self.decoder_geo = nn.Linear(embed_size, num_geo)

        if args.cat_loss_type == 'id_loss':
            self.decoder_cat = nn.Linear(embed_size, num_cat)
        elif args.cat_loss_type == 'embedding_loss':
            self.decoder_cat = nn.Linear(embed_size, 768)
        else:
            print('cat_loss_type error')
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.decoder_poi.bias.data.zero_()
        self.decoder_poi.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = src * math.sqrt(self.embed_size)
        src = self.pos_encoder(src)
        x = self.transformer_encoder(src, src_mask)
        out_poi = self.decoder_poi(x)
        if self.use_time:
            out_time = self.decoder_time(x)
        if self.geo_graph_enabled:
            out_geo = self.decoder_geo(x)

        out_cat = self.decoder_cat(x)

        if self.use_time and self.geo_graph_enabled:
            return out_poi, out_time, out_cat, out_geo
        else:
            return out_poi, out_time, out_cat
