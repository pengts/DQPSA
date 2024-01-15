import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from adaptive_span import AdaptiveSpan


def _skew(X, pad_value):
    """shift every row 1 step to right"""
    # X = B x M x L
    B, M, L = X.size()
    X = F.pad(X, (0, M + 1), value=pad_value)  # B x M x (L+M+1)
    X = X.view(B, -1)  # B x ML+MM+M
    X = X[:, :-M]  # B x ML+MM
    X = X.view(B, M, M + L)  # B x M x L+M
    return X

def _unskew(X):
    """reverse _skew operation"""
    # X = B x M x L+M
    B, M, L = X.size()
    L -= M
    X = X.view(B, -1)  # B x ML+MM
    X = F.pad(X, (0, M))  # B x ML+MM+M
    X = X.view(B, M, M + L + 1)  # B x M x L+M+1
    X = X[:, :, :L]  # B x M x L
    return X

class SeqAttention(nn.Module):
    """Sequential self-attention layer.
    Each token will attend to its previous fixed number of steps.
    Note that attention doesn't include the current step itself.
    """
    def __init__(self, hidden_size, nb_heads, attn_span,
                 dropout,adapt_span_params, **kargs):
        nn.Module.__init__(self)
        # pdb.set_trace()

        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size # size of a single head
        self.attn_span = attn_span
        self.adapt_span_enabled = adapt_span_params['adapt_span_enabled']
        if self.adapt_span_enabled:
            self.adaptive_span = AdaptiveSpan(attn_span=attn_span, nb_heads=nb_heads,
                                              **adapt_span_params, **kargs)

        self.persistent_memory = None

    def forward(self, query, key, value, key_pe,output_attentions=False):
        # query size = B x M x H
        # key, value sizes = B x (M+L) x H
        # compute attention from context
        # B x M (dest) x (M+L) (src)
        attn_cont = torch.matmul(query, key.transpose(-1, -2))
        attn_cont = _unskew(attn_cont)  # B x M x L

        # compute the effect of position embedding
        attn_pos = torch.matmul(query, key_pe)  # B x M x L_pos
        attn = attn_cont + attn_pos

        if self.persistent_memory is not None:
            attn, pers_mem_out = self.persistent_memory(query, attn)
        else:
            attn = attn / math.sqrt(self.hidden_size)  # B x M X L_pos
            attn = F.softmax(attn, dim=-1)
            
            if self.adapt_span_enabled:
                # trim attention lengths according to the learned span
                attn = self.adaptive_span(attn)

        attn = self.dropout(attn)  # B x M X L_pos

        attn_cont = _skew(attn, 0)  # B x M X (L+M)
        
        out = torch.matmul(attn_cont, value)  # B x M x H
        # pdb.set_trace()


        if self.persistent_memory is not None:
            out = out + pers_mem_out
        if output_attentions:
            L=attn_cont.size()[1]
            
            return out, attn_cont[:,:,-L:]
        
        else:
            return out

    def get_cache_size(self):
        if self.adapt_span_enabled:
            return self.adaptive_span.get_cache_size()
        else:
            return self.attn_span


class MultiHeadSeqAttention(nn.Module):
    def __init__(self, hidden_size, nb_heads, **kargs):
        nn.Module.__init__(self)
        # pdb.set_trace()
        assert hidden_size % nb_heads == 0
        self.nb_heads = nb_heads
        self.head_dim = hidden_size // nb_heads
        self.attn = SeqAttention(
            hidden_size=self.head_dim, nb_heads=nb_heads, **kargs)
        self.proj_query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_val = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_key = nn.Linear(hidden_size, hidden_size, bias=False)

    def head_reshape(self, x):
        K = self.nb_heads
        D = self.head_dim
        x = x.view(x.size()[:-1] + (K, D))  # B x (M+L) x K x D
        x = x.transpose(1, 2).contiguous()  # B x K x (M+L) x D
        x = x.view(-1, x.size(-2), x.size(-1))  # B_K x (M+L) x D
        return x

    def forward(self, query, key, value, key_pe,output_attentions=False):
        B = query.size(0)
        K = self.nb_heads
        D = self.head_dim
        M = query.size(1)

        query = self.proj_query(query)
        query = self.head_reshape(query)
        value = self.proj_val(value)
        value = self.head_reshape(value)
        key = self.proj_key(key)
        key = self.head_reshape(key)


        if output_attentions:
            out,attentions = self.attn(query, key, value, key_pe,output_attentions)  # B_K x M x D
        else:
            out = self.attn(query, key, value, key_pe,output_attentions)  # B_K x M x D

        out = out.view(B, K, M, D)  # B x K x M x D
        out = out.transpose(1, 2).contiguous()  # B x M x K x D
        out = out.view(B, M, -1)  # B x M x K_D
        out = self.proj_out(out)
        # pdb.set_trace()
        if output_attentions:
            return out, attentions
        else:
            return out


class FeedForwardLayer(nn.Module):
    def __init__(self, hidden_size, inner_hidden_size, dropout, **kargs):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(hidden_size, inner_hidden_size)
        self.fc2 = nn.Linear(inner_hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h):
        h1 = F.relu(self.fc1(h))
        h1 = self.dropout(h1)
        h2 = self.fc2(h1)
        return h2


class FSA_layer(nn.Module):
    def __init__(self, hidden_size,**kargs):
        nn.Module.__init__(self)
        # pdb.set_trace()
        self.attn_span=kargs['attn_span']
        self.hidden_size=hidden_size
        self.attn = MultiHeadSeqAttention(hidden_size=hidden_size, **kargs)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ff = FeedForwardLayer(hidden_size=hidden_size, **kargs)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.key_pe = nn.Parameter(
            torch.randn(1, hidden_size // kargs['nb_heads'], kargs['attn_span']))
        # self.h_cache=torch.zeros(16,kargs['attn_span'],hidden_size).cuda()
    def forward(self, h,output_attentions=False):
        # h = B x M x H
        # h_cache = B x L x H
        B=h.shape[0]
        self.h_cache=torch.zeros(B,self.attn_span,self.hidden_size).to(h.device)
        h_all = torch.cat([self.h_cache, h], dim=1)  # B x (M+L) x H
        attn_out = self.attn(h, h_all, h_all, self.key_pe,output_attentions)

        if output_attentions:
            attn_out,attentions = self.attn(h, h_all, h_all, self.key_pe,output_attentions)
        else:
            attn_out = self.attn(h, h_all, h_all, self.key_pe,output_attentions)

        h = self.norm1(h + attn_out)  # B x M x H
        ff_out = self.ff(h)
        out = self.norm2(h + ff_out)  # B x M x H
        if output_attentions:
            return out,attentions
        else:
            return out
