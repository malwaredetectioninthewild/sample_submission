
import torch
from torch import nn
from torch.nn import functional as F
import math


class Backbone_ResNet(nn.Module):
    def __init__(self, in_dim, n_layers=3, hidden_size=256, hidden_factor=1.5, hidden_dropout=0.5, 
                 residual_dropout=0.2, keep_feats=None):
        
        super(Backbone_ResNet, self).__init__()

        self.activation = F.relu
        self.rdo = residual_dropout
        self.hdo = hidden_dropout
        self.keep_feats = keep_feats

        self.first_layer = nn.Linear(in_dim, hidden_size)

        inner_hidden = int(hidden_size * hidden_factor)

        self.layers = nn.ModuleList()

        for _ in range(n_layers):

            cur_layer = nn.ModuleDict({'norm': nn.BatchNorm1d(hidden_size), 
                                       'linear0': nn.Linear(hidden_size, inner_hidden), 
                                       'linear1': nn.Linear(inner_hidden, hidden_size)})
            
            self.layers.append(cur_layer)

        self.last_normalization = nn.BatchNorm1d(hidden_size)
        self.feat_dim = hidden_size
        self.init = None

    def forward_w_intermediate(self, x):

        if hasattr(self, 'keep_feats') and self.keep_feats is not None:
            x = x[:, self.keep_feats]

        all_reprs = []
        x = self.first_layer(x)
        all_reprs.append(x)

        for layer in self.layers:
            z = x
            z = layer['norm'](z)
            z = layer['linear0'](z)
            z = self.activation(z)
            if self.hdo > 0:
                z = F.dropout(z, self.hdo, self.training)
            z = layer['linear1'](z)
            if self.rdo:
                z = F.dropout(z, self.rdo, self.training)
            x = x + z
            all_reprs.append(x)

        x = self.last_normalization(x)
        x = self.activation(x)
        return x, all_reprs

    def forward(self, x):
        if hasattr(self, 'keep_feats') and self.keep_feats is not None:
            x = x[:, self.keep_feats]

        x = self.first_layer(x)
        for layer in self.layers:
            z = x
            z = layer['norm'](z)
            z = layer['linear0'](z)
            z = self.activation(z)
            if self.hdo > 0:
                z = F.dropout(z, self.hdo, self.training)
            z = layer['linear1'](z)
            if self.rdo:
                z = F.dropout(z, self.rdo, self.training)
            x = x + z

        x = self.last_normalization(x)
        x = self.activation(x)
        return x


def get_resnet_backbone(cfg):
    # classifier params
    assert 'num_tokens' in cfg, 'num_tokens should be given in config (feature vector dim)'
    in_dim = cfg['num_tokens']
    n_layers = cfg.get('num_layers', 3)
    hidden_size = cfg.get('hidden_size', 256)
    hidden_factor = cfg.get('hidden_factor', 1.5)
    hidden_dropout = cfg.get('hidden_dropout', 0.5)
    res_dropout = cfg.get('res_dropout', 0.2)
    keep_feats = cfg.get('keep_feats', None)

    backbone = Backbone_ResNet(in_dim=in_dim, n_layers=n_layers, hidden_size=hidden_size, 
                            hidden_factor=hidden_factor, hidden_dropout=hidden_dropout, 
                            residual_dropout=res_dropout, keep_feats=keep_feats)

    return backbone


class Self_Attention(nn.Module):
    def __init__(self, query_dim):
        # assume: query_dim = key/value_dim
        super(Self_Attention, self).__init__()
        self.scale = 1. / math.sqrt(query_dim)

    def forward(self, query, key, value):

        # query == hidden: (batch_size, hidden_dim * 2)
        # key/value == gru_output: (batch_size, sentence_length, hidden_dim * 2)
        query = query.unsqueeze(1) # (batch_size, 1, hidden_dim * 2)
        key = key.permute(0, 2, 1) # (batch_size, hidden_dim * 2, sentence_length)

        # bmm: batch matrix-matrix multiplication
        attention_weight = torch.bmm(query, key) # (batch_size, 1, sentence_length)
        attention_weight = F.softmax(attention_weight.mul_(self.scale), dim=2) # normalize sentence_length's dimension
        attention_output = torch.bmm(attention_weight, value) # (batch_size, 1, hidden_dim * 2)
        attention_output = attention_output.squeeze(1) # (batch_size, hidden_dim * 2)

        return attention_output, attention_weight.squeeze(1)


class Backbone_CNN_RNN_Attention(torch.nn.Module):
    def __init__(self, num_tokens, embed_size, num_kernels, kernel_size, pool_size, hidden_size, num_layers):
        super(Backbone_CNN_RNN_Attention, self).__init__()
        
        self.word_embeddings = nn.Embedding(num_tokens, embed_size, padding_idx=0)
        self.conv = nn.Conv1d(embed_size, num_kernels, kernel_size, padding='same')
        self.pool = nn.MaxPool1d(kernel_size=pool_size)
        self.rnn = nn.GRU(num_kernels, hidden_size, num_layers, bidirectional=True, batch_first=True)
        self.attention = Self_Attention(hidden_size * 2)
        self.linear = nn.Linear(hidden_size*2, hidden_size)
        self.feat_dim = hidden_size
        self.init = None

    def forward(self, x):  
        
        # batch_size, num_tokens
        x = self.word_embeddings(x)
        # batch_size, num_tokens, embedding_size
        x = x.permute(0,2,1) 
        # batch_size, embedding_size, num_tokens
        x = self.conv(x)
        # batch_size, conv_num_kernels, num_tokens
        x = self.pool(x)
        # batch_size, conv_num_kernels, num_tokens/pool_size
        x = x.permute(0,2,1)
        # batch_size, num_tokens/pool_size, conv_num_kernels
        x, hidden = self.rnn(x)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        x, attn_weights = self.attention(query=hidden, key=x, value=x)

        x = self.linear(x)

        return x

# num_tokens, embed_size, num_kernels, kernel_size, hidden_size, num_layers
def get_attention_backbone(cfg):
    num_tokens = cfg.get('num_tokens', 2**14)
    embed_size = cfg.get('embed_size', 256)
    num_kernels = cfg.get('num_kernels', 100)
    kernel_size = cfg.get('kernel_size', 4)
    pool_size = cfg.get('pool_size', 4)
    hidden_size = cfg.get('hidden_size', 32)
    num_layers = cfg.get('num_layers', 2)

    backbone = Backbone_CNN_RNN_Attention(num_tokens=num_tokens, embed_size=embed_size, 
                                          num_kernels=num_kernels, kernel_size=kernel_size, 
                                          pool_size=pool_size, hidden_size=hidden_size, num_layers=num_layers)

    return backbone