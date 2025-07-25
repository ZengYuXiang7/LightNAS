from copy import deepcopy as cp

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def clones(module, N):
    return nn.ModuleList([cp(module) for _ in range(N)])

def attention(query, key, value, dropout = None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) #(b, n_head, l_q, d_per_head) * (b, n_head, d_per_head, l_k)
    attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        attn = dropout(attn) ##(b, n_head, l_q, l_k)
    return torch.matmul(attn, value), attn 

class MultiHeadAttention(nn.Module):
    def __init__(self, config, q_learnable):
        super(MultiHeadAttention, self).__init__()
        self.q_learnable = q_learnable
        self.d_model = config.graph_d_model
        self.n_head = config.graph_n_head
        self.d_k = config.graph_d_model // config.graph_n_head # default: 32
        if self.q_learnable:
            self.linears = clones(nn.Linear(self.d_model, self.d_model), 3)  
        else:
            self.linears = clones(nn.Linear(self.d_model, self.d_model), 4)  
        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, query, key, value):
        batch_size = query.size(0)

        if self.q_learnable:
            key , value = [l(x).view(batch_size, -1, self.n_head, self.d_k).transpose(1,2) for l, x in zip(self.linears, (key, value))]
            query = query.view(batch_size, -1, self.n_head, self.d_k).transpose(1,2) 
        else:    
            query, key , value = [l(x).view(batch_size, -1, self.n_head, self.d_k).transpose(1,2) for l, x in zip(self.linears, (query, key, value))]
        x, attn = attention(query, key, value, dropout = self.dropout) #x(b, n_head, l_q, d_k)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head * self.d_k)
        return self.linears[-1](x), attn
    
    
#Different Attention Blocks, All Based on MultiHeadAttention
class SelfAttentionBlock(nn.Module):
    def __init__(self, config):
        super(SelfAttentionBlock, self).__init__()
        self.norm = nn.LayerNorm(config.graph_d_model)
        self.attn = MultiHeadAttention(config, q_learnable=False)
        self.dropout = nn.Dropout(p = config.dropout) 
        self.drop_path = DropPath(config.drop_path_rate) if config.drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        x_ = self.norm(x)
        x_ , attn = self.attn(x_, x_, x_)
        return self.drop_path(x_) + x, attn  #L_v4.2.2

class CrossAttentionBlock(nn.Module):
    def __init__(self, config):
        super(CrossAttentionBlock, self).__init__()
        self.norm = nn.LayerNorm(config.graph_d_model)
        self.attn = MultiHeadAttention(config, q_learnable=True)
        self.dropout = nn.Dropout(p = config.dropout) 
        self.drop_path = DropPath(config.drop_path_rate) if config.drop_path_rate > 0. else nn.Identity()

    def forward(self, x, learnt_q):
        x_ = self.norm(x)
        x_ , attn = self.attn(learnt_q, x_, x_)
        # In multi_stage' attention, no residual connection is used because of the change in output shape
        return self.drop_path(x_), attn 

#Blocks Used in Encoder
class FuseFeatureBlock(nn.Module):
    def __init__(self, config):
        super(FuseFeatureBlock, self).__init__()
        self.norm_kv = nn.LayerNorm(config.graph_d_model)
        self.norm_q = nn.LayerNorm(config.graph_d_model)
        self.fuse_attn = MultiHeadAttention(config, q_learnable=False)
        self.feed_forward = FeedForwardBlock(config)

    def forward(self, memory, q):
        x_ = self.norm_kv(memory)
        q_ = self.norm_q(q)
        x , attn = self.fuse_attn(q_, x_, x_)
        x = self.feed_forward(x)
        return x, attn 

class EncoderBlock(nn.Module):
    def __init__(self, config):
        super(EncoderBlock, self).__init__()
        self.self_attn = SelfAttentionBlock(config)
        self.feed_forward = FeedForwardBlock(config)

    def forward(self, x):
        x, attn = self.self_attn(x)
        x = self.feed_forward(x) 
        return x, attn

class FuseStageBlock(nn.Module):
    def __init__(self, config, stg_id, dp_rates):
        super(FuseStageBlock, self).__init__()
        self.n_self_attn = config.depths[stg_id] - 1
        self.self_attns = nn.ModuleList()
        self.feed_forwards = nn.ModuleList()
        for i,r in enumerate(dp_rates):
            config.drop_path_rate = r
            self.feed_forwards.append(FeedForwardBlock(config))
            if i==0:
                self.cross_attn = CrossAttentionBlock(config)
            else:
                self.self_attns.append(SelfAttentionBlock(config))

    def forward(self, kv, q):
        x, attn = self.cross_attn(kv, q)
        x = self.feed_forwards[0](x)
        for i in range(self.n_self_attn):
            x, attn = self.self_attns[i](x)
            x = self.feed_forwards[i+1](x)
        return x, attn

#FFN 
class PositionwiseFeedForward(nn.Module):
    def __init__(self, config):
        super(PositionwiseFeedForward, self).__init__()

        self.w_1 = nn.Linear(config.graph_d_model, config.graph_d_ff)
        self.w_2 = nn.Linear(config.graph_d_ff, config.graph_d_model)
        self.dropout = nn.Dropout(p = config.dropout)
        if config.act_function.lower() == 'relu':
            self.act = torch.nn.ReLU()
        elif config.act_function.lower() == 'gelu':
            self.act = gelu
        else:
            raise ValueError("Unsupported activation: %s" % config.act_function)

    def forward(self, x):
        return self.w_2(self.dropout(self.act(self.w_1(x))))


class FeedForwardBlock(nn.Module):
    def __init__(self, config):
        super(FeedForwardBlock, self).__init__()
        self.norm = nn.LayerNorm(config.graph_d_model)
        self.feed_forward = PositionwiseFeedForward(config)
        self.dropout = nn.Dropout(p = config.dropout)
        self.drop_path = DropPath(config.drop_path_rate) if config.drop_path_rate > 0. else nn.Identity()
    def forward(self, x):
        x_ = self.norm(x) 
        x_ = self.feed_forward(x_)
        return self.drop_path(x_) + x



import math
import torch
import torch.nn as nn
import numpy as np


def tokenizer(ops, matrix, dim_x, dim_r, dim_p, embed_type): 
    # encode operation
    fn, _ = get_embedder(dim_x, embed_type=embed_type)
    code_ops_tmp = [fn(torch.tensor([op], dtype=torch.float32)) for op in ops]
    code_ops_tmp.append(fn(torch.tensor([1e5], dtype=torch.float32)))
    code_ops = torch.stack(code_ops_tmp, dim=0) #(len, dim_x)

    # encode self position
    fn, _ = get_embedder(dim_p, embed_type=embed_type)
    code_pos_tmp = [fn(torch.tensor([i], dtype=torch.float32)) for i in range(len(ops))]
    code_pos_tmp.append(fn(torch.tensor([1e5], dtype=torch.float32)))
    code_pos = torch.stack(code_pos_tmp, dim=0) #(len, dim_p)

    # encode data source of each node
    fn, _ = get_embedder(dim_r, embed_type=embed_type)
    code_sour_tmp = [fn(torch.tensor([-1], dtype=torch.float32))]
    for i in range(1, len(ops)):
        i_sour = 0
        for j in range(i):
            if matrix[j][i] == 1:
                i_sour += fn(torch.tensor([j], dtype=torch.float32))
        code_sour_tmp.append(i_sour)
    code_sour_tmp.append(fn(torch.tensor([1e5], dtype=torch.float32)))
    code_sour = torch.stack(code_sour_tmp, dim=0) #(len, dim_r)
    code = torch.cat([code_ops, code_pos, code_sour], dim=-1)
    return code
    
    
def get_embedder(multires, embed_type='nerf', input_type='tensor'):
    embed_kwargs = {
                'input_type' : input_type,
                'embedding_type' : embed_type,
                'input_dims' : 1,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
    }
    if input_type=='tensor': 
        embed_kwargs['periodic_fns'] = [torch.sin, torch.cos] 
        embedder_obj = Embedder(**embed_kwargs)
        embed = lambda x, eo=embedder_obj : eo.embed_tensor(x)
    else:
        embed_kwargs['periodic_fns'] = [np.sin, np.cos]
        embedder_obj = Embedder(**embed_kwargs)
        embed = lambda x, eo=embedder_obj : eo.embed(x)
    
    return embed, embedder_obj.out_dim


class Embedder():
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
            
        max_freq = self.kwargs['max_freq_log2'] # max_freq=multires-1
        N_freqs = self.kwargs['num_freqs'] # N_freqs=multires
        
        dty = self.kwargs['input_type']
        if self.kwargs['embedding_type'] == 'nerf':
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs) if dty=='tensor'\
                            else np.linspace(2.**0., 2.**max_freq, num=N_freqs)
            
            for freq in freq_bands:
                for p_fn in self.kwargs['periodic_fns']: #p_fn=torch.sin, p_fn=torch.cos
                    embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * math.pi * freq))
                    out_dim += d 
        
        elif self.kwargs['embedding_type'] == 'trans':
            dim = self.kwargs['num_freqs']
            freq_bands = [ 1 / (10000**(j/dim)) for j in range(dim)]
            for freq in freq_bands: #
                for p_fn in self.kwargs['periodic_fns']: #p_fn=torch.sin, p_fn=torch.cos
                    embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                    out_dim += d 

        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed_tensor(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)
    
    def embed(self, inputs):
        return np.concatenate([fn(inputs) for fn in self.embed_fns])

#Main class
class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.num_stage = len(config.depths)
        self.num_layers = sum(config.depths)
        self.norm = nn.LayerNorm(config.graph_d_model)
        
        # stochastic depth 
        dpr =  [x.item() for x in torch.linspace(0, config.drop_path_rate, self.num_layers)]
        
        #1st stage: Encoder
        self.layers = nn.ModuleList()
        for i in range(config.depths[0]):
            config.drop_path_rate = dpr[i]
            self.layers.append(EncoderBlock(config))

        if self.num_stage > 1:
            #Rest stage: information fusion
            self.fuseUnit = nn.ModuleList()
            self.fuseStages = nn.ModuleList()
            self.fuseStages.append(FuseStageBlock(config, stg_id=1, dp_rates=dpr[sum(config.depths[:1]):sum(config.depths[:2])]))
            for i in range(2, self.num_stage):
                self.fuseUnit.append(FuseFeatureBlock(config))
                self.fuseStages.append(FuseStageBlock(config, stg_id=i, dp_rates=dpr[sum(config.depths[:i]):sum(config.depths[:i+1])]))

            self.learnt_q = nn.ParameterList([nn.Parameter(torch.randn(1, 2**(3-s), config.graph_d_model)) for s in range(1, self.num_stage)])

    def forward(self, x):
        B, _, _ = x.shape

        #1st stage: Encoder
        for i, layer in enumerate(self.layers):
            x, attn = layer(x) #EncoderBlock()
            
        x_ = x
        #Rest stage: information fusion
        if self.num_stage > 1:
            memory = x
            q, attn = self.fuseStages[0](memory, self.learnt_q[0].repeat(B,1,1,1)) #q(b,4,d)
            for i in range(self.num_stage-2):
                kv, attn = self.fuseUnit[i](memory, q)
                q, attn = self.fuseStages[i+1](kv, self.learnt_q[i+1].repeat(B,1,1,1)) #q(b,2,d), q(b,1,d)
            x_ = q
        output = self.norm(x_)
        
        return output


class RegHead(nn.Module):
    def __init__(self, config):
        super(RegHead, self).__init__()
        self.config = config
        if self.config.avg_tokens:
            self.pool = nn.AdaptiveAvgPool1d(1)
        self.layer = nn.Linear(config.d_model, 1)
        self.dataset = config.dataset
        if self.dataset == 'nnlqp':
            mlp_hiddens = [config.d_model//(2**i) for i in range(4)]
            self.mlp = []
            dim = config.d_model
            for hidden_size in mlp_hiddens:
                self.mlp.append(
                    nn.Sequential(
                        nn.Linear(dim, hidden_size),
                        nn.ReLU(inplace=False),
                        nn.Dropout(p=config.dropout)
                    )
                )
                dim = hidden_size
            self.mlp.append(nn.Linear(dim, 1))
            self.mlp = nn.Sequential(*self.mlp)

    def forward(self, x, sf): #x(b/n_gpu, l, d)
        if self.config.avg_tokens:
            x_ = self.pool(x.permute(0, 2, 1)).permute(0, 2, 1).squeeze(1)
        else:
            x_ = x[:,-1,:] #(b,d)
        
        # if self.dataset == 'nnlqp':
            # x_ = torch.cat([x_, sf], dim=-1)
        
        res = self.mlp(x_) if self.dataset == 'nnlqp' else F.sigmoid(self.layer(x_))
        return res
    

class NarFormer(torch.nn.Module):
    def __init__(self, enc_in, config):
        super(NarFormer, self).__init__()
        self.config = config
        self.d_model = config.d_model
        self.transformer = Encoder(config)
        self.mlp = RegHead(config)
        
    def forward(self, x, static_feats=None):
        x_enc = self.transformer(x) #multi_stage:aev(b, 1, d)
        y = self.mlp(x_enc, static_feats)
        return y
