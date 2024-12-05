import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import einsum
from einops import rearrange
from mamba_ssm import Mamba
    
class CVQ(nn.Module):

    def __init__(self, num_embed, embed_dim, beta, distance='cos', 
                 anchor='closest', first_batch=False, contras_loss=True):
        super().__init__()

        self.num_embed = num_embed
        self.embed_dim = embed_dim
        self.beta = beta
        self.distance = distance
        self.anchor = anchor
        self.first_batch = first_batch
        self.contras_loss = contras_loss
        self.decay = 0.99
        self.init = False

        self.embedding = nn.Embedding(self.num_embed, self.embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embed, 1.0 / self.num_embed)
        self.register_buffer("embed_prob", torch.zeros(self.num_embed))

    
    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        # z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.contiguous().view(-1, self.embed_dim)

        # clculate the distance
        if self.distance == 'l2':
            # l2 distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
            d = - torch.sum(z_flattened.detach() ** 2, dim=1, keepdim=True) - \
                torch.sum(self.embedding.weight ** 2, dim=1) + \
                2 * torch.einsum('bd, dn-> bn', z_flattened.detach(), rearrange(self.embedding.weight, 'n d-> d n'))
        elif self.distance == 'cos':
            # cosine distances from z to embeddings e_j 
            normed_z_flattened = F.normalize(z_flattened, dim=1).detach()
            normed_codebook = F.normalize(self.embedding.weight, dim=1)
            d = torch.einsum('bd,dn->bn', normed_z_flattened, rearrange(normed_codebook, 'n d -> d n'))

        # encoding
        sort_distance, indices = d.sort(dim=1)
        # look up the closest point for the indices
        encoding_indices = indices[:,-1]
        encodings = torch.zeros(encoding_indices.unsqueeze(1).shape[0], self.num_embed, device=z.device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)

        # quantise and unflatten
        z_q = torch.matmul(encodings, self.embedding.weight).view(z.shape)
        # compute loss for embedding
        loss = self.beta * torch.mean((z_q.detach()-z)**2) + torch.mean((z_q - z.detach()) ** 2)
        # preserve gradients
        z_q = z + (z_q - z).detach()
        # reshape back to match original input shape
        # z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        # count
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        min_encodings = encodings

        # online clustered reinitialisation for unoptimized points
        if self.training:
            # calculate the average usage of code entries
            self.embed_prob.mul_(self.decay).add_(avg_probs, alpha= 1 - self.decay)
            # running average updates
            if self.anchor in ['closest', 'random', 'probrandom'] and (not self.init):
                # closest sampling
                if self.anchor == 'closest':
                    sort_distance, indices = d.sort(dim=0)
                    random_feat = z_flattened.detach()[indices[-1,:]]
                # probabilitical based random sampling
                elif self.anchor == 'probrandom':
                    norm_distance = F.softmax(d.t(), dim=1)
                    prob = torch.multinomial(norm_distance, num_samples=1).view(-1)
                    random_feat = z_flattened.detach()[prob]
                # decay parameter based on the average usage
                decay = torch.exp(-(self.embed_prob*self.num_embed*10)/(1-self.decay)-1e-3).unsqueeze(1).repeat(1, self.embed_dim)
                self.embedding.weight.data = self.embedding.weight.data * (1 - decay) + random_feat * decay
                if self.first_batch:
                    self.init = True
            # contrastive loss
            if self.contras_loss:
                sort_distance, indices = d.sort(dim=0)
                dis_pos = sort_distance[-max(1, int(sort_distance.size(0)/self.num_embed)):,:].mean(dim=0, keepdim=True)
                dis_neg = sort_distance[:int(sort_distance.size(0)*1/2),:]
                dis = torch.cat([dis_pos, dis_neg], dim=0).t() / 0.07
                contra_loss = F.cross_entropy(dis, torch.zeros((dis.size(0),), dtype=torch.long, device=dis.device))
                loss +=  contra_loss

        # return z_q, loss, (perplexity, min_encodings, encoding_indices)
        return z_q, loss, encoding_indices #(perplexity, min_encodings, encoding_indices)
    
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        # 定义嵌入向量
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, x):
        # 添加 batch 和特征维度
        flat_x = x.view(-1, self.embedding_dim)

        # 计算到嵌入的欧几里得距离
        distances = (flat_x ** 2).sum(dim=1, keepdim=True) - 2 * flat_x @ self.embedding.weight.t() + \
                    (self.embedding.weight ** 2).sum(dim=1)
        
        # 找到最近的嵌入
        indices = torch.argmin(distances, dim=1).unsqueeze(1)
        quantized = self.embedding(indices).view_as(x)

        # 损失：压缩表示和解码后的表示一致
        e_latent_loss = torch.mean((quantized.detach() - x) ** 2)
        q_latent_loss = torch.mean((quantized - x.detach()) ** 2)
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # 保持梯度
        quantized = x + (quantized - x).detach()
        return quantized, loss, indices

# 定义 VQ-VAE 模型
class VQVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, state_num_embeddings, transition_num_embeddings,layer, embedding_dim, commitment_cost,model = 'gru'):
        super(VQVAE, self).__init__()
        self.model = model
        if self.model not in ['gru','lstm','rnn','mamba']:
            raise KeyError(f'{self.model} not in list')
        
        self.embedding_dim = embedding_dim
        spatial_transformer_encoders = nn.TransformerEncoderLayer(d_model=input_dim, nhead=4, dim_feedforward=hidden_dim, batch_first=True)
        temporal_transformer_encoders = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=hidden_dim, batch_first=True)

        self.state_spatial_encoder = nn.TransformerEncoder(spatial_transformer_encoders,num_layers=2,norm=nn.LayerNorm(input_dim))
        self.state_temporal_encoder = nn.TransformerEncoder(temporal_transformer_encoders,num_layers=2,norm=nn.LayerNorm(64))

        self.spatial_attn = nn.Sequential(
            nn.Linear(64, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 64),
            nn.Sigmoid()
        )

        self.temporal_attn = nn.Sequential(
            nn.Linear(100, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 100),
            nn.Sigmoid()
        )

        if self.model == 'rnn':
            self.transition_encoder = nn.RNN(input_dim, hidden_dim, num_layers=layer, batch_first=True)
            self.transition_fc_encoder = nn.Linear(hidden_dim, embedding_dim)  # 双向 GRU 输出拼接
        elif self.model == 'gru':
            self.transition_encoder = nn.GRU(input_dim, hidden_dim, num_layers=layer, batch_first=True)
            self.transition_fc_encoder = nn.Linear(hidden_dim, embedding_dim)  # 双向 GRU 输出拼接
        elif self.model == 'lstm':
            self.transition_encoder = nn.LSTM(input_dim, hidden_dim, num_layers=layer, batch_first=True)
            self.transition_fc_encoder = nn.Linear(hidden_dim, embedding_dim)  # 双向 GRU 输出拼接
        elif self.model == 'mamba':
            self.transition_encoder = Mamba(d_model=input_dim, d_state=4, d_conv=3, expand=layer)
            self.transition_fc_encoder = nn.Linear(input_dim, embedding_dim)  # 双向 GRU 输出拼接
        self.state_proj = nn.Linear(input_dim, hidden_dim)

        # self.encoder = nn.GRU(input_dim, hidden_dim, num_layers=4, batch_first=True, bidirectional=True)
        self.state_fc_encoder = nn.Linear(hidden_dim, embedding_dim)  # 双向 GRU 输出拼接

        self.state_quantizer = CVQ(state_num_embeddings, embedding_dim, commitment_cost)
        self.transition_quantizer = CVQ(transition_num_embeddings, embedding_dim, commitment_cost)
        # VectorQuantizer(state_num_embeddings, embedding_dim, commitment_cost)
        # self.transition_quantizer = VectorQuantizer(transition_num_embeddings, embedding_dim, commitment_cost)

        self.state_fc_decoder = nn.Linear(embedding_dim, hidden_dim)
        self.transition_fc_decoder = nn.Linear(embedding_dim, hidden_dim)

        transformer_decoders = nn.TransformerEncoderLayer(d_model=hidden_dim * 2, nhead=4, dim_feedforward=hidden_dim * 2, batch_first=True)
        self.decoder = nn.TransformerEncoder(transformer_decoders,num_layers=2,norm=nn.LayerNorm(hidden_dim * 2))
        self.predict = nn.Linear(hidden_dim * 2, input_dim)
    def forward_token(self,x):

        temporal_x = self.state_spatial_encoder(x) # N, 64, 100
        spatial_x = self.state_temporal_encoder(x.permute(0,2,1)) # N, 100, 64

        temporal_attn = self.spatial_attn(temporal_x.mean(-1))[:,None] # N, 1, 64
        spatial_attn = self.temporal_attn(spatial_x.mean(-1))[:,None] # N, 1, 100
        print(temporal_x.shape,spatial_attn.shape,spatial_x.shape,temporal_attn.shape,x.shape);exit()
        spa_tem_out = temporal_x * spatial_attn + spatial_x * temporal_attn + x

        if self.model == 'mamba':
            transitions = self.transition_encoder(spa_tem_out)
        else:
            transitions, hn = self.transition_encoder(spa_tem_out)
        states = self.state_proj(spa_tem_out)

        states = self.state_fc_encoder(states)
        transitions = self.transition_fc_encoder(transitions)

        quantized_state, loss_state, token_state = self.state_quantizer(states)
        quantized_transition, loss_transition, token_transition = self.transition_quantizer(transitions)
        return quantized_state, quantized_transition, token_state, token_transition
    
    def forward(self, x):

        temporal_x = self.state_spatial_encoder(x) # N, 64, 100
        spatial_x = self.state_temporal_encoder(x.permute(0,2,1)) # N, 100, 64

        temporal_attn = self.spatial_attn(temporal_x.mean(-1))[:,None] # N, 1, 64
        spatial_attn = self.temporal_attn(spatial_x.mean(-1))[:,None] # N, 1, 100

        # print(temporal_x.shape,spatial_attn.shape,spatial_x.shape,temporal_attn.shape,x.shape);exit()
        spa_tem_out = temporal_x * spatial_attn + (spatial_x * temporal_attn).permute(0,2,1) + x

        # transitions, hn = self.transition_encoder(x)
        if self.model == 'mamba':
            transitions = self.transition_encoder(spa_tem_out)
        else:
            transitions, hn = self.transition_encoder(spa_tem_out)
        states = self.state_proj(spa_tem_out)

        states = self.state_fc_encoder(states)
        transitions = self.transition_fc_encoder(transitions)

        # gru_out = gru_out.permute(0, 2, 1).contiguous()  # 转为 (batch, embedding_dim, time)
        # print("before quantizer:", states.shape, transitions.shape) # torch.Size([256, 64, 512]) torch.Size([256, 64, 512])
        # print(self.embedding_dim);exit()
        quantized_state, loss_state, _ = self.state_quantizer(states)
        quantized_transition, loss_transition, _ = self.transition_quantizer(transitions)

        # 解码器
        decoded_state = self.state_fc_decoder(quantized_state)#.permute(0, 2, 1)  # 转回 (batch, time, hidden_dim * 2)
        decoded_transition = self.transition_fc_decoder(quantized_transition)#.permute(0, 2, 1)  # 转回 (batch, time, hidden_dim * 2)

        decoded = torch.cat([decoded_state, decoded_transition], -1)
        x_reconstructed_state = self.decoder(decoded)#.permute(0, 2, 1)
        x_reconstructed_state = self.predict(x_reconstructed_state)
        return x_reconstructed_state, loss_state, loss_transition