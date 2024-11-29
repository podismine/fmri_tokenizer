import torch
import torch.nn as nn

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
    def __init__(self, input_dim, hidden_dim, state_num_embeddings, transition_num_embeddings, embedding_dim, commitment_cost):
        super(VQVAE, self).__init__()
        self.embedding_dim = embedding_dim
        transformer_encoders = nn.TransformerEncoderLayer(d_model=input_dim, nhead=2, dim_feedforward=hidden_dim, batch_first=True)
        self.state_encoder = nn.TransformerEncoder(transformer_encoders,num_layers=2,norm=nn.LayerNorm(input_dim))

        self.transition_encoder = nn.GRU(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.state_proj = nn.Linear(input_dim, hidden_dim)

        # self.encoder = nn.GRU(input_dim, hidden_dim, num_layers=4, batch_first=True, bidirectional=True)
        self.state_fc_encoder = nn.Linear(hidden_dim, embedding_dim)  # 双向 GRU 输出拼接
        self.transition_fc_encoder = nn.Linear(hidden_dim, embedding_dim)  # 双向 GRU 输出拼接

        self.state_quantizer = VectorQuantizer(state_num_embeddings, embedding_dim, commitment_cost)
        self.transition_quantizer = VectorQuantizer(transition_num_embeddings, embedding_dim, commitment_cost)

        self.state_fc_decoder = nn.Linear(embedding_dim, hidden_dim)
        self.transition_fc_decoder = nn.Linear(embedding_dim, hidden_dim)

        transformer_decoders = nn.TransformerEncoderLayer(d_model=hidden_dim * 2, nhead=2, dim_feedforward=hidden_dim * 2, batch_first=True)
        self.decoder = nn.TransformerEncoder(transformer_decoders,num_layers=2,norm=nn.LayerNorm(hidden_dim * 2))
        self.predict = nn.Linear(hidden_dim * 2, input_dim)
    def forward_token(self,x):
        x = self.state_encoder(x)
        transitions, hn = self.transition_encoder(x)
        states = self.state_proj(x)

        states = self.state_fc_encoder(states)
        transitions = self.transition_fc_encoder(transitions)

        # gru_out = gru_out.permute(0, 2, 1).contiguous()  # 转为 (batch, embedding_dim, time)
        # print("before quantizer:", states.shape, transitions.shape) # torch.Size([256, 64, 512]) torch.Size([256, 64, 512])
        # print(self.embedding_dim);exit()
        quantized_state, loss_state, token_state = self.state_quantizer(states)
        quantized_transition, loss_transition, token_transition = self.transition_quantizer(transitions)
        return quantized_state, quantized_transition, token_state, token_transition
    def forward(self, x):

        x = self.state_encoder(x)
        transitions, hn = self.transition_encoder(x)
        states = self.state_proj(x)

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