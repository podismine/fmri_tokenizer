import torch
import torch.nn as nn

# 定义 Vector Quantizer
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
    def __init__(self, input_dim, hidden_dim, num_embeddings, embedding_dim, commitment_cost):
        super(VQVAE, self).__init__()
        # self.encoder = nn.Sequential(
        #     nn.Conv1d(input_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.Conv1d(hidden_dim, embedding_dim, kernel_size=4, stride=2, padding=1),
        #     nn.ReLU(),
        # )

        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers=4, batch_first=True, bidirectional=True)
        self.fc_encoder = nn.Linear(hidden_dim * 2, embedding_dim)  # 双向 GRU 输出拼接

        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose1d(embedding_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
        #     nn.LeakyReLU(),
        #     nn.BatchNorm1d(hidden_dim),
        #     nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
        #     nn.LeakyReLU(),
        #     nn.BatchNorm1d(hidden_dim),
        #     nn.ConvTranspose1d(hidden_dim, input_dim, kernel_size=4, stride=2, padding=1),
        # )

        self.fc_decoder = nn.Linear(embedding_dim, hidden_dim * 2)
        self.decoder = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, input_dim, kernel_size=1, stride=1, padding=0),
        )
    def forward_token(self,x):
        gru_out, _ = self.encoder(x)  # (batch, time, hidden_dim * 2)
        gru_out = self.fc_encoder(gru_out)  # (batch, time, embedding_dim)
        quantized, loss, _ = self.quantizer(gru_out)
        return quantized
    def forward(self, x):
        # GRU 编码器
        gru_out, _ = self.encoder(x)  # (batch, time, hidden_dim * 2)
        gru_out = self.fc_encoder(gru_out)  # (batch, time, embedding_dim)

        # 量化
        # gru_out = gru_out.permute(0, 2, 1).contiguous()  # 转为 (batch, embedding_dim, time)
        quantized, loss, _ = self.quantizer(gru_out)

        # 解码器
        decoded = self.fc_decoder(quantized).permute(0, 2, 1)  # 转回 (batch, time, hidden_dim * 2)
        x_reconstructed = self.decoder(decoded).permute(0, 2, 1)  # 输出 (batch, input_dim, time)
        return x_reconstructed, loss