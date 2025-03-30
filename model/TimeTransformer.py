import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings

# # 忽略嵌套张量相关的警告
# warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.transformer")

class PositionalEncoding(nn.Module):
    """
    位置编码模块
    """
    def __init__(self, d_model, max_len=5000):
        """
        初始化位置编码模块

        参数:
            d_model: 隐藏层维度
            max_len: 最大序列长度
        """
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x:  (batch_size, seq_len, d_model)
        # output:  (batch_size, seq_len, d_model)
        x = x + self.pe[:x.size(0), :x.size(1), :]
        return x


class TransformerTimeSeriesPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_heads, ff_dim, dropout, target_len):
        """
        因果Transformer模型

        参数:
            input_dim: 输入特征维度
            hidden_dim: Transformer隐藏层维度
            output_dim: 输出特征维度
            num_layers: Transformer层数
            num_heads: 注意力头数
            ff_dim: 前馈网络维度
            dropout: Dropout率
        """
        super(TransformerTimeSeriesPredictor, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.target_len = target_len

        # 输入嵌入层
        self.input_embedding = nn.Linear(input_dim, hidden_dim)

        # 位置编码层
        self.positional_encoding = PositionalEncoding(hidden_dim)

        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Transformer解码器层
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 输出层
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x:  (batch_size, seq_len, input_dim)
        # output:  (batch_size, target_len, output_dim)
        # 输入嵌入
        x = self.input_embedding(x)  # (batch_size, seq_len, hidden_dim)
        # 添加位置编码
        x = self.positional_encoding(x)  # (batch_size, seq_len, hidden_dim)
        # Transformer编码器
        encoder_output = self.transformer_encoder(x)  # (batch_size, seq_len, hidden_dim)
        # 创建目标序列的初始输入（通常为全零或随机初始化）
        target = torch.zeros(x.size(0), self.target_len, self.hidden_dim, device=x.device)  # (batch_size, target_len, hidden_dim)
        # Transformer解码器
        decoder_output = self.transformer_decoder(target, encoder_output)  # (batch_size, target_len, hidden_dim)
        # 输出层
        output = self.output_layer(decoder_output)  # (batch_size, seq_len, output_dim)
        return output


if __name__ == "__main__":
    # 配置模型参数
    input_dim = 10  # 输入特征维度
    hidden_dim = 512  # Transformer隐藏层维度
    output_dim = 10  # 输出特征维度
    num_layers = 6  # Transformer层数
    num_heads = 8  # 注意力头数
    ff_dim = 2048  # 前馈网络维度
    dropout = 0.1  # Dropout率
    seq_len = 64  # 序列长度
    target_len = 128

    # 创建模型
    model = TransformerTimeSeriesPredictor(input_dim, hidden_dim, output_dim, num_layers, num_heads, ff_dim, dropout, target_len)

    # 创建一个随机输入张量
    batch_size = 32  # 批量大小
    x = torch.randn(batch_size, seq_len, input_dim)  # (batch_size, seq_len, input_dim)

    # 前向传播
    output = model(x)  # (batch_size, seq_len, output_dim)

    # 打印输出形状
    print("Output shape:", output.shape)
