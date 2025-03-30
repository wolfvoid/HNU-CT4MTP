import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMTimeSeriesPredictor(nn.Module):
    """
    LSTM时序预测模型
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.1):
        """
        初始化LSTM时序预测模型

        参数:
            input_dim: 输入特征维度 (C)
            hidden_dim: LSTM隐藏层维度
            output_dim: 输出特征维度 (C)
            num_layers: LSTM层数
            dropout: Dropout率
        """
        super(LSTMTimeSeriesPredictor, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # LSTM层
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)

        # 输出层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, target_len):
        # LSTM层
        # x: (batch_size, t1, input_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)  # (batch_size, t1, hidden_dim), (num_layers, batch_size, hidden_dim), (num_layers, batch_size, hidden_dim)

        # 使用最后一个时间步的隐藏状态和细胞状态作为初始状态
        # h_n 的形状为 (num_layers, batch_size, hidden_dim)
        # c_n 的形状为 (num_layers, batch_size, hidden_dim)
        # 这些状态将被用作目标序列生成的初始状态

        # 初始化目标序列的输入
        target_input = x[:, -1, :].unsqueeze(1)  # (batch_size, 1, input_dim)

        # 逐步生成目标序列
        outputs = []
        for _ in range(target_len):
            target_out, (h_n, c_n) = self.lstm(target_input, (h_n, c_n))  # (batch_size, 1, hidden_dim)
            target_out = self.fc(target_out)  # (batch_size, 1, output_dim)
            outputs.append(target_out)
            target_input = target_out  # 使用上一个时间步的输出作为下一个时间步的输入

        # 将所有时间步的输出拼接起来
        output = torch.cat(outputs, dim=1)  # (batch_size, target_len, output_dim)
        return output


if __name__ == "__main__":
    # 配置模型参数
    input_dim = 10  # 输入特征维度 (C)
    hidden_dim = 128  # LSTM隐藏层维度
    output_dim = 10  # 输出特征维度 (C)
    num_layers = 2  # LSTM层数
    dropout = 0.1  # Dropout率

    # 创建模型
    model = LSTMTimeSeriesPredictor(input_dim, hidden_dim, output_dim, num_layers, dropout)

    # 创建一个随机输入张量
    batch_size = 32  # 批量大小
    t1 = 20  # 输入序列长度
    target_len = 10  # 目标输出序列长度
    x = torch.randn(batch_size, t1, input_dim)  # (batch_size, t1, input_dim)

    # 前向传播
    output = model(x, target_len)  # (batch_size, target_len, output_dim)

    # 打印输出形状
    print("Output shape:", output.shape)
