import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ProbSparseAttention(nn.Module):
    def __init__(self, d_model, n_heads, factor=5):
        super(ProbSparseAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.factor = factor
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        d = self.d_model // H

        queries = self.query_linear(queries).view(B, L, H, d)
        keys = self.key_linear(keys).view(B, S, H, d)
        values = self.value_linear(values).view(B, S, H, d)

        # ProbSparse Attention Mechanism
        u = self.factor * math.log(L)
        U = int(u)
        scores = torch.einsum("blhd,bshd->bhls", queries, keys)
        top_k = torch.topk(scores, k=U, dim=-1)[1]
        sparse_scores = torch.zeros_like(scores).scatter_(-1, top_k, scores.gather(-1, top_k))

        attention = F.softmax(sparse_scores, dim=-1)
        attention = self.dropout(attention)
        output = torch.einsum("bhls,bshd->blhd", attention, values).reshape(B, L, self.d_model)
        return output

class InformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, factor=5):
        super(InformerEncoderLayer, self).__init__()
        self.self_attn = ProbSparseAttention(d_model, n_heads, factor)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, src):
        src2 = self.self_attn(src, src, src)
        src = src + self.dropout(src2)
        src = self.layer_norm1(src)
        src2 = self.feed_forward(src)
        src = src + self.dropout(src2)
        src = self.layer_norm2(src)
        return src

class InformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, num_layers, factor=5):
        super(InformerEncoder, self).__init__()
        self.layers = nn.ModuleList([InformerEncoderLayer(d_model, n_heads, factor) for _ in range(num_layers)])

    def forward(self, src):
        for layer in self.layers:
            src = layer(src)
        return src

class InformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, factor=5):
        super(InformerDecoderLayer, self).__init__()
        self.self_attn = ProbSparseAttention(d_model, n_heads, factor)
        self.cross_attn = ProbSparseAttention(d_model, n_heads, factor)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, tgt, memory):
        tgt2 = self.self_attn(tgt, tgt, tgt)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.layer_norm1(tgt)
        tgt2 = self.cross_attn(tgt, memory, memory)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.layer_norm2(tgt)
        tgt2 = self.feed_forward(tgt)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.layer_norm3(tgt)
        return tgt

class InformerDecoder(nn.Module):
    def __init__(self, d_model, n_heads, num_layers, factor=5, output_dim=1):
        super(InformerDecoder, self).__init__()
        self.layers = nn.ModuleList([InformerDecoderLayer(d_model, n_heads, factor) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, output_dim)  # Assuming univariate output

    def forward(self, tgt, memory):
        for layer in self.layers:
            tgt = layer(tgt, memory)
        output = self.fc_out(tgt)
        return output

class Informer(nn.Module):
    def __init__(self, d_model, n_heads, num_encoder_layers, num_decoder_layers, factor=5, output_dim=1):
        super(Informer, self).__init__()
        self.encoder = InformerEncoder(d_model, n_heads, num_encoder_layers, factor)
        self.decoder = InformerDecoder(d_model, n_heads, num_decoder_layers, factor, output_dim)

    def forward(self, src, tgt):
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        return output

# Example usage
if __name__ == "__main__":
    d_model = 512
    n_heads = 8
    num_encoder_layers = 3
    num_decoder_layers = 2
    factor = 5
    batch_size = 32
    seq_len = 48  # Input sequence length
    label_len = 48  # Label sequence length
    pred_len = 24  # Prediction length
    num_variables = 10  # Number of variables

    model = Informer(d_model, n_heads, num_encoder_layers, num_decoder_layers, factor, output_dim=num_variables)

    src = torch.randn(batch_size, seq_len, d_model)  # (Batch, Seq_len, d_model)
    tgt = torch.randn(batch_size, label_len, d_model)  # (Batch, Label_len, d_model)

    output = model(src, tgt)  # (Batch, Pred_len, num_variables)
    print(output.shape)  # Should be (batch_size, pred_len, num_variables)
