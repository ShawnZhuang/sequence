import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        初始化多头注意力模块。

        参数:
        - d_model: 模型的维度（如 512）
        - num_heads: 注意力头的数量（如 8）
        - dropout: Dropout 概率
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度

        # 定义线性变换层（W_q, W_k, W_v, W_o）
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)  # 可选：在残差连接后使用 LayerNorm

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        缩放点积注意力计算。

        参数:
        - Q: 查询矩阵 (batch_size, num_heads, seq_len, d_k)
        - K: 键矩阵     (batch_size, num_heads, seq_len, d_k)
        - V: 值矩阵     (batch_size, num_heads, seq_len, d_k)
        - mask: 掩码张量 (可选)，用于屏蔽填充或未来位置

        返回:
        - 上下文张量 (batch_size, num_heads, seq_len, d_k)
        - 注意力权重 (batch_size, num_heads, seq_len, seq_len)
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        
        if mask is not None:
            # 将 mask 中为 True 的位置设为 -inf，防止信息泄露
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        output = torch.matmul(attn, V)
        return output, attn

    def split_heads(self, x):
        """
        将输入按头数分割。
        x: (batch_size, seq_len, d_model)
        返回: (batch_size, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        """
        合并所有注意力头。
        x: (batch_size, num_heads, seq_len, d_k)
        返回: (batch_size, seq_len, d_model)
        """
        batch_size, _, seq_len, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, query, key, value, mask=None):
        """
        前向传播。

        参数:
        - query, key, value: 形状均为 (batch_size, seq_len, d_model)
        - mask: 注意力掩码，形状为 (batch_size, 1, seq_len, seq_len) 或类似

        返回:
        - 输出张量 (batch_size, seq_len, d_model)
        - 注意力权重（可用于可视化）
        """
        residual = query  # 残差连接用

        # 线性变换
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # 分割头
        Q = self.split_heads(Q)  # (batch_size, num_heads, seq_len, d_k)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # 缩放点积注意力
        x, attn = self.scaled_dot_product_attention(Q, K, V, mask)

        # 合并头
        x = self.combine_heads(x)  # (batch_size, seq_len, d_model)

        # 线性输出
        output = self.W_o(x)  # (batch_size, seq_len, d_model)

        # 残差连接 + LayerNorm（常见于 Transformer）
        output = self.layer_norm(output + residual)
        
        return output, attn


# ======================
# 使用示例
# ======================
if __name__ == "__main__":
    # 参数设置
    d_model = 512
    num_heads = 8
    batch_size = 2
    seq_len = 10

    # 创建模型
    attn_layer = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

    # 随机输入
    x = torch.randn(batch_size, seq_len, d_model)

    # 可选：创建一个因果掩码（用于解码器）
    def generate_causal_mask(size):
        """生成上三角为0的因果掩码"""
        mask = torch.tril(torch.ones(size, size))  # 下三角包括对角线为1
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, size, size)

    mask = generate_causal_mask(seq_len)  # (1, 1, seq_len, seq_len)

    # 前向传播
    output, attention_weights = attn_layer(x, x, x, mask=mask)

    print(f"输入形状: {x.shape}")                    # [2, 10, 512]
    print(f"输出形状: {output.shape}")              # [2, 10, 512]
    print(f"注意力权重形状: {attention_weights.shape}") # [2, 8, 10, 10]

# ONNX 导出路径
    onnx_file_path = "multihead_attention.onnx"

    # inferred_model = onnx.shape_inference.infer_shapes(onnx_model)


    dummy_input = torch.randn(1, 10, 512)  # (batch, seq_len, d_model)
    dummy_mask = torch.ones(1, 1, 10, 10).bool()  # 示例 mask


    # 使用 torch.onnx.export 导出
    torch.onnx.export(
        attn_layer,
        (dummy_input, dummy_input, dummy_input, dummy_mask),  # (query, key, value, mask)
        onnx_file_path,
        export_params=True,  # 存储训练好的权重
        opset_version=13,   # 推荐使用 13 或更高（支持更多算子）
        do_constant_folding=True,  # 优化常量
        input_names=["query", "key", "value", "mask"],
        output_names=["output"],
        dynamic_axes={
            "query": {0: "batch_size", 1: "seq_len"},
            "key": {0: "batch_size", 1: "seq_len"},
            "value": {0: "batch_size", 1: "seq_len"},
            "mask": {2: "tgt_len", 3: "src_len"},
            "output": {0: "batch_size", 1: "seq_len"},
        },  # 支持动态 batch 和序列长度
    )

    print(f"✅ ONNX 模型已成功导出到: {onnx_file_path}")


