import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttentionWithKVCache(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, kv_cache=None, use_cache=False):
        """
        x: (B, T, d_model)  当前输入的 token（训练时是完整序列，推理时是单个 token）
        kv_cache: tuple of (cached_k, cached_v) each (B, n_heads, cache_len, d_k)
        use_cache: bool, 是否使用 KVCache（推理时为 True）
        
        Returns:
            output: (B, T, d_model)
            new_kv_cache: tuple (new_k, new_v) if use_cache else None
        """
        B, T, _ = x.shape

        # 线性变换
        Q = self.W_q(x)  # (B, T, d_model)
        K = self.W_k(x)  # (B, T, d_model)
        V = self.W_v(x)  # (B, T, d_model)

        # 拆分为多头
        Q = Q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # (B, n_heads, T, d_k)
        K = K.view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # (B, n_heads, T, d_k)
        V = V.view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # (B, n_heads, T, d_k)

        if use_cache and kv_cache is not None:
            cached_k, cached_v = kv_cache  # (B, n_heads, cache_len, d_k)
            # 拼接缓存的 K 和 V
            K = torch.cat([cached_k, K], dim=2)  # (B, n_heads, cache_len + T, d_k)
            V = torch.cat([cached_v, V], dim=2)
            new_kv_cache = (K, V)
        else:
            new_kv_cache = (K, V) if use_cache else None

        # Scaled Dot-Product Attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)  # (B, n_heads, T, T_cache)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)  # (B, n_heads, T, d_k)

        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.d_model)
        output = self.W_o(attn_output)

        return output, new_kv_cache
    




if __name__ == "__main__":
    # 参数
    d_model = 512
    n_heads = 8
    seq_len = 10
    batch_size = 1

    # 初始化模型
    model = MultiHeadAttentionWithKVCache(d_model, n_heads)

    # 模拟输入序列（逐 token 推理）
    x_full = torch.randn(batch_size, seq_len, d_model)

    kv_cache = None
    outputs = []

    for t in range(seq_len):
        x_t = x_full[:, t:t+1, :]  # (1, 1, d_model)
        out, kv_cache = model(x_t, kv_cache=kv_cache, use_cache=True)
        outputs.append(out)

    # 最终输出
    final_output = torch.cat(outputs, dim=1)  # (1, 10, 512)
    print("Output shape:", final_output.shape)