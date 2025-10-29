import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 2048):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_seq_len = max_seq_len
        
        # 线性变换层
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        
        # 缓存相关的属性
        self.k_cache = None
        self.v_cache = None
        self.current_seq_len = 0
        
    def _reset_cache(self):
        """重置KV缓存"""
        self.k_cache = None
        self.v_cache = None
        self.current_seq_len = 0
        
    def _init_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        """初始化KV缓存"""
        cache_shape = (batch_size, self.n_heads, self.max_seq_len, self.head_dim)
        self.k_cache = torch.zeros(cache_shape, device=device, dtype=dtype)
        self.v_cache = torch.zeros(cache_shape, device=device, dtype=dtype)
        self.current_seq_len = 0
        
    def _update_cache(self, k: torch.Tensor, v: torch.Tensor, start_pos: int):
        """更新KV缓存"""
        if self.k_cache is None or self.v_cache is None:
            self._init_cache(k.size(0), k.device, k.dtype)
            
        # 将新的k, v存入缓存
        seq_len = k.size(2)
        self.k_cache[:, :, start_pos:start_pos + seq_len] = k
        self.v_cache[:, :, start_pos:start_pos + seq_len] = v
        self.current_seq_len = start_pos + seq_len
        
    def _scaled_dot_product_attention(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """缩放点积注意力"""
        # q, k, v: [batch_size, n_heads, seq_len, head_dim]
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用mask（如果提供）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # softmax得到注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        
        # 加权求和
        output = torch.matmul(attn_weights, v)
        
        return output, attn_weights
    
    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
        start_pos: int = 0,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            use_cache: 是否使用KV缓存
            start_pos: 当前序列在缓存中的起始位置
            mask: 注意力mask
            
        Returns:
            output: 输出张量 [batch_size, seq_len, d_model]
            attn_weights: 注意力权重 [batch_size, n_heads, seq_len, seq_len]
        """
        print("shape of x is ",x.shape)
        import pdb; pdb.set_trace()
        batch_size, seq_len, _ = x.shape[-2]
        
        # 线性变换
        q = self.wq(x)  # [batch_size, seq_len, d_model]
        k = self.wk(x)
        v = self.wv(x)
        
        # 重整形为多头
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        if use_cache:
            if self.k_cache is not None and self.v_cache is not None:
                # 使用缓存中的k, v
                cache_k = self.k_cache[:, :, :start_pos]
                cache_v = self.v_cache[:, :, :start_pos]
                
                # 拼接新的k, v到缓存
                k = torch.cat([cache_k, k], dim=2)
                v = torch.cat([cache_v, v], dim=2)
                
            # 更新缓存
            self._update_cache(k, v, start_pos)
        
        # 计算注意力
        attn_output, attn_weights = self._scaled_dot_product_attention(q, k, v, mask)
        
        # 重整形回原始形状
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # 输出线性变换
        output = self.wo(attn_output)
        
        return output, attn_weights

def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """创建因果mask（用于自回归生成）"""
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, seq_len]

# 测试示例
def test_attention_with_kv_cache():
    # 参数设置
    d_model = 512
    n_heads = 8
    batch_size = 2
    seq_len = 10
    max_seq_len = 100
    
    # 创建注意力层
    attention = MultiHeadAttention(d_model, n_heads, max_seq_len)
    
    # 模拟输入
    x = torch.randn(batch_size, seq_len, d_model)
    
    print("=== 第一次前向传播（不使用缓存）===")
    output1, attn_weights1 = attention(x, use_cache=False)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output1.shape}")
    print(f"注意力权重形状: {attn_weights1.shape}")
    
    print("\n=== 第二次前向传播（使用缓存，模拟自回归生成）===")
    # 模拟自回归生成：每次一个token
    for i in range(3):
        # 模拟新生成的token（batch_size=2, seq_len=1）
        new_token = torch.randn(batch_size, 1, d_model)
        
        # 使用缓存进行前向传播
        output, attn_weights = attention(
            new_token, 
            use_cache=True, 
            start_pos=i  # 当前token位置
        )
        
        print(f"步骤 {i+1}: 新token形状 {new_token.shape}, 输出形状 {output.shape}")
        print(f"          缓存序列长度: {attention.current_seq_len}")
    
    print("\n=== 重置缓存测试 ===")
    attention._reset_cache()
    print(f"重置后缓存序列长度: {attention.current_seq_len}")

# 更复杂的示例：模拟完整的自回归生成过程
def simulate_autoregressive_generation():
    print("\n" + "="*50)
    print("模拟完整的自回归生成过程")
    print("="*50)
    
    d_model = 256
    n_heads = 4
    vocab_size = 1000
    max_seq_len = 50
    
    # 创建模型组件
    attention = MultiHeadAttention(d_model, n_heads, max_seq_len)
    embedding = nn.Embedding(vocab_size, d_model)
    lm_head = nn.Linear(d_model, vocab_size)
    
    # 模拟输入序列
    input_ids = torch.randint(0, vocab_size, (1, 5))  # [1, 5]
    print(f"初始输入序列长度: {input_ids.shape[1]}")
    
    # 编码输入序列
    with torch.no_grad():
        # 第一次前向传播（编码阶段）
        x = embedding(input_ids)
        output, _ = attention(x, use_cache=True, start_pos=0)
        logits = lm_head(output[:, -1:])  # 只取最后一个token的logits
        
        # 模拟生成5个新token
        generated_tokens = []
        for i in range(5):
            # 选择最可能的下一个token（这里简单使用argmax）
            next_token = torch.argmax(logits[:, -1:], dim=-1)
            generated_tokens.append(next_token.item())
            
            # 准备下一次前向传播的输入
            next_embedding = embedding(next_token).unsqueeze(1)
            
            # 使用KV缓存进行前向传播
            output, _ = attention(
                next_embedding, 
                use_cache=True, 
                start_pos=input_ids.shape[1] + i
            )
            
            # 预测下一个token
            logits = lm_head(output)
            
            print(f"生成步骤 {i+1}: 生成token {next_token.item()}, 缓存长度: {attention.current_seq_len}")
    
    print(f"生成的tokens: {generated_tokens}")

if __name__ == "__main__":
    test_attention_with_kv_cache()
    simulate_autoregressive_generation()