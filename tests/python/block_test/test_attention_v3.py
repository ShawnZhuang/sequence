import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class OutputLayer(nn.Module):
    """完整的输出层，包含LayerNorm、线性投影和损失计算"""
    
    def __init__(self, d_model: int, vocab_size: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # 输出层组件
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
    def forward(self, hidden_states: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            hidden_states: 注意力层的输出 [batch_size, seq_len, d_model]
            labels: 目标标签 [batch_size, seq_len]，包含-100的位置会被忽略
            
        Returns:
            logits: 预测logits [batch_size, seq_len, vocab_size]
            loss: 如果提供了labels则返回损失值，否则为None
        """
        # LayerNorm和Dropout
        x = self.layer_norm(hidden_states)
        x = self.dropout(x)
        
        # 线性投影到词汇表大小
        logits = self.lm_head(x)  # [batch_size, seq_len, vocab_size]
        
        # 计算损失（如果提供了labels）
        loss = None
        if labels is not None:
            # 将logits和labels重塑为CrossEntropyLoss需要的形状
            # logits: [batch_size * seq_len, vocab_size]
            # labels: [batch_size * seq_len]
            loss = self.criterion(
                logits.view(-1, logits.size(-1)), 
                labels.view(-1)
            )
        
        return logits, loss

class LanguageModelHead(nn.Module):
    """语言模型头部，支持权重共享"""
    
    def __init__(self, d_model: int, vocab_size: int, tie_weights: bool = False, embedding_layer: Optional[nn.Module] = None):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.tie_weights = tie_weights
        
        # 输出层
        self.layer_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # 权重共享（如果指定了embedding层）
        if tie_weights and embedding_layer is not None:
            self.lm_head.weight = embedding_layer.weight
            
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """前向传播，返回logits"""
        x = self.layer_norm(hidden_states)
        logits = self.lm_head(x)
        return logits

class TransformerLM(nn.Module):
    """完整的Transformer语言模型"""
    
    def __init__(self, vocab_size: int, d_model: int = 512, n_heads: int = 8, 
                 n_layers: int = 6, max_seq_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # 词嵌入层
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # 注意力层（多层）
        from attention_with_cache import MultiHeadAttention  # 导入之前实现的注意力层
        
        self.layers = nn.ModuleList([
            MultiHeadAttention(d_model, n_heads, max_seq_len) 
            for _ in range(n_layers)
        ])
        
        # 输出层（使用权重共享）
        self.output_layer = OutputLayer(d_model, vocab_size, dropout)
        
        # 初始化权重
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """权重初始化"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
            
    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None,
                use_cache: bool = False, start_pos: int = 0) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        完整的前向传播
        
        Args:
            input_ids: 输入token IDs [batch_size, seq_len]
            labels: 目标标签 [batch_size, seq_len]
            use_cache: 是否使用KV缓存
            start_pos: 缓存起始位置
            
        Returns:
            logits: 预测logits [batch_size, seq_len, vocab_size]
            loss: 交叉熵损失（如果提供了labels）
        """
        batch_size, seq_len = input_ids.shape
        
        # 创建位置编码
        positions = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)
        
        # 词嵌入 + 位置编码
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.position_embedding(positions)
        hidden_states = token_embeds + pos_embeds
        
        # 多层注意力
        for layer in self.layers:
            hidden_states, _ = layer(
                hidden_states, 
                use_cache=use_cache, 
                start_pos=start_pos
            )
        
        # 输出层
        logits, loss = self.output_layer(hidden_states, labels)
        
        return logits, loss
    
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, temperature: float = 1.0):
        """生成文本"""
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # 前向传播（使用缓存）
                logits, _ = self(input_ids, use_cache=True)
                
                # 取最后一个token的logits
                next_token_logits = logits[:, -1, :] / temperature
                
                # 应用softmax得到概率
                probs = F.softmax(next_token_logits, dim=-1)
                
                # 采样下一个token
                next_token = torch.multinomial(probs, num_samples=1)
                
                # 添加到输入序列
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # 如果生成了结束符，提前终止
                if next_token.item() == 0:  # 假设0是结束符
                    break
        
        return input_ids

class CrossEntropyLossWithZLoss(nn.Module):
    """带Z-Loss的交叉熵损失，提高训练稳定性"""
    
    def __init__(self, ignore_index: int = -100, z_loss_weight: float = 0.01):
        super().__init__()
        self.ignore_index = ignore_index
        self.z_loss_weight = z_loss_weight
        
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        计算带Z-Loss的交叉熵损失
        
        Args:
            logits: [batch_size, seq_len, vocab_size]
            labels: [batch_size, seq_len]
        """
        # 标准交叉熵损失
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=self.ignore_index
        )
        
        # Z-Loss（防止logits变得过大，提高数值稳定性）
        log_z = torch.logsumexp(logits, dim=-1)
        z_loss = torch.mean(log_z ** 2) * self.z_loss_weight
        
        return ce_loss + z_loss

# 测试代码
def test_output_layer():
    print("=== 测试输出层 ===")
    
    batch_size = 2
    seq_len = 10
    d_model = 512
    vocab_size = 10000
    
    # 创建输出层
    output_layer = OutputLayer(d_model, vocab_size)
    
    # 模拟注意力输出
    hidden_states = torch.randn(batch_size, seq_len, d_model)
    
    # 模拟标签（包含-100作为padding）
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels[0, 5:] = -100  # 设置一些padding位置
    
    print(f"隐藏状态形状: {hidden_states.shape}")
    print(f"标签形状: {labels.shape}")
    
    # 前向传播
    logits, loss = output_layer(hidden_states, labels)
    
    print(f"Logits形状: {logits.shape}")
    print(f"损失值: {loss.item():.4f}")
    
    return logits, loss

def test_complete_pipeline():
    print("\n=== 测试完整流程 ===")
    
    vocab_size = 5000
    batch_size = 2
    seq_len = 8
    
    # 创建完整模型
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=256,
        n_heads=4,
        n_layers=2
    )
    
    # 模拟输入数据
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print(f"输入形状: {input_ids.shape}")
    print(f"标签形状: {labels.shape}")
    
    # 前向传播
    logits, loss = model(input_ids, labels)
    
    print(f"模型输出Logits形状: {logits.shape}")
    print(f"总损失: {loss.item():.4f}")
    
    # 测试生成
    print("\n=== 测试文本生成 ===")
    prompt = torch.randint(0, vocab_size, (1, 5))
    generated = model.generate(prompt, max_new_tokens=3)
    print(f"生成结果形状: {generated.shape}")

def test_advanced_loss():
    print("\n=== 测试高级损失函数 ===")
    
    batch_size = 2
    seq_len = 5
    vocab_size = 1000
    d_model = 256
    
    # 创建带Z-Loss的损失函数
    criterion = CrossEntropyLossWithZLoss()
    
    # 模拟logits和labels
    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # 计算损失
    loss = criterion(logits, labels)
    
    print(f"带Z-Loss的交叉熵损失: {loss.item():.4f}")
    
    # 比较标准交叉熵损失
    standard_loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        labels.view(-1)
    )
    print(f"标准交叉熵损失: {standard_loss.item():.4f}")

if __name__ == "__main__":
    # 运行测试
    test_output_layer()
    test_complete_pipeline() 
    test_advanced_loss()
    
    # 性能测试
    print("\n=== 性能测试 ===")
    import time
    
    model = TransformerLM(vocab_size=10000, d_model=512, n_layers=4)
    input_ids = torch.randint(0, 10000, (4, 32))
    
    start_time = time.time()
    with torch.no_grad():
        logits, loss = model(input_ids)
    end_time = time.time()
    
    print(f"推理时间: {(end_time - start_time)*1000:.2f} ms")
    print(f"内存占用: {logits.element_size() * logits.nelement() / 1024 / 1024:.2f} MB")