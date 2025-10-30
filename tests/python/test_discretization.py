import torch

d_hidden = 3
steps = 5
batch = 4




def upper_bound (values: torch.Tensor, min_val: float = -1, max_val: float=1, steps: int=10) -> torch.Tensor:
    shared_seq = torch.linspace(min_val, max_val, steps=steps)  # [5]
    # values: [batch, d_hidden]
    idx = torch.searchsorted(shared_seq, values, side='right')  # ✅ 直接广播！
    # print("shared_seq:\n", shared_seq.shape)
    # print("idx:\n", idx)
    result = shared_seq[idx.clamp(max=steps-1)]  # [batch, d_hidden]
    return result

# 完全无循环、无显式 brc 扩展


def upper_bound_by_sorted_seq(values: torch.Tensor, seqs: list) -> torch.Tensor:
    """
    将连续值离散化为最近的预定义序列值。
    Args:
        values: torch.Tensor, 形状为 (batch, d_hidden)
        seqs: list of torch.Tensor, 每个元素形状为 (steps,)
    Returns:
        torch.Tensor, 离散化后的值，形状为 (batch, d_hidden)
    """
    batch, d_hidden = values.shape
    steps = seqs[0].shape[0]
    assert(len(seqs) == d_hidden), "seqs 长度必须等于 d_hidden"
    result = torch.empty(batch, d_hidden, device=values.device)

    for i in range(d_hidden):
        seq_i = seqs[i]  # 第 i 个维度的序列 [steps]
        val_i = values[:, i]  # 所有 batch 在第 i 维的值 [batch]
        
        # 核心：searchsorted 是向量化的！
        idx_i = torch.searchsorted(seq_i, val_i, side='right')  # [batch]
        
        # clamp 到有效范围，越界自动取最后一个
        idx_i_clamped = idx_i.clamp(max=steps-1)
        
        # 索引赋值
        result[:, i] = seq_i[idx_i_clamped]  # [batch]

    return result

# 每个维度的 linspace（可以不同）
linspace_vec = torch.linspace(-1, 1, steps=steps)  # [5]
# 如果不同维度不同，可以用 list of tensors
seqs = [linspace_vec for _ in range(d_hidden)]  # [d_hidden] 个 [steps]

# 查询值
torch.manual_seed(42)
values = torch.randn(batch, d_hidden) * 1.5  # 有些会超出 [-1,1]

# 存储结果
# result = torch.empty(batch, d_hidden, device=values.device)

# 只对 d_hidden 循环（通常很小，比如 8, 64, 128），不是对 batch
# for i in range(d_hidden):
#     seq_i = seqs[i]  # 第 i 个维度的序列 [steps]
#     val_i = values[:, i]  # 所有 batch 在第 i 维的值 [batch]
    
#     # 核心：searchsorted 是向量化的！
#     idx_i = torch.searchsorted(seq_i, val_i, side='right')  # [batch]
    
#     # clamp 到有效范围，越界自动取最后一个
#     idx_i_clamped = idx_i.clamp(max=steps-1)
    
#     # 索引赋值
#     result[:, i] = seq_i[idx_i_clamped]  # [batch]
result= upper_bound_by_sorted_seq(values, seqs)
print("values:\n", values)
print("result:\n", result)
result2= upper_bound(values,-1, 1 ,10)
print("result2:\n", result2)


