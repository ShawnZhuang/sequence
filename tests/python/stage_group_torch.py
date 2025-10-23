import torch
import numpy as np
from collections import defaultdict

def analyze_strategies_torch(probabilities, actions, profits):
    """
    使用 PyTorch 向量化实现：聚合每个 action 的期望收益和期望损失
    
    参数:
        probabilities: list of float
        actions:       list of str (可重复)
        profits:       list of float
    
    返回:
        list of dict: 排序后的结果 [{'action': ..., 'expected_profit': ..., 'expected_loss': ...}, ...]
    """
    assert len(probabilities) == len(actions) == len(profits)

    # 转为 Tensor
    probs = torch.tensor(probabilities, dtype=torch.float32)
    profits = torch.tensor(profits, dtype=torch.float32)

    # 获取唯一的 action 及其映射
    unique_actions = list(set(actions))
    action_to_idx = {act: idx for idx, act in enumerate(unique_actions)}
    num_actions = len(unique_actions)

    # 将 actions 转为索引张量
    action_indices = torch.tensor([action_to_idx[act] for act in actions], dtype=torch.long)

    # 初始化聚合张量: [num_actions]
    expected_profit = torch.zeros(num_actions)
    expected_loss = torch.zeros(num_actions)

    # 向量化计算期望收益
    expected_profit.scatter_add_(0, action_indices, probs * profits)

    # 计算期望损失：只对 profit < 0 的项，加上 prob * |profit|
    loss_mask = profits < 0
    abs_losses = probs * profits  # 正的损失贡献
    expected_loss.scatter_add_(0, action_indices, abs_losses * loss_mask)

    # 转回 CPU 和 Python 列表
    exp_profit_np = expected_profit.numpy()
    exp_loss_np = expected_loss.numpy()

    # 构建结果并排序（Python 层排序不影响性能瓶颈）
    results = [
        {
            'action': act,
            'expected_profit': exp_profit_np[i],
            'expected_loss': exp_loss_np[i]
        }
        for i, act in enumerate(unique_actions)
    ]

    # 排序：期望收益降序，期望损失升序
    results.sort(key=lambda x: (-x['expected_profit'], x['expected_loss']))
    return results


import random

# 设置随机种子以复现
random.seed(42)

N = 64
actions_pool = ['买入', '卖出', '持有', '减仓', '加仓']

actions = [random.choice(actions_pool) for _ in range(N)]
profits = []
for a in actions:
    if a == '买入':
        profits.append(random.choices([-50, 20, 80], weights=[0.2, 0.3, 0.5])[0])
    elif a == '卖出':
        profits.append(random.choices([10, 30, -40], weights=[0.3, 0.4, 0.3])[0])
    elif a == '持有':
        profits.append(random.choices([-20, 0, 25], weights=[0.2, 0.5, 0.3])[0])
    elif a == '加仓':
        profits.append(random.choices([-80, -30, 50, 100], weights=[0.1, 0.2, 0.3, 0.4])[0])
    elif a == '减仓':
        profits.append(random.choices([-10, 15, 40], weights=[0.1, 0.4, 0.5])[0])

# 归一化概率
raw_probs = [random.uniform(0.5, 3.0) for _ in range(N)]
probabilities = [p / sum(raw_probs) for p in raw_probs]

# 使用 PyTorch 版本计算
results = analyze_strategies_torch(probabilities, actions, profits)

# 输出
print(f"{'Action':<4} {'Exp Profit':<12} {'Exp Loss':<12}")
print("-" * 30)
for res in results:
    print(f"{res['action']:<6} {res['expected_profit']:>10.6f} {res['expected_loss']:>10.6f}")