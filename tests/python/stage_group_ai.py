import random
def analyze_strategies(probabilities, actions, profits):
    """
    计算每个 action 的总期望收益和总期望损失（支持重复 action），并排序。

    参数:
        probabilities (list of float): 每个事件发生的概率
        actions (list): 对应的动作（可重复）
        profits (list of float): 对应的收益（负值为损失）

    返回:
        list of dict: 每个唯一 action 的汇总结果，按期望收益降序、期望损失升序排列
    """
    if not (len(probabilities) == len(actions) == len(profits)):
        raise ValueError("probabilities, actions, profits 的长度必须一致")

    # 使用字典聚合相同 action 的期望
    aggregated = {}

    for prob, action, profit in zip(probabilities, actions, profits):
        expected_profit = prob * profit
        expected_loss = prob * profit if profit < 0 else 0
        # max_loss = min(profit, 0)

        if action not in aggregated:
            aggregated[action] = {
                'action': action,
                'expected_profit': 0.0,
                'expected_loss': 0.0
            }
        
        aggregated[action]['expected_profit'] += expected_profit
        aggregated[action]['expected_loss'] += expected_loss
        # aggregated[action]['max_loss'] = min(aggregated[action].get('max_loss',0), max_loss)

    # 转为列表并排序：期望收益降序，期望损失升序
    results = list(aggregated.values())
    sorted_results = sorted(results, key=lambda x: ( -x['expected_loss'], -x['expected_profit']))

    return sorted_results


def demo_data(N):
    actions_pool = [1,2,-1, -2, 0]

    # 生成 64 个事件
    actions = [random.choice(actions_pool) for _ in range(N)]

    # 收益：根据 action 设定一定分布倾向
    profits = []
    for a in actions:
        if a > 0:
            profits.append(random.choices([-50, 20, 80], weights=[0.2, 0.3, 0.5])[0])
        elif a <0:
            profits.append(random.choices([10, 30, -40], weights=[0.3, 0.4, 0.3])[0])
        else :
            profits.append(random.choices([-20, 0, 25], weights=[0.2, 0.5, 0.3])[0])

    # 概率：随机生成并归一化，使总和为 1（表示互斥事件的完整分布）
    raw_probs = [random.uniform(0.5, 3.0) for _ in range(N)]  # 原始权重
    total = sum(raw_probs)
    probabilities = [p / total for p in raw_probs]  # 归一化，sum=1

    # 确保浮点精度下接近 1
    assert abs(sum(probabilities) - 1.0) < 1e-10
    return probabilities, actions, profits

# 示例：包含重复 action
if __name__ == "__main__":
    # random.seed(42)  # 可复现 
    probabilities, actions, profits  =demo_data(64)
    result = analyze_strategies(probabilities, actions, profits)

    for res in result:
        print(f"Action: {res['action']}, "
        f"期望收益: {res['expected_profit']:.2f}, "
        f"期望损失: {res['expected_loss']:.2f}, "
        # f"最大损失: {res['max_loss']:.2f}"
        )