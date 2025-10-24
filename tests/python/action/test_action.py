import numpy as np
from action import StockInvestmentPlanner, InvestmentPlanPrinter, calculate_daily_plan
import pytest


def test_cal_action_size():
    planner = StockInvestmentPlanner(max_action_size=10, transaction_cost=0.001)

    price = 3
    actions = planner.calculate_action_range(current_shares=3, cash=20, price=price)
    # assert actions == [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
    assert len(actions)<=10
    actions = planner.calculate_action_range(current_shares=10, cash=5, price=price)
    # assert actions == [-10, -8, -6, -4, -2, 0]
    assert len(actions)<=10
    actions = planner.calculate_action_range(current_shares=5, cash=100, price=price)
    assert len(actions)<=10
    # assert actions == [-10, -8, -6, -4, -2, 0]


def test_act():
    """主函数"""
    # 测试数据
    np.random.seed(42)
    days = 10
    prices = [100]
    for i in range(days - 1):
        change = np.random.normal(0, 2)
        prices.append(max(prices[-1] + change, 50))

    initial_shares = 100
    initial_cash = 10000

    print(f"股票价格序列 ({days}天):")
    for i, price in enumerate(prices, 1):
        print(f"第{i:2d}天: {price:.2f}")

    print(f"\n初始状态: 持仓{initial_shares}股, 现金{initial_cash:.2f}")
    print("初始资产:",initial_shares*prices[0]+initial_cash)

    # 计算最佳交易序列
    planner = StockInvestmentPlanner()
    trades = planner.calculate_investment_plan(prices, initial_shares, initial_cash)
    print(trades)

#     # 打印交易序列
#     InvestmentPlanPrinter.print_trades(trades)

#     # 计算每日计划
#     daily_plan = calculate_daily_plan(prices, trades, initial_shares, initial_cash)

#     # 打印每日计划
#     InvestmentPlanPrinter.print_daily_plan(daily_plan)

#     # 打印总结
#     InvestmentPlanPrinter.print_summary(
#         daily_plan, initial_cash, initial_shares, prices[0]
#     )


if __name__ == "__main__":
    test_act()
    # test_cal_action_size()
