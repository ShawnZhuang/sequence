
import numpy as np
from action import StockInvestmentPlanner, InvestmentPlanPrinter, calculate_daily_plan

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
    
    # 计算最佳交易序列
    planner = StockInvestmentPlanner()
    trades = planner.calculate_investment_plan(prices, initial_shares, initial_cash)
    
    # 打印交易序列
    InvestmentPlanPrinter.print_trades(trades)
    
    # 计算每日计划
    daily_plan = calculate_daily_plan(prices, trades, initial_shares, initial_cash)
    
    # 打印每日计划
    InvestmentPlanPrinter.print_daily_plan(daily_plan)
    
    # 打印总结
    InvestmentPlanPrinter.print_summary(daily_plan, initial_cash, initial_shares, prices[0])


if __name__ == "__main__":
    test_act()