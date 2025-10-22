import numpy as np
from typing import List, Tuple



 

class StockInvestmentPlanner:
    def __init__(self, transaction_cost=0.001, max_position_ratio=0.8):
        """
        初始化投资规划器
        
        Args:
            transaction_cost: 交易成本率
            max_position_ratio: 最大持仓比例
        """
        self.transaction_cost = transaction_cost
        self.max_position_ratio = max_position_ratio
    
    def calculate_investment_plan(self, prices: List[float], 
                                current_shares: int, 
                                remaining_cash: float) -> dict:
        """
        计算最佳投资计划
        
        Args:
            prices: 32天的股票价格序列
            current_shares: 当前持有股票数量
            remaining_cash: 剩余现金
            
        Returns:
            包含最佳投资计划的字典
            
        for act in actions:
            dp[][remain_share][remain_cach] = dp[][remain_share][remain_cach] + reward(act)
            
        """
        days = len(prices)        
        # 估算最大可能持仓量
        max_possible_shares = current_shares + int(remaining_cash / prices[0]) + 100
        max_possible_shares = min(max_possible_shares, 10000)  # 设置上限避免内存过大
        
        # 动态规划表: dp[day][shares] = 最大组合价值
        dp = np.full((days, max_possible_shares), -np.inf)
        # 路径记录: path[day][shares] = (前一天股份, 交易量)
        path = np.full((days, max_possible_shares, 2), -1, dtype=int)
        
        # 初始化第0天
        initial_value = remaining_cash + current_shares * prices[0]
        if current_shares < max_possible_shares:
            dp[0][current_shares] = initial_value
        
        # 动态规划
        for day in range(days - 1):
            for shares in range(max_possible_shares):
                if dp[day][shares] == -np.inf:
                    continue
                
                # 当前现金 = 总价值 - 股票价值
                current_cash = dp[day][shares] - shares * prices[day]
                if current_cash < 0:
                    continue
                
                # 计算最大可买入股数
                max_buy_shares = int(current_cash * self.max_position_ratio / (prices[day] * (1 + self.transaction_cost)))
                max_buy_shares = min(max_buy_shares, max_possible_shares - shares - 1)
                
                # 考虑所有可能的交易决策 (-shares 到 max_buy_shares)
                for trade in range(-shares, max_buy_shares + 1):
                    if trade == 0:  # 不交易
                        new_shares = shares
                        transaction_cost_amount = 0
                        new_cash = current_cash
                    else:
                        new_shares = shares + trade
                        transaction_cost_amount = abs(trade) * prices[day] * self.transaction_cost
                        
                        if trade > 0:  # 买入
                            cost = trade * prices[day] + transaction_cost_amount
                            if cost > current_cash:
                                continue
                            new_cash = current_cash - cost
                        else:  # 卖出
                            revenue = -trade * prices[day] - transaction_cost_amount
                            new_cash = current_cash + revenue
                    
                    # 确保新股份在合理范围内
                    if new_shares < 0 or new_shares >= max_possible_shares:
                        continue
                    
                    # 计算新一天的总价值
                    total_value = new_cash + new_shares * prices[day + 1]
                    
                    if total_value > dp[day + 1][new_shares]:
                        dp[day + 1][new_shares] = total_value
                        path[day + 1][new_shares] = [shares, trade]  # 修正：只存储2个值
        
        # 回溯找到最佳路径
        best_plan = self._backtrack_path(prices, dp, path, current_shares, remaining_cash, max_possible_shares)
        return best_plan
    
    def _backtrack_path(self, prices: List[float], dp: np.ndarray, 
                       path: np.ndarray, initial_shares: int, initial_cash: float,
                       max_possible_shares: int) -> dict:
        """回溯找到最佳投资路径"""
        days = len(prices)
        
        # 找到最后一天的最佳状态
        best_final_value = -np.inf
        best_final_shares = 0
        
        for shares in range(max_possible_shares):
            if dp[days - 1][shares] > best_final_value:
                best_final_value = dp[days - 1][shares]
                best_final_shares = shares
        
        # 回溯路径
        trades = [0] * days  # 初始化所有天数为不交易
        current_shares = best_final_shares
        
        # 从最后一天回溯到第一天
        for day in range(days - 1, 0, -1):
            prev_shares, trade = path[day][current_shares]
            if prev_shares != -1:  # 如果有有效路径
                trades[day - 1] = trade  # 记录交易（发生在day-1天）
                current_shares = prev_shares
        
        # 计算每日持仓和交易
        daily_plan = []
        current_shares = initial_shares
        current_cash = initial_cash
        
        for day in range(days):
            trade_amount = trades[day] if day < len(trades) else 0
            
            # 执行交易
            if trade_amount != 0:
                trade_value = trade_amount * prices[day]
                transaction_cost = abs(trade_amount) * prices[day] * self.transaction_cost
                
                if trade_amount > 0:  # 买入
                    current_cash -= trade_value + transaction_cost
                else:  # 卖出
                    current_cash += -trade_value - transaction_cost  # trade_amount是负数
                
                current_shares += trade_amount
            
            portfolio_value = current_cash + current_shares * prices[day]
            
            daily_plan.append({
                'day': day + 1,
                'price': prices[day],
                'action': 'BUY' if trade_amount > 0 else 'SELL' if trade_amount < 0 else 'HOLD',
                'shares_traded': abs(trade_amount),
                'shares_held': current_shares,
                'cash': current_cash,
                'portfolio_value': portfolio_value
            })
        
        return {
            'initial_cash': initial_cash,
            'initial_shares': initial_shares,
            'final_portfolio_value': best_final_value,
            'total_return': (best_final_value - (initial_cash + initial_shares * prices[0])) / 
                           (initial_cash + initial_shares * prices[0]) * 100,
            'daily_plan': daily_plan,
            'summary': {
                'total_buy_trades': sum(1 for day in daily_plan if day['action'] == 'BUY'),
                'total_sell_trades': sum(1 for day in daily_plan if day['action'] == 'SELL'),
                'final_shares': current_shares,
                'final_cash': current_cash
            }
        }

def main():
    # 示例数据
    np.random.seed(42)
    
    # 生成32天的模拟价格序列 (起始价格100)
    base_price = 100
    prices = [base_price]
    for i in range(5):
        change = np.random.normal(0, 2)  # 每日价格变化
        new_price = max(prices[-1] + change, 1)  # 价格不能低于1
        prices.append(new_price)
    
    # 初始状态
    current_shares = 100
    remaining_cash = 10000
    
    print("股票价格序列 (32天):")
    for i, price in enumerate(prices, 1):
        print(f"第{i:2d}天: {price:.2f}")
    
    print(f"\n初始状态:")
    print(f"持有股票: {current_shares}股")
    print(f"剩余现金: {remaining_cash:.2f}")
    print(f"初始总资产: {remaining_cash + current_shares * prices[0]:.2f}")
    
    # 计算投资计划
    planner = StockInvestmentPlanner()
    plan = planner.calculate_investment_plan(prices, current_shares, remaining_cash)
    
    print(f"\n最佳投资计划结果:")
    print(f"最终投资组合价值: {plan['final_portfolio_value']:.2f}")
    print(f"总收益率: {plan['total_return']:.2f}%")
    
    print(f"\n详细投资计划:")
    print("天数 | 价格 | 操作 | 交易股数 | 持仓股数 | 现金 | 组合价值")
    print("-" * 70)
    
    for day_plan in plan['daily_plan']:
        print(f"{day_plan['day']:2d} | {day_plan['price']:6.2f} | "
              f"{day_plan['action']:4} | {day_plan['shares_traded']:8d} | "
              f"{day_plan['shares_held']:8d} | {day_plan['cash']:8.2f} | "
              f"{day_plan['portfolio_value']:10.2f}")
    
    print(f"\n投资总结:")
    print(f"总买入交易次数: {plan['summary']['total_buy_trades']}")
    print(f"总卖出交易次数: {plan['summary']['total_sell_trades']}")
    print(f"最终持有股数: {plan['summary']['final_shares']}")
    print(f"最终现金: {plan['summary']['final_cash']:.2f}")

if __name__ == "__main__":
    main()