import numpy as np
from typing import List, Dict, Any

class StockInvestmentPlanner:
    def __init__(self, transaction_cost=0.001, max_position_ratio=0.8,max_action_size=10):
        """
        初始化投资规划器
        
        Args:
            transaction_cost: 交易成本率
            max_position_ratio: 最大持仓比例
        """
        self.transaction_cost = transaction_cost
        self.max_position_ratio = max_position_ratio
        self.max_action_size= max_action_size
    
    def calculate_investment_plan(self, prices: List[float], 
                                current_shares: int, 
                                remaining_cash: float) -> List[int]:
        """
        计算最佳投资计划，返回交易量的列表
        
        Args:
            prices: 股票价格序列
            current_shares: 当前持有股票数量
            remaining_cash: 剩余现金
            
        Returns:
            交易量列表，正数表示买入，负数表示卖出，0表示不交易

        状态: 每天的持仓量
        动作: 买入、卖出或持有一定数量的股票
        转移: 根据当前价格和交易成本更新现金和持仓
        奖励: 最终投资组合的总价值     
        dp[day][shares] = 最大组合价值
        for action in possible_actions:
            new_shares = shares + action
            new_cash = current_cash - action * prices[day] - transaction_costs
            dp[day+1][new_shares] = max(dp[day+1][new_shares], new_cash + new_shares * prices[day+1])
            prev[day+1][new_shares] = (shares, action)
        """
        days = len(prices)
        if days == 0:
            raise ValueError("价格序列不能为空")

        dp = [dict() for _ in range(days)]
        # 路径记录: prev[day][shares] = (前一天股份, 交易量)
        # prev = np.full((days, max_possible_shares, 2), -1, dtype=int)
        # 初始化第0天
        dp[0][current_shares] = remaining_cash
        
        # 动态规划
        for day in range(days - 1):
            for shares, cash in dp[day].items():          
                next_actions=self.calculate_action_range(shares,cash,prices[day+1])
                for trade in next_actions:
                    transaction_cost_amount= abs(trade) * prices[day] * self.transaction_cost
                    if trade == 0:  # 不交易
                        new_shares = shares
                        cost = 0
                        new_cash = cash
                    else:
                        new_shares = shares + trade                        
                        if trade > 0:  # 买入
                            cost = trade * prices[day] + transaction_cost_amount
                            if cost > cash:
                                continue
                            new_cash = cash - cost
                        else:  # 卖出
                            new_cash = cash + trade * prices[day] - transaction_cost_amount
                    
                    
                    # 计算新状态的价值
                    # total_value = new_cash + new_shares * prices[day + 1]
                    
                    if new_cash > dp[day + 1].get(new_shares,0):
                        dp[day + 1][new_shares] = new_cash
                        # prev[day + 1][new_shares] = [shares, trade]  # 记录前继状态
        
        # 重建最佳路径，返回交易量列表

        return dp[days - 1]
        # return self._rebuild_trades(prices, dp, prev, current_shares, max_possible_shares)
    
    def calculate_action_range(self, current_shares: int, cash: float, price: float) -> List[int]:
        """计算当前状态下的可行动作范围"""
        max_buy = int(cash  / (price * (1 + self.transaction_cost)))
        min_sell = -current_shares
        step= (max_buy-min_sell+ self.max_action_size-1)// self.max_action_size
        # print(f"计算可行动作范围: 当前持仓={current_shares}, 现金={cash:.2f}, 价格={price:.2f}, 最大买入={max_buy}, 最小卖出={min_sell}, 步长={step}")
        return list(range(min_sell, max_buy + 1,step))

    def _calculate_realistic_max_shares(self, current_shares: int, cash: float, min_price: float) -> int:
        """计算实际可能的最大持仓量"""
        max_shares_from_cash = int(cash / (min_price * (1 + self.transaction_cost)))
        max_shares = current_shares + max_shares_from_cash + 20
        return min(max_shares, 2000)
    
    def _rebuild_trades(self, prices: List[float], dp: np.ndarray, 
                       prev: np.ndarray, initial_shares: int,
                       max_shares: int) -> List[int]:
        """重建最佳交易序列"""
        days = len(prices)
        
        # 找到最佳最终状态
        best_value = -np.inf
        best_shares = 0
        for shares in range(max_shares):
            if dp[days - 1][shares] > best_value:
                best_value = dp[days - 1][shares]
                best_shares = shares
        
        # 从最终状态向前重建路径，获取交易序列
        trades = [0] * days  # 初始化所有天数为不交易
        current_shares = best_shares
        
        for day in range(days - 1, 0, -1):
            prev_shares, trade = prev[day][current_shares]
            if prev_shares != -1:
                trades[day - 1] = trade  # 交易发生在day-1天
                current_shares = prev_shares
        
        return trades


def calculate_daily_plan(prices: List[float], 
                        trades: List[int], 
                        initial_shares: int, 
                        initial_cash: float,
                        transaction_cost: float = 0.001) -> List[Dict[str, Any]]:
    """
    根据交易序列计算每日投资计划
    
    Args:
        prices: 股票价格序列
        trades: 交易量列表，正数表示买入，负数表示卖出，0表示不交易
        initial_shares: 初始持有股票数量
        initial_cash: 初始现金
        transaction_cost: 交易成本率
        
    Returns:
        每日投资计划列表
    """
    daily_plan = []
    current_shares = initial_shares
    current_cash = initial_cash
    
    for day, trade_amount in enumerate(trades):
        price = prices[day]
        
        # 执行交易
        if trade_amount != 0:
            transaction_cost_amount = abs(trade_amount) * price * transaction_cost
            
            if trade_amount > 0:  # 买入
                cost = trade_amount * price + transaction_cost_amount
                if cost <= current_cash:  # 确保有足够现金
                    current_cash -= cost
                    current_shares += trade_amount
                else:
                    # 现金不足，改为不交易
                    trade_amount = 0
            else:  # 卖出
                if abs(trade_amount) <= current_shares:  # 确保有足够股票
                    revenue = abs(trade_amount) * price - transaction_cost_amount
                    current_cash += revenue
                    current_shares += trade_amount  # trade_amount是负数
                else:
                    # 股票不足，改为不交易
                    trade_amount = 0
        
        # 确定操作类型
        if trade_amount > 0:
            action = 'BUY'
        elif trade_amount < 0:
            action = 'SELL'
        else:
            action = 'HOLD'
        
        portfolio_value = current_cash + current_shares * price
        
        daily_plan.append({
            'day': day + 1,
            'price': price,
            'action': action,
            'shares_traded': abs(trade_amount),
            'shares_held': current_shares,
            'cash': current_cash,
            'portfolio_value': portfolio_value
        })
    
    return daily_plan


def calculate_final_value(prices: List[float], 
                         trades: List[int], 
                         initial_shares: int, 
                         initial_cash: float,
                         transaction_cost: float = 0.001) -> float:
    """
    计算最终投资组合价值
    
    Args:
        prices: 股票价格序列
        trades: 交易量列表
        initial_shares: 初始持有股票数量
        initial_cash: 初始现金
        transaction_cost: 交易成本率
        
    Returns:
        最终投资组合价值
    """
    current_shares = initial_shares
    current_cash = initial_cash
    
    for day, trade_amount in enumerate(trades):
        price = prices[day]
        
        # 执行交易
        if trade_amount != 0:
            transaction_cost_amount = abs(trade_amount) * price * transaction_cost
            
            if trade_amount > 0:  # 买入
                cost = trade_amount * price + transaction_cost_amount
                if cost <= current_cash:
                    current_cash -= cost
                    current_shares += trade_amount
            else:  # 卖出
                if abs(trade_amount) <= current_shares:
                    revenue = abs(trade_amount) * price - transaction_cost_amount
                    current_cash += revenue
                    current_shares += trade_amount
    
    # 最终价值 = 现金 + 股票价值
    return current_cash + current_shares * prices[-1]


class InvestmentPlanPrinter:
    """投资计划打印器"""
    
    @staticmethod
    def print_trades(trades: List[int]):
        """打印交易序列"""
        print(f"\n最佳交易序列 ({len(trades)}天):")
        print("天数 | 交易量 | 操作")
        print("-" * 25)
        for day, trade in enumerate(trades, 1):
            if trade > 0:
                action = f"买入 {trade}"
            elif trade < 0:
                action = f"卖出 {abs(trade)}"
            else:
                action = "持有"
            print(f"{day:2d} | {trade:6d} | {action}")
    
    @staticmethod
    def print_daily_plan(daily_plan: List[Dict[str, Any]], max_display: int = 15):
        """打印每日投资计划"""
        days = len(daily_plan)
        
        print(f"\n详细投资计划 ({days}天):")
        print("天数 | 价格 | 操作 | 交易股数 | 持仓股数 | 现金 | 组合价值")
        print("-" * 70)
        
        # 显示策略：有交易的日子 + 首尾几天
        if days > max_display:
            trade_days = [day for day in daily_plan if day['action'] != 'HOLD']
            first_few = daily_plan[:2]
            last_few = daily_plan[-2:]
            
            display_days = first_few + trade_days + last_few
            # 去重并排序
            seen = set()
            unique_days = []
            for day in display_days:
                if day['day'] not in seen:
                    unique_days.append(day)
                    seen.add(day['day'])
            unique_days.sort(key=lambda x: x['day'])
            
            if len(unique_days) < len(daily_plan):
                print(f"(显示{len(unique_days)}个关键交易日)")
        else:
            unique_days = daily_plan
        
        for day_plan in unique_days:
            print(f"{day_plan['day']:2d} | {day_plan['price']:6.2f} | "
                  f"{day_plan['action']:4} | {day_plan['shares_traded']:8d} | "
                  f"{day_plan['shares_held']:8d} | {day_plan['cash']:8.2f} | "
                  f"{day_plan['portfolio_value']:10.2f}")
    
    @staticmethod
    def print_summary(daily_plan: List[Dict[str, Any]], initial_cash: float, initial_shares: int, first_price: float):
        """打印投资总结"""
        total_buy = sum(1 for day in daily_plan if day['action'] == 'BUY')
        total_sell = sum(1 for day in daily_plan if day['action'] == 'SELL')
        final_shares = daily_plan[-1]['shares_held'] if daily_plan else 0
        final_cash = daily_plan[-1]['cash'] if daily_plan else 0
        final_value = daily_plan[-1]['portfolio_value'] if daily_plan else 0
        
        initial_total = initial_cash + initial_shares * first_price
        total_return = (final_value - initial_total) / initial_total * 100 if initial_total > 0 else 0
        
        print(f"\n投资总结:")
        print(f"总买入交易次数: {total_buy}")
        print(f"总卖出交易次数: {total_sell}")
        print(f"最终持有股数: {final_shares}")
        print(f"最终现金: {final_cash:.2f}")
        print(f"最终组合价值: {final_value:.2f}")
        print(f"总收益率: {total_return:.2f}%")

