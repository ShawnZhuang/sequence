import akshare as ak
import pandas as pd
import numpy as np
from pybroker import Strategy, StrategyConfig, register_data_source
from datetime import datetime

# -----------------------------
# Step 1: 使用 register_data_source 定义 akshare 数据源
# -----------------------------
@register_data_source(name='akshare')
def akshare_data(symbols, start_date, end_date):
    """
    使用 akshare 获取 A 股日线数据
    返回符合 PyBroker 格式的 DataFrame
    """
    data = []
    for symbol in symbols:
        # 提取股票代码（如 600519.SH -> 600519）
        code = symbol.split('.')[0]
        try:
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_date.strftime('%Y%m%d'),
                end_date=end_date.strftime('%Y%m%d'),
                adjust="qfq"  # 前复权
            )
            if df.empty:
                print(f"⚠️ 未获取到 {symbol} 的数据")
                continue

            # 重命名列
            df.rename(columns={
                "日期": "date",
                "开盘": "open",
                "最高": "high",
                "最低": "low",
                "收盘": "close",
                "成交量": "volume"
            }, inplace=True)

            df['date'] = pd.to_datetime(df['date'])
            df['symbol'] = symbol  # 注意：必须保留 symbol 列

            # 选择并排序字段
            df = df[['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']]
            df.dropna(inplace=True)
            data.append(df)
        except Exception as e:
            print(f"❌ 获取 {symbol} 数据失败: {e}")

    if data:
        return pd.concat(data, ignore_index=True)
    else:
        return pd.DataFrame()

# -----------------------------
# Step 2: 定义策略逻辑
# -----------------------------
def buy_breakout_high(ctx):
    if ctx.long_pos():  # 已持仓则跳过
        return
    if ctx.bars < 6:     # 确保有足够的历史数据
        return

    # 前5日最高价
    lookback_high = np.max(ctx.high[-6:-1])
    current_close = ctx.close[-1]

    if current_close > lookback_high:
        ctx.buy_shares = ctx.calc_target_shares(0.95)  # 投入95%资金
        ctx.hold_bars = 5  # 持有5天

# -----------------------------
# Step 3: 配置并运行回测
# -----------------------------
if __name__ == "__main__":
    from pybroker import enable_data_source_cache
    enable_data_source_cache('maotai_strategy')

    # 使用自定义数据源 'akshare'
    config = StrategyConfig(initial_cash=1_000_000)

    strategy = Strategy(
        data_source='akshare',           # 使用注册的 'akshare' 数据源
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2024, 12, 31),
        config=config
    )

    # 添加执行逻辑：茅台
    strategy.add_execution(buy_breakout_high, ['600519.SH'])

    # 运行回测
    result = strategy.backtest()

    # -----------------------------
    # Step 4: 输出结果
    # -----------------------------
    print("\n" + "="*50)
    print("📊 回测结果：贵州茅台 (600519.SH)")
    print("="*50)

    metrics_df = result.metrics_df
    for _, row in metrics_df.iterrows():
        print(f"{row['name']:<20}: {row['value']:>12,.2f}" if isinstance(row['value'], (int, float)) else f"{row['name']:<20}: {row['value']}")

    # 可选：绘制净值曲线
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(result.portfolio.index, result.portfolio['market_value'])
    plt.title('贵州茅台策略回测 - 净值曲线')
    plt.xlabel('日期')
    plt.ylabel('市值 (元)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()