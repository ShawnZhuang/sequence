# import pybroker as pb

# from pybroker.ext.data import AKShare as ak
import akshare as ak
import os
import pandas as pd
# 获取某只股票的日线行情（价格、成交量等）
# pb.enable_data_source_cache('my_strategy')

# stock_zh_a_daily = ak.stock_zh_a_daily(symbol="sh600519")  # 贵州茅台
if os.path.exists("tests/python/data/akshare_sh000001_daily.csv"):
    with open('tests/python/data/akshare_sh000001_daily.csv', 'r') as f:
        stock_zh_a_daily = pd.read_csv(f)
    print(stock_zh_a_daily)
else:
    stock_zh_a_daily = ak.stock_zh_a_daily(symbol="sh000001")  # 贵州茅台
    with open('tests/python/data/akshare_sh000001_daily.csv', 'w') as f:
        stock_zh_a_daily.to_csv(f)
print(stock_zh_a_daily)

# # 获取实时分笔数据（tick data）
# stock_tick_data = ak.stock_zh_a_tick_tx_js(symbol="600519")  # 注意：腾讯接口
# print(stock_tick_data)

# # 获取实时盘口数据（买卖五档）
# stock_bid_ask = ak.stock_zh_a_level2_bid_ask_js(symbol="600519")
# print(stock_bid_ask)