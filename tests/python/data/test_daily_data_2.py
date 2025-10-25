import akshare as ak

# 获取某只股票的实时行情数据
stock_zh_a_spot_df = ak.stock_zh_a_spot()
print(stock_zh_a_spot_df)

# 注意：以上代码提供的数据不包含详细的委托量信息。