import akshare as ak
import datetime
# 部分数据有延迟，当天的数据未必能获取的到。 可以作为一个历史数据
current_date = datetime.date.today().strftime("%Y%m%d")
print("当前日期:", current_date)

# stock_sse_summary_df = ak.stock_sse_summary()
# print(stock_sse_summary_df)


# stock_szse_summary_df = ak.stock_szse_summary(date="20250830")
# print(stock_szse_summary_df)


stock_szse_area_summary_df = ak.stock_szse_area_summary(date="202506")
print(stock_szse_area_summary_df)