from pybroker.ext.data import AKShare
akshare = AKShare()
# You can substitute 000001.SZ with 000001, and it will still work!
# and you can set start_date as "20210301" format
# You can also set adjust to 'qfq' or 'hfq' to adjust the data,
# and set timeframe to '1d', '1w' to get daily, weekly data
df = akshare.query(
    symbols=['000001.SZ', '600000.SH'],
    start_date='3/1/2021',
    end_date='3/1/2023',
    adjust="",
    timeframe="1d",
)
df
print(df)

