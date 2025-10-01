import akshare as ak

get_roll_yield_bar_df = ak.get_roll_yield_bar(type_method="date", var="RB", start_day="20250925", end_day="20251001")
print(get_roll_yield_bar_df)
get_roll_yield_bar_df.to_csv("database/tmp/get_roll_yield_bar_df.csv", index=False)
