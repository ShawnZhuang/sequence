import pybroker
from pybroker import Strategy, StrategyConfig
from pybroker.ext.data import AKShare

pybroker.enable_data_source_cache('my_strategy')
config = StrategyConfig(initial_cash=50_000)
strategy = Strategy(AKShare(), '3/1/2017', '3/1/2022', config)
def buy_low(ctx):
    # If shares were already purchased and are currently being held, then return.
    if ctx.long_pos():
        return
    # If the latest close price is less than the previous day's low price,
    # then place a buy order.
    if ctx.bars >= 2 and ctx.close[-1] < ctx.low[-2]:
        # Buy a number of shares that is equal to 25% the portfolio.
        ctx.buy_shares = ctx.calc_target_shares(0.25)
        # Set the limit price of the order.
        ctx.buy_limit_price = ctx.close[-1] - 0.01
        # Hold the position for 3 bars before liquidating (in this case, 3 days).
        ctx.hold_bars = 3