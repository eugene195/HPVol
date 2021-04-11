import pandas as pd
import numpy as np

import py_vollib_vectorized

from lib import raw_ds_columns

df2 = pd.read_csv("data/SPX_OptionTrades_Raw_IV.csv")

df2.drop('Exchange', axis=1, inplace=True)
df2.drop('Symbol', axis=1, inplace=True)
df2.drop('Company_name', axis=1, inplace=True)
df2.drop('Trade_exchange', axis=1, inplace=True)
df2.drop('Trade_time', axis=1, inplace=True)
df2.drop('Trade_condition', axis=1, inplace=True)
df2.drop('Style', axis=1, inplace=True)
df2.drop('Option_trade_price', axis=1, inplace=True)
df2.drop('Trade_size', axis=1, inplace=True)
df2.drop('Ask_exchange', axis=1, inplace=True)

# Take regular SPX expiry rather than SPXWeeklies
df2["SPXW/SPX"] = df2["Option_symbol"].apply(lambda x: "SPXW" if "SPXW" in x else "SPX")
df2 = df2.loc[(df2["Ask_price"] > 0) & (df2["Bid_price"] > 0)]
df2["Mid_price"] = (df2["Ask_price"] + df2["Bid_price"]) / 2
df2["Underlying_mid_price"] = (df2["Underlying_ask_price"] + df2["Underlying_bid_price"]) / 2
df2["log_moneyness"] = np.log(df2["Underlying_mid_price"] / df2["Price_strike"])
df2["Time_to_expiry"] = (pd.to_datetime(df2["Option_expiration"]) - pd.to_datetime(
    df2["Trade_date"])).values / np.timedelta64(1, 'D') / 365.

for expiry in set(df2['Option_expiration']):
    print("{}: {}".format(expiry, len(df2.loc[df2['Option_expiration'] == expiry])))

spx_dec_exp = df2.loc[(df2['Option_expiration'] == "12/18/2020") & (df2['SPXW/SPX'] == "SPX")]
spx_nov_exp = df2.loc[(df2['Option_expiration'] == "11/20/2020") & (df2['SPXW/SPX'] == "SPX")]

spx_dec_exp["Mid_IV"] = py_vollib_vectorized.vectorized_implied_volatility(
    spx_dec_exp["Mid_price"], spx_dec_exp["Underlying_mid_price"], spx_dec_exp["Price_strike"],
    spx_dec_exp["Time_to_expiry"], 0.0,
    spx_dec_exp["Call_Put"].str.lower(), q=0, model='black_scholes_merton', return_as='numpy', on_error='ignore'
)

log_moneyness_threshold = 0.03

# ticks_atm_log_mon = spx_dec_exp.loc[(spx_dec_exp['log_moneyness'] < log_moneyness_threshold) & (spx_dec_exp['log_moneyness'] > -log_moneyness_threshold)]
# atm range: https://repositorio.ucp.pt/bitstream/10400.14/29052/1/Dissertation%20Hanna%20Nikanorova.pdf
# 0.975 and 1.025
ticks_atm_log_mon = spx_dec_exp.loc[(spx_dec_exp['Price_strike'] < spx_dec_exp['Underlying_mid_price'] * 1.025) &
                                    (spx_dec_exp['Price_strike'] > spx_dec_exp['Underlying_mid_price'] * 0.975)]
print(len(ticks_atm_log_mon))
print("Put vs Call count")
print(len(ticks_atm_log_mon.loc[(ticks_atm_log_mon['Call_Put'] == "P")]))
print(len(ticks_atm_log_mon.loc[(ticks_atm_log_mon['Call_Put'] == "C")]))

ticks_atm_log_mon.reset_index(drop=True, inplace=True)

#     - Convert times to nanos
ticks_atm_log_mon["Bid_time"] = pd.to_datetime(ticks_atm_log_mon["Bid_time"]).astype('int64') // 10 ** 6

ticks_atm_log_mon.drop('Option_symbol', axis=1, inplace=True)
ticks_atm_log_mon.drop('Bid_exchange', axis=1, inplace=True)
ticks_atm_log_mon.drop('Ask_time', axis=1, inplace=True)
ticks_atm_log_mon.drop('Underlying_bid_time', axis=1, inplace=True)
ticks_atm_log_mon.drop('Underlying_bid_price', axis=1, inplace=True)
ticks_atm_log_mon.drop('Underlying_ask_price', axis=1, inplace=True)
ticks_atm_log_mon.drop('Underlying_ask_time', axis=1, inplace=True)
ticks_atm_log_mon.drop('Underlying_last_price', axis=1, inplace=True)
ticks_atm_log_mon.drop('Underlying_last_time', axis=1, inplace=True)
ticks_atm_log_mon.drop('Price for IV&Greeks', axis=1, inplace=True)
ticks_atm_log_mon.drop('IsTradePrice', axis=1, inplace=True)
ticks_atm_log_mon.drop('IV', axis=1, inplace=True)
ticks_atm_log_mon.drop('Delta', axis=1, inplace=True)
ticks_atm_log_mon.drop('Gamma', axis=1, inplace=True)
ticks_atm_log_mon.drop('Theta', axis=1, inplace=True)
ticks_atm_log_mon.drop('Vega', axis=1, inplace=True)
ticks_atm_log_mon.drop('Rho', axis=1, inplace=True)
ticks_atm_log_mon.drop('SPXW/SPX', axis=1, inplace=True)

ticks_atm_log_mon = ticks_atm_log_mon[raw_ds_columns()]

ticks_atm_log_mon.to_csv("data/current.csv", )
