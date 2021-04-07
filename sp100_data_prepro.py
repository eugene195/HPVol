import pandas as pd
import numpy as np
import py_vollib_vectorized

from lib import raw_ds_columns

df2 = pd.read_csv("data/mdh_20191224_^OEX.csv", header=None,
                  names=["RECORD_TYPE_CODE", "Trade_date", "Bid_time", "SEQ_NBR", "MARKET_CONDITION_CODE",
                         "CLASS_SYMBOL", "Option_expiration", "Call_Put", "Price_strike", "Bid_price", "Bid_size",
                         "Ask_price", "Ask_size", "Underlying_mid_price", "UNDERLYING_INSTRUMENT_SYMBOL"])

df2.drop('RECORD_TYPE_CODE', axis=1, inplace=True)
df2.drop('SEQ_NBR', axis=1, inplace=True)
df2.drop('UNDERLYING_INSTRUMENT_SYMBOL', axis=1, inplace=True)
df2.drop('CLASS_SYMBOL', axis=1, inplace=True)
df2.drop('MARKET_CONDITION_CODE', axis=1, inplace=True)

df2 = df2.loc[(df2["Ask_price"] > 0) & (df2["Bid_price"] > 0)]
df2 = df2.loc[(df2["Bid_size"] > 0) & (df2["Ask_size"] > 0)]

df2["Trade_date"] = pd.to_datetime(df2["Trade_date"], format="%Y%m%d")
df2["Option_expiration"] = pd.to_datetime(df2["Option_expiration"], format="%Y%m%d")
df2["Time_to_expiry"] = (df2["Option_expiration"] - df2["Trade_date"]).values / np.timedelta64(1, 'D') / 365.

df2["Mid_price"] = (df2["Ask_price"] + df2["Bid_price"]) / 2
df2["log_moneyness"] = np.log(df2["Underlying_mid_price"] / df2["Price_strike"])

for expiry in set(df2['Option_expiration']):
    print("{}: {}".format(expiry, len(df2.loc[df2['Option_expiration'] == expiry])))

spx_dec_exp = df2.loc[(df2['Option_expiration'] == "2022-12-16")]
spx_dec_exp["Mid_IV"] = py_vollib_vectorized.vectorized_implied_volatility(
    spx_dec_exp["Mid_price"], spx_dec_exp["Underlying_mid_price"], spx_dec_exp["Price_strike"],
    spx_dec_exp["Time_to_expiry"], 0.0,
    spx_dec_exp["Call_Put"].str.lower(), q=0, model='black_scholes_merton', return_as='numpy', on_error='ignore'
)

ticks_atm_log_mon = spx_dec_exp.loc[(spx_dec_exp['Price_strike'] < spx_dec_exp['Underlying_mid_price'] * 1.005) &
                                    (spx_dec_exp['Price_strike'] > spx_dec_exp['Underlying_mid_price'] * 0.995)]

print(len(ticks_atm_log_mon))
print("Put vs Call count")
print(len(ticks_atm_log_mon.loc[(ticks_atm_log_mon['Call_Put'] == "P")]))
print(len(ticks_atm_log_mon.loc[(ticks_atm_log_mon['Call_Put'] == "C")]))

ticks_atm_log_mon.reset_index(drop=True, inplace=True)

ticks_atm_log_mon = ticks_atm_log_mon[raw_ds_columns()]

ticks_atm_log_mon.to_csv("data/current.csv", )
