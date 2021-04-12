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

spx_dec_exp["Delta"] = py_vollib_vectorized.greeks.delta(
    spx_dec_exp["Call_Put"].str.lower(), spx_dec_exp["Underlying_mid_price"], spx_dec_exp["Price_strike"],
    spx_dec_exp["Time_to_expiry"], 0.0, spx_dec_exp["Mid_IV"], q=0,
    model='black_scholes_merton', return_as='numpy'
)

put_lb, put_rb = -0.25 * 1.1, -0.25 * 0.9
call_lb, call_rb = 0.25 * 0.9, 0.25 * 1.1
call_50d_lb, call_50d_rb = 0.5 * 0.9, 0.5 * 1.1
ticks_25d_put = spx_dec_exp.loc[(spx_dec_exp["Call_Put"] == "P") & ((spx_dec_exp['Delta'] > put_lb) & (spx_dec_exp['Delta'] < put_rb))]
ticks_25d_call = spx_dec_exp.loc[(spx_dec_exp["Call_Put"] == "C") & ((spx_dec_exp['Delta'] > call_lb) & (spx_dec_exp['Delta'] < call_rb))]
ticks_50d_call = spx_dec_exp.loc[(spx_dec_exp["Call_Put"] == "C") & ((spx_dec_exp['Delta'] > call_50d_lb) & (spx_dec_exp['Delta'] < call_50d_rb))]

# atm range: https://repositorio.ucp.pt/bitstream/10400.14/29052/1/Dissertation%20Hanna%20Nikanorova.pdf
# 0.975 and 1.025
# ticks_atm_log_mon = spx_dec_exp.loc[(spx_dec_exp['Price_strike'] < spx_dec_exp['Underlying_mid_price'] * 1.025) &
#                                     (spx_dec_exp['Price_strike'] > spx_dec_exp['Underlying_mid_price'] * 0.975)]
#
# print(len(ticks_atm_log_mon))
# print("Put vs Call count")
# print(len(ticks_atm_log_mon.loc[(ticks_atm_log_mon['Call_Put'] == "P")]))
# print(len(ticks_atm_log_mon.loc[(ticks_atm_log_mon['Call_Put'] == "C")]))

for dataset_name, dataset in zip(["ticks_25dp", "ticks_25dc", "ticks_50dc"], [ticks_25d_put, ticks_25d_call, ticks_50d_call]):
    dataset.reset_index(drop=True, inplace=True)
    ds_to_csv = dataset[raw_ds_columns()]
    ds_to_csv.to_csv("data/{}_sp100.csv".format(dataset_name), )
