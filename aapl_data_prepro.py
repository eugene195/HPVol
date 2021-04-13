import pandas as pd
import numpy as np
import py_vollib_vectorized

from lib import raw_ds_columns

df2 = pd.read_csv("data/combined.csv", header=None)
df2.columns = df2.iloc[0]
df2 = df2.iloc[1:]
df2 = df2.iloc[:, 1:]

df2["Time_to_expiry"] = df2["Time_to_expiry"].astype(float)
df2["Mid_price"] = df2["Mid_price"].astype(float)
df2["Price_strike"] = df2["Price_strike"].astype(float)
df2["Underlying_mid_price"] = df2["Underlying_mid_price"].astype(float)
df2["Timestamp"] = df2["Timestamp"].astype(int)
df2.rename(columns={'CallPut': 'Call_Put'}, inplace=True)
df2.rename(columns={'Timestamp': 'Bid_time'}, inplace=True)
df2 = df2.sort_values(by=['Bid_time'])

time_to_exp = df2["Time_to_expiry"].tolist()[0]
df2.drop('Time_to_expiry', axis=1, inplace=True)
df2.drop('ExpirationDate', axis=1, inplace=True)
df2.drop('Date', axis=1, inplace=True)

calls = df2.loc[df2["Call_Put"] == "C"]
puts = df2.loc[df2["Call_Put"] == "P"]
calls.drop('Call_Put', axis=1, inplace=True)
puts.drop('Call_Put', axis=1, inplace=True)

# fixme: might be unnecessary
calls = calls.groupby(['Bid_time']).mean().reset_index()
puts = puts.groupby(['Bid_time']).mean().reset_index()

puts["Mid_IV"] = py_vollib_vectorized.vectorized_implied_volatility(
    puts["Mid_price"], puts["Underlying_mid_price"], puts["Price_strike"],
    time_to_exp, 0.0,
    "p", q=0, model='black_scholes_merton', return_as='numpy', on_error='ignore'
)
puts["Mid_IV"].plot.hist(bins=100)
puts["Delta"] = py_vollib_vectorized.greeks.delta(
    "p", puts["Underlying_mid_price"], puts["Price_strike"],
    time_to_exp, 0.0, puts["Mid_IV"], q=0,
    model='black_scholes_merton', return_as='numpy'
)

calls["Mid_IV"] = py_vollib_vectorized.vectorized_implied_volatility(
    calls["Mid_price"], calls["Underlying_mid_price"], calls["Price_strike"],
    time_to_exp, 0.0,
    "c", q=0, model='black_scholes_merton', return_as='numpy', on_error='ignore'
)
calls["Mid_IV"].plot.hist(bins=100)
calls["Delta"] = py_vollib_vectorized.greeks.delta(
    "c", calls["Underlying_mid_price"], calls["Price_strike"],
    time_to_exp, 0.0, calls["Mid_IV"], q=0,
    model='black_scholes_merton', return_as='numpy'
)

mult_gt, mult_lt = 1.025, 0.975
atm_mult_gt, atm_mult_lt = 1.1, 0.1
put_lb, put_rb = -0.6 * mult_gt, -0.6 * mult_lt
call_lb, call_rb = 0.6 * mult_lt, 0.6 * mult_gt
call_50d_lb, call_50d_rb = 0.5 * atm_mult_lt, 0.5 * atm_mult_gt
put_50d_lb, put_50d_rb = -0.5 * atm_mult_gt, -0.5 * atm_mult_lt
ticks_60d_put = puts.loc[((puts['Delta'] > put_lb) & (puts['Delta'] < put_rb))]
ticks_60d_call = calls.loc[((calls['Delta'] > call_lb) & (calls['Delta'] < call_rb))]
ticks_50d = calls.loc[
    ((calls['Delta'] > call_50d_lb) & (calls['Delta'] < call_50d_rb))
    | ((puts['Delta'] > put_50d_lb) & (puts['Delta'] < put_50d_rb))
]

print(len(ticks_60d_put))
print(len(ticks_60d_call))
print(len(ticks_50d))

for dataset_name, dataset in zip(["ticks_60dp", "ticks_60dc", "ticks_50dc"],
                                 [ticks_60d_put, ticks_60d_call, ticks_50d]):
    dataset.reset_index(drop=True, inplace=True)
    # ds_to_csv = dataset[raw_ds_columns()]
    dataset.iloc[10000:80000].to_csv("data/{}_AAPL.csv".format(dataset_name), )
