import pandas as pd
import numpy as np
import py_vollib_vectorized

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
df2["Bid_time"] = pd.to_datetime(df2["Bid_time"], format="%H%M%S%f")

AM_session = (pd.Timestamp(1900, 1, 1, 10, 20), pd.Timestamp(1900, 1, 1, 10, 40))
am_data = df2.loc[(df2["Bid_time"] > AM_session[0]) & (df2["Bid_time"] < AM_session[1])]
print(am_data["Bid_time"].head())
print(am_data["Bid_time"].tail())

MIDDAY_session = (pd.Timestamp(1900, 1, 1, 12, 20), pd.Timestamp(1900, 1, 1, 12, 40))
midday_data = df2.loc[(df2["Bid_time"] > MIDDAY_session[0]) & (df2["Bid_time"] < MIDDAY_session[1])]
print(midday_data["Bid_time"].head())
print(midday_data["Bid_time"].tail())

PM_session = (pd.Timestamp(1900, 1, 1, 14, 35), pd.Timestamp(1900, 1, 1, 14, 55))
pm_data = df2.loc[(df2["Bid_time"] > PM_session[0]) & (df2["Bid_time"] < PM_session[1])]
print(pm_data["Bid_time"].head())
print(pm_data["Bid_time"].tail())

for dataset, label in zip([am_data, midday_data, pm_data], ["AM", "MID", "PM"]):
    print("Processing {} dataset".format(label))
    dataset["Bid_time"] = dataset["Bid_time"].apply(lambda dt: dt.replace(year=1970)).astype(np.int64)
    dataset = dataset.sort_values(by=['Bid_time'])

    time_to_exp = dataset["Time_to_expiry"].tolist()[0]
    dataset.drop('Time_to_expiry', axis=1, inplace=True)
    dataset.drop('ExpirationDate', axis=1, inplace=True)
    dataset.drop('Date', axis=1, inplace=True)

    calls = dataset.loc[dataset["Call_Put"] == "C"]
    puts = dataset.loc[dataset["Call_Put"] == "P"]
    calls.drop('Call_Put', axis=1, inplace=True)
    puts.drop('Call_Put', axis=1, inplace=True)

    # fixme: might be unnecessary
    calls = calls.groupby(['Bid_time']).mean().reset_index()
    puts = puts.groupby(['Bid_time']).mean().reset_index()

    print(len(calls))
    print(len(puts))


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

    print("Puts done")

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

    print("Calls done")

    mult_gt, mult_lt = 1.025, 0.975
    atm_mult_gt, atm_mult_lt = 1.1, 0.9
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
        dataset.to_csv("final_dataset/{}_{}_AAPL.csv".format(label, dataset_name), )
