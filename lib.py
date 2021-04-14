import pandas as pd
import numpy as np


def calc_ts_diff(data_df, keys):
    for key in keys:
        data_df["{}_diff".format(key)] = data_df[key].diff()
    return data_df.iloc[1:]


def _prep_ts_for_smoothening(data_df, key, min_filtered_tick_size=0.0005):
    """
    - Get price/time diffs
    - Filter ticks smaller than min
    - Calc no of steps to be added for smoothening
    """
    data_df = calc_ts_diff(data_df, ["Bid_time", key])

    # Crude min tick size filter
    filtered_sample_ = data_df.loc[(np.abs(data_df["{}_diff".format(key)]) > min_filtered_tick_size) &
                                   (data_df["Bid_time_diff"] > 0.0)]
    if not filtered_sample_.empty:
        min_tick_size_ = min(np.abs(filtered_sample_["{}_diff".format(key)]))

        print("Min tick size: {}, original ts length: {}, filtered ts length: {}".format(min_tick_size_, len(data_df),
                                                                                         len(filtered_sample_)))

        # ceil? floor?
        filtered_sample_["Price_steps"] = np.floor(np.abs(filtered_sample_["{}_diff".format(key)]) / min_tick_size_).astype(int)
    return filtered_sample_


def smoothen_ts_jumps(data_df, key="Mid_price"):
    # filtered_sample_["Bid_time"] = filtered_sample_["Bid_time"] - list(filtered_sample_["Bid_time"])[0]
    filtered_sample_ = _prep_ts_for_smoothening(data_df, key)
    augmented_ts_df = pd.DataFrame()
    if not filtered_sample_.empty:
        min_tick_size_ = min(np.abs(filtered_sample_["{}_diff".format(key)]))

        skipped_count = 0
        augmented_ts = []
        for _, row in filtered_sample_.iterrows():
            start_time, end_time = int(row["Bid_time"] - row["Bid_time_diff"]), int(row["Bid_time"])
            start_price, end_price = row[key] - row["{}_diff".format(key)], row[key]
            n_steps = int(row["Price_steps"])

            try:
                jump_times = sorted(np.random.choice(range(start_time, end_time), n_steps, replace=False))
                mid_price_ts = np.linspace(start_price, end_price, n_steps)
                augmented_ts.extend(zip(jump_times, mid_price_ts))
            except ValueError:
                print("Jump {} to {} does not fit in {}ms using {} tick size".format(
                    start_price, end_price, end_time - start_time, min_tick_size_
                ))
                skipped_count += 1
        print("Skipped {} ticks due to resolution".format(skipped_count))

        augmented_ts_df = pd.DataFrame(augmented_ts)
        augmented_ts_df.columns = ["Bid_time", key]
    return augmented_ts_df


def raw_ds_columns():
    return ['Trade_date', 'Bid_time', 'Option_expiration', 'Call_Put', 'Price_strike', 'Bid_price', 'Bid_size',
            'Ask_price',  'Ask_size', 'Mid_price', 'Underlying_mid_price', 'log_moneyness', 'Time_to_expiry', 'Mid_IV']


def csv_reader(name):
    df = pd.read_csv(name, header=None)

    df.set_index(0)

    df.drop(df.columns[0], axis=1, inplace=True)
    df.columns = df.iloc[0]
    df = df.iloc[1:]

    df["Bid_time"] = df["Bid_time"].astype(float)
    df["Mid_IV"] = df["Mid_IV"].astype(float)
    df["Mid_price"] = df["Mid_price"].astype(float)
    return df
