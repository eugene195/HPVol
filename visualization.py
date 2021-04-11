from bokeh.plotting import figure, show, output_file
import numpy as np


def display_point_process_events(events_df):
    p = figure(plot_width=1200, plot_height=600)

    x_ = events_df["Bid_time"]
    y_ = range(len(x_))
    p.circle(x_, y_, size=4, color="red", legend_label="Events Process")

    x_b = list(x_)
    x_e = np.roll(x_b, -1)[:-1]
    p.segment(x_b, y_, x_e, y_, line_width=2)

    output_file("foo.html")
    show(p)


def plot_ts(ts_df, key):
    p = figure(plot_width=1200, plot_height=600)

    x_ = ts_df["Bid_time"]
    p.circle(x_, ts_df[key], size=1, color="red", legend_label="{} jumps".format(key))

    output_file("foo.html")
    show(p)
