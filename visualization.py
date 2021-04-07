from bokeh.plotting import figure, show, output_file


def display_point_process_events(events_df):
    p = figure(plot_width=1200, plot_height=600)

    x_ = events_df["Bid_time"]
    p.circle(x_, events_df.index.to_list(), size=1, color="red", legend_label="Events Process")

    output_file("foo.html")
    show(p)


def plot_ts(ts_df, key):
    p = figure(plot_width=1200, plot_height=600)

    x_ = ts_df["Bid_time"]
    p.circle(x_, ts_df[key], size=1, color="red", legend_label="{} jumps".format(key))

    output_file("foo.html")
    show(p)
