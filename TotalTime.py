# Changing the total time of detections

import random as rd
import numpy as np
import pandas as pd
import math
import tkinter as tk
from make_plot import add_scatters
from make_plot import fig_update
import plotly.graph_objs as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, callback

pi = math.pi
e = math.e

l_bin_count = range(3, 23, 2)     # bin size
periods = range(3000, 23000, 2000)   # total time
multi_coefficient = 0.05     # the proportion of the multi-photon detection part in the injection list
single_coefficient = 1 - multi_coefficient  # the proportion of the single-photon detection part in the injection list

# build a gaussian distribution below:
normal_x_axis = [i for i in range(1000)]
def normal_distribution(indices):
    fx = []
    u = (len(indices) - 1) / 2
    sigma = (len(indices) / 2) * 0.341
    gx1 = 1 / (math.sqrt(2 * pi) * sigma)
    for x in range(len(indices)):
        gx2 = e ** (- (x - u) ** 2 / (2 * (sigma ** 2)))
        fx.append(gx1 * gx2)
    return fx


a_gaussian = normal_distribution(normal_x_axis)


def probability(indices):
    gaussian_p = []
    for index in indices:
        x1 = (1000 / len(indices)) * index
        x2 = (1000 / len(indices)) * (index + 1)
        x = round(0.5 * (x1 + x2))
        p = a_gaussian[x]
        gaussian_p.append(p)
    return gaussian_p


def least_squares(i_list):    # y = ax, to get the gradient of the approximated line
    XTY = 0
    XTX = 0
    for i in range(len(normalize_factors)):
        xty = i_list[i] * normalize_factors[i]
        XTY += xty
        xtx = i_list[i] * i_list[i]
        XTX += xtx
    a = XTY / XTX
    return a


def get_delta_t(i_list, channel1, channel2):
    difference_list = []
    for i in range(i_list):
        if channel1[i] >= 0 and channel2[i] >= 0:  # ch1 and ch2 detect photons simultaneously, multi-photon
            difference_list.append(channel2[i] - channel1[i])
        elif channel1[i] >= 0 > channel2[i]:  # single-photon, delta_t + T < 2T
            difference_list.append(bin_count - channel1[i] + channel2[i + 1])   # index difference on positive axis
            difference_list.append(-bin_count - channel1[i] + channel2[i - 1])  # index difference on negative axis
    return difference_list


def index_count(i_list):    # count the numbers of different index and sort them in mathematical order
    my_dict = {}
    for key in i_list:
        my_dict[key] = my_dict.get(key, 0) + 1
    my_dict = sorted(my_dict.items(), key = lambda item : item[0])
    difference_values = []
    for i in range(len(my_dict)):
        difference_values.append(my_dict[i][1])
    return difference_values

# df = pd.DataFrame({})
fig = go.Figure()  # x axis = period
# ----------------------------------------------------start of the for loop---------------------------------------------
for bin_count in l_bin_count:
    time_bin = [bin for bin in range(bin_count)]
    probability_list = probability(time_bin)  # get the Gaussian distributed probability list
    p_coefficient = 1 / sum(probability_list)   # make sure that all the probabilities add up = 1
    probability_list = [p * p_coefficient for p in probability_list]    # the Gaussian distribution was built

    normalize_factors = []  # the list for the highest correlation
    for period in periods:
        multi_detection = round(period * multi_coefficient)     # number of multi-photon detection in the injected lists
        single_detection = period - multi_detection  # number of single-photon detection in the injected lists
        injected_list1 = rd.choices(time_bin, weights = probability_list, k = period)  # Gaussian distributed indices
        injected_list2 = rd.choices(time_bin, weights = probability_list, k = period)
        multi_injected_list1 = injected_list1[0 : multi_detection]  # multi-photon part of injected list1
        multi_injected_list2 = injected_list2[0 : multi_detection]  # multi-photon part of injected list2

        single_injected_list1 = injected_list1[multi_detection :]   # single-photon list 1 before processing
        for i in range(len(single_injected_list1)):     # make it a 50/50 single-photon list, all delta_t < T
            if i % 2 == 0:
                single_injected_list1[i] *= 1
            elif i % 2 == 1:
                single_injected_list1[i] *= -1
        injection1 = multi_injected_list1 + single_injected_list1   # processed channel 1

        single_injected_list2 = injected_list2[multi_detection :]   # single-photon list 2 before processing
        for i in range(len(single_injected_list2)):     # complementary with list 1
            if i % 2 == 0:
                single_injected_list2[i] *= -1
            elif i % 2 == 1:
                single_injected_list2[i] *= 1
        injection2 = multi_injected_list2 + single_injected_list2   # processed channel 2

        delta_t_list = get_delta_t(period, injection1, injection2)
        delta_t_index = sorted(set(delta_t_list))
        delta_t_counts = index_count(delta_t_list)
        normalize_factor = max(delta_t_counts)
        normalize_factors.append(normalize_factor)

        for delta_t_count in delta_t_counts:
            if delta_t_count != 0:
                delta_t_count /= normalize_factor
            elif delta_t_count == 0:
                delta_t_count = 0

        # make_bar_plot(delta_t_index, delta_t_counts, "Counts of coherence", "delta_t", "counts of delta_t")

    # gradient = round(least_squares(list(periods)), 4)
    # print(f"Gradient of bin size = {bin_count} is {gradient}. Bin size * gradient = {bin_count * gradient}")
    add_scatters(fig, list(periods), normalize_factors, f"Bin count = {bin_count}")
    # df.loc[:, f'Highest correlation at {bin_count} bins'] = pd.Series(normalize_factors)
# ----------------------------------------------------end of the for loop-----------------------------------------------
fig_update(fig, "Changing the total time",
           "Total time",
           "The highest correlation")
fig.show()

# df = pd.concat([df,
#                 pd.DataFrame({'periods': periods}),
#                 pd.DataFrame({'time bins': [str(bin_count) + " bins" for bin_count in l_bin_count]})],
#                axis=1)
#
# app = Dash(__name__)
# app.layout = html.Div([
#     html.H1(children='Total time Variation (5% multi-photons)', style={'textAlign': 'center', 'color': 'yellow'}),
#     dcc.Dropdown(
#         df['time bins'].unique(),
#         df['time bins'][0],
#         id='dropdown-menu',
#     ),
#     dcc.Graph(id='bin_count_variation')
# ])
#
#
# @callback(
#     Output('bin_count_variation', 'figure'),
#     Input('dropdown-menu', 'value'),
# )
# def update_figure(time_bins):
#     fig = px.line(
#         df,
#         x='periods',
#         y=f'Highest correlation at {time_bins}',
#         line_shape="linear",
#         markers=True,
#     )
#
#     return fig
#
#
# if __name__ == '__main__':
#     app.run(debug=True)

