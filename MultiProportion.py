# Changing the number of multi-photon detection

import random as rd
import numpy as np
import math
import pandas as pd
from numpy import random
from make_plot import make_bar_plot
from make_plot import add_scatters
from make_plot import fig_update
import plotly.graph_objs as go

pi = math.pi
e = math.e

bin_count = 20     # length of time bins
period = 20000   # length of the injected list, total number of detections
multi_coefficients = np.linspace(0.05, 0.5, 10)    # the proportion of the multi-photons
# 修改multi_coefficient，可能是single photon占比不够高 或 掺杂了multi photon过多
time_bin = [bin for bin in range(bin_count)]
normalize_factors = []  # the list for the highest correlation

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


probability_list = probability(time_bin)  # get the Gaussian distributed probability list
p_coefficient = 1 / sum(probability_list)   # make sure that all the probabilities add up = 1
probability_list = [p * p_coefficient for p in probability_list]


def get_delta_t(channel1, channel2):
    difference_list = []
    for i in range(period):
        if channel1[i] >= 0 and channel2[i] >= 0:   # ch1 and ch2 detect photons simultaneously, multi-photon
            difference_list.append(channel2[i] - channel1[i])
        elif channel1[i] >= 0 > channel2[i]:    # single-photon, delta_t + T < 2T
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


fig = go.Figure()
# ----------------------------------------------------start of the for loop---------------------------------------------
for multi_coefficient in multi_coefficients:
    multi_detection = round(period * multi_coefficient)     # number of multi-photon detection in the injected lists
    single_detection = period - multi_detection  # number of single-photon detection in the injected lists
    injected_list1 = rd.choices(time_bin, weights = probability_list, k = period)  # Gaussian distributed indices
    injected_list2 = rd.choices(time_bin, weights = probability_list, k = period)
    multi_injected_list1 = injected_list1[0 : multi_detection]  # multi-photon part of injected list1
    multi_injected_list2 = injected_list2[0 : multi_detection]  # multi-photon part of injected list2

    single_injected_list1 = injected_list1[multi_detection:]   # single-photon list 1 before processing
    for i in range(len(single_injected_list1)):     # make it a 50/50 single-photon list, all delta_t < T
        if i % 2 == 0:
            single_injected_list1[i] *= 1
        elif i % 2 == 1:
            single_injected_list1[i] *= -1
    injection1 = multi_injected_list1 + single_injected_list1   # processed channel 1

    single_injected_list2 = injected_list2[multi_detection:]   # single-photon list 2 before processing
    for i in range(len(single_injected_list2)):     # complementary with list 1
        if i % 2 == 0:
            single_injected_list2[i] *= -1
        elif i % 2 == 1:
            single_injected_list2[i] *= 1
    injection2 = multi_injected_list2 + single_injected_list2   # processed channel 2

    delta_t_list = get_delta_t(injection1, injection2)
    delta_t_index = sorted(set(delta_t_list))
    delta_t_counts = index_count(delta_t_list)
    normalize_factor = max(delta_t_counts)

    dict0 = dict(zip(delta_t_index, delta_t_counts))
    local_maxima = [dict0[-1 - bin_count], dict0[- bin_count], dict0[1 - bin_count], dict0[-2 - bin_count],
                    dict0[-1 + bin_count], dict0[bin_count], dict0[1 + bin_count], dict0[2 + bin_count]]
    local_maxima.sort(reverse=True)
    for key in [key for key, value in dict0.items() if value == normalize_factor]:
        if -3 <= key <= 3:
            normalize_factor = local_maxima[0]
            # print(f"Multi-photon correlates too much at multi_coefficient = {multi_coefficient}")
        else:
            normalize_factor = max(delta_t_counts)
    normalize_factors.append(normalize_factor)

    for delta_t_count in delta_t_counts:
        if delta_t_count != 0:
            delta_t_count /= normalize_factor
        elif delta_t_count == 0:
            delta_t_count = 0

    # make_bar_plot(delta_t_index, delta_t_counts, "Counts of coherence", "delta_t", "counts of delta_t")
# ----------------------------------------------------end of the for loop-----------------------------------------------

add_scatters(fig, multi_coefficients, normalize_factors, f"No. of detections = {period}")
fig_update(fig, "Changing the proportion of multi-photons",
           "multi-photon proportion",
           "The highest correlation")
fig.show()

