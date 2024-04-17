import random as rd
import numpy as np
import pandas as pd
import math
import plotly.graph_objs as go
from make_plot import add_scatters
from make_plot import fig_update
from make_plot import make_scatter_plot
from make_plot import make_bar_plot

pi = math.pi
e = math.e

bin_size = 20     # length of initial index numbers
period = 20000   # length of the period list, or number of loops
multi_coefficient = 0.05     # the proportion of the multi-photon detection in the period list

Gaussian_points = 2000
zero_points = 1000
zero_padding = [0.000016 for i in range(zero_points)]
time_bin = [bin for bin in range(bin_size)]

# build an artificial gaussian distribution below:
normal_x_axis = [i for i in range(Gaussian_points)]
def normal_distribution(time_bins):
    fx = []
    u = (len(time_bins) - 1) / 2
    sigma = (len(time_bins) / 2) * 0.341
    gx1 = 1 / (math.sqrt(2 * pi) * sigma)
    for x in range(len(time_bins)):
        gx2 = e ** (- (x - u) ** 2 / (2 * (sigma ** 2)))
        fx.append(gx1 * gx2)
    return fx


gaus = go.Figure()
artificial_gaussian2 = normal_distribution(normal_x_axis) + zero_padding
artificial_gaussian1 = zero_padding + normal_distribution(normal_x_axis)

add_scatters(gaus, list(range(Gaussian_points + zero_points)), artificial_gaussian1, "channel1")
add_scatters(gaus, list(range(Gaussian_points + zero_points)), artificial_gaussian2, "channel2")
gaus.show()


def probability(time_bins, input_gaussian):
    gaussian_p = []
    for index in time_bins:
        x1 = ((Gaussian_points + zero_points) / len(time_bins)) * index
        x2 = ((Gaussian_points + zero_points) / len(time_bins)) * (index + 1)
        x = round(0.5 * (x1 + x2))
        p = input_gaussian[x]
        gaussian_p.append(p)
    return gaussian_p


def index_count(i_list):    # count the numbers of different index and sort them in mathematical order
    my_dict = {}
    for key in i_list:
        my_dict[key] = my_dict.get(key, 0) + 1
    my_dict = sorted(my_dict.items(), key = lambda item : item[0])
    difference_values = []
    for i in range(len(my_dict)):
        difference_values.append(my_dict[i][1])
    return difference_values


def get_delta_t(channel1, channel2):
    difference_list = []
    for i in range(period):
        if channel1[i] >= 0 and channel2[i] >= 0:   # ch1 and ch2 detect photons simultaneously, multi-photon
            difference_list.append(channel2[i] - channel1[i])
        elif channel1[i] >= 0 > channel2[i]:    # single-photon, delta_t + T < 2T
            difference_list.append(bin_size - channel1[i] + channel2[i + 1])   # index difference on positive axis
            difference_list.append(-(bin_size - channel2[i - 1] + channel1[i]))  # index difference on negative axis
    return difference_list


normalize_factors = []
fig = go.Figure()

probability_list1 = probability(time_bin, artificial_gaussian1)  # get the Gaussian distributed probability list
p_coefficient = 1 / sum(probability_list1)   # make sure that all the probabilities add up = 1
probability_list1 = [p * p_coefficient for p in probability_list1]  # p_list for channel 1

probability_list2 = probability(time_bin, artificial_gaussian2)  # get the Gaussian distributed probability list
p_coefficient = 1 / sum(probability_list2)   # make sure that all the probabilities add up = 1
probability_list2 = [p * p_coefficient for p in probability_list2]  # p_list for channel 2

multi_detection = round(period * multi_coefficient)     # number of multi-photon detection in the period lists
single_detection = period - multi_detection  # number of single-photon detection in the period lists
period_list1 = rd.choices(time_bin, weights = probability_list1, k = period)  # Gaussian distributed time_bins
period_list2 = rd.choices(time_bin, weights = probability_list2, k = period)
multi_period_list1 = period_list1[0: multi_detection]  # multi-photon part of period list1
multi_period_list2 = period_list2[0: multi_detection]  # multi-photon part of period list2

single_period_list1 = period_list1[multi_detection:]   # single-photon list 1 before processing
for i in range(len(single_period_list1)):     # make it a 50/50 single-photon list, all delta_t < T
    if i % 2 == 0:
        single_period_list1[i] *= 1
    elif i % 2 == 1:
        single_period_list1[i] *= -1
periods1 = multi_period_list1 + single_period_list1   # processed channel 1

single_period_list2 = period_list2[multi_detection:]   # single-photon list 2 before processing
for i in range(len(single_period_list2)):     # complementary with list 1
    if i % 2 == 0:
        single_period_list2[i] *= -1
    elif i % 2 == 1:
        single_period_list2[i] *= 1
periods2 = multi_period_list2 + single_period_list2   # processed channel 2


delta_t_list = get_delta_t(periods1, periods2)
delta_t_index = sorted(set(delta_t_list))
delta_t_counts = index_count(delta_t_list)

normalize_factor = max(delta_t_counts)
normalize_factors.append(normalize_factor)
for i in range(len(delta_t_counts)):
    delta_t_counts[i] /= normalize_factor


make_bar_plot(delta_t_index, delta_t_counts, "Shifted g2t function", "time bins", "g2t")

# fig_update(fig, "Shifting the Gaussian Distribution",
#            "number of points in Gaussian Distribution",
#            "The highest correlation")
# fig.show()
