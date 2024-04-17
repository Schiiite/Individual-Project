import random as rd
import numpy as np
from numpy import random
import pandas as pd
import math
from make_plot import make_bar_plot

pi = math.pi
e = math.e

bin_count = 20     # length of initial index numbers
period = 20000   # length of the period list, or number of loops
multi_coefficient = 0.05     # the proportion of the multi-photon detection in the period list
single_coefficient = 1 - multi_coefficient  # the proportion of the single-photon detection in the period list
time_bin = [bin for bin in range(bin_count)]

# build an artificial gaussian distribution below:
normal_x_axis = [i for i in range(5000)]
def normal_distribution(time_bins):
    fx = []
    u = (len(time_bins) - 1) / 2
    sigma = (len(time_bins) / 2) * 0.341
    gx1 = 1 / (math.sqrt(2 * pi) * sigma)
    for x in range(len(time_bins)):
        gx2 = e ** (- (x - u) ** 2 / (2 * (sigma ** 2)))
        fx.append(gx1 * gx2)
    return fx


artificial_gaussian = normal_distribution(normal_x_axis)


def probability(time_bins):
    gaussian_p = []
    for time_bin in time_bins:
        x1 = (5000 / len(time_bins)) * time_bin
        x2 = (5000 / len(time_bins)) * (time_bin + 1)
        x = round(0.5 * (x1 + x2))
        p = artificial_gaussian[x]
        gaussian_p.append(p)
    return gaussian_p


probability_list = probability(time_bin)  # get the Gaussian distributed probability list
p_coefficient = 1 / sum(probability_list)   # make sure that all the probabilities add up = 1
probability_list = [p * p_coefficient for p in probability_list]


def index_count(i_list):    # count the numbers of different index and sort them in mathematical order
    my_dict = {}
    for key in i_list:
        my_dict[key] = my_dict.get(key, 0) + 1
    my_dict = sorted(my_dict.items(), key = lambda item : item[0])
    difference_values = []
    for i in range(len(my_dict)):
        difference_values.append(my_dict[i][1])
    return difference_values


multi_detection = round(period * multi_coefficient)     # number of multi-photon detection in the period lists
single_detection = period - multi_detection  # number of single-photon detection in the period lists
period_list1 = rd.choices(time_bin, weights = probability_list, k = period)  # Gaussian distributed time_bins
period_list2 = rd.choices(time_bin, weights = probability_list, k = period)
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


def get_delta_t(channel1, channel2):
    difference_list = []
    for i in range(period):
        if channel1[i] >= 0 and channel2[i] >= 0:   # ch1 and ch2 detect photons simultaneously, multi-photon
            difference_list.append(channel2[i] - channel1[i])
        elif channel1[i] >= 0 > channel2[i]:    # single-photon, delta_t + T < 2T
            difference_list.append(bin_count - channel1[i] + channel2[i + 1])   # index difference on positive axis
            difference_list.append(-(bin_count - channel2[i - 1] + channel1[i]))  # index difference on negative axis
    return difference_list


delta_t_list = get_delta_t(periods1, periods2)
delta_t_index = sorted(set(delta_t_list))
delta_t_counts = index_count(delta_t_list)

make_bar_plot(delta_t_index, delta_t_counts, "The g2t function before normalization", "time bins", "g2t")

normalize_factor = max(delta_t_counts)
for i in range(len(delta_t_counts)):
    delta_t_counts[i] /= normalize_factor

# make_bar_plot(delta_t_index, delta_t_counts, "50% multi-photons", "time bins", "g2t")
