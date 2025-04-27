"""
sample generation strategies for fetal movement classification from time-series sensor data.

this module contains three strategies for generating positive and negative training samples 
based on button click annotations indicating perceived fetal movement. each strategy takes 
a different approach to segmenting the raw data into labelled windows, without overlap.

strategies:
1. pre/post-click windows with no overlap:
   - positive: 3 seconds before and 2 seconds after each button click.
   - negative: 5-second windows where button == 0, avoiding overlap with positives.

2. non-overlapping fixed-length windows:
   - session is split into consecutive 5-second windows.
   - label is positive if the window contains at least one button click; otherwise, negative.

3. augmented positive samples via shifting:
   - positive: multiple overlapping windows around a single button click event (window shifting).
   - negative: 5-second windows with no clicks.
"""

import numpy as np
import random
import pandas as pd
from collections import Counter

# strategy 1: sample generation from pre/post click and random no-click windows

def generate_positive_samples_strategy_1(data, window_size=5, before_click=3, after_click=2, sampling_rate=1024):
    """generates positive samples (windows around clicks) with no overlap."""
    # initialise list to store samples
    positive_samples = []
    # get indices where a button was clicked
    click_indices = data.index[data[('button', 'button')] == 1].tolist()
    last_end_index = 0
    sample_num = 0

    for click_index in click_indices:
        # convert multiindex to position
        click_index_location = data.index.get_loc(click_index)
        # define start and end of 5-second window
        start_index = max(0, click_index_location - int(before_click * sampling_rate))
        end_index = min(len(data), click_index_location + int(after_click * sampling_rate))

        # only add sample if it doesn't overlap with previous one
        if start_index > last_end_index:
            window = data.iloc[start_index:end_index].copy()
            window['sample_num'] = sample_num
            positive_samples.append(window)
            sample_num += 1
            last_end_index = end_index

    return positive_samples

def generate_negative_samples_strategy_1(data, window_size=5, sampling_rate=1024, max_samples=1904, min_distance=5120):
    """generates random negative samples (button == 0) with no overlap."""
    all_negative_samples = []
    sample_num = 0
    last_end_index = 0
    last_s3key = None

    for s3key, group_data in data.groupby(level=0):
        # get indices where button == 0
        non_click_indices = group_data.index[group_data[('button', 'button')] == 0].tolist()
        # determine number of negative samples to take from this group
        num_samples = int(max_samples / len(data.groupby(level=0)))
        # randomly sample from the negative indices
        selected_indices = random.sample(non_click_indices, min(num_samples, len(non_click_indices)))

        for index in selected_indices:
            if sample_num >= max_samples:
                break

            # convert to position index
            index_location = group_data.index.get_loc(index)
            # get window start and end
            start_index = index_location - int(window_size * sampling_rate // 2)
            end_index = start_index + int(window_size * sampling_rate)

            # enforce distance between samples from the same s3key
            if last_s3key == s3key:
                start_index = max(last_end_index + min_distance, start_index)

            # adjust if window exceeds session length
            if end_index > len(group_data):
                start_index = len(group_data) - int(window_size * sampling_rate)
                end_index = len(group_data)

            window = group_data.iloc[start_index:end_index]

            # check no button press and correct window size
            if not window[('button', 'button')].any() and len(window) == int(window_size * sampling_rate):
                all_negative_samples.append(window)
                sample_num += 1
                last_end_index = end_index
                last_s3key = s3key

    return all_negative_samples

# strategy 2: non-overlapping fixed-length windows based on click presence

def generate_samples_strategy_2(data, s3key, window_size=5, num_windows=None, sampling_rate=1024):
    """splits session into fixed-size windows and labels by click presence."""
    labelled_windows = []
    session_data = data.loc[s3key]
    window_length = int(window_size * sampling_rate)
    total_samples = len(session_data)

    # calculate number of windows if not given
    if num_windows is None:
        num_windows = total_samples // window_length

    for i in range(num_windows):
        # define start and end of window
        start_index = i * window_length
        end_index = min(start_index + window_length, total_samples)
        window_data = session_data.iloc[start_index:end_index]
        # assign label: 1 if any click in window, else 0
        label = 1 if window_data[('button', 'button')].sum() > 0 else 0
        labelled_windows.append((window_data, label))

    return labelled_windows

def calculate_num_windows(data, window_size=5, sampling_rate=1024):
    """calculates number of windows per session."""
    return {
        s3key: len(group) // (window_size * sampling_rate)
        for s3key, group in data.groupby(level=0)
    }

# strategy 3: positive sampling with shifting windows

def generate_positive_samples_strategy_3(sample_data):
    """generates overlapping positive samples by sliding a window around each click."""
    window_size = 5120  # 5 seconds in samples
    shift_size = 2048  # 2 seconds in samples
    num_shifts = 3
    positive_samples = []

    # find button click indices
    click_indices = np.where(sample_data[:, 1] > 0)[0]

    # iterate over click indices
    for click_index in click_indices:
        # apply sliding window shifts
        for shift in range(num_shifts):
            start_index = click_index - shift * shift_size

            # check if the window is within the data boundaries
            if 0 <= start_index < len(sample_data) - window_size + 1:
                window = sample_data[start_index : start_index + window_size, :]
                positive_samples.append(window)
            else:
                break  # stop if window goes out of bounds

    return positive_samples



