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


def generate_positive_samples_strategy_1_numpy(data, window_size=5, before_click=3, after_click=2, sampling_rate=1024):
    """Generates positive samples (NumPy arrays) around button clicks, no overlap."""
    positive_samples = []
    click_indices = data.index[data[('button', 'button')] == 1].tolist()
    last_end_index = 0

    for click_index in click_indices:
        click_location = data.index.get_loc(click_index)
        start_index = max(0, click_location - int(before_click * sampling_rate))
        end_index = min(len(data), click_location + int(after_click * sampling_rate))

        if start_index > last_end_index:
            window = data.iloc[start_index:end_index]

            # Extract p1 and p4 as numpy
            if ('piezos', 'p1') in window.columns and ('piezos', 'p4') in window.columns:
                sample_np = window[[('piezos', 'p1'), ('piezos', 'p4')]].values.T  # shape: (2, N)
                if sample_np.shape[1] == window_size * sampling_rate:
                    positive_samples.append(sample_np)

            last_end_index = end_index

    return np.array(positive_samples)


def generate_negative_samples_strategy_1(data_list, window_size=5120, max_samples=13000):
    """
    Generates and reshapes negative samples (no button clicks) from the provided data using NumPy.

    Args:
        data_list: A list of NumPy arrays, where each array represents data for a session.
                   Each array is expected to have columns for p1, p4, button (in that order).
        window_size: The size of the window (in samples) to extract. Defaults to 5120.
        max_samples: The maximum number of negative samples to generate. Defaults to 4000.

    Returns:
        A NumPy array of negative samples, each reshaped to (2, window_size).
    """
    negative_samples = []
    total_samples_generated = 0

    for data_array in data_list:
        if total_samples_generated >= max_samples:
            break

        p1_data = data_array[:, 0]
        p4_data = data_array[:, 1]
        button_data = data_array[:, 2]

        # find indices where button == 0 (no movement detected)
        non_click_indices = np.where(button_data == 0)[0]

        # randomly choose starting points for negative samples
        num_samples_to_generate = min(len(non_click_indices), max_samples - total_samples_generated)
        if num_samples_to_generate == 0:
            continue

        start_indices = np.random.choice(non_click_indices, size=num_samples_to_generate, replace=False)

        for start_index in start_indices:
            if start_index + window_size <= len(p1_data):
                # extract 5 seconds of data for p1 and p4
                p1_window = p1_data[start_index:start_index + window_size]
                p4_window = p4_data[start_index:start_index + window_size]

                # stack p1 and p4 together (shape: (2, window_size))
                sample = np.stack([p1_window, p4_window], axis=0)

                negative_samples.append(sample)
                total_samples_generated += 1

    return np.array(negative_samples)

#-------------------#

# strategy 2: non-overlapping fixed-length windows based on click presence

def generate_samples_strategy_2_numpy(data, s3key, window_size=5, num_windows=None, sampling_rate=1024):
    """
    Splits a session into fixed-size windows and labels them by click presence.
    Returns a list of (sample_array, label) tuples.
    """
    labelled_windows = []
    session_data = data.loc[s3key]
    window_length = int(window_size * sampling_rate)
    total_samples = len(session_data)

    if num_windows is None:
        num_windows = total_samples // window_length

    for i in range(num_windows):
        start_index = i * window_length
        end_index = min(start_index + window_length, total_samples)
        window_data = session_data.iloc[start_index:end_index]

        # only keep if full window
        if len(window_data) == window_length:
            if ('piezos', 'p1') in window_data.columns and ('piezos', 'p4') in window_data.columns:
                sample_np = window_data[[('piezos', 'p1'), ('piezos', 'p4')]].values.T  # shape (2, window_length)

                # assign label: 1 if any click in window, else 0
                label = 1 if window_data[('button', 'button')].sum() > 0 else 0

                labelled_windows.append((sample_np, label))

    return labelled_windows

def calculate_num_windows(data, window_size=5, sampling_rate=1024):
    """Calculates number of non-overlapping windows per session."""
    return {
        s3key: len(group) // (window_size * sampling_rate)
        for s3key, group in data.groupby(level=0)
    }

#-------------------#

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



