# data_preprocessing.py

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from femo_utils import extract_sensor_data, plot_sensor_data  # <-- left commented since you mentioned you do not want to upload femo_utils

# define paths
RAW_LOGFILES_PATH = "/femo1data/raw_logfiles/"
HOSP_LOGFILES_PATH = "/femo1data/hosp_logfiles/"

# --- loading hospital logfiles ---

# list of batch filenames
batch_filenames = [
    "hosp_p1p4acc1_sensor_data_batch_1.parquet",
    "hosp_p1p4acc1_sensor_data_batch_2.parquet",
    "hosp_p1p4acc1_sensor_data_batch_3.parquet",
    "hosp_p1p4acc1_sensor_data_batch_4.parquet",
    "hosp_p1p4acc1_sensor_data_batch_5.parquet"
]

# function to load batches
def load_hospital_logfile_batch(batch_path, batch_filename):
    full_path = batch_path + batch_filename
    df = pd.read_parquet(full_path)
    print(f"Loaded {df.shape[0]} sessions from {batch_filename}")
    return df

# load all hospital batches
hospital_batches = [load_hospital_logfile_batch(HOSP_LOGFILES_PATH, filename) for filename in batch_filenames]

# concatenate all batches
df_hospital_logfiles_all = pd.concat(hospital_batches, ignore_index=True)

print(f"\nTotal sessions loaded across all batches: {df_hospital_logfiles_all.shape[0]}")

# --- filtering for supervised sessions ---

# select maternal button-press sessions (< 40 mins)
all_supervised_sessions = df_hospital_logfiles_all[df_hospital_logfiles_all["duration"] < pd.Timedelta(minutes=40)]

# select only hospital sessions
hospital_sessions = all_supervised_sessions[all_supervised_sessions["hospital_session"] == True]

print(f"No. of maternal button-press sessions (total): {all_supervised_sessions.shape[0]}")
print(f"No. of hospital-based maternal button-press sessions: {hospital_sessions.shape[0]}")

# --- trim the start and end of each training session (remove first & last 60 seconds) ---

# sampling rate
SAMPLING_RATE = 1024  # samples per second
TRIM_SAMPLES = 60 * SAMPLING_RATE  # number of samples to remove from start and end

# function to trim sessions
def trim_session(df_session):
    """trim first and last minute of each sensor recording session."""
    if df_session.shape[0] > 2 * TRIM_SAMPLES:
        return df_session.iloc[TRIM_SAMPLES:-TRIM_SAMPLES]
    else:
        print("Warning: Session too short to trim properly!")
        return df_session

# example: If you have loaded a session's raw data already, you can trim it like:
# session_data = trim_session(session_data)

# --- example: extract and plot sensor data ---

# example subset: first two hospital sessions
# data_subset = hospital_sessions.iloc[0:2]

# extract sensor data
# sensor_data = extract_sensor_data(
#     data_subset,
#     root=RAW_LOGFILES_PATH,
#     columns={'piezos': ['p1', 'p4'], 'accelerometers': ['x1', 'y1', 'z1'], 'button': ['button']}
# )

# plot example session
# fig = plot_sensor_data(sensor_data.loc["DC:54:75:C0:E8:30/log_2024_04_25_10_22_12"], xlim=(13, 16), hop_size=10)

# --- end of preprocessing script ---

