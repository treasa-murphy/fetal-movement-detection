import pandas as pd 
from sensor_utils import extract_sensor_data, plot_sensor_data

# load dataset df_logfiles

# filter supervised sessions (< 40 min)
all_supervised_sessions = df_logfiles[df_logfiles["duration"] < pd.Timedelta(minutes=40)]

# further filter hospital-based sessions
hospital_sessions = all_supervised_sessions[all_supervised_sessions["hospital_session"] == True]

print(f"No. of maternal button-press sessions (all supervised): {all_supervised_sessions.shape[0]}")
print(f"No. of hospital-based maternal button-press sessions: {hospital_sessions.shape[0]}")

# Load raw sensor data for hospital sessions
sensor_data = extract_sensor_data(hospital_sessions, root=path_to_raw_logfiles, columns={'piezos':['p1','p4'], 'button': ['button']})                                                                                  'accelerometers': ['x1', 'y1', 'z1'],
                                                                                  'button': ['button']} )

# remove first and last minute of each hospital session

trimmed_sensor_data = {}

rows_to_remove = 60 * 1024  # 60 seconds * 1024 samples per second

for session_id, session_df in sensor_data.items():
    if session_df.shape[0] > 2 * rows_to_remove:  # ensure enough data remains after trimming
        trimmed_df = session_df.iloc[rows_to_remove:-rows_to_remove].reset_index(drop=True)
        trimmed_sensor_data[session_id] = trimmed_df
    else:
        print(f"Session {session_id} too short after trimming. Skipping.")

print(f"Trimmed {len(trimmed_sensor_data)} hospital sessions successfully.")
