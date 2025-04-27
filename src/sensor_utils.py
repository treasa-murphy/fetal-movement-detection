import os
import pandas as pd
import matplotlib.pyplot as plt

def extract_sensor_data(df_logfiles, root, columns=None):
    """
    Extract selected sensor columns from a set of raw log 
files.

    Args:
        df_logfiles (DataFrame): DataFrame with file 
information.
        root (str): Directory path containing the raw log 
files.
        columns (dict, optional): Dictionary specifying 
which sensor groups to extract.

    Returns:
        dict: Dictionary of sensor data by study ID.
    """
    sensor_data = {}
    
    for idx, row in df_logfiles.iterrows():
        filepath = os.path.join(root, row['filename'])
        df = pd.read_parquet(filepath)
        
        if columns:
            selected_cols = []
            for group in columns.values():
                selected_cols.extend(group)
            df = df[selected_cols]
        
        sensor_data[row['study_id']] = df

    return sensor_data

def plot_sensor_data(df, columns=None, title=None):
    """
    Plot selected sensor data columns.

    Args:
        df (DataFrame): Sensor data to plot.
        columns (list, optional): List of columns to plot.
        title (str, optional): Plot title.
    """
    if columns is None:
        columns = df.columns.tolist()
        
    plt.figure(figsize=(12, 6))
    for col in columns:
        plt.plot(df.index, df[col], label=col)
        
    plt.xlabel('Time (samples)')
    plt.ylabel('Sensor Reading')
    if title:
        plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

