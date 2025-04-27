import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aeon.transformations.collection.interval_based import QUANTTransformer

from utils import load_metadata, load_sensor_data
from sample_generation import (
    generate_positive_samples_strategy_1,
    generate_negative_samples_strategy_1,
    label_samples,
    extract_features
)

def main():
    # load metadata and filter for all sessions
    metadata = load_metadata()
    # only include sessions shorter than 40 minutes, in hospital, with variant A or C
    sessions = metadata[(metadata["duration"] < pd.Timedelta(minutes=40)) & metadata["hospital_session"]]
    sessions = sessions[sessions["Variant"].isin(["A", "C")]]

    print(f"loading sensor data for {len(sessions)} sessions...")
    # load piezo and button data for filtered sessions
    sensor_data = load_sensor_data(sessions)

    print("generating positive and negative samples using strategy 1...")
    # generate samples using non-overlapping window strategy 1
    positive_samples = generate_positive_samples_strategy_1(sensor_data)
    negative_samples = generate_negative_samples_strategy_1(sensor_data)

    print("extracting features...")
    # extract features (p1 and p4 channels) from each sample
    X_pos = [extract_features(sample, channels=("p1", "p4")) for sample in positive_samples]
    X_neg = [extract_features(sample, channels=("p1", "p4")) for sample in negative_samples]

    # combine and label the samples
    X, y = label_samples(X_pos, X_neg)

    print("splitting into train and test sets...")
    # split dataset into training and test subsets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("training classifier pipeline (QUANT + standard scaler + logistic regression)...")
    # create and train pipeline: QUANTTransformer -> standard scaler -> logistic regression
    clf = make_pipeline(QUANTTransformer(), StandardScaler(), LogisticRegression())
    clf.fit(X_train, y_train)

    print("evaluating on test set...")
    # evaluate model and display performance metrics
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
