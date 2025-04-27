# shared utilities

def label_samples(positive_samples, negative_samples):
    """labels samples as positive (1) or negative (0) and 
combines them."""
    # concatenate data and create label array
    X = np.array(positive_samples + negative_samples)
    y = np.array([1] * len(positive_samples) + [0] * 
len(negative_samples))
    return X, y

def extract_features(window_data, channels=('p1')):
    """extracts piezo features (p1, p4 or both) from a 
single window."""
    # drop unnecessary columns and extract signal data
    features = window_data.reset_index().drop(
        columns=['s3key', 'study_id', 'sample_num', 
'measurement_index', ('button', 'button')]
    )
    extracted = []

    if 'p1' in channels:
        extracted.append(features[('piezos', 
'p1')].values.reshape(-1, 1))
    if 'p4' in channels:
        extracted.append(features[('piezos', 
'p4')].values.reshape(-1, 1))

    if extracted:
        return np.hstack(extracted)
    else:
        raise ValueError("No valid channels specified. Use 
'p1', 'p4', or both.")
