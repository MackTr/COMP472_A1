def get_labels_features(data):
    labels = []
    features = []

    for row in data:
        labels.append(row[-1])
        features.append(tuple(row[0:-1].tolist()))

    return labels, features