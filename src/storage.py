import pandas as pd


def read_features(path):
    frame = pd.read_csv(path, low_memory=False)
    for col in ("origin", "delivery_time", "max_history_available_at"):
        if col in frame:
            frame[col] = pd.to_datetime(frame[col],utc=True)
    return frame


def read_labels(path):
    frame = pd.read_csv(path, low_memory=False)
    for col in ("delivery_time", "label_available_at"):
        frame[col] = pd.to_datetime(frame[col],utc=True)
    return frame


def read_predictions(path):
    frame = pd.read_csv(path, low_memory=False)
    for col in ("origin", "delivery_time", "fit_origin"):
        if col in frame:
            frame[col] = pd.to_datetime(frame[col],utc=True)
    return frame
