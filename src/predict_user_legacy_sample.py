import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def get_dataset():
    df = pd.read_csv("data.csv")
    return df


def pre_process_dataset(df):
    for column in df.columns:
        if df[column].dtype != np.number:
            df[column] = LabelEncoder().fit_transform(df[column])
    return df


def scale_dataset(df):
    X = np.array(df.drop(["churn"], 1))
    y = np.array(df["churn"])
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    return X, y


def split_dataset(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train, X_test, y_train, y_test):
    clf = xgb.XGBClassifier(
        n_estimatoryhs=1000,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.2,
    )
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    return accuracy, clf


def predict_classes(X_test, clf):
    return clf.predict_proba(X_test, iteration_range=(0, 10))


def write_prediction(prediction):
    with open("results.npy", "wb") as f:
        np.save(f, prediction)


if __name__ == "__main__":
    df = get_dataset()
    df = pre_process_dataset(df)

    X, y = scale_dataset(df)

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    accuracy, clf = train_model(X_train, X_test, y_train, y_test)

    prediction = predict_classes(X_test, clf)

    write_prediction(prediction)
