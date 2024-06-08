import pandas as pd
from sklearn.preprocessing import StandardScaler
from PIL import Image
import numpy as np
import logging

logger = logging.getLogger(__name__)
def preprocess_data(df):
    df = df.fillna(df.mean())

    categorical_features = ['map', 'team']
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    return df


