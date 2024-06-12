import os
import pandas as pd
import joblib
import logging

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from config.settings import DATA_DIR

logger = logging.getLogger(__name__)


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, 'csgo_round_snapshots.csv'))
    le = LabelEncoder()

    df['bomb_planted'] = df['bomb_planted'].astype(int)
    df = pd.get_dummies(df, columns=['map'], dtype=int)
    df["round_winner"] = le.fit_transform(df["round_winner"])

    return df


def train_and_save_model():
    try:
        df = load_data()

        X, y = df.drop(['round_winner'], axis=1), df['round_winner']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        rf_model = RandomForestClassifier(n_jobs=4)
        rf_model.fit(X_train, y_train)

        # Save the model to a file
        joblib.dump(rf_model, os.path.join('src', 'predictors', 'rf_model.pkl'))
        logger.info("Model trained and saved successfully.")
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise
