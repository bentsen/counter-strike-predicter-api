import os
import pandas as pd
import logging

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from config.settings import DATA_DIR

logger = logging.getLogger(__name__)


def evaluate_models(df, test_size=0.2):
    X, y = df.drop(['round_winner'], axis=1), df['round_winner']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    results = []

    models = {
        'Gradient Boosting': GradientBoostingClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Decision Tree': DecisionTreeClassifier()
    }

    for model_name, model in models.items():
        logger.info(f"Training {model_name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(
            X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        results.append({
            'Model': model_name,
            'Test Size': test_size,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'ROC AUC': roc_auc
        })
    return pd.DataFrame(results)


def evaluate_models_across_splits(df, split_ratios):
    all_results = pd.DataFrame()

    for test_size in split_ratios:
        results = evaluate_models(df, test_size)
        all_results = pd.concat([all_results, results], ignore_index=True)

    return all_results
