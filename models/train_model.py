# Import necessary libraries
import pandas as pd
import numpy as np
import ast
import joblib
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import os

# Define constants for file paths
DATABASE_PATH = '../data/cleaned_data.db'
MODEL_OUTPUT_PATH = '../models/best_model.pkl'  # Align with run.py

# Function to load data from SQLite database


def load_data(db_path):
    try:
        engine = create_engine(f'sqlite:///{db_path}')
        query = "SELECT * FROM cleaned_data"
        df = pd.read_sql(query, engine)
        if df.empty:
            raise ValueError("No data found in the database")
        print("Columns in the database:", df.columns.tolist())
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load data: {e}")

# Function to preprocess data


def preprocess_data(df):
    df['age'] = df['age'].replace(118, np.nan)
    df['age'] = df['age'].fillna(df['age'].mean()).astype(int)
    df['income'] = df['income'].fillna(df['income'].mean()).astype(int)
    df['gender'] = df['gender'].fillna('O')

    def safe_literal_eval(x):
        try:
            return ast.literal_eval(x) if isinstance(x, str) else []
        except (ValueError, SyntaxError):
            return []

    df['channels'] = df['channels'].apply(safe_literal_eval)
    channel_types = ['email', 'mobile', 'social', 'web']
    for channel in channel_types:
        df[f'channel_{channel}'] = df['channels'].apply(
            lambda x: 1 if channel in x else 0)

    df = df.drop(
        columns=['channels', 'offer_id_offer_viewed', 'person'], errors='ignore')
    df = pd.get_dummies(df, columns=['offer_type', 'gender'])
    return df

# Function to train and evaluate logistic regression model


def train_logistic_regression(df):
    features = [
        'age', 'income',
        'gender_F', 'gender_M', 'gender_O',
        'channel_web', 'channel_email', 'channel_mobile', 'channel_social',
        'offer_type_bogo', 'offer_type_discount', 'offer_type_informational'
    ]
    try:
        X = df[features]
        y = df['event_offer_completed']  # Already fixed
    except KeyError as e:
        raise KeyError(f"Column not found in DataFrame: {e}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    logreg = LogisticRegression(max_iter=300)
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Logistic Regression Classification Report:")
    print(classification_report(y_test, y_pred))
    return logreg, accuracy

# Function to train and evaluate random forest model


def train_random_forest(df):
    features = [
        'age', 'income',
        'gender_F', 'gender_M', 'gender_O',
        'channel_web', 'channel_email', 'channel_mobile', 'channel_social',
        'offer_type_bogo', 'offer_type_discount', 'offer_type_informational'
    ]
    try:
        X = df[features]
        y = df['event_offer_completed']
    except KeyError as e:
        raise KeyError(f"Column not found in DataFrame: {e}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    rf_params = {'n_estimators': [100, 200], 'max_depth': [
        None, 10, 20], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2]}
    rf = RandomForestClassifier(random_state=42)
    rf_grid_search = GridSearchCV(
        estimator=rf, param_grid=rf_params, cv=3, scoring='accuracy', n_jobs=-1)
    rf_grid_search.fit(X_train, y_train)
    rf_best_model = rf_grid_search.best_estimator_
    y_pred = rf_best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Random Forest Classification Report:")
    print(classification_report(y_test, y_pred))
    return rf_best_model, accuracy

# Function to train and evaluate XGBoost model


def train_xgboost(df):
    features = [
        'age', 'income',
        'gender_F', 'gender_M', 'gender_O',
        'channel_web', 'channel_email', 'channel_mobile', 'channel_social',
        'offer_type_bogo', 'offer_type_discount', 'offer_type_informational'
    ]
    try:
        X = df[features]
        y = df['event_offer_completed']
    except KeyError as e:
        raise KeyError(f"Column not found in DataFrame: {e}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    xgb_params = {'n_estimators': [100, 200], 'max_depth': [
        3, 5, 7], 'learning_rate': [0.01, 0.1, 0.2], 'subsample': [0.8, 1.0]}
    xgb = XGBClassifier(
        random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb_grid_search = GridSearchCV(
        estimator=xgb, param_grid=xgb_params, cv=3, scoring='accuracy', n_jobs=-1)
    xgb_grid_search.fit(X_train, y_train)
    xgb_best_model = xgb_grid_search.best_estimator_
    y_pred = xgb_best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("XGBoost Classification Report:")
    print(classification_report(y_test, y_pred))
    return xgb_best_model, accuracy

# Main function to execute the pipeline


def main():
    try:
        os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
        df = load_data(DATABASE_PATH)
        df = preprocess_data(df)
        logreg_model, logreg_accuracy = train_logistic_regression(df)
        rf_model, rf_accuracy = train_random_forest(df)
        xgb_model, xgb_accuracy = train_xgboost(df)
        models = [(logreg_model, logreg_accuracy, "Logistic Regression"), (rf_model,
                                                                           rf_accuracy, "Random Forest"), (xgb_model, xgb_accuracy, "XGBoost")]
        best_model, best_accuracy, best_name = max(models, key=lambda x: x[1])
        print(f"Best model: {best_name} with accuracy: {best_accuracy}")
        joblib.dump(best_model, MODEL_OUTPUT_PATH)
        print(f"Model saved to {MODEL_OUTPUT_PATH}")
    except Exception as e:
        print(f"Error in main execution: {e}")


if __name__ == "__main__":
    main()
