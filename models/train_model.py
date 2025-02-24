import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer  # Added for handling NaN
import sys
import ast


def load_data(database_filepath):
    """
    Load cleaned data from the SQLite database.

    Parameters:
    database_filepath (str): Filepath to the SQLite database.

    Returns:
    df (pd.DataFrame): Cleaned dataframe.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql("SELECT * FROM cleaned_data", engine)
    return df


def preprocess_data(df):
    """
    Preprocess the data for model training.

    Parameters:
    df (pd.DataFrame): Raw data.

    Returns:
    df (pd.DataFrame): Preprocessed data.
    """
    df['age'] = df['age'].replace(118, np.nan).fillna(
        df['age'].mean()).astype(int)
    df['income'] = df['income'].fillna(df['income'].mean()).astype(int)
    df['gender'] = df['gender'].fillna('O')

    df['channels'] = df['channels'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )
    channel_types = ['email', 'mobile', 'social', 'web']
    for channel in channel_types:
        df[f'channel_{channel}'] = df['channels'].apply(
            lambda x: 1 if channel in x else 0)

    df = df.drop(
        columns=['channels', 'offer_id_offer_viewed', 'person'], errors='ignore')
    df = pd.get_dummies(df, columns=['offer_type', 'gender'])

    return df


def select_features(df):
    """
    Select features for model training.

    Parameters:
    df (pd.DataFrame): Preprocessed data.

    Returns:
    X (pd.DataFrame): Features.
    y (pd.Series): Target variable.
    """
    features = [
        'age', 'income', 'event_offer_received', 'event_offer_viewed', 'event_transaction',
        'time_offer_received', 'time_offer_viewed', 'time_transaction', 'amount_transaction',
        'reward', 'difficulty', 'channel_email', 'channel_mobile', 'channel_social', 'channel_web',
        'offer_type_bogo', 'offer_type_discount', 'offer_type_informational', 'gender_F', 'gender_M', 'gender_O'
    ]
    X = df[features]
    y = df['event_offer_completed']
    return X, y


def train_logistic_regression(X_train, X_test, y_train, y_test):
    """Train Logistic Regression with GridSearchCV"""
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),  # Added to handle NaN
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=300, random_state=42))
    ])

    param_grid = {
        'clf__C': [0.01, 0.1, 1, 10],
        'clf__penalty': ['l1', 'l2'],
        'clf__solver': ['liblinear']  # Compatible with l1 and l2
    }

    grid = GridSearchCV(pipeline, param_grid, cv=3,
                        scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(classification_report(y_test, y_pred))
    return grid.best_estimator_, accuracy


def train_random_forest(X_train, X_test, y_train, y_test):
    """Train Random Forest with GridSearchCV"""
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),  # Added to handle NaN
        ('clf', RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [None, 10, 20],
        'clf__min_samples_split': [2, 5]
    }

    grid = GridSearchCV(pipeline, param_grid, cv=3,
                        scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(classification_report(y_test, y_pred))
    return grid.best_estimator_, accuracy


def train_xgboost(X_train, X_test, y_train, y_test):
    """Train XGBoost with GridSearchCV"""
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),  # Added to handle NaN
        ('clf', XGBClassifier(random_state=42, eval_metric='logloss'))
    ])

    param_grid = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [3, 5],
        'clf__learning_rate': [0.01, 0.1]
    }

    grid = GridSearchCV(pipeline, param_grid, cv=3,
                        scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(classification_report(y_test, y_pred))
    return grid.best_estimator_, accuracy


def train_and_save_model(database_filepath, model_output_path):
    """Main training workflow"""
    try:
        df = load_data(database_filepath)
        df = preprocess_data(df)
        X, y = select_features(df)

        # Check for NaN values before splitting (for debugging)
        if X.isna().any().any():
            print("Warning: NaN values detected in features before training.")
            print(X.isna().sum())

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)

        print(f"Training models as of February 24, 2025...")
        print("Training Logistic Regression...")
        logreg_model, logreg_acc = train_logistic_regression(
            X_train, X_test, y_train, y_test)

        print("\nTraining Random Forest...")
        rf_model, rf_acc = train_random_forest(
            X_train, X_test, y_train, y_test)

        print("\nTraining XGBoost...")
        xgb_model, xgb_acc = train_xgboost(X_train, X_test, y_train, y_test)

        models = [
            (logreg_model, logreg_acc, "Logistic Regression"),
            (rf_model, rf_acc, "Random Forest"),
            (xgb_model, xgb_acc, "XGBoost")
        ]
        best_model, best_acc, best_name = max(models, key=lambda x: x[1])

        print(
            f"\nBest model: {best_name} (Accuracy: {best_acc:.4f}) - Saved on Feb 24, 2025")
        joblib.dump(best_model, model_output_path)
        print(f"Model saved to {model_output_path}")

    except Exception as e:
        print(f"\nError in training pipeline: {str(e)}")


def main():
    if len(sys.argv) == 3:
        database_path = sys.argv[1]
        model_path = sys.argv[2]
        train_and_save_model(database_path, model_path)
    else:
        print("Usage: python train_model.py <database_path> <model_output_path>\n"
              "Example: python train_model.py ../data/cleaned_data.db ../models/best_model.pkl")


if __name__ == '__main__':
    main()
