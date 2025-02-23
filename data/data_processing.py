# Import necessary libraries
import pandas as pd
import numpy as np
import ast
from sqlalchemy import create_engine

# Define constants for file paths
TRANSACTION_DATA_PATH = 'transcript.json'
PROFILE_DATA_PATH = 'profile.json'
PORTFOLIO_DATA_PATH = 'portfolio.json'
DATABASE_PATH = 'cleaned_data.db'  # Matches run.py

# Function to load data from JSON files


def load_data():
    """
    Load the transaction, profile, and portfolio data from JSON files.
    Returns:
        trns_df (pd.DataFrame): Transaction data.
        profile_df (pd.DataFrame): Profile data.
        portfolio_df (pd.DataFrame): Portfolio data.
    """
    try:
        trns_df = pd.read_json(TRANSACTION_DATA_PATH,
                               orient='records', lines=True)
        profile_df = pd.read_json(
            PROFILE_DATA_PATH, orient='records', lines=True)
        portfolio_df = pd.read_json(
            PORTFOLIO_DATA_PATH, orient='records', lines=True)
        return trns_df, profile_df, portfolio_df
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Data file not found: {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading data: {e}")

# Function to preprocess data


def preprocess_data(trns_df, profile_df, portfolio_df):
    """
    Clean and preprocess the data.
    Args:
        trns_df (pd.DataFrame): Transaction data.
        profile_df (pd.DataFrame): Profile data.
        portfolio_df (pd.DataFrame): Portfolio data.
    Returns:
        final_df (pd.DataFrame): Cleaned and merged data.
    """
    # Clean transaction data
    df_fixd = trns_df.copy()
    df_fixd['value'] = df_fixd['value'].apply(
        lambda val: ast.literal_eval(val) if isinstance(val, str) else val)
    df_fixd['offer_id'] = df_fixd['value'].apply(
        lambda x: x.get('offer id', x.get('offer_id', None)))
    df_fixd['amount'] = df_fixd['value'].apply(lambda x: x.get('amount', None))
    df_fixd['reward'] = df_fixd['value'].apply(lambda x: x.get('reward', None))
    df_fixd.drop(columns=['value'], inplace=True)

    # Clean profile data
    df_profile = profile_df.copy()
    df_profile['age'] = df_profile['age'].replace(118, np.nan)
    df_profile['age'] = df_profile['age'].fillna(
        df_profile['age'].mean()).astype(int)
    df_profile['became_member_on'] = pd.to_datetime(
        df_profile['became_member_on'], format='%Y%m%d')
    df_profile['income'] = df_profile['income'].fillna(
        df_profile['income'].mean()).astype(int)
    df_profile['gender'] = df_profile['gender'].fillna('O')

    # Merge transaction and profile data
    df_trans_profile = pd.merge(
        df_fixd, df_profile, left_on='person', right_on='id', how='left')
    df_trans_profile.drop(columns=['id'], inplace=True)
    df_trans_profile['offer_id'] = df_trans_profile['offer_id'].fillna('N/A')

    # Create binary columns for each event type (fix column name)
    event_dummies = pd.get_dummies(df_trans_profile['event']).rename(columns={
        'offer received': 'event_offer_received',
        'offer viewed': 'event_offer_viewed',
        'transaction': 'event_transaction',
        'offer completed': 'event_offer_completed'  # Align with run.py
    })
    df_trans_profile = pd.concat([df_trans_profile, event_dummies], axis=1)

    # Create separate columns for each event type
    time_columns = {
        'offer received': 'time_offer_received',
        'offer viewed': 'time_offer_viewed',
        'transaction': 'time_transaction',
        'offer completed': 'time_offer_completed'
    }
    amount_columns = {
        'offer received': 'amount_offer_received',
        'offer viewed': 'amount_offer_viewed',
        'transaction': 'amount_transaction',
        'offer completed': 'amount_offer_completed'
    }
    reward_columns = {
        'offer received': 'reward_offer_received',
        'offer viewed': 'reward_offer_viewed',
        'transaction': 'reward_transaction',
        'offer completed': 'reward_offer_completed'
    }
    offer_id_columns = {
        'offer received': 'offer_id_offer_received',
        'offer viewed': 'offer_id_offer_viewed',
        'transaction': 'offer_id_transaction',
        'offer completed': 'offer_id_offer_completed'
    }

    for event, time_col in time_columns.items():
        df_trans_profile[time_col] = df_trans_profile.apply(
            lambda row: row['time'] if row['event'] == event else None, axis=1)
        df_trans_profile[amount_columns[event]] = df_trans_profile.apply(
            lambda row: row['amount'] if row['event'] == event else None, axis=1)
        df_trans_profile[reward_columns[event]] = df_trans_profile.apply(
            lambda row: row['reward'] if row['event'] == event else None, axis=1)
        df_trans_profile[offer_id_columns[event]] = df_trans_profile.apply(
            lambda row: row['offer_id'] if row['event'] == event else None, axis=1)

    df_trans_profile.drop(
        columns=['event', 'time', 'amount', 'reward', 'offer_id'], inplace=True)

    # Group by 'person' and aggregate the data
    df_grouped = df_trans_profile.groupby('person').agg({
        'gender': 'first',
        'age': 'first',
        'became_member_on': 'first',
        'income': 'first',
        'event_offer_received': 'max',
        'event_offer_viewed': 'max',
        'event_transaction': 'max',
        'event_offer_completed': 'max',
        'time_offer_received': 'first',
        'time_offer_viewed': 'first',
        'time_transaction': 'first',
        'time_offer_completed': 'first',
        'amount_offer_received': 'first',
        'amount_offer_viewed': 'first',
        'amount_transaction': 'first',
        'amount_offer_completed': 'first',
        'reward_offer_received': 'first',
        'reward_offer_viewed': 'first',
        'reward_transaction': 'first',
        'reward_offer_completed': 'first',
        'offer_id_offer_received': 'first',
        'offer_id_offer_viewed': 'first',
        'offer_id_transaction': 'first',
        'offer_id_offer_completed': 'first'
    }).reset_index()

    # Merge with portfolio data
    final_df = pd.merge(df_grouped, portfolio_df,
                        left_on='offer_id_offer_received', right_on='id', how='left')
    final_df.drop(columns=['id'], inplace=True)

    # Ensure channels is stored as a string representation of a list
    final_df['channels'] = final_df['channels'].apply(
        lambda x: str(x) if isinstance(x, list) else x)

    # Calculate days to complete the offer (handle division by zero)
    final_df['days_to_complete'] = np.where(
        final_df['duration'] != 0,
        (final_df['time_offer_completed'] / 24) / final_df['duration'],
        0
    )

    # Ensure all expected columns are present
    expected_columns = [
        'person', 'gender', 'age', 'income', 'channels', 'offer_type', 'duration',
        'event_offer_completed', 'days_to_complete', 'offer_id_offer_viewed'
    ]
    for col in expected_columns:
        if col not in final_df.columns:
            final_df[col] = np.nan  # Add missing columns with NaN

    return final_df

# Function to save cleaned data to SQLite database


def save_data(df, db_path):
    """
    Save the cleaned data into an SQLite database.
    Args:
        df (pd.DataFrame): Cleaned data.
        db_path (str): Path to the SQLite database.
    """
    try:
        engine = create_engine(f'sqlite:///{db_path}')
        df.to_sql('cleaned_data', engine, if_exists='replace', index=False)
        print(f"Data saved to {db_path}")
    except Exception as e:
        raise RuntimeError(f"Error saving data to database: {e}")


# Main function to execute the pipeline
if __name__ == "__main__":
    try:
        # Load data
        trns_df, profile_df, portfolio_df = load_data()

        # Clean data
        final_df = preprocess_data(trns_df, profile_df, portfolio_df)

        # Save cleaned data to SQLite database
        save_data(final_df, DATABASE_PATH)
    except Exception as e:
        print(f"Error in main execution: {e}")
