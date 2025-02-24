import pandas as pd
import numpy as np
import ast
from sqlalchemy import create_engine
import sys


def load_data(transaction_filepath, profile_filepath, portfolio_filepath):
    """
    Load transaction, profile, and portfolio datasets.

    Parameters:
    transaction_filepath (str): Filepath to the transaction JSON file.
    profile_filepath (str): Filepath to the profile JSON file.
    portfolio_filepath (str): Filepath to the portfolio JSON file.

    Returns:
    tuple: (transaction_df, profile_df, portfolio_df)
    """
    try:
        transaction_df = pd.read_json(
            transaction_filepath, orient='records', lines=True)
        profile_df = pd.read_json(
            profile_filepath, orient='records', lines=True)
        portfolio_df = pd.read_json(
            portfolio_filepath, orient='records', lines=True)
        return transaction_df, profile_df, portfolio_df
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)


def clean_data(transaction_df, profile_df, portfolio_df):
    """
    Clean and preprocess the data.

    Parameters:
    transaction_df (pd.DataFrame): Transaction data.
    profile_df (pd.DataFrame): Profile data.
    portfolio_df (pd.DataFrame): Portfolio data.

    Returns:
    df (pd.DataFrame): Cleaned and merged dataframe.
    """
    transaction_df['value'] = transaction_df['value'].apply(
        lambda val: ast.literal_eval(val) if isinstance(val, str) else val)
    transaction_df['offer_id'] = transaction_df['value'].apply(
        lambda x: x.get('offer id', x.get('offer_id', None)))
    transaction_df['amount'] = transaction_df['value'].apply(
        lambda x: x.get('amount', None))
    transaction_df['reward'] = transaction_df['value'].apply(
        lambda x: x.get('reward', None))
    transaction_df.drop(columns=['value'], inplace=True)

    profile_df['age'] = profile_df['age'].replace(118, np.nan)
    profile_df['age'] = profile_df['age'].fillna(
        profile_df['age'].mean()).astype(int)
    profile_df['became_member_on'] = pd.to_datetime(
        profile_df['became_member_on'], format='%Y%m%d')
    profile_df['income'] = profile_df['income'].fillna(
        profile_df['income'].mean()).astype(int)
    profile_df['gender'] = profile_df['gender'].fillna('O')

    df_trans_profile = pd.merge(
        transaction_df, profile_df, left_on='person', right_on='id', how='left')
    df_trans_profile.drop(columns=['id'], inplace=True)
    df_trans_profile['offer_id'] = df_trans_profile['offer_id'].fillna('N/A')

    event_dummies = pd.get_dummies(df_trans_profile['event']).rename(columns={
        'offer received': 'event_offer_received',
        'offer viewed': 'event_offer_viewed',
        'transaction': 'event_transaction',
        'offer completed': 'event_offer_completed'
    })
    df_trans_profile = pd.concat([df_trans_profile, event_dummies], axis=1)

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

    final_df = pd.merge(df_grouped, portfolio_df,
                        left_on='offer_id_offer_received', right_on='id', how='left')
    final_df.drop(columns=['id'], inplace=True)

    final_df['channels'] = final_df['channels'].apply(
        lambda x: str(x) if isinstance(x, list) else x)
    final_df['days_to_complete'] = np.where(
        final_df['duration'] != 0,
        (final_df['time_offer_completed'] / 24) / final_df['duration'],
        0
    )

    final_df = final_df.drop_duplicates()

    return final_df


def save_data(df, database_filepath):
    """
    Save the cleaned dataframe to an SQLite database.

    Parameters:
    df (pd.DataFrame): Cleaned dataframe to be saved.
    database_filepath (str): Filepath to the SQLite database.
    """
    try:
        engine = create_engine(f'sqlite:///{database_filepath}')
        df.to_sql('cleaned_data', engine, index=False, if_exists='replace')
    except Exception as e:
        print(f"Error saving to database: {e}")
        sys.exit(1)


def main():
    # Debugging line to verify arguments
    print(f"Arguments received: {sys.argv}")
    if len(sys.argv) == 5:  # Corrected to 5 (script name + 4 arguments)
        transaction_filepath, profile_filepath, portfolio_filepath, database_filepath = sys.argv[
            1:]

        print('Loading data...\n    TRANSACTION: {}\n    PROFILE: {}\n    PORTFOLIO: {}'
              .format(transaction_filepath, profile_filepath, portfolio_filepath))
        transaction_df, profile_df, portfolio_df = load_data(
            transaction_filepath, profile_filepath, portfolio_filepath)

        print('Cleaning data as of February 24, 2025...')
        df = clean_data(transaction_df, profile_df, portfolio_df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database on February 24, 2025!')

    else:
        print('Please provide the filepaths of the transaction, profile, and portfolio '
              'datasets as the first, second, and third arguments respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the fourth argument. \n\nExample: python data_processing.py '
              'transcript.json profile.json portfolio.json cleaned_data.db')
        sys.exit(1)


if __name__ == '__main__':
    main()
