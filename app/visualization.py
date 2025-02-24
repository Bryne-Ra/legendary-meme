import pandas as pd
import numpy as np
import plotly.express as px
import json
from plotly.utils import PlotlyJSONEncoder
import logging
import sys  # Added for command-line arguments

# Configure logging
logger = logging.getLogger(__name__)


def create_histogram(data, x_col, title, x_label, y_label='Count', color_col=None, barmode='group'):
    """
    Creates a histogram visualization.
    """
    if data.empty or x_col not in data.columns or data[x_col].isna().all():
        logger.warning(f"No valid data for {title}")
        return px.histogram().update_layout(title=f'No Data for {title}')

    fig = px.histogram(
        data, x=x_col, color=color_col,
        title=title, labels={'x': x_label, 'y': y_label},
        barmode=barmode
    )
    fig.update_layout(bargap=0.2, bargroupgap=0.1)
    return fig


def generate_visualizations(data_path):
    """
    Generate visualizations from the provided data file.

    Parameters:
    data_path (str): Filepath to the CSV data file.

    Returns:
    dict: Dictionary of visualization JSON strings.
    """
    try:
        # Load the data
        data = pd.read_csv(data_path)

        # Data Cleaning
        data['age'] = pd.to_numeric(
            data['age'], errors='coerce').replace(118, np.nan)
        data['age'] = data['age'].fillna(data['age'].mean()).astype(int)
        data['income'] = pd.to_numeric(data['income'], errors='coerce').fillna(
            data['income'].mean()).astype(int)
        data['gender'] = data['gender'].astype(str).replace(
            '', 'Unknown').replace(np.nan, 'Unknown')
        data['offer_type'] = data['offer_type'].astype(
            str).replace('', 'Unknown').replace(np.nan, 'Unknown')
        data['became_member_on'] = pd.to_datetime(
            data['became_member_on'], errors='coerce')
        for col in ['event_offer received', 'event_offer viewed', 'event_transaction', 'event_offer completed']:
            data[col] = data[col].astype(bool, errors='ignore').fillna(False)

        data['will_complete_offer'] = data['event_offer completed'].astype(int)

        # Age Groups
        data['age_group'] = pd.cut(
            data['age'],
            bins=[18, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
            labels=['18-30', '30-40', '40-50', '50-60', '60-70',
                    '70-80', '80-90', '90-100', '100-110', '110-120']
        )

        # Income Groups
        data['income_group'] = pd.cut(
            data['income'],
            bins=[0, 30000, 60000, 90000, 120000, 150000],
            labels=['0-30k', '30k-60k', '60k-90k', '90k-120k', '120k-150k']
        )

        visualizations = {
            'age_distribution': create_histogram(data, 'age_group', 'Age Distribution', 'Age Group'),
            'gender_distribution': create_histogram(data, 'gender', 'Gender Distribution', 'Gender'),
            'income_distribution': create_histogram(data, 'income_group', 'Income Distribution', 'Income Group'),
            'offer_completion_by_gender': create_histogram(data, 'gender', 'Offer Completion by Gender', 'Gender', color_col='will_complete_offer'),
            'offer_completion_by_age': create_histogram(data, 'age_group', 'Offer Completion by Age Group', 'Age Group', color_col='will_complete_offer'),
            'offer_completion_by_income': create_histogram(data, 'income_group', 'Offer Completion by Income Group', 'Income Group', color_col='will_complete_offer')
        }

        # Convert figures to JSON
        return {key: json.dumps(fig, cls=PlotlyJSONEncoder) for key, fig in visualizations.items()}

    except Exception as e:
        logger.error(
            f"Error generating visualizations: {str(e)}", exc_info=True)
        return {"error": str(e)}


def main():
    if len(sys.argv) == 2:  # Expecting script name + data path
        data_path = sys.argv[1]
        visualizations = generate_visualizations(data_path)
        for name, viz in visualizations.items():
            print(f"{name}: {viz[:200]}...")
    else:
        print("Usage: python visualization.py <data_path>\n"
              "Example: python visualization.py ../data/final_df.csv")


if __name__ == "__main__":
    main()
